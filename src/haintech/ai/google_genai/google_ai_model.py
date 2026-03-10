import logging
import re
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, override

from google import genai
from google.genai.types import (
    AutomaticFunctionCallingConfig,
    Blob,
    Content,
    ContentOrDict,
    FunctionCall,
    FunctionDeclaration,
    FunctionResponse,
    GenerateContentConfig,
    GenerateContentResponse,
    GenerationConfig,
    Part,
    Schema,
    Tool,
    ToolListUnion,
    Type,
)

from haintech.ai.exceptions import UnsupportedMimeTypeError

from ..base import BaseAIModel
from ..model import (
    AIChatResponse,
    AIContext,
    AIFunction,
    AIModelInteraction,
    AIModelInteractionMessage,
    AIModelToolCall,
    AIPrompt,
    RAGItem,
)

_log = logging.getLogger(__name__)


class GoogleAIParameters(GenerationConfig):
    pass


class GoogleAIModel(BaseAIModel):
    """Google AI implementation of BaseAIModel"""

    _configured = False

    def __init__(
        self,
        model_name: str = "gemini-2.5-flash",
        parameters: Optional[GenerationConfig | Dict[str, Any]] = None,
        api_key: Optional[str] = None,
    ):
        if api_key:
            self.setup(api_key=api_key)
        self.model_name = model_name
        if not parameters:
            self.parameters = GenerationConfig()
        elif isinstance(parameters, GenerationConfig):
            self.parameters = parameters
        else:
            self.parameters = GenerationConfig.model_validate(parameters)

    @classmethod
    def setup(cls, api_key: Optional[str] = None):
        if not cls._configured:
            cls.client = genai.Client(api_key=api_key)
            cls._configured = True
            _log.debug("Google AI Model configured")

    @classmethod
    def get_model_names(cls) -> List[str]:
        if not cls._configured:
            cls.setup()
        ret = []
        for m in cls.client.models.list():
            if m.supported_actions and "generateContent" in m.supported_actions:
                if m.name:
                    model_id = m.name[7:] if m.name.startswith("models/") else m.name
                    ret.append(model_id)
        return ret

    @override
    def get_chat_response(
        self,
        system_prompt: str | None = None,
        history: Optional[Iterable[AIModelInteractionMessage]] = None,
        context: Optional[AIContext] = None,
        message: Optional[AIModelInteractionMessage] = None,
        functions: Optional[Dict[Callable, Any]] = None,
        interaction_logger: Optional[Callable[[AIModelInteraction], None]] = None,
        response_format: Literal["text", "json"] = "text",
    ) -> AIChatResponse:
        history = list(history or [])
        parameters = self._prepare_parameters(system_prompt, history, context, message, functions, response_format)
        response = self._get_chat_response(**parameters)
        if interaction_logger:
            interaction_logger(
                AIModelInteraction(
                    model=self.model_name,
                    message=message,
                    context=context,
                    history=history,
                    response=response,
                )
            )
        return response

    @override
    async def get_chat_response_async(
        self,
        system_prompt: Optional[str | AIPrompt] = None,
        history: Optional[Iterable[AIModelInteractionMessage]] = None,
        context: Optional[AIContext] = None,
        message: Optional[AIModelInteractionMessage] = None,
        functions: Optional[Dict[Callable, Any]] = None,
        interaction_logger: Optional[Callable[[AIModelInteraction], None]] = None,
        response_format: Literal["text", "json"] = "text",
    ) -> AIChatResponse:
        history = list(history or [])
        parameters = self._prepare_parameters(system_prompt, history, context, message, functions, response_format)
        response = await self._get_chat_response_async(**parameters)
        if interaction_logger:
            interaction_logger(
                AIModelInteraction(
                    model=self.model_name,
                    message=message,
                    context=context,
                    history=history,
                    response=response,
                )
            )
        return response

    def _prepare_parameters(
        self,
        system_prompt: str | AIPrompt | None,
        history: List[AIModelInteractionMessage],
        context: AIContext | None,
        message: AIModelInteractionMessage | None,
        functions: Dict[Callable, Any] | None,
        response_format: Literal["text", "json"] = "text",
    ) -> Dict[str, Any]:
        if not self._configured:
            self.setup()

        msg_list: list[ContentOrDict] = []
        if not message:
            message = history[-1]
            msg_list = [self._create_content_from_message(m) for m in history[:-1]]
        else:
            msg_list = [self._create_content_from_message(m) for m in history]
            if isinstance(message, str):
                message = AIModelInteractionMessage(role="user", content=message)

        system_instructions = self._prompt_to_str(system_prompt) if system_prompt else ""
        if context:
            if system_instructions:
                system_instructions += "\n\n"
            system_instructions += self._context_to_str(context)

        tools: ToolListUnion | None = None
        if functions:
            tool = Tool(function_declarations=list(functions.values()))
            tools = [tool]
        config = GenerateContentConfig(
            **self.parameters.model_dump(exclude_none=True),
            system_instruction=system_instructions,
            response_mime_type="text/plain" if response_format == "text" else "application/json",
            tools=tools,
            automatic_function_calling=AutomaticFunctionCallingConfig(disable=True),
        )

        return {
            "config": config,
            "history": msg_list,
            "message": self._create_content_from_message(message).parts,
        }

    def _get_chat_response(
        self,
        config: GenerateContentConfig,
        history: list[ContentOrDict] | None,
        message: list[Part] | Part,
    ) -> AIChatResponse:
        try:
            chat = self.client.chats.create(model=self.model_name, config=config, history=history)
            native_response = chat.send_message(message)
            return self._create_response_from_content_response(native_response)
        except Exception as e:
            if "Unsupported MIME type" in str(e):
                raise UnsupportedMimeTypeError()
            else:
                raise e

    async def _get_chat_response_async(
        self,
        config: GenerateContentConfig,
        history: list[ContentOrDict] | None,
        message: list[Part] | Part,
    ) -> AIChatResponse:
        try:
            chat = self.client.aio.chats.create(model=self.model_name, config=config, history=history)
            native_response = await chat.send_message(message)
            return self._create_response_from_content_response(native_response)
        except Exception as e:
            if "Unsupported MIME type" in str(e):
                raise UnsupportedMimeTypeError()
            else:
                raise e

    @classmethod
    def _prompt_to_str(cls, prompt: str | AIPrompt) -> str:
        if isinstance(prompt, str):
            return prompt
        ret = ""
        if prompt.persona or prompt.objective:
            ret += f"<OBJECTIVE_AND_PERSONA>\n{prompt.persona} {prompt.objective}\n</OBJECTIVE_AND_PERSONA>\n"
        if prompt.instructions:
            ret += f"<INSTRUCTIONS>\n{prompt.instructions}\n</INSTRUCTIONS>\n"
        if prompt.constraints:
            ret += f"<CONSTRAINTS>\n{prompt.constraints}\n</CONSTRAINTS>\n"
        if prompt.context:
            ret += f"<CONTEXT>\n{prompt.context}\n</CONTEXT>\n"
        if prompt.documents:
            ret += "<DOCUMENTS>\n"
            for d in prompt.documents:
                if isinstance(d, RAGItem):
                    r: RAGItem = d
                    ret += f'<DOCUMENT title="{r.title}">\n{r.content}\n</DOCUMENT>\n'
                else:
                    ret += f"<DOCUMENT>\n{d}</DOCUMENT>\n"
            ret += "</DOCUMENTS>\n"
        if prompt.examples:
            ret += f"<FEW_SHOT_EXAMPLES>\n{prompt.examples}\n</FEW_SHOT_EXAMPLES>\n"
        if prompt.recap:
            ret += f"<RECAP>\n{prompt.recap}\n</RECAP>\n"
        return ret

    @classmethod
    def _create_content_from_message(cls, i_message: AIModelInteractionMessage) -> Content:
        """Converts AIModelInteractionMessage to protos.Content required by Google

        Args:
            i_message: AIModelInteractionMessage
        Returns:
            Content
        """
        if i_message.role == "tool":
            return cls._create_content_from_function_response(i_message)
        parts = [Part(text=i_message.content)] if i_message.content else []
        text_blob_contents = ""
        for blob in i_message.blobs or []:
            _log.warning("name=%s, type=%s", blob.name, blob.content_type)
            if blob.content_type and blob.content_type.startswith("text/"):
                text_blob_contents += f"\n<file {'name="' + blob.name + '"' if blob.name else ''}>\n"
                text_blob_contents += blob.content.decode("utf-8")
                text_blob_contents += "\n</file>\n"
            else:
                parts.append(Part(inline_data=Blob(data=blob.content, mime_type=blob.content_type)))
        if text_blob_contents:
            parts.append(Part(text=text_blob_contents))
        for f in cls._create_parts_from_tool_calls(i_message):
            parts.append(f)
        return Content(
            role="model" if i_message.role == "assistant" else "user",
            parts=parts,
        )

    @classmethod
    def _create_content_from_function_response(cls, i_message: AIModelInteractionMessage) -> Content:
        """Converts AIModelInteractionMessage with tool call result to protos.Content

        Args:
            i_message: AIModelInteractionMessage
        Returns:
            protos.Content
        """
        tool_call_id = i_message.tool_call_id
        if tool_call_id:
            # Jeśli kończy się na __liczba, wydziel nazwę
            match = re.match(r"^(.*)__\d+$", tool_call_id)
            name = match.group(1) if match else tool_call_id
        else:
            name = None
        return Content(
            role="user",
            parts=[
                Part(
                    function_response=FunctionResponse(
                        id=tool_call_id,
                        name=name,
                        response={"response": i_message.content},
                    )
                )
            ],
        )

    @classmethod
    def _create_parts_from_tool_calls(cls, i_message: AIModelInteractionMessage) -> Iterable[Part]:
        """Converts AIModelInteractionMessage with tool call to protos.Part

        Args:
            i_message: AIModelInteractionMessage
        Returns:
            Iterable[Part]
        """
        if not i_message.tool_calls:
            return []
        return (
            Part(function_call=FunctionCall(name=tc.function_name, args=tc.arguments)) for tc in i_message.tool_calls
        )

    @classmethod
    @override
    def model_function_definition(cls, ai_function: AIFunction) -> FunctionDeclaration:
        parameters = Schema(type=Type.OBJECT, properties={}, required=[])
        assert parameters.properties is not None
        assert parameters.required is not None
        for param in ai_function.parameters:
            parameters.properties[param.name] = Schema(type=Type.STRING, description=param.description)
            if param.required:
                parameters.required.append(param.name)
        return FunctionDeclaration(
            name=ai_function.name,
            description=ai_function.description,
            parameters=parameters,
        )

    @classmethod
    def _create_response_from_content_response(cls, n_resp: GenerateContentResponse) -> AIChatResponse:
        """Converts protos.GenerateContentResponse to AIChatResponse

        Args:
            n_resp: protos.GenerateContentResponse
        Returns:
            AIChatResponse
        """
        tool_calls = []
        texts = []
        name_indices = {}
        assert n_resp.candidates
        assert n_resp.candidates[0].content
        assert n_resp.candidates[0].content.parts
        for part in n_resp.candidates[0].content.parts:
            if part.function_call:
                fc = part.function_call
                name = fc.name
                # If id is provided, use it
                if fc.id:
                    tool_id = fc.id
                # If not, use name and add numbering
                else:
                    idx = name_indices.get(name, 1)
                    tool_id = f"{name}__{idx}"
                    name_indices[name] = idx + 1
                tool_calls.append(
                    AIModelToolCall(
                        id=tool_id,
                        function_name=name,  # type: ignore
                        arguments=dict(fc.args),  # type: ignore
                    )
                )
            if part.text:
                texts.append(part.text)
        return AIChatResponse(
            content="\n".join(texts) or None,
            tool_calls=tool_calls or None,
        )

    try:
        from agents.mcp import MCPServer
        from mcp import Tool as MCPTool

        def prepare_mcp_tool_definition(self, tool: MCPTool) -> FunctionDeclaration:
            """Creates a FunctionDefinition from an MCP Tool.

            It can be overriden if other models expect different definition

            Args:
                tool: The MCP Tool to create the FunctionDefinition from.
            Returns:
                A FunctionDefinition object representing the tool.
            """
            parameters = Schema(type=Type.OBJECT, properties={}, required=[])
            assert parameters.properties is not None
            for param_name, param in tool.inputSchema["properties"].items():
                match param["type"]:
                    case "integer":
                        type = Type.INTEGER
                    case "boolean":
                        type = Type.BOOLEAN
                    case _:
                        type = Type.STRING
                parameters.properties[param_name] = Schema(type=type)
            parameters.required = tool.inputSchema["required"] if "required" in tool.inputSchema else []
            return FunctionDeclaration(
                name=tool.name,
                description=tool.description,
                parameters=parameters,
            )

            # return {
            #     "name": tool.name,
            #     "description": tool.description,
            #     "parameters": {
            #         "type": "object",
            #         "properties": ,
            #         "required": tool.inputSchema["required"],
            #     },
            # }

    except ImportError:
        pass
