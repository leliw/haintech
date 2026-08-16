import base64
import logging
import re
from collections.abc import Callable, Iterable, Sequence
from typing import Any, Literal, override

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
    Model,
    Part,
    Schema,
    Tool,
    ToolListUnion,
)
from pydantic import BaseModel

from haintech.ai.exceptions import UnsupportedMimeTypeError
from haintech.helpers import get_inner_type, is_list_type

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

    _api_key: str | None = None
    _client: genai.Client | None = None
    _models_list: list[Model] | None = None

    @classmethod
    def get_vendor_name(cls) -> str:
        return "google"

    @classmethod
    def setup(cls, api_key: str | None = None):
        cls._api_key = api_key

    def __init__(
        self,
        model_name: str = "gemini-3.5-flash-lite",
        parameters: GenerationConfig | dict[str, Any] | None = None,
        api_key: str | None = None,
    ):
        if api_key:
            self._api_key = api_key
        self.model_name = model_name
        if not parameters:
            self.parameters = GenerationConfig()
        elif isinstance(parameters, GenerationConfig):
            self.parameters = parameters
        else:
            self.parameters = GenerationConfig.model_validate(parameters)

    @classmethod
    def get_client(cls) -> genai.Client:
        if not cls._client:
            cls._client = genai.Client(api_key=cls._api_key)
        return cls._client

    @override
    @classmethod
    def get_model_names(cls, task: str = "chat") -> list[str]:
        if not cls._models_list:
            cls._models_list = list(cls.get_client().models.list())
        ret = []
        for m in cls._models_list:
            if not m.supported_actions or "generateContent" not in m.supported_actions or not m.name:
                continue

            name = m.name.removeprefix("models/")

            # Filter out snapshots, experimental, and obsolete models
            if (
                "exp" in name
                or name.endswith(("-001", "-latest"))
                or name.startswith(("gemini-robotics", "gemini-2.0-flash-lite-preview"))
                or cls._ends_with_month_year(name)
            ):
                continue

            # Filter by task capability
            if task == "chat":
                if any(k in name for k in ("tts", "gemma", "image")):
                    continue
                ret.append(name)
            elif task == "image" and "image" in name or task == "embedding" and "embedding" in name:
                ret.append(name)
        return ret

    @override
    @classmethod
    async def get_model_names_async(cls, task: str = "chat") -> list[str]:
        if not cls._models_list:
            cls._models_list = []
            async for model in await cls.get_client().aio.models.list():
                cls._models_list.append(model)
        return cls.get_model_names(task)

    @classmethod
    def _ends_with_month_year(cls, s: str) -> bool:
        pattern = r"-\d{2}-\d{4}$"
        return bool(re.search(pattern, s))

    @override
    def _prepare_response_format(
        self,
        response_format: Literal["text", "json"]
        | type[Sequence[BaseModel | str | int | float | bool]]
        | type[BaseModel] = "text",
    ) -> dict:
        if response_format == "text":
            ret = {"type": "text"}
        elif response_format == "json":
            ret = {"type": "json_object"}
        elif isinstance(response_format, type) and issubclass(response_format, BaseModel):
            ret = response_format.model_json_schema()
        elif is_list_type(response_format):
            inner_type = get_inner_type(response_format)
            if inner_type is str:
                inner_ret = {"type": "array", "items": {"type": "string"}}
            elif inner_type is int:
                inner_ret = {"type": "array", "items": {"type": "integer"}}
            elif inner_type is float:
                inner_ret = {"type": "array", "items": {"type": "number"}}
            elif inner_type is bool:
                inner_ret = {"type": "array", "items": {"type": "boolean"}}
            elif issubclass(inner_type, BaseModel):
                inner_ret = {"type": "array", "items": inner_type.model_json_schema()}
            else:
                raise ValueError(f"Unsupported inner response format: {inner_type}")
            ret = {
                "type": "object",
                "properties": {"list": inner_ret},
                "additionalProperties": False,
            }
        else:
            raise ValueError(f"Unsupported response format: {response_format}")
        return ret

    @override
    def get_chat_response(
        self,
        system_prompt: str | AIPrompt | None = None,
        history: Iterable[AIModelInteractionMessage] | None = None,
        context: AIContext | None = None,
        message: AIModelInteractionMessage | None = None,
        functions: dict[Callable, Any] | None = None,
        interaction_logger: Callable[[AIModelInteraction], None] | None = None,
        response_format: Literal["text", "json"] | dict = "text",
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
        system_prompt: str | AIPrompt | None = None,
        history: Iterable[AIModelInteractionMessage] | None = None,
        context: AIContext | None = None,
        message: AIModelInteractionMessage | None = None,
        functions: dict[Callable, Any] | None = None,
        interaction_logger: Callable[[AIModelInteraction], None] | None = None,
        response_format: Literal["text", "json"] | dict = "text",
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
        history: list[AIModelInteractionMessage],
        context: AIContext | None,
        message: AIModelInteractionMessage | None,
        functions: dict[Callable, Any] | None,
        response_format: Literal["text", "json"] | dict = "text",
    ) -> dict[str, Any]:
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
            response_json_schema=response_format if isinstance(response_format, dict) else None,
            tools=tools,
            automatic_function_calling=AutomaticFunctionCallingConfig(disable=True),
        )

        ret = {
            "config": config,
            "history": msg_list,
            "message": self._create_content_from_message(message).parts,
        }
        _log.debug("===========>\n %s\n <==========", ret)
        return ret

    def _get_chat_response(
        self,
        config: GenerateContentConfig,
        history: list[ContentOrDict] | None,
        message: list[Part] | Part,
    ) -> AIChatResponse:
        try:
            chat = self.get_client().chats.create(model=self.model_name, config=config, history=history)
            native_response = chat.send_message(message)
            return self._create_response_from_content_response(native_response)
        except Exception as e:
            if "Unsupported MIME type" in str(e):
                raise UnsupportedMimeTypeError()
            else:
                raise

    async def _get_chat_response_async(
        self,
        config: GenerateContentConfig,
        history: list[ContentOrDict] | None,
        message: list[Part] | Part,
    ) -> AIChatResponse:
        try:
            chat = self.get_client().aio.chats.create(model=self.model_name, config=config, history=history)
            native_response = await chat.send_message(message)
            return self._create_response_from_content_response(native_response)
        except Exception as e:
            if "Unsupported MIME type" in str(e):
                raise UnsupportedMimeTypeError()
            else:
                raise

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
        parts = []
        text_blob_contents = ""
        for blob in i_message.blobs or []:
            _log.debug("name=%s, type=%s", blob.name, blob.content_type)
            is_text = (blob.content_type and blob.content_type.startswith("text/")) or (
                blob.content and b"\x00" not in blob.content
            )
            if is_text:
                text_blob_contents += f"\n<file {'name="' + blob.name + '"' if blob.name else ''}>\n"
                text_blob_contents += blob.content.decode("utf-8")
                text_blob_contents += "\n</file>\n"
            else:
                parts.append(Part(inline_data=Blob(data=blob.content, mime_type=blob.content_type)))
        parts.extend(cls._create_parts_from_tool_calls(i_message))
        if i_message.content:
            if text_blob_contents:
                text_blob_contents += "\n"
            text_blob_contents += i_message.content
        if text_blob_contents:
            parts.append(Part(text=text_blob_contents))
            # parts[0].text += text_blob_contents
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
            Part(
                function_call=FunctionCall(name=tc.function_name, args=tc.arguments),
                thought_signature=base64.b64decode(tc.thought_signature) if tc.thought_signature else None,
            )
            for tc in i_message.tool_calls
        )

    @classmethod
    @override
    def model_function_definition(cls, ai_function: AIFunction) -> FunctionDeclaration:
        parameters = Schema(type=genai.types.Type.OBJECT, properties={}, required=[])
        assert parameters.properties is not None
        assert parameters.required is not None
        for param in ai_function.parameters:
            parameters.properties[param.name] = Schema(type=genai.types.Type.STRING, description=param.description)
            if param.required:
                parameters.required.append(param.name)
        return FunctionDeclaration(
            name=ai_function.name,
            description=ai_function.description,
            parameters=parameters,
        )

    def _create_response_from_content_response(self, n_resp: GenerateContentResponse) -> AIChatResponse:
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
                        thought_signature=base64.b64encode(part.thought_signature).decode("utf-8")
                        if part.thought_signature
                        else None,
                    )
                )
            if part.text:
                texts.append(part.text)
        usage = n_resp.usage_metadata
        return AIChatResponse(
            content="\n".join(texts) or None,
            tool_calls=tool_calls or None,
            vendor_model_name=self.get_vendor_model_name(),
            input_tokens=usage.prompt_token_count if usage else None,
            input_tokens_cached=usage.cached_content_token_count if usage else None,
            reasoning_tokens=usage.thoughts_token_count if usage else None,
            tool_use_tokens= usage.tool_use_prompt_token_count if usage else None,
            output_tokens= usage.candidates_token_count if usage else None,

        )

    try:
        from agents.mcp import MCPServer
        from mcp import Tool as MCPTool

        def prepare_mcp_tool_definition(self, tool: MCPTool) -> FunctionDeclaration:
            """Creates a FunctionDefinition from an MCP Tool.

            It can be overridden if other models expect different definition

            Args:
                tool: The MCP Tool to create the FunctionDefinition from.
            Returns:
                A FunctionDefinition object representing the tool.
            """
            parameters = Schema(type=genai.types.Type.OBJECT, properties={}, required=[])
            assert parameters.properties is not None
            for param_name, param in tool.input_schema.get("properties", {}).items():
                match param["type"]:
                    case "integer":
                        param_type = genai.types.Type.INTEGER
                    case "boolean":
                        param_type = genai.types.Type.BOOLEAN
                    case _:
                        param_type = genai.types.Type.STRING
                parameters.properties[param_name] = Schema(type=param_type)
            parameters.required = tool.input_schema.get("required", [])
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
