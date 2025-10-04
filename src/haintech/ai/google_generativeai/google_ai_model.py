import logging
import re
from itertools import chain
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, override

import google.generativeai as genai
from google.generativeai import protos
from google.generativeai.client import configure as genai_configure
from google.generativeai.generative_models import GenerativeModel
from google.generativeai.types import GenerationConfig, generation_types

from haintech.ai.model import AIModelToolCall, AIFunction, AIPrompt, RAGItem

from ..base import BaseAIModel
from ..model import (
    AIChatResponse,
    AIContext,
    AIModelInteraction,
    AIModelInteractionMessage,
)


class GoogleAIModel(BaseAIModel):
    """Google AI implementation of BaseAIModel"""

    _log = logging.getLogger(__name__)

    def __init__(
        self,
        model_name: str = "gemini-2.5-flash",
        parameters: Optional[GenerationConfig | Dict[str, Any]] = None,
        api_key: Optional[str] = None,
    ):
        if api_key:
            genai_configure(api_key=api_key)
        self.model_name = model_name
        self.parameters = parameters

    def get_model_names(self) -> List[str]:
        return [m.name for m in genai.list_models() if "generateContent" in m.supported_generation_methods]  # type: ignore

    @override
    def get_chat_response(
        self,
        system_prompt: Optional[str | AIPrompt] = None,
        history: Optional[Iterable[AIModelInteractionMessage]] = None,
        context: Optional[AIContext] = None,
        message: Optional[AIModelInteractionMessage] = None,
        functions: Optional[Dict[Callable, Any]] = None,
        interaction_logger: Optional[Callable[[AIModelInteraction], None]] = None,
        response_format: Literal["text", "json"] = "text",
    ) -> AIChatResponse:
        history = history or []
        if not isinstance(history, list):
            history = list(history)
        if not message:
            message = history.pop()
        elif isinstance(message, str):
            message = AIModelInteractionMessage(role="user", content=message)
        msg_list = (self._create_content_from_message(m) for m in history)
        if system_prompt:
            msg = AIModelInteractionMessage(role="system", content=self._prompt_to_str(system_prompt))
            msg_list = chain([self._create_content_from_message(msg)], msg_list)
        if context:
            msg = AIModelInteractionMessage(role="system", content=self._context_to_str(context))
            msg_list = chain(msg_list, [self._create_content_from_message(msg)])

        model = GenerativeModel(
            model_name=self.model_name,
            generation_config=self.parameters,  # type: ignore
            # system_instruction=context,
        )
        generation_config = GenerationConfig(
            response_mime_type="text/plain" if response_format == "text" else "application/json"
        )
        chat = model.start_chat(history=msg_list)
        native_response = chat.send_message(
            self._create_content_from_message(message),
            generation_config=generation_config,
            tools=functions.values() if functions else None,
        )
        response = self._create_response_from_content_response(native_response)
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
                    ret += f'<DOCUMENT title="{r.title}">\n{r.content}\n<DOCUMENT>\n'
                else:
                    ret += f"<DOCUMENT>\n{d}</DOCUMENT>\n"
            ret += f"{prompt.documents}\n"
            ret += "</DOCUMENTS>\n"
        if prompt.examples:
            ret += f"<FEW_SHOT_EXAMPLES>\n{prompt.examples}\n</FEW_SHOT_EXAMPLES>\n"
        if prompt.recap:
            ret += f"<RECAP>\n{prompt.recap}\n</RECAP>\n"
        return ret


    @classmethod
    def _create_content_from_message(cls, i_message: AIModelInteractionMessage) -> protos.Content:
        """Converts AIModelInteractionMessage to protos.Content required by Google

        Args:
            i_message: AIModelInteractionMessage
        Returns:
            protos.Content
        """
        if i_message.role == "tool":
            return cls._create_content_from_function_response(i_message)
        parts = [protos.Part(text=i_message.content)] if i_message.content else []
        for f in cls._create_parts_from_tool_calls(i_message):
            parts.append(f)
        return protos.Content(
            role="model" if i_message.role == "assistant" else "user",
            parts=parts,
        )

    @classmethod
    def _create_content_from_function_response(cls, i_message: AIModelInteractionMessage) -> protos.Content:
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
        return protos.Content(
            role="user",
            parts=[
                protos.Part(
                    function_response=protos.FunctionResponse(
                        id=tool_call_id,
                        name=name,
                        response={"response": i_message.content},
                    )
                )
            ],
        )

    @classmethod
    def _create_parts_from_tool_calls(cls, i_message: AIModelInteractionMessage) -> Iterable[protos.Part]:
        """Converts AIModelInteractionMessage with tool call to protos.Part

        Args:
            i_message: AIModelInteractionMessage
        Returns:
            Iterable[protos.Part]
        """
        if not i_message.tool_calls:
            return []
        return (
            protos.Part(function_call=protos.FunctionCall(name=tc.function_name, args=tc.arguments))
            for tc in i_message.tool_calls
        )

    @classmethod
    @override
    def model_function_definition(cls, ai_function: AIFunction) -> protos.FunctionDeclaration:
        parameters = protos.Schema(type=protos.Type.OBJECT, properties={}, required=[])
        for param in ai_function.parameters:
            parameters.properties[param.name] = protos.Schema(type=protos.Type.STRING, description=param.description)
            if param.required:
                parameters.required.append(param.name)
        return protos.FunctionDeclaration(
            name=ai_function.name,
            description=ai_function.description,
            parameters=parameters,
        )

    @classmethod
    def _create_response_from_content_response(cls, n_resp: generation_types.GenerateContentResponse) -> AIChatResponse:
        """Converts protos.GenerateContentResponse to AIChatResponse

        Args:
            n_resp: protos.GenerateContentResponse
        Returns:
            AIChatResponse
        """
        tool_calls = []
        texts = []
        name_indices = {}
        for part in n_resp.parts:
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
                        function_name=name,
                        arguments=dict(fc.args),
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

        def prepare_mcp_tool_definition(self, tool: MCPTool) -> protos.FunctionDeclaration:
            """Creates a FunctionDefinition from an MCP Tool.

            It can be overriden if other models expect different definition

            Args:
                tool: The MCP Tool to create the FunctionDefinition from.
            Returns:
                A FunctionDefinition object representing the tool.
            """
            parameters = protos.Schema(type=protos.Type.OBJECT, properties={}, required=[])
            for param_name, param in tool.inputSchema["properties"].items():
                match param["type"]:
                    case "integer":
                        type = protos.Type.INTEGER
                    case "boolean":
                        type = protos.Type.BOOLEAN
                    case _:
                        type = protos.Type.STRING
                parameters.properties[param_name] = protos.Schema(type=type)
            parameters.required = tool.inputSchema["required"]
            return protos.FunctionDeclaration(
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
