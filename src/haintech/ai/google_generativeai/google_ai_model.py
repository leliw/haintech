import logging
from itertools import chain
from typing import Callable, Dict, Iterator, override

from google.generativeai import (
    GenerationConfig,
    GenerativeModel,
    protos,
)
from google.generativeai import (
    configure as genai_configure,
)
from google.generativeai.types import generation_types

from haintech.ai.model import AIChatResponseToolCall, AIFunction, AIPrompt, RAGItem

from ..base import BaseAIModel
from ..model import (
    AIChatResponse,
    AIModelInteraction,
    AIModelInteractionMessage,
)


class GoogleAIModel(BaseAIModel):
    """Google AI implementation of BaseAIModel"""

    _log = logging.getLogger(__name__)

    def __init__(
        self,
        model_name: str = "gemini-1.5-flash",
        parameters: GenerationConfig = None,
        api_key: str = None,
    ):
        if api_key:
            genai_configure(api_key=api_key)
        self.model_name = model_name
        self.parameters = parameters

    @override
    def get_chat_response(
        self,
        message: str | AIModelInteractionMessage = None,
        context: str | AIPrompt = None,
        history: Iterator[AIModelInteractionMessage] = None,
        functions: Dict[Callable, protos.FunctionDeclaration] = None,
        interaction_logger: Callable[[AIModelInteraction], None] = None,
    ) -> AIChatResponse:
        history = history or []
        if not isinstance(history, list):
            history = list(history)
        if not message:
            message = history.pop()
        elif isinstance(message, str):
            message = AIModelInteractionMessage(role="user", content=message)
        msg_list = (self._create_content_from_message(m) for m in history)
        context = self._prompt_to_str(context) if context else None
        if context:
            msg_ctx = AIModelInteractionMessage(role="system", content=context)
            msg_list = chain([self._create_content_from_message(msg_ctx)], msg_list)
        model = GenerativeModel(
            model_name=self.model_name,
            generation_config=self.parameters,
            # system_instruction=context,
        )
        chat = model.start_chat(history=msg_list)
        native_response = chat.send_message(
            self._create_content_from_message(message),
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
    def _create_content_from_message(
        cls, i_message: AIModelInteractionMessage
    ) -> protos.Content:
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
    def _create_content_from_function_response(
        cls, i_message: AIModelInteractionMessage
    ) -> protos.Content:
        """Converts AIModelInteractionMessage with tool call result to protos.Content

        Args:
            i_message: AIModelInteractionMessage
        Returns:
            protos.Content
        """
        return protos.Content(
            role="user",
            parts=[
                protos.Part(
                    function_response=protos.FunctionResponse(
                        name=i_message.tool_call_id,
                        response={"response": i_message.content},
                    )
                )
            ],
        )

    @classmethod
    def _create_parts_from_tool_calls(
        cls, i_message: AIModelInteractionMessage
    ) -> Iterator[protos.Part]:
        """Converts AIModelInteractionMessage with tool call to protos.Part

        Args:
            i_message: AIModelInteractionMessage
        Returns:
            Iterator[protos.Part]
        """
        if not i_message.tool_calls:
            return []
        return (
            protos.Part(
                function_call=protos.FunctionCall(
                    name=tc.function_name, args=tc.arguments
                )
            )
            for tc in i_message.tool_calls
        )

    @classmethod
    @override
    def model_function_definition(
        cls, ai_function: AIFunction
    ) -> protos.FunctionDeclaration:
        parameters = protos.Schema(type=protos.Type.OBJECT, properties={}, required=[])
        for param in ai_function.parameters:
            parameters.properties[param.name] = protos.Schema(
                type=protos.Type.STRING, description=param.description
            )
            if param.required:
                parameters.required.append(param.name)
        return protos.FunctionDeclaration(
            name=ai_function.name,
            description=ai_function.description,
            parameters=parameters,
        )

    @classmethod
    def _create_response_from_content_response(
        cls, n_resp: generation_types.GenerateContentResponse
    ) -> AIChatResponse:
        """Converts protos.GenerateContentResponse to AIChatResponse

        Args:
            n_resp: protos.GenerateContentResponse
        Returns:
            AIChatResponse
        """
        tool_calls = []
        texts = []
        for part in n_resp.parts:
            if part.function_call:
                tool_calls.append(
                    AIChatResponseToolCall(
                        id=part.function_call.name,
                        function_name=part.function_call.name,
                        arguments=dict(part.function_call.args),
                    )
                )
            if part.text:
                texts.append(part.text)
        return AIChatResponse(
            content="\n".join(texts) or None,
            tool_calls=tool_calls or None,
        )
