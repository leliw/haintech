import json
import logging
from typing import Any, Callable, Dict, Iterable, Literal, Optional, override

from openai.types.chat import (
    ChatCompletionMessage,
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
)

from haintech.ai import (
    AIChatResponse,
    AIChatResponseToolCall,
    AIModelInteraction,
    AIModelInteractionMessage,
    AIPrompt,
    BaseAIModel,
)


class AntrhopicAIModel(BaseAIModel):
    _log = logging.getLogger(__name__)

    def __init__(
        self,
        model_name: str = "claude-sonnet-4-20250514",
        parameters: Optional[Dict[str, str | int | float]] = None,
    ):
        try:
            import anthropic

            self.client = anthropic.Anthropic()
            self.async_client = anthropic.AsyncAnthropic()
            self.not_given = anthropic.NOT_GIVEN
            self.model_name = model_name
            self.parameters = parameters or {}
            if "max_tokens" not in self.parameters:
                self.parameters["max_tokens"] = 1000
        except ImportError as e:
            raise ImportError(
                "The 'anthropic' library is not installed. Please install it to use AntrhopicAIModel (e.g., `pip install haintech[anthropic]`)."
            ) from e


    @override
    def get_chat_response(
        self,
        message: Optional[AIModelInteractionMessage] = None,
        prompt: Optional[str | AIPrompt] = None,
        context: Optional[str | AIPrompt] = None,
        history: Optional[Iterable[AIModelInteractionMessage]] = None,
        functions: Optional[Dict[Callable, Any]] = None,
        interaction_logger: Optional[Callable[[AIModelInteraction], None]] = None,
        response_format: Literal["text", "json"] = "text",
    ) -> AIChatResponse:
        if not isinstance(history, list):
            history = list(history or [])
        parameters, ai_model_interaction = self._prepare_parameters(
            message, prompt, context, history, functions, response_format
        )
        try:
            resp = self.client.messages.create(**parameters)
            response = self._create_ai_chat_response(resp.content[0])
            return response
        except Exception as e:
            self._log.error("Error: %s", e)
            response = AIChatResponse(content=str(e))
        finally:
            if interaction_logger:
                ai_model_interaction.response = response
                interaction_logger(ai_model_interaction)

    def _prepare_parameters(
        self,
        message: Optional[AIModelInteractionMessage] = None,
        prompt: Optional[str | AIPrompt] = None,
        context: Optional[str | AIPrompt] = None,
        history: Optional[Iterable[AIModelInteractionMessage]] = None,
        functions: Optional[Dict[Callable, Any]] = None,
        response_format: Literal["text", "json"] = "text",
    ):
        history = list(history or [])
        msg_list = [self._create_message(m) for m in history]
        if prompt:
            context = self._prompt_to_str(prompt)
        else:
            context = self._prompt_to_str(context) if context else None
        if message:
            if isinstance(message, str):
                message = AIModelInteractionMessage(role="user", content=message)
            msg_list.append(self._create_message(message))
        elif history:
            message = history.pop()
        else:
            raise ValueError("No message provided")
        tools = []
        if functions:
            for f in functions:
                definition = functions[f]
                tools.append(
                    ChatCompletionToolParam(type="function", function=definition)
                )
        if response_format == "json":
            response_format_param = {"type": "json_object"}
        else:
            response_format_param = {"type": "text"}

        ai_model_interaction = AIModelInteraction(
            model=self.model_name,
            message=message,
            prompt=prompt,
            context=context if not prompt else None,
            history=history,
            tools=tools,
            response_format=response_format_param,
        )
        system_prompt = self._prompt_to_str(prompt) if prompt else self._prompt_to_str(context)
        return (
            {
                "model": self.model_name,
                "system": system_prompt or self.not_given,
                "messages": msg_list,
                "tools": tools,
            }
            | (self.parameters or {}),  # Unpack parameters if not None
            ai_model_interaction,
        )

    @classmethod
    def _create_message(
        cls, interaction_message: AIModelInteractionMessage
    ) -> ChatCompletionMessageParam:
        ret = {
            "role": interaction_message.role,
            "content": interaction_message.content,
            # "tool_call_id": interaction_message.tool_call_id,
            # "tool_calls": (
            #     [
            #         {
            #             "id": tool_call.id,
            #             "type": "function",
            #             "function": {
            #                 "name": tool_call.function_name,
            #                 "arguments": json.dumps(tool_call.arguments),
            #             },
            #         }
            #         for tool_call in interaction_message.tool_calls  # type: ignore
            #     ]
            #     if interaction_message.tool_calls
            #     else None
            # ),
        }
        cls._log.debug("Creating message: %s", interaction_message)
        return ret

    @classmethod
    def _create_ai_chat_response(cls, m_resp: ChatCompletionMessage) -> AIChatResponse:
        # if m_resp.tool_calls:
        #     tool_calls = [
        #         AIChatResponseToolCall(
        #             id=tool_call.id,
        #             function_name=tool_call.function.name,
        #             arguments=json.loads(tool_call.function.arguments),
        #         )
        #         for tool_call in m_resp.tool_calls  # type: ignore
        #     ]
        # else:
        tool_calls = None
        return AIChatResponse(content=m_resp.text, tool_calls=tool_calls)
