import json
import logging
from typing import Any, Callable, Dict, Iterable, Literal, Optional, override

from openai import AsyncOpenAI, OpenAI
from openai.types.chat import (
    ChatCompletionMessage,
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
)
from openai.types.shared_params import FunctionDefinition

from ..base import BaseAIModel
from ..model import (
    AIChatResponse,
    AIModelToolCall,
    AIModelInteraction,
    AIModelInteractionMessage,
    AIPrompt,
)
from .model import OpenAIParameters


class OpenAIModel(BaseAIModel):
    """OpenAI implementation of BaseAIModel"""

    _log = logging.getLogger(__name__)

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        parameters: Optional[OpenAIParameters | Dict[str, Any]] = None,
    ):
        self.openai = OpenAI()
        self.async_openai = AsyncOpenAI()
        self.model_name = model_name
        if parameters and isinstance(parameters, dict):
            parameters = OpenAIParameters(**parameters)
        self.parameters = parameters or OpenAIParameters()

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
            history, prompt, context, message, functions, response_format
        )
        try:
            resp = self.openai.chat.completions.create(**parameters)
            response = self._create_ai_chat_response(resp.choices[0].message)
            return response
        except Exception as e:
            self._log.error("Error: %s", e)
            response = AIChatResponse(content=str(e))
            return response
        finally:
            if interaction_logger:
                ai_model_interaction.response = response
                interaction_logger(ai_model_interaction)

    @override
    async def get_chat_response_async(
        self,
        message: Optional[str | AIModelInteractionMessage] = None,
        prompt: Optional[AIPrompt] = None,
        context: Optional[str | AIPrompt] = None,
        history: Optional[Iterable[AIModelInteractionMessage]] = None,
        functions: Optional[Dict[str, FunctionDefinition]] = None,
        interaction_logger: Optional[Callable[[AIModelInteraction], None]] = None,
        response_format: Literal["text", "json"] = "text",
    ) -> AIChatResponse:
        if not isinstance(history, list):
            history = list(history or [])
        parameters, ai_model_interaction = self._prepare_parameters(
            history, prompt, context, message, functions, response_format
        )
        try:
            resp = await self.async_openai.chat.completions.create(**parameters)
            response = self._create_ai_chat_response(resp.choices[0].message)
            return response
        except Exception as e:
            self._log.error("Error: %s", e)
            response = AIChatResponse(content=str(e))
            return response
        finally:
            if interaction_logger:
                ai_model_interaction.response = response
                interaction_logger(ai_model_interaction)

    def _prepare_parameters(
        self,
        history: Optional[Iterable[AIModelInteractionMessage]] = None,
        prompt: Optional[AIPrompt] = None,
        context: Optional[str | AIPrompt] = None,
        message: Optional[str | AIModelInteractionMessage] = None,
        functions: Optional[Dict[str, FunctionDefinition]] = None,
        response_format: Literal["text", "json"] = "text",
    ):
        msg_list = [self._create_message(m) for m in history or []]
        if prompt:
            context = self._prompt_to_str(prompt)
        else:
            context = self._prompt_to_str(context) if context else None
        if context:
            msg_list.insert(
                0,
                self._create_message(
                    AIModelInteractionMessage(
                        role="system", content=self._prompt_to_str(context)
                    )
                ),
            )
        if message:
            if isinstance(message, str):
                message = AIModelInteractionMessage(role="user", content=message)
            msg_list.append(self._create_message(message))
        else:
            message = history.pop()
        if functions:
            tools = []
            for f in functions:
                definition = functions[f]
                tools.append(
                    ChatCompletionToolParam(type="function", function=definition)
                )
        else:
            tools = None
        if response_format == "json":
            response_format = {"type": "json_object"}
        else:
            response_format = {"type": "text"}

        ai_model_interaction = AIModelInteraction(
            model=self.model_name,
            message=message,
            prompt=prompt,
            context=context if not prompt else None,
            history=history,
            tools=tools,
            response_format=response_format,
        )
        return (
            {
                "model": self.model_name,
                "messages": msg_list,
                "tools": tools,
                "response_format": response_format,
                **self.parameters.model_dump(),
            },
            ai_model_interaction,
        )

    @classmethod
    def _create_message(
        cls, interaction_message: AIModelInteractionMessage
    ) -> ChatCompletionMessageParam:
        ret = {
            "role": interaction_message.role,
            "content": interaction_message.content,
            "tool_call_id": interaction_message.tool_call_id,
            "tool_calls": (
                [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.function_name,
                            "arguments": json.dumps(tool_call.arguments),
                        },
                    }
                    for tool_call in interaction_message.tool_calls  # type: ignore
                ]
                if interaction_message.tool_calls
                else None
            ),
        }
        cls._log.debug("Creating message: %s", interaction_message)
        return ret

    @classmethod
    def _create_ai_chat_response(cls, m_resp: ChatCompletionMessage) -> AIChatResponse:
        if m_resp.tool_calls:
            tool_calls = [
                AIModelToolCall(
                    id=tool_call.id,
                    function_name=tool_call.function.name,
                    arguments=json.loads(tool_call.function.arguments),
                )
                for tool_call in m_resp.tool_calls  # type: ignore
            ]
        else:
            tool_calls = None
        return AIChatResponse(content=m_resp.content, tool_calls=tool_calls)
