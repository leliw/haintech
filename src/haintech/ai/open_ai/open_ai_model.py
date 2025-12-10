import json
import logging
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, override

from openai import AsyncOpenAI, OpenAI
from openai.types.chat import (
    ChatCompletionMessage,
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
)

from ..base import BaseAIModel
from ..model import (
    AIChatResponse,
    AIContext,
    AIModelInteraction,
    AIModelInteractionMessage,
    AIModelToolCall,
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

    def get_model_names(self) -> List[str]:
        return [m.id for m in self.openai.models.list().data] 
    
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
        if not isinstance(history, list):
            history = list(history or [])
        parameters, ai_model_interaction = self._prepare_parameters(
            system_prompt, history, context, message, functions, response_format
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
        system_prompt: Optional[str | AIPrompt] = None,
        history: Optional[Iterable[AIModelInteractionMessage]] = None,
        context: Optional[AIContext] = None,
        message: Optional[AIModelInteractionMessage] = None,
        functions: Optional[Dict[Callable, Any]] = None,
        interaction_logger: Optional[Callable[[AIModelInteraction], None]] = None,
        response_format: Literal["text", "json"] = "text",
    ) -> AIChatResponse:
        if not isinstance(history, list):
            history = list(history or [])
        parameters, ai_model_interaction = self._prepare_parameters(
            system_prompt, history, context, message, functions, response_format
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
        system_prompt: Optional[str | AIPrompt] = None,
        history: Optional[Iterable[AIModelInteractionMessage]] = None,
        context: Optional[AIContext] = None,
        message: Optional[AIModelInteractionMessage] = None,
        functions: Optional[Dict[Callable, Any]] = None,
        response_format: Literal["text", "json"] = "text",
    ):
        if not isinstance(history, list):
            history = list(history or [])
        if message:
            if isinstance(message, str):
                message = AIModelInteractionMessage(role="user", content=message)
        else:
            message = history.pop()
        msg_list = []
        if system_prompt:
            msg_list.append(
                self._create_message(
                    AIModelInteractionMessage(role="system", content=self._prompt_to_str(system_prompt))
                )
            )
        for m in history:
            msg_list.append(self._create_message(m))

        if context:
            msg_list.append(
                self._create_message(AIModelInteractionMessage(role="system", content=self._context_to_str(context)))
            )
        msg_list.append(self._create_message(message))
        if functions:
            tools = []
            for f in functions:
                definition = functions[f]
                tools.append(ChatCompletionToolParam(type="function", function=definition))
        else:
            tools = None
        if response_format == "json":
            response_format_dict = {"type": "json_object"}
        else:
            response_format_dict = {"type": "text"}

        ai_model_interaction = AIModelInteraction(
            model=self.model_name,
            message=message,
            prompt=system_prompt if isinstance(system_prompt, AIPrompt) else None,
            context=context,
            history=history,
            tools=tools,
            response_format=response_format_dict,
        )
        parameters_dict = self.parameters.get_for_model(self.model_name) if isinstance(self.parameters, OpenAIParameters) else {}
        return (
            {
                "model": self.model_name,
                "messages": msg_list,
                "tools": tools,
                "response_format": response_format_dict,
                **parameters_dict,
            },
            ai_model_interaction,
        )

    @classmethod
    def _create_message(cls, interaction_message: AIModelInteractionMessage) -> ChatCompletionMessageParam:
        ret = {
            "role": interaction_message.role,
            "content": interaction_message.content,
        }
        if interaction_message.tool_call_id:
            ret["tool_call_id"] = interaction_message.tool_call_id
        if interaction_message.tool_calls:
            ret["tool_calls"] = [
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
        cls._log.debug("Creating message: %s", interaction_message)
        return ret  # type: ignore

    @classmethod
    def _create_ai_chat_response(cls, m_resp: ChatCompletionMessage) -> AIChatResponse:
        if m_resp.tool_calls:
            tool_calls = [
                AIModelToolCall(
                    id=tool_call.id,
                    function_name=tool_call.function.name,  # type: ignore
                    arguments=json.loads(tool_call.function.arguments),  # type: ignore
                )
                for tool_call in m_resp.tool_calls
            ]
        else:
            tool_calls = None
        return AIChatResponse(content=m_resp.content, tool_calls=tool_calls)
