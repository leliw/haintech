import json
import logging
from typing import Callable, Dict, Iterator, override

from openai import OpenAI
from openai.types.chat import (
    ChatCompletionMessage,
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
)
from openai.types.shared_params import FunctionDefinition

from ..base import BaseAIModel
from ..model import (
    AIChatResponse,
    AIChatResponseToolCall,
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
        parameters: OpenAIParameters | Dict[str, str | int | float] = None,
    ):
        self.openai = OpenAI()
        self.model_name = model_name
        if parameters and isinstance(parameters, dict):
            parameters = OpenAIParameters(**parameters)
        self.parameters = parameters or OpenAIParameters()

    @override
    def get_chat_response(
        self,
        message: str | AIModelInteractionMessage = None,
        context: str | AIPrompt = None,
        history: Iterator[AIModelInteractionMessage] = None,
        functions: Dict[str, FunctionDefinition] = None,
        interaction_logger: Callable[[AIModelInteraction], None] = None,
    ) -> AIChatResponse:
        history = history or []
        if not isinstance(history, list):
            history = list(history)
        msg_list = [self._create_message(m) for m in history]
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
        try:
            resp = self.openai.chat.completions.create(
                model=self.model_name,
                messages=msg_list,
                tools=tools,
                response_format={"type": "text"},
                **self.parameters.model_dump(),
            )
            response = self._create_ai_chat_response(resp.choices[0].message)
        except Exception as e:
            self._log.error("Error: %s", e)
            response = AIChatResponse(content=str(e))
        if interaction_logger:
            interaction_logger(
                AIModelInteraction(
                    model=self.model_name,
                    message=message,
                    context=context,
                    history=history,
                    tools=tools,
                    response_format={"type": "text"},
                    response=response,
                )
            )
        return response

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
                AIChatResponseToolCall(
                    id=tool_call.id,
                    function_name=tool_call.function.name,
                    arguments=json.loads(tool_call.function.arguments),
                )
                for tool_call in m_resp.tool_calls  # type: ignore
            ]
        else:
            tool_calls = None
        return AIChatResponse(content=m_resp.content, tool_calls=tool_calls)
