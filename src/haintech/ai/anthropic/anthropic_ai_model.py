import json
import logging
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, override

import anthropic
from pydantic import BaseModel

from haintech.ai import (
    AIChatResponse,
    AIModelInteraction,
    AIModelInteractionMessage,
    AIPrompt,
    BaseAIModel,
)
from haintech.ai.model import AIChatResponseToolCall, AIFunction, AIModelInteractionTool


class AnthropicAIModel(BaseAIModel):
    _log = logging.getLogger(__name__)

    def __init__(
        self,
        model_name: str = "claude-sonnet-4-20250514",
        parameters: Optional[Dict[str, str | int | float]] = None,
    ):
        try:
            import anthropic

            self.NOT_GIVEN = anthropic.NOT_GIVEN

            self.client = anthropic.Anthropic()
            self.async_client = anthropic.AsyncAnthropic()
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
        parameters, ai_model_interaction = self._prepare_parameters(
            message, prompt, context, history, functions, response_format
        )
        try:
            resp: anthropic.types.Message = self.client.messages.create(**parameters)
            response = self._create_ai_chat_response(resp.content)  # type: ignore
        except Exception as e:
            self._log.error("Error: %s", e)
            response = AIChatResponse(content=str(e))
        if interaction_logger:
            ai_model_interaction.response = response
            interaction_logger(ai_model_interaction)
        return response

    @override
    async def get_chat_response_async(
        self,
        message: Optional[AIModelInteractionMessage] = None,
        prompt: Optional[str | AIPrompt] = None,
        context: Optional[str | AIPrompt] = None,
        history: Optional[Iterable[AIModelInteractionMessage]] = None,
        functions: Optional[Dict[Callable, Any]] = None,
        interaction_logger: Optional[Callable[[AIModelInteraction], None]] = None,
        response_format: Literal["text", "json"] = "text",
    ) -> AIChatResponse:
        parameters, ai_model_interaction = self._prepare_parameters(
            message, prompt, context, history, functions, response_format
        )
        try:
            resp: anthropic.types.Message = await self.async_client.messages.create(
                **parameters
            )
            response = self._create_ai_chat_response(resp.content)  # type: ignore
        except Exception as e:
            self._log.error("Error: %s", e)
            response = AIChatResponse(content=str(e))
        if interaction_logger:
            ai_model_interaction.response = response
            interaction_logger(ai_model_interaction)
        return response

    def _prepare_parameters(
        self,
        message: Optional[AIModelInteractionMessage] = None,
        prompt: Optional[str | AIPrompt] = None,
        context: Optional[str | AIPrompt] = None,
        history: Optional[Iterable[AIModelInteractionMessage]] = None,
        functions: Optional[Dict[Callable, Any]] = None,
        response_format: Literal["text", "json"] = "text",
    ):
        if not isinstance(history, list):
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
                tools.append(definition)
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
            tools=[
                AIModelInteractionTool(type="function", function=tool) for tool in tools
            ],
            response_format=response_format_param,
        )
        system_prompt = (
            self._prompt_to_str(prompt) if prompt else self._prompt_to_str(context)
        )
        return (
            {
                "model": self.model_name,
                "system": system_prompt or anthropic.NOT_GIVEN,
                "messages": msg_list,
                "tools": tools,
            }
            | (self.parameters or {}),  # Unpack parameters if not None
            ai_model_interaction,
        )

    @classmethod
    def _create_message(
        cls, interaction_message: AIModelInteractionMessage
    ) -> Dict[str, Any]:
        if interaction_message.tool_call_id:
            ret = {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": interaction_message.tool_call_id,
                        "content": interaction_message.content,
                    }
                ],
            }
        else:
            role = (
                interaction_message.role
                if interaction_message.role != "tool"
                else "user"
            )
            content = []
            if interaction_message.content:
                content.append(
                    {
                        "type": "text",
                        "text": interaction_message.content,
                    }
                )
            if interaction_message.tool_calls:
                for t in interaction_message.tool_calls:
                    content.append(
                        {
                            "type": "tool_use",
                            "id": t.id,
                            "name": t.function_name,
                            "input": t.arguments,
                        }
                    )
            ret = {"role": role, "content": content}
        cls._log.debug("Creating message: %s", interaction_message)
        return ret

    @classmethod
    def _create_ai_chat_response(cls, lm_resp: List[BaseModel]) -> AIChatResponse:
        content = None
        tool_calls = []
        for m_resp in lm_resp:
            m_resp = m_resp.model_dump()
            if "text" in m_resp:
                content = m_resp["text"]
            elif "id" in m_resp and "name" in m_resp and "input" in m_resp:
                tool_calls.append(
                    AIChatResponseToolCall(
                        id=m_resp["id"],
                        function_name=m_resp["name"],
                        arguments=m_resp["input"],
                    )
                )
        return AIChatResponse(content=content, tool_calls=tool_calls)

    @classmethod
    def model_function_definition(cls, ai_function: AIFunction) -> Dict[str, Any]:
        parameters: Dict[str, Any] = {
            "type": "object",
            "properties": {},
            "required": [],
        }
        for param in ai_function.parameters:
            param_name = param.name
            param_description = param.description
            parameters["properties"][param_name] = {
                "type": "string",
                "description": param_description,
            }
            if param.required:
                parameters["required"].append(param_name)

        parameters["additionalProperties"] = (
            False  # Ensure no extra properties are allowed
        )

        return {
            "name": ai_function.name,
            "description": ai_function.description,
            "input_schema": parameters,
        }
