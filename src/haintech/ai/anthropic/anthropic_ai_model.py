import logging
from collections.abc import Callable, Iterable
from typing import Any, Literal, override

import anthropic
from anthropic.types import ModelInfo
from pydantic import BaseModel

from haintech.ai import (
    AIChatResponse,
    AIModelInteraction,
    AIModelInteractionMessage,
    AIPrompt,
    BaseAIModel,
)
from haintech.ai.model import AIContext, AIFunction, AIModelInteractionTool, AIModelToolCall

_log = logging.getLogger(__name__)


class AnthropicAIModel(BaseAIModel):
    _client: anthropic.Anthropic | None = None
    _async_client: anthropic.AsyncAnthropic | None = None

    _models: list[ModelInfo] | None = None

    @classmethod
    def get_vendor_name(cls) -> str:
        return "anthropic"

    @classmethod
    def get_client(cls) -> anthropic.Anthropic:
        if not cls._client:
            cls._client = anthropic.Anthropic()
        return cls._client

    @classmethod
    def get_async_client(cls) -> anthropic.AsyncAnthropic:
        if not cls._async_client:
            cls._async_client = anthropic.AsyncAnthropic()
        return cls._async_client

    def __init__(
        self,
        model_name: str = "claude-haiku-4-5-20251001",
        parameters: dict[str, str | int | float] | None = None,
    ):
        super().__init__(model_name)
        self.NOT_GIVEN = anthropic.NOT_GIVEN
        self.parameters = parameters or {}
        if "max_tokens" not in self.parameters:
            self.parameters["max_tokens"] = 1000

    @classmethod
    @override
    def get_model_names(cls, task: str = "chat") -> list[str]:
        if not cls._models:
            cls._models = list(cls.get_client().models.list())
        return [m.id for m in cls._models]

    @override
    def get_chat_response(
        self,
        system_prompt: str | AIPrompt | None = None,
        history: Iterable[AIModelInteractionMessage] | None = None,
        context: AIContext | None = None,
        message: AIModelInteractionMessage | None = None,
        functions: dict[Callable, Any] | None = None,
        interaction_logger: Callable[[AIModelInteraction], None] | None = None,
        response_format: Literal["text", "json"] = "text",
    ) -> AIChatResponse:
        if not isinstance(history, list):
            history = list(history or [])
        parameters, ai_model_interaction = self._prepare_parameters(
            system_prompt, history, context, message, functions, response_format
        )
        try:
            resp: anthropic.types.Message = self.get_client().messages.create(**parameters)  # type: ignore
            response = self._create_ai_chat_response(resp.content)  # type: ignore
        except Exception as e:  # noqa: BLE001
            _log.error("Error: %s", e)
            response = AIChatResponse(content=str(e))
        if interaction_logger:
            ai_model_interaction.response = response
            interaction_logger(ai_model_interaction)
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
        response_format: Literal["text", "json"] = "text",
    ) -> AIChatResponse:
        if not isinstance(history, list):
            history = list(history or [])
        parameters, ai_model_interaction = self._prepare_parameters(
            system_prompt, history, context, message, functions, response_format
        )
        try:
            resp: anthropic.types.Message = await self.get_async_client().messages.create(**parameters)  # type: ignore
            response = self._create_ai_chat_response(resp.content)  # type: ignore
        except Exception as e:  # noqa: BLE001
            _log.error("Error: %s", e)
            response = AIChatResponse(content=str(e))
        if interaction_logger:
            ai_model_interaction.response = response
            interaction_logger(ai_model_interaction)
        return response

    def _prepare_parameters(
        self,
        system_prompt: str | AIPrompt | None,
        history: list[AIModelInteractionMessage],
        context: AIContext | None = None,
        message: AIModelInteractionMessage | None = None,
        functions: dict[Callable, Any] | None = None,
        response_format: Literal["text", "json"] = "text",
    ):
        _log.debug("Preparing parameters for Anthropic model")
        if not message:
            if history:
                message = history[-1]
                msg_list = [self._create_message(m) for m in history[:-1]]
            else:
                raise ValueError("No message provided")
        else:
            msg_list = [self._create_message(m) for m in history]
        if isinstance(message, str):
            message = AIModelInteractionMessage(role="user", content=message)
        if message:
            msg_list.append(self._create_message(message, context))
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
            prompt=system_prompt,
            context=context,
            history=history,
            tools=[AIModelInteractionTool(type="function", function=tool) for tool in tools],
            response_format=response_format_param,
        )
        system = self._prompt_to_str(system_prompt) if isinstance(system_prompt, AIPrompt) else system_prompt
        params = {
            "model": self.model_name,
            "system": system or anthropic.NOT_GIVEN,
            "messages": msg_list,
        }
        if tools:
            params["tools"] = tools
        return (
            params | (self.parameters or {}),  # Unpack parameters if not None
            ai_model_interaction,
        )

    @classmethod
    def _create_message(
        cls, interaction_message: AIModelInteractionMessage, context: AIContext | None = None
    ) -> dict[str, Any]:
        _log.debug("Creating message: %s", interaction_message)
        if context:
            _log.debug("With context: %s", context)

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
            role = interaction_message.role if interaction_message.role != "tool" else "user"
            content = []
            if interaction_message.content:
                content.append(
                    {
                        "type": "text",
                        "text": interaction_message.content,
                    }
                )
            if context:
                content.append(
                    {
                        "type": "text",
                        "text": cls._context_to_str(context),
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
        return ret

    def _create_ai_chat_response(self, lm_resp: list[BaseModel]) -> AIChatResponse:
        content_parts = []
        tool_calls = []
        for m_resp in lm_resp:
            m_resp = m_resp.model_dump()
            if "text" in m_resp:
                content_parts.append(m_resp["text"])
            elif "id" in m_resp and "name" in m_resp and "input" in m_resp:
                tool_calls.append(
                    AIModelToolCall(
                        id=m_resp["id"], function_name=m_resp["name"], arguments=m_resp["input"], thought_signature=None
                    )
                )
        content = "".join(content_parts) if content_parts else None
        return AIChatResponse(
            vendor_model_name=self.get_vendor_model_name(),
            content=content,
            tool_calls=tool_calls,
        )

    @classmethod
    def model_function_definition(cls, ai_function: AIFunction) -> dict[str, Any]:
        parameters: dict[str, Any] = {
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

        parameters["additionalProperties"] = False  # Ensure no extra properties are allowed

        return {
            "name": ai_function.name,
            "description": ai_function.description,
            "input_schema": parameters,
        }

    try:
        from agents.mcp import MCPServer
        from mcp import Tool as MCPTool

        def prepare_mcp_tool_definition(self, tool: MCPTool) -> dict[str, Any]:
            """Creates a FunctionDefinition from an MCP Tool.

            It can be overriden if other models expect different definition

            Args:
                tool: The MCP Tool to create the FunctionDefinition from.
            Returns:
                A FunctionDefinition object representing the tool.
            """
            if tool.input_schema is None:
                raise ValueError(f"Tool {tool.name} has no input_schema")
            ret = {
                "name": tool.name,
                "description": tool.description,
            }
            if tool.input_schema.get("properties"):
                ret["input_schema"] = {
                    "type": "object",
                    "properties": tool.input_schema["properties"],
                    "required": tool.input_schema["required"],
                }
            return ret
    except ImportError:
        pass
