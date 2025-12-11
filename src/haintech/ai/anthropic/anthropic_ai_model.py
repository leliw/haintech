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
from haintech.ai.model import AIContext, AIFunction, AIModelInteractionTool, AIModelToolCall


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
        system_prompt: Optional[str | AIPrompt] = None,
        history: Optional[Iterable[AIModelInteractionMessage]] = None,
        context: Optional[AIContext] = None,
        message: Optional[AIModelInteractionMessage] = None,
        functions: Optional[Dict[Callable, Any]] = None,
        interaction_logger: Optional[Callable[[AIModelInteraction], None]] = None,
        response_format: Literal["text", "json"] = "text",
    ) -> AIChatResponse:
        parameters, ai_model_interaction = self._prepare_parameters(
            system_prompt, history, context, message, functions, response_format
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
        system_prompt: Optional[str | AIPrompt] = None,
        history: Optional[Iterable[AIModelInteractionMessage]] = None,
        context: Optional[AIContext] = None,
        message: Optional[AIModelInteractionMessage] = None,
        functions: Optional[Dict[Callable, Any]] = None,
        interaction_logger: Optional[Callable[[AIModelInteraction], None]] = None,
        response_format: Literal["text", "json"] = "text",
    ) -> AIChatResponse:
        parameters, ai_model_interaction = self._prepare_parameters(
            system_prompt, history, context, message, functions, response_format
        )
        try:
            resp: anthropic.types.Message = await self.async_client.messages.create(**parameters)
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
        system_prompt: Optional[str | AIPrompt] = None,
        history: Optional[Iterable[AIModelInteractionMessage]] = None,
        context: Optional[AIContext] = None,
        message: Optional[AIModelInteractionMessage] = None,
        functions: Optional[Dict[Callable, Any]] = None,
        response_format: Literal["text", "json"] = "text",
    ):
        self._log.debug("Preparing parameters for Anthropic model")
        if not isinstance(history, list):
            history = list(history or [])
        if not message:
            if history:
                message = history.pop()
            else:
                raise ValueError("No message provided")
        msg_list = [self._create_message(m) for m in history]
        if isinstance(message, str):
            message = AIModelInteractionMessage(role="user", content=message)
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
        return (
            {
                "model": self.model_name,
                "system": system or anthropic.NOT_GIVEN,
                "messages": msg_list,
                "tools": tools,
            }
            | (self.parameters or {}),  # Unpack parameters if not None
            ai_model_interaction,
        )

    @classmethod
    def _create_message(
        cls, interaction_message: AIModelInteractionMessage, context: Optional[AIContext] = None
    ) -> Dict[str, Any]:
        cls._log.debug("Creating message: %s", interaction_message)
        if context:
            cls._log.debug("With context: %s", context)

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
                    AIModelToolCall(
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

        parameters["additionalProperties"] = False  # Ensure no extra properties are allowed

        return {
            "name": ai_function.name,
            "description": ai_function.description,
            "input_schema": parameters,
        }

    try:
        from agents.mcp import MCPServer
        from mcp import Tool as MCPTool

        def prepare_mcp_tool_definition(self, tool: MCPTool) -> Dict[str, Any]:
            """Creates a FunctionDefinition from an MCP Tool.

            It can be overriden if other models expect different definition

            Args:
                tool: The MCP Tool to create the FunctionDefinition from.
            Returns:
                A FunctionDefinition object representing the tool.
            """
            if tool.inputSchema is None:
                raise ValueError(f"Tool {tool.name} has no inputSchema")
            ret = {
                "name": tool.name,
                "description": tool.description,
            }
            if tool.inputSchema["properties"]:
                ret["input_schema"] = {
                    "type": "object",
                    "properties": tool.inputSchema["properties"],
                    "required": tool.inputSchema["required"],
                }
            return ret
    except ImportError:
        pass
