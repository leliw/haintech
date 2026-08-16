import json
import logging
from collections.abc import Callable, Iterable, Sequence
from itertools import chain
from typing import Any, Literal, override

from openai import AsyncOpenAI, OpenAI
from openai.types import Model
from openai.types.responses import (
    EasyInputMessageParam,
    FunctionToolParam,
    Response,
    ResponseFunctionToolCallParam,
    ResponseInputParam,
)
from openai.types.responses.response_input_param import FunctionCallOutput
from pydantic import BaseModel

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
)
from .model import ResponsesAIParameters

_log = logging.getLogger(__name__)


class ResponsesAIModel(BaseAIModel):
    """OpenAI implementation of BaseAIModel"""

    _api_key: str | None = None
    _openai: OpenAI | None = None
    _async_openai: AsyncOpenAI | None = None
    _models_list: list[Model] | None = None

    @classmethod
    def get_vendor_name(cls) -> str:
        return "openai"

    @classmethod
    def setup(cls, api_key: str | None = None):
        cls._api_key = api_key

    def __init__(
        self,
        model_name: str = "gpt-5.4-nano",
        parameters: ResponsesAIParameters | dict[str, Any] | None = None,
    ):
        self.model_name = model_name
        self.parameters = parameters or ResponsesAIParameters()

    @classmethod
    def get_openai(cls) -> OpenAI:
        if not cls._openai:
            cls._openai = OpenAI(api_key=cls._api_key)
        return cls._openai

    @classmethod
    def get_async_openai(cls) -> AsyncOpenAI:
        if not cls._async_openai:
            cls._async_openai = AsyncOpenAI(api_key=cls._api_key)
        return cls._async_openai

    @override
    @classmethod
    def get_model_names(cls, task: str = "chat") -> list[str]:
        if not cls._models_list:
            cls._models_list = cls.get_openai().models.list().data
        raw_models = [m.id for m in cls._models_list]
        ret = []

        for name in raw_models:
            # Filter out snapshot dates and deprecated models
            if (
                cls._ends_with_date(name)
                or name.endswith("-latest")
                or name.startswith(("babbage", "gpt-4-", "davinci"))
            ):
                continue

            # Task filtering
            if task == "chat":
                non_chat_keywords = (
                    "tts",
                    "image-",
                    "embedding-",
                    "sora-",
                    "whisper-",
                    "audio",
                    "transcribe",
                    "realtime",
                    "search",
                    "codex",
                    "turbo",
                    "dall-e",
                )
                if any(k in name for k in non_chat_keywords):
                    continue
                ret.append(name)
            elif task == "image":
                if "dall-e" in name:
                    ret.append(name)
            elif task == "embedding":
                if "embedding" in name:
                    ret.append(name)

        return ret

    @override
    @classmethod
    async def get_model_names_async(cls, task: str = "chat") -> list[str]:
        if not cls._models_list:
            cls._models_list = (await cls.get_async_openai().models.list()).data
        return cls.get_model_names(task)

    @classmethod
    def _ends_with_date(cls, s: str) -> bool:
        import re

        pattern = r"-\d{4}-\d{2}-\d{2}$"
        return bool(re.search(pattern, s))

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
        if not isinstance(history, list):
            history = list(history or [])
        parameters, ai_model_interaction = self._prepare_parameters(
            system_prompt, history, context, message, functions, response_format
        )
        try:
            resp: Response = self.get_openai().responses.create(**parameters)
            response = self._create_ai_chat_response(resp)
            return response
        except Exception as e:  # noqa: BLE001
            _log.error("Error: %s", e)
            response = AIChatResponse(content=str(e))
            return response
        finally:
            if interaction_logger:
                ai_model_interaction.response = response
                interaction_logger(ai_model_interaction)

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
        if not isinstance(history, list):
            history = list(history or [])
        parameters, ai_model_interaction = self._prepare_parameters(
            system_prompt, history, context, message, functions, response_format
        )
        try:
            resp = await self.get_async_openai().responses.create(**parameters)
            response = self._create_ai_chat_response(resp)
            return response
        except Exception as e:  # noqa: BLE001
            _log.error("Error: %s", e)
            response = AIChatResponse(content=str(e))
            return response
        finally:
            if interaction_logger:
                ai_model_interaction.response = response
                interaction_logger(ai_model_interaction)

    def _prepare_parameters(
        self,
        system_prompt: str | AIPrompt | None,
        history: list[AIModelInteractionMessage],
        context: AIContext | None = None,
        message: AIModelInteractionMessage | None = None,
        functions: dict[Callable, Any] | None = None,
        response_format: Literal["text", "json"] | dict = "text",
    ):
        input = self._create_input(
            chain(
                (
                    [AIModelInteractionMessage(role="system", content=self._prompt_to_str(system_prompt))]
                    if system_prompt
                    else []
                ),
                history,
                ([AIModelInteractionMessage(role="system", content=self._context_to_str(context))] if context else []),
                ([message] if message else []),
            )
        )

        if functions:
            tools = list(functions.values())
        else:
            tools = None

        if response_format == "text":
            response_format_dict = None
        elif response_format == "json":
            response_format_dict = {"format": {"type": "json_object"}}
        elif isinstance(response_format, dict):
            response_format_dict = response_format
        else:
            raise ValueError(f"Unsupported response format: {response_format}")

        ai_model_interaction = AIModelInteraction(
            model=self.model_name,
            message=AIModelInteractionMessage(role="user", content=message) if isinstance(message, str) else message,
            prompt=system_prompt if isinstance(system_prompt, AIPrompt) else None,
            context=context,
            history=history,
            # tools=tools,
            response_format=response_format_dict,
        )
        parameters_dict = (
            self.parameters.get_for_model(self.model_name) if isinstance(self.parameters, ResponsesAIParameters) else {}
        )
        ret = {
            "model": self.model_name,
            "input": input,
            "tools": tools,
            "text": response_format_dict,
            **parameters_dict,
        }
        if _log.isEnabledFor(logging.DEBUG):
            _log.debug("===========>\n %s\n <==========", ret)
        return (ret, ai_model_interaction)

    @staticmethod
    def _create_input(messages: Iterable[AIModelInteractionMessage]) -> ResponseInputParam:
        ret = []
        for m in messages:
            if m.role in ["user", "system", "assistant", "developer"] and m.content:
                text_blob_contents = ""
                for blob in m.blobs or []:
                    _log.debug("name=%s, type=%s", blob.name, blob.content_type)
                    is_text = (blob.content_type and blob.content_type.startswith("text/")) or (
                        blob.content and b"\x00" not in blob.content
                    )
                    if is_text:
                        text_blob_contents += "\n<file>\n"
                        if blob.name:
                            text_blob_contents += f"<name>{blob.name}</name>\n"
                        text_blob_contents += "<content>\n"
                        text_blob_contents += blob.content.decode("utf-8")
                        text_blob_contents += "\n</content>\n"
                        text_blob_contents += "</file>\n"
                    else:
                        _log.warning("Unsupported blob type: %s", blob.content_type)
                if text_blob_contents:
                    text_blob_contents += "\n"
                text_blob_contents += m.content
                ret.append(EasyInputMessageParam(role=m.role, content=text_blob_contents))  # pyright: ignore[reportArgumentType]
            if m.tool_calls:
                for c in m.tool_calls:
                    if not c.id:
                        raise RuntimeError("Tool calls must have an id")
                    ret.append(
                        ResponseFunctionToolCallParam(
                            type="function_call", call_id=c.id, name=c.function_name, arguments=json.dumps(c.arguments)
                        )
                    )
            if m.tool_call_id:
                ret.append(
                    FunctionCallOutput(
                        type="function_call_output",
                        call_id=m.tool_call_id,
                        output=m.content or "",
                    )
                )
        return ret

    @classmethod
    def _create_message(cls, m: AIModelInteractionMessage) -> ResponseInputParam:
        text_blob_contents = ""
        for blob in m.blobs or []:
            _log.debug("name=%s, type=%s", blob.name, blob.content_type)
            is_text = (blob.content_type and blob.content_type.startswith("text/")) or (
                blob.content and b"\x00" not in blob.content
            )
            if is_text:
                text_blob_contents += "\n<file>\n"
                if blob.name:
                    text_blob_contents += f"<name>{blob.name}</name>\n"
                text_blob_contents += "<content>\n"
                text_blob_contents += blob.content.decode("utf-8")
                text_blob_contents += "\n</content>\n"
                text_blob_contents += "</file>\n"
            else:
                _log.warning("Unsupported blob type: %s", blob.content_type)
        if m.content:
            if text_blob_contents:
                text_blob_contents += "\n"
            text_blob_contents += m.content

        if m.tool_call_id:
            ret: dict[str, Any] = {}
            ret["type"] = "function_call_output"
            ret["call_id"] = m.tool_call_id
            ret["output"] = m.content
        elif m.tool_calls:
            ret = {}
            ret["tool_calls"] = [
                {
                    "call_id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.function_name,
                        "arguments": json.dumps(tool_call.arguments),
                    },
                }
                for tool_call in m.tool_calls  # type: ignore
            ]
        else:
            ret: dict[str, Any] = {
                "role": m.role,
                "content": text_blob_contents,
            }
        _log.debug("Creating message: %s", m)
        return ret  # type: ignore

    def _create_ai_chat_response(self, resp: Response) -> AIChatResponse:
        input_tokens = resp.usage.input_tokens if resp.usage else None
        input_tokens_cached = resp.usage.input_tokens_details.cached_tokens if resp.usage else None
        reasoning_tokens = resp.usage.output_tokens_details.reasoning_tokens if resp.usage else None
        output_tokens = resp.usage.output_tokens if resp.usage else None

        tool_calls = []
        for item in resp.output:
            if item.type == "function_call":
                call = AIModelToolCall(
                    id=item.call_id,
                    function_name=item.name,
                    arguments=json.loads(item.arguments),
                    thought_signature=None,
                )
                tool_calls.append(call)
        return AIChatResponse(
            content=resp.output_text,
            tool_calls=tool_calls or None,
            vendor_model_name=self.get_vendor_model_name(),
            input_tokens=input_tokens,
            input_tokens_cached=input_tokens_cached,
            reasoning_tokens=reasoning_tokens,
            output_tokens=output_tokens,
        )

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
            schema = self.prepare_schema(response_format)
            ret = {
                "type": "json_schema",
                "strict": True,
                "name": response_format.__name__,
                "schema": schema,
            }
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
                schema = self.prepare_schema(inner_type)
                inner_ret = {"type": "array", "items": schema}
            else:
                raise ValueError(f"Unsupported inner response format: {inner_type}")
            ret = {
                "type": "json_schema",
                "name": "list_wrapper",
                "schema": {
                    "type": "object",
                    "properties": {"list": inner_ret},
                    "required": ["list"],
                    "additionalProperties": False,
                },
            }
        else:
            raise ValueError(f"Unsupported response format: {response_format}")
        return {"format": ret}

    @classmethod
    def prepare_function_definition(
        cls,
        func: Callable[..., Any],
        name: str | None = None,
        description: str | None = None,
    ) -> FunctionToolParam:
        ai_function = cls.create_ai_function(func)
        if name:
            ai_function.name = name
        if description:
            ai_function.description = description
        return cls.model_function_definition(ai_function)

    @classmethod
    def model_function_definition(cls, ai_function: AIFunction) -> FunctionToolParam:
        """Creates an OpenAI FunctionDefinition from a Python callable.

        It can be overriden if other models expect different definition
        Args:
            ai_function: AIFunction object
        Returns:
            A FunctionDefinition object representing the callable.  Returns None if input is invalid.
        Raises:
            TypeError: If input is not a callable.
            ValueError: If the function signature is invalid or missing required information.
        """
        parameters: dict[str, Any] = {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        }
        for param in ai_function.parameters:
            param_name = param.name
            param_description = param.description
            parameters["properties"][param_name] = {
                "type": (
                    # Work around for not required parameters - union with null
                    "string" if param.required else ["string", "null"]
                ),
                "description": param_description,
            }  # Default type is string, could be improved.
            parameters["required"].append(param_name)

        return FunctionToolParam(
            type="function",
            name=ai_function.name,
            description=ai_function.description or "",
            parameters=parameters,
            strict=True,
        )

    try:
        from agents.mcp import MCPServer
        from mcp import Tool as MCPTool

        def prepare_mcp_tool_definition(self, tool: MCPTool) -> FunctionToolParam:
            """Creates a FunctionDefinition from an MCP Tool.

            It can be overriden if other models expect different definition

            Args:
                tool: The MCP Tool to create the FunctionDefinition from.
            Returns:
                A FunctionDefinition object representing the tool.
            """
            try:
                properties = tool.input_schema.get("properties", {})
                required = tool.input_schema.get("required", [])
                # if len(properties) != len(required):
                #     required = list(properties.keys()) # Ensure all properties are marked as required for strict mode
                # for _, v in properties.items():
                #     if "items" in v:
                #         items = v.get("items", {})
                #         if "properties" in items:
                #             items["additionalProperties"] = False   # Required by OpenAI
                ret = FunctionToolParam(
                    name=tool.name,
                    description=tool.description,
                    type="function",
                    parameters={
                        "type": "object",
                        "properties": properties,
                        "required": required,
                        "additionalProperties": False,
                    },
                    strict=False,
                )
                return ret
            except Exception as e:
                print(f"Error preparing MCP tool definition: {e}")
                raise

    except ImportError:
        pass

    @staticmethod
    def prepare_schema(model: type[BaseModel]) -> dict:
        schema = model.model_json_schema()
        for v in schema["properties"].values():
            v.pop("title", None)
        schema.pop("title", None)
        schema.pop("description", None)
        schema["additionalProperties"] = False
        return schema
