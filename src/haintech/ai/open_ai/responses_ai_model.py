import json
import logging
from itertools import chain
from typing import Any, Callable, Dict, Iterable, Literal, Optional, Sequence, Type, override

from openai import AsyncOpenAI, OpenAI
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

    _configured = False

    @classmethod
    def setup(cls):
        if not cls._configured:
            cls.openai = OpenAI()
            cls.async_openai = AsyncOpenAI()
            cls._configured = True
            _log.debug("OpenAI AI Model configured")

    def __init__(
        self,
        model_name: str = "gpt-5.4-nano",
        parameters: ResponsesAIParameters | Dict[str, Any] | None = None,
    ):
        self.setup()
        self.model_name = model_name
        self.parameters = parameters or ResponsesAIParameters()

    @classmethod
    def get_model_names(cls) -> list[str]:
        cls.setup()
        return [m.id for m in cls.openai.models.list().data]

    @override
    def get_chat_response(
        self,
        system_prompt: Optional[str | AIPrompt] = None,
        history: Optional[Iterable[AIModelInteractionMessage]] = None,
        context: Optional[AIContext] = None,
        message: Optional[AIModelInteractionMessage] = None,
        functions: Optional[Dict[Callable, Any]] = None,
        interaction_logger: Optional[Callable[[AIModelInteraction], None]] = None,
        response_format: Literal["text", "json"] | dict = "text",
    ) -> AIChatResponse:
        if not isinstance(history, list):
            history = list(history or [])
        parameters, ai_model_interaction = self._prepare_parameters(
            system_prompt, history, context, message, functions, response_format
        )
        try:
            resp: Response = self.openai.responses.create(**parameters)
            response = self._create_ai_chat_response(resp)
            return response
        except Exception as e:
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
        system_prompt: Optional[str | AIPrompt] = None,
        history: Optional[Iterable[AIModelInteractionMessage]] = None,
        context: Optional[AIContext] = None,
        message: Optional[AIModelInteractionMessage] = None,
        functions: Optional[Dict[Callable, Any]] = None,
        interaction_logger: Optional[Callable[[AIModelInteraction], None]] = None,
        response_format: Literal["text", "json"] | dict = "text",
    ) -> AIChatResponse:
        if not isinstance(history, list):
            history = list(history or [])
        parameters, ai_model_interaction = self._prepare_parameters(
            system_prompt, history, context, message, functions, response_format
        )
        try:
            resp = await self.async_openai.responses.create(**parameters)
            response = self._create_ai_chat_response(resp)
            return response
        except Exception as e:
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
            tools = []
            for f, d in functions.items():
                tools.append(d)
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
                ret.append(EasyInputMessageParam(role=m.role, content=text_blob_contents)) # pyright: ignore[reportArgumentType]
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

    @classmethod
    def _create_ai_chat_response(cls, resp: Response) -> AIChatResponse:
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
            input_tokens=input_tokens,
            input_tokens_cached=input_tokens_cached,
            reasoning_tokens=reasoning_tokens,
            output_tokens=output_tokens,
        )

    def _prepare_response_format(
        self,
        response_format: Literal["text", "json"]
        | Type[Sequence[BaseModel | str | int | float | bool]]
        | Type[BaseModel] = "text",
    ) -> dict | None:
        if response_format == "text":
            ret = None
        elif response_format == "json":
            ret = {"type": "json_object"}
        elif isinstance(response_format, type) and issubclass(response_format, BaseModel):
            schema = prepare_schema(response_format)
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
                schema = prepare_schema(inner_type)
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
        return {"format": ret} if ret else None

    @classmethod
    def prepare_function_definition(
        cls,
        func: Callable[..., Any],
        name: Optional[str] = None,
        description: Optional[str] = None,
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
        parameters: Dict[str, Any] = {
            "type": "object",
            "properties": {},
            "required": [],
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

        parameters["additionalProperties"] = False  # Ensure no extra properties are allowed

        return FunctionToolParam(
            type="function",
            name=ai_function.name,
            description=ai_function.description or "",
            parameters=parameters,
            strict=True,
        )


def prepare_schema(model: Type[BaseModel]) -> dict:
    schema = model.model_json_schema()
    for _, v in schema["properties"].items():
        v.pop("title", None)
    schema.pop("title", None)
    schema.pop("description", None)
    schema["additionalProperties"] = False
    return schema
