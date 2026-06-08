import json
import logging
from typing import Any, Callable, Dict, Iterable, Literal, Optional, Sequence, Type, override

from openai import AsyncOpenAI, OpenAI
from openai.types.responses import EasyInputMessageParam, FunctionToolParam, Response
from pydantic import BaseModel

from haintech.helpers import get_inner_type, is_list_type

from ..base import BaseAIModel
from ..model import (
    AIChatResponse,
    AIContext,
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
        model_name: str = "gpt-4o-mini",
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
            resp = self.openai.responses.create(**parameters)
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
        context: Optional[AIContext] = None,
        message: Optional[AIModelInteractionMessage] = None,
        functions: Optional[Dict[Callable, Any]] = None,
        response_format: Literal["text", "json"] | dict = "text",
    ):
        msg_list = []
        if system_prompt:
            msg_list.append(
                self._create_message(
                    AIModelInteractionMessage(role="system", content=self._prompt_to_str(system_prompt))
                )
            )
        if not message:
            message = history[-1]
            for m in history[:-1]:
                msg_list.append(self._create_message(m))
        else:
            for m in history:
                msg_list.append(self._create_message(m))
            if isinstance(message, str):
                message = AIModelInteractionMessage(role="user", content=message)

        if context:
            msg_list.append(
                self._create_message(AIModelInteractionMessage(role="system", content=self._context_to_str(context)))
            )
        msg_list.append(self._create_message(message))
        if functions:
            tools = []
            for f in functions:
                definition = functions[f]
                tools.append(FunctionToolParam(type="function", function=definition))
        else:
            tools = None

        if response_format == "text":
            response_format_dict = None
        elif response_format == "json":
            response_format_dict = {"type": "json_object"}
        elif isinstance(response_format, dict):
            response_format_dict = response_format
        else:
            raise ValueError(f"Unsupported response format: {response_format}")

        ai_model_interaction = AIModelInteraction(
            model=self.model_name,
            message=message,
            prompt=system_prompt if isinstance(system_prompt, AIPrompt) else None,
            context=context,
            history=history,
            tools=tools,
            response_format=response_format_dict,
        )
        parameters_dict = (
            self.parameters.get_for_model(self.model_name) if isinstance(self.parameters, ResponsesAIParameters) else {}
        )
        ret = {
            "model": self.model_name,
            "input": msg_list,
            "tools": tools,
            "text": response_format_dict,
            **parameters_dict,
        }
        if _log.isEnabledFor(logging.DEBUG):
            _log.debug("===========>\n %s\n <==========", ret)
        return (ret, ai_model_interaction)

    @classmethod
    def _create_message(cls, i_message: AIModelInteractionMessage) -> EasyInputMessageParam:
        text_blob_contents = ""
        for blob in i_message.blobs or []:
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
        if i_message.content:
            if text_blob_contents:
                text_blob_contents += "\n"
            text_blob_contents += i_message.content

        ret: dict[str, Any] = {
            "role": i_message.role,
            "content": text_blob_contents,
        }
        if i_message.tool_call_id:
            ret["tool_call_id"] = i_message.tool_call_id
        if i_message.tool_calls:
            ret["tool_calls"] = [
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.function_name,
                        "arguments": json.dumps(tool_call.arguments),
                    },
                }
                for tool_call in i_message.tool_calls  # type: ignore
            ]
        _log.debug("Creating message: %s", i_message)
        return ret  # type: ignore

    @classmethod
    def _create_ai_chat_response(cls, resp: Response) -> AIChatResponse:
        input_tokens = resp.usage.input_tokens if resp.usage else None
        input_tokens_cached = resp.usage.input_tokens_details.cached_tokens if resp.usage else None
        reasoning_tokens = resp.usage.output_tokens_details.reasoning_tokens if resp.usage else None
        output_tokens = resp.usage.output_tokens if resp.usage else None

        if resp.tools:
            tool_calls = [
                AIModelToolCall(
                    id=tool_call.id,
                    function_name=tool_call.function.name,  # type: ignore
                    arguments=json.loads(tool_call.function.arguments),  # type: ignore
                )
                for tool_call in resp.tools
            ]
        else:
            tool_calls = None
        return AIChatResponse(
            content=resp.output_text,
            tool_calls=tool_calls,
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


def prepare_schema(model: Type[BaseModel]) -> dict:
    schema = model.model_json_schema()
    for _, v in schema["properties"].items():
        v.pop("title", None)
    schema.pop("title", None)
    schema.pop("description", None)
    schema["additionalProperties"] = False
    return schema
