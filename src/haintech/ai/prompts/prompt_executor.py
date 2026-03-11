import json
import logging
from typing import Callable, Literal, Type

from haintech.ai import AIChatResponse, AIModelInteractionMessage, BaseAIModel
from haintech.ai.model import AIModelInteraction
from pydantic import BaseModel, ValidationError

from .prompt_service import PromptService

_log = logging.getLogger(__name__)


class PromptExecutor:
    def __init__(
        self,
        ai_model: BaseAIModel,
        prompt_service: PromptService,
        interaction_logger: Callable[[AIModelInteraction], None] | None = None,
    ):
        self.ai_model = ai_model
        self.prompt_service = prompt_service
        self.interaction_logger = interaction_logger

    def execute(self, prompt_name: str, response_format: Literal["text", "json"] = "text", **kwargs) -> AIChatResponse:
        system, user = self.prompt_service.render(prompt_name, **kwargs)
        m_resp = self.ai_model.get_chat_response(
            system_prompt=system,
            message=AIModelInteractionMessage(role="user", content=user),
            response_format=response_format,
            interaction_logger=self.interaction_logger,
        )
        if m_resp.content is None:
            raise ValueError(f"AI model returned empty content for prompt: {prompt_name}")
        return m_resp

    def execute_list(self, prompt_name: str, **kwargs) -> list[str]:
        response = self.execute(prompt_name, response_format="json", **kwargs)
        return self._prepare_response_list(response)

    def _prepare_response_list(self, m_resp: AIChatResponse) -> list[str]:
        if m_resp.content is None:
            raise ValueError("AI model returned empty content")

        try:
            ret = json.loads(m_resp.content)
            if isinstance(ret, dict) and len(ret) == 1:
                ret = list(ret.values())[0]
            if isinstance(ret, dict) and all(isinstance(v, list) for v in ret.values()):
                return sum(ret.values(), [])
            if isinstance(ret, list):
                return ret
            else:
                raise ValueError(f"Invalid response format: {m_resp.content}")
        except (json.JSONDecodeError, ValidationError) as e:
            _log.warning(e)
            _log.warning("JSON content: %s", m_resp.content)
            raise e

    def execute_typed[T: BaseModel](self, prompt_name: str, clazz: Type[T], **kwargs) -> T:
        output_class = self.prompt_service.get_output_class(prompt_name)
        if output_class:
            json_schema = output_class.model_json_schema()
        else:
            json_schema = clazz.model_json_schema()
        for attempt in range(3):
            try:
                response = self.execute(prompt_name, response_format="json", json_schema=json_schema, **kwargs)
                if output_class:
                    ret = self._prepare_response_typed(output_class, response)
                    return ret.convert(**kwargs)
                else:
                    return self._prepare_response_typed(clazz, response)
            except (json.JSONDecodeError, ValidationError) as e:
                if attempt == 2:
                    raise e
        raise ValueError(f"Invalid response format: {response.content}")

    def _prepare_response_typed[T: BaseModel](self, clazz: Type[T], m_resp: AIChatResponse) -> T:
        if m_resp.content is None:
            raise ValueError("AI model returned empty content")
        try:
            response = json.loads(m_resp.content)
            if isinstance(response, list) and len(response) == 1:
                response = response[0]
            return clazz.model_validate(response)
        except (json.JSONDecodeError, ValidationError) as e:
            _log.warning(e)
            _log.warning("JSON content: %s", m_resp.content)
            raise e

    def execute_typed_list[T: BaseModel](self, prompt_name: str, clazz: Type[T], **kwargs) -> list[T]:
        json_schema = {"type": "array", "items": clazz.model_json_schema()}
        response = self.execute(prompt_name, response_format="json", json_schema=json_schema, **kwargs)
        return self._prepare_response_typed_list(clazz, response)

    def _prepare_response_typed_list[T: BaseModel](self, clazz: Type[T], m_resp: AIChatResponse) -> list[T]:
        if m_resp.content is None:
            raise ValueError("AI model returned empty content")
        try:
            response = json.loads(m_resp.content)
            if isinstance(response, dict) and len(response) == 1:
                response = list(response.values())[0]
            elif isinstance(response, dict) and len(response) == 2 and "items" in response:
                response = response.get("items", [])
            ret = []
            for j in response:
                ret.append(clazz.model_validate(j))
            return ret
        except (json.JSONDecodeError, ValidationError) as e:
            _log.warning(e)
            _log.warning("JSON content: %s", m_resp.content)
            raise e
