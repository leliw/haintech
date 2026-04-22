import logging
from typing import Callable, Literal, Type

from ampf.base import Blob, BlobLocation

from haintech.ai import BaseAIModel
from haintech.ai.model import AIModelInteraction
from pydantic import BaseModel

from haintech.ai.prompts.prompt_model import BaseOutput

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

    def execute(
        self,
        prompt_name: str,
        response_format: Literal["text", "json"] | dict = "text",
        blob_locations: list[BlobLocation] | None = None,
        blobs: list[Blob] | None = None,
        **kwargs,
    ) -> str:
        system, user = self.prompt_service.render(prompt_name, **kwargs)
        resp = self.ai_model.get_response(
            system_prompt=system,
            message=user, 
            blob_locations=blob_locations , 
            blobs=blobs,
            response_format=response_format,
        )
        return resp


    def execute_list[T: str | int | float | bool](
        self,
        prompt_name: str,
        type: Type[T] = str,
        blob_locations: list[BlobLocation] | None = None,
        blobs: list[Blob] | None = None,
        **kwargs,
    ) -> list[T]:
        
        system, user = self.prompt_service.render(prompt_name, **kwargs)
        resp = self.ai_model.get_response_list(
            system_prompt=system,
            message=user, 
            blob_locations=blob_locations , 
            blobs=blobs,
            type=type,
        )
        return resp        


    def execute_typed[T: BaseModel](
        self,
        prompt_name: str,
        clazz: Type[T],
        blob_locations: list[BlobLocation] | None = None,
        blobs: list[Blob] | None = None,
        **kwargs,
    ) -> T:
        output_class = self.prompt_service.get_output_class(prompt_name)
        if output_class:
            json_schema = output_class.model_json_schema()
        else:
            json_schema = clazz.model_json_schema()
        kwargs["json_schema"] = json_schema
        system, user = self.prompt_service.render(prompt_name, **kwargs)
        ret = self.ai_model.get_response_typed(
            system_prompt=system,
            message=user, 
            blob_locations=blob_locations , 
            blobs=blobs,
            response_format=output_class or clazz,
        )
        if output_class and isinstance(ret, BaseOutput):
            return ret.convert(**kwargs)
        elif isinstance(ret, clazz):
            return ret
        raise ValueError(f"Invalid response format: {ret}")

    def execute_typed_list[T: BaseModel](
        self,
        prompt_name: str,
        clazz: Type[T],
        blob_locations: list[BlobLocation] | None = None,
        blobs: list[Blob] | None = None,
        **kwargs,
    ) -> list[T]:
        json_schema = {"type": "array", "items": clazz.model_json_schema()}
        kwargs["json_schema"] = json_schema
        system, user = self.prompt_service.render(prompt_name, **kwargs)
        ret = self.ai_model.get_response_list_typed(
            system_prompt=system,
            message=user, 
            blob_locations=blob_locations , 
            blobs=blobs,
            response_format= clazz,
        )
        return ret


    async def execute_async(
        self,
        prompt_name: str,
        response_format: Literal["text", "json"] | dict = "text",
        blob_locations: list[BlobLocation] | None = None,
        blobs: list[Blob] | None = None,
        **kwargs,
    ) -> str:
        system, user = self.prompt_service.render(prompt_name, **kwargs)
        resp = await self.ai_model.get_response_async(
            system_prompt=system,
            message=user, 
            blob_locations=blob_locations , 
            blobs=blobs,
            response_format=response_format,
        )
        return resp


    async def execute_list_async[T: str | int | float | bool](
        self,
        prompt_name: str,
        type: Type[T] = str,
        blob_locations: list[BlobLocation] | None = None,
        blobs: list[Blob] | None = None,
        **kwargs,
    ) -> list[T]:
        
        system, user = self.prompt_service.render(prompt_name, **kwargs)
        resp = await self.ai_model.get_response_list_async(
            system_prompt=system,
            message=user, 
            blob_locations=blob_locations , 
            blobs=blobs,
            type=type,
        )
        return resp        


    async def execute_typed_async[T: BaseModel](
        self,
        prompt_name: str,
        clazz: Type[T],
        blob_locations: list[BlobLocation] | None = None,
        blobs: list[Blob] | None = None,
        **kwargs,
    ) -> T:
        output_class = self.prompt_service.get_output_class(prompt_name)
        if output_class:
            json_schema = output_class.model_json_schema()
        else:
            json_schema = clazz.model_json_schema()
        kwargs["json_schema"] = json_schema
        system, user = self.prompt_service.render(prompt_name, **kwargs)
        ret = await self.ai_model.get_response_typed_async(
            system_prompt=system,
            message=user, 
            blob_locations=blob_locations , 
            blobs=blobs,
            response_format=output_class or clazz,
        )
        if output_class and isinstance(ret, BaseOutput):
            return ret.convert(**kwargs)
        elif isinstance(ret, clazz):
            return ret
        raise ValueError(f"Invalid response format: {ret}")

    async def execute_typed_list_async[T: BaseModel](
        self,
        prompt_name: str,
        clazz: Type[T],
        blob_locations: list[BlobLocation] | None = None,
        blobs: list[Blob] | None = None,
        **kwargs,
    ) -> list[T]:
        json_schema = {"type": "array", "items": clazz.model_json_schema()}
        kwargs["json_schema"] = json_schema
        system, user = self.prompt_service.render(prompt_name, **kwargs)
        ret = await self.ai_model.get_response_list_typed_async(
            system_prompt=system,
            message=user, 
            blob_locations=blob_locations , 
            blobs=blobs,
            response_format= clazz,
        )
        return ret
