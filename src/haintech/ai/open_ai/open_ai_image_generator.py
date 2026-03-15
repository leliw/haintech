import base64
import logging
from typing import Any, AsyncGenerator, Generator

from ampf.base import BlobCreate
from openai import AsyncOpenAI, OpenAI

from haintech.ai.base.base_image_generator import BaseImageGenerator

_log = logging.getLogger(__name__)


class OpenAIImageGenerator(BaseImageGenerator):
    _configured = False

    def __init__(self, model_name: str = "gpt-image-1-mini", api_key: str | None = None):
        if api_key:
            self.setup(api_key=api_key)
        self.model_name = model_name

    @classmethod
    def setup(cls, api_key: str | None = None):
        if not cls._configured:
            cls.client = OpenAI(api_key=api_key)
            cls.async_openai = AsyncOpenAI(api_key=api_key)
            cls._configured = True
            _log.debug("OpenAI AI Model configured")

    def _prepare_parameters(self) -> dict[str, Any]:
        if not self._configured:
            self.setup()
        if "dall-e" in self.model_name:
            return {"response_format": "b64_json"}
        else:
            return {}

    def generate(self, prompt: str) -> Generator[BlobCreate]:
        parameters = self._prepare_parameters()
        result = self.client.images.generate(
            model=self.model_name,
            prompt=prompt,
            size="1024x1024",
            n=1,
            **parameters,
        )
        if result.data:
            for data in result.data:
                image_bytes = base64.b64decode(data.b64_json)
                yield BlobCreate.from_content(
                    content_type=f"image/{result.output_format or 'png'}",
                    content=image_bytes,
                )

    async def generate_async(self, prompt: str) -> AsyncGenerator[BlobCreate]:
        parameters = self._prepare_parameters()
        result = await self.async_openai.images.generate(
            model=self.model_name,
            prompt=prompt,
            size="1024x1024",
            n=1,
            **parameters,
        )
        if result.data:
            for data in result.data:
                image_bytes = base64.b64decode(data.b64_json)
                yield BlobCreate.from_content(
                    content_type=f"image/{result.output_format or 'png'}",
                    content=image_bytes,
                )
