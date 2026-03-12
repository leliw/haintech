import logging
from typing import AsyncGenerator, Generator

from ampf.base import BlobCreate
from google import genai
from google.genai import types

from haintech.ai.base.base_image_generator import BaseImageGenerator


_log = logging.getLogger(__name__)


class GenAIImageGenerator(BaseImageGenerator):
    _configured = False

    def __init__(self, model_name: str = "gemini-2.5-flash-image", api_key: str | None = None):
        if api_key:
            self.setup(api_key=api_key)
        self.model_name = model_name

    @classmethod
    def setup(cls, api_key: str | None = None):
        if not cls._configured:
            cls.client = genai.Client(api_key=api_key)
            cls._configured = True
            _log.debug("Google AI Model configured")

    def generate(self, prompt: str) -> Generator[BlobCreate]:
        if not self._configured:
            self.setup()
        contents = [types.Content(role="user", parts=[types.Part.from_text(text=prompt)])]
        generate_content_config = types.GenerateContentConfig(
            response_modalities=["IMAGE"], image_config=types.ImageConfig(aspect_ratio="1:1")
        )
        for chunk in self.client.models.generate_content_stream(
            model=self.model_name,
            contents=contents,
            config=generate_content_config,
        ):
            if (
                chunk.candidates is None
                or chunk.candidates[0].content is None
                or chunk.candidates[0].content.parts is None
            ):
                continue
            if chunk.candidates[0].content.parts[0].inline_data:
                inline_data = chunk.candidates[0].content.parts[0].inline_data
                if inline_data.data and inline_data.mime_type:
                    yield BlobCreate.from_content(
                        content_type=inline_data.mime_type,
                        content=inline_data.data,
                    )
                else:
                    _log.warning("No binary data!?")
            else:
                _log.info(chunk.text)

    async def generate_async(self, prompt: str) -> AsyncGenerator[BlobCreate]:
        if not self._configured:
            self.setup()
        contents = [types.Content(role="user", parts=[types.Part.from_text(text=prompt)])]
        generate_content_config = types.GenerateContentConfig(
            response_modalities=["IMAGE"], image_config=types.ImageConfig(aspect_ratio="1:1")
        )
        async for chunk in await self.client.aio.models.generate_content_stream(
            model=self.model_name,
            contents=contents,
            config=generate_content_config,
        ):
            if (
                chunk.candidates is None
                or chunk.candidates[0].content is None
                or chunk.candidates[0].content.parts is None
            ):
                continue
            if chunk.candidates[0].content.parts[0].inline_data:
                inline_data = chunk.candidates[0].content.parts[0].inline_data
                if inline_data.data and inline_data.mime_type:
                    yield BlobCreate.from_content(
                        content_type=inline_data.mime_type,
                        content=inline_data.data,
                    )
                else:
                    _log.warning("No binary data!?")
            else:
                _log.info(chunk.text)
