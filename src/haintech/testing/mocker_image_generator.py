import base64
import json
import logging
from contextlib import contextmanager
from typing import AsyncGenerator, Generator

import pytest
from ampf.base import BaseBlobMetadata, BlobCreate
from pydantic import BaseModel, field_serializer, field_validator
from pytest_mock.plugin import MockerFixture

from haintech.ai import BaseImageGenerator

try:
    from haintech.ai.google_genai import GenAIImageGenerator  # noqa: F401, F811

    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False

try:
    from haintech.ai.open_ai import OpenAIImageGenerator  # noqa: F401

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


_log = logging.getLogger(__name__)


class ImageGeneratorCall(BaseModel):
    prompt: str | None = None
    response: bytes

    @field_serializer("response")
    def serialize_blobs(self, value: bytes) -> str:
        return base64.b64encode(value).decode("utf-8")

    @field_validator("response", mode="before")
    @classmethod
    def decode_blobs(cls, value: str | bytes) -> bytes:
        if isinstance(value, bytes):
            return value
        else:
            return base64.b64decode(value)


class MockerImageGenerator:
    _mocked_methods: list[str] = []

    if GOOGLE_AVAILABLE:
        _mocked_methods.append("haintech.ai.google_genai.GenAIImageGenerator.generate")
        _mocked_methods.append("haintech.ai.google_genai.GenAIImageGenerator.generate_async")
    if OPENAI_AVAILABLE:
        _mocked_methods.append("haintech.ai.open_ai.OpenAIImageGenerator.generate")
        _mocked_methods.append("haintech.ai.open_ai.OpenAIImageGenerator.generate_async")

    def __init__(self, mocker: MockerFixture):
        self.mocker = mocker
        self.org_ai_model = None
        self.responses: list[ImageGeneratorCall] = []
        self.setup()

    def setup(self) -> None:
        for method in self._mocked_methods:
            if method.endswith("_async"):
                self.mocker.patch(method, side_effect=self.generate_async)
            else:
                self.mocker.patch(method, side_effect=self.generate)

    @contextmanager
    def record(self, ai_model: BaseImageGenerator | None = None):
        """Records all AI responses and prints them to console."""
        if not GOOGLE_AVAILABLE:
            raise RuntimeError("Google GenAI is not available. Please install the required dependencies.")
        from haintech.ai.google_genai import GenAIImageGenerator

        self.org_ai_model = ai_model or GenAIImageGenerator()
        yield
        print(
            json.dumps(
                [call.model_dump(mode="python", exclude_none=True, exclude_unset=True) for call in self.responses],
                indent=2,
                ensure_ascii=False,
            )
        )
        assert False

    def add(
        self,
        response: bytes,
        prompt: str | None = None,
    ) -> None:
        self.responses.append(ImageGeneratorCall(prompt=prompt, response=response))

    def generate(self, prompt: str) -> Generator[BlobCreate]:
        #####################

        if self.org_ai_model:
            self.mocker.stopall()
            response = next(self.org_ai_model.generate(prompt=prompt))
            if not isinstance(response.content, bytes):
                raise ValueError("Expected response content to be bytes")
            call = ImageGeneratorCall(prompt=prompt, response=response.content)
            _log.info("AI response: %s", response)
            self.responses.append(call)
            self.setup()
        elif not self.responses:
            raise RuntimeError("No mocked AI responses available.")
        else:
            call = self.responses.pop(0)
            if call.prompt:
                assert prompt
                assert prompt == call.prompt, f"Expected prompt content '{call.prompt}', got '{prompt}'"

            response = BlobCreate(content=call.response, metadata=BaseBlobMetadata(content_type="image/jpeg"))

        ####################
        yield response

    async def generate_async(self, prompt: str) -> AsyncGenerator[BlobCreate]:
        for i in self.generate(prompt):
            yield i

    def add_calls(self, calls: list[dict]) -> None:
        for call in calls:
            self.responses.append(ImageGeneratorCall.model_validate(call))


@pytest.fixture
def mocker_image_generator(mocker: MockerFixture) -> MockerImageGenerator:
    return MockerImageGenerator(mocker)
