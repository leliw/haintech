import pytest

from haintech.ai.base.base_image_generator import BaseImageGenerator
from haintech.ai.google_genai.genai_image_generator import GenAIImageGenerator
from haintech.ai.open_ai.open_ai_image_generator import OpenAIImageGenerator


@pytest.fixture(scope="module", params=[OpenAIImageGenerator, GenAIImageGenerator])
def service(request) -> BaseImageGenerator:
    return request.param()


@pytest.mark.skip(reason="It generates costs.")
def test_generate(service: BaseImageGenerator):
    blob_create = next(service.generate("Big black dog."))

    assert blob_create.metadata
    assert blob_create.metadata.content_type
    assert blob_create.content


@pytest.mark.skip(reason="It generates costs.")
@pytest.mark.asyncio
async def test_generate_async(service: BaseImageGenerator):
    async for blob_create in service.generate_async("Big black dog."):
        assert blob_create.metadata
        assert blob_create.metadata.content_type
        assert blob_create.content
