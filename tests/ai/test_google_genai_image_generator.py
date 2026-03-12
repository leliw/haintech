from ampf.local import LocalFactory
from ampf.base import BaseBlobStorage, Blob
import pytest

from haintech.ai.google_genai.genai_image_generator import GenAIImageGenerator


@pytest.fixture(scope="module", params=["gemini-2.5-flash-image", "gemini-3.1-flash-image-preview"])
def service(request) -> GenAIImageGenerator:
    return GenAIImageGenerator(model_name=request.param)


@pytest.fixture
def blob_storage():
    factory = LocalFactory("./tests/data/")
    return factory.create_blob_storage("images")


@pytest.mark.skip(reason="It generates costs.")
def test_generate(service: GenAIImageGenerator, blob_storage: BaseBlobStorage):
    blob_create = next(service.generate("Big black dog."))

    assert blob_create.metadata
    assert blob_create.metadata.content_type
    assert blob_create.content

    blob_create.name = service.model_name + ".png"
    blob_storage.upload(Blob.create(blob_create))


@pytest.mark.skip(reason="It generates costs.")
@pytest.mark.asyncio
async def test_generate_async(service: GenAIImageGenerator, blob_storage: BaseBlobStorage):
    async for blob_create in service.generate_async("Big black dog."):
        assert blob_create.metadata
        assert blob_create.metadata.content_type
        assert blob_create.content

        blob_create.name = f"async_{service.model_name}.png"
        blob_storage.upload(Blob.create(blob_create))
