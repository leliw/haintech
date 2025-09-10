import pytest
from ampf.base import Blob
from ampf.in_memory import InMemoryAsyncFactory
from pydantic import BaseModel

from haintech.pipelines import Pipeline
from haintech.pipelines.ampf import BlobStorageWriter


class D(BaseModel):
    page_no: int
    content: str


@pytest.fixture
def metadata():
    return D(page_no=1, content="test")


@pytest.fixture
def factory():
    return InMemoryAsyncFactory()


@pytest.mark.asyncio
async def test_pipe_file_name_key(factory, metadata):
    # Given: Storage
    storage = factory.create_blob_storage("test", D, "text/plain")
    # And: Pipeline with BlobStorageWriter
    pl = Pipeline(
        [
            BlobStorageWriter[D](storage),
        ]
    )
    # When: Run pipeline with blob and metadata
    await pl.run_and_return([Blob(name="1", data=b"test", metadata=metadata)])
    # Then: Blob and metadata are uploaded
    blobs = list(storage.list_blobs())
    assert len(blobs) == 1
    assert blobs[0].name == "1"
    blob = await storage.download_async(blobs[0].name)
    assert  blob.data.getvalue() == b"test"

