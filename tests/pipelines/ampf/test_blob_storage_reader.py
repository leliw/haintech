import pytest
from ampf.in_memory import InMemoryFactory
from pydantic import BaseModel

from haintech.pipelines import Pipeline
from haintech.pipelines.ampf import BlobStorageReader


class D(BaseModel):
    page_no: int
    content: str


@pytest.fixture
def metadata():
    return D(page_no=1, content="test")


@pytest.fixture
def factory():
    return InMemoryFactory()


@pytest.mark.asyncio
async def test_pipe_file_name_key(factory, metadata):
    # Given: Storage with uploaded blob
    storage = factory.create_blob_storage("test", D, "text/plain")
    storage.upload_blob(1, b"test", metadata)
    # And: Pipeline with BlobStorageReader
    pl = Pipeline(
        [
            BlobStorageReader(storage),
        ]
    )
    # When: Run pipeline with key of stored blob
    ret = await pl.run_and_return(1)
    # Then: Returns blom and metadata
    assert ret[0] == "test"
    assert ret[1] == metadata
