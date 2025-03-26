from pydantic import BaseModel
import pytest

from haintech.pipelines import Pipeline
from haintech.pipelines.ampf import BlobStorageWriter

from ampf.in_memory import InMemoryFactory


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
    # Given: Storage
    storage = factory.create_blob_storage("test", D, "text/plain")
    # And: Pipeline with BlobStorageWriter 
    pl = Pipeline(
        [
            BlobStorageWriter(storage, "page_no"),
        ]
    )
    # When: Run pipeline with blob and metadata
    await pl.run_and_return(("test", metadata))
    # Then: Blob and metadata are uploaded
    keys = list(storage.keys())
    assert len(keys) == 1
    assert keys[0] == 1
    assert storage.download_blob(keys[0]) == b"test"
    assert storage.get_metadata(keys[0]) == metadata


@pytest.mark.asyncio
async def test_pipe_file_name_expr(factory, metadata):
    # Given:
    storage = factory.create_blob_storage("test2", D)
    pl = Pipeline(
        [
            BlobStorageWriter(storage, lambda x: f"xx{x.page_no}"),
        ]
    )
    # When:
    await pl.run_and_return(("test", metadata))
    # Then:
    keys = list(storage.keys())
    assert len(keys) == 1
    assert keys[0] == "xx1"
    assert storage.download_blob(keys[0]) == b"test"
    assert storage.get_metadata(keys[0]) == metadata
