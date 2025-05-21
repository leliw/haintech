import pytest
from ampf.base import CollectionDef, Blob
from ampf.local_async import AsyncLocalFactory
from pydantic import BaseModel

from haintech.pipelines import Pipeline
from haintech.pipelines.ampf import BlobStorageIterator, BlobStorageWriter
from haintech.pipelines.progress_tracker import ProgressTracker


@pytest.fixture
def factory(tmp_path):
    return AsyncLocalFactory(tmp_path)


class D(BaseModel):
    page_no: str
    content: str


@pytest.fixture
def data1():
    return Blob[D](key="1", data=b"1", metadata=D(page_no="1", content="test"))


@pytest.fixture
def data2():
    return Blob[D](key="2", data=b"2", metadata=D(page_no="2", content="test"))


@pytest.mark.asyncio
async def test_blob_iterator(factory, data1, data2):
    # Given: Storage with saved blobs
    storage = factory.create_blob_storage("test", D)
    await storage.upload_async(data1)
    await storage.upload_async(data2)
    # And: Pipeline with BlobStorageIterator
    pl = Pipeline(
        [
            BlobStorageIterator(storage),
        ]
    )
    # When: Run pipeline without any data
    ret = await pl.run_and_return(None)
    ret.sort(key=lambda x: x.metadata.page_no)
    # Then: Returns all blobs from storage
    assert ret == [data1, data2]


@pytest.mark.asyncio
async def test_iterator_with_progress_tracker(factory, data1, data2):
    # Given: Storage with saved blobs
    storage = factory.create_blob_storage("test", D)
    await storage.upload_async(data1)
    await storage.upload_async(data2)
    # And: Progress tracker
    pt = ProgressTracker()
    # And: Pipeline with StorageIterator and StorageWriter
    pl = Pipeline(
        [
            BlobStorageIterator(storage, progress_tracker=pt),
            BlobStorageWriter(storage, progress_tracker=pt),
        ]
    )
    # When: Run pipeline without any data
    ret = await pl.run_and_return(None)
    ret.sort(key=lambda x: x.metadata.page_no)
    # Then: Returns all blobs from storage
    assert ret == [data1, data2]
    # And: Progress tracker is completed
    assert pt.total_steps == 2
    assert pt.is_complete()


class C(BaseModel):
    name: str


@pytest.mark.asyncio
async def test_lambda_storage(factory, data1, data2):
    # Given: Storage with saved data
    storage = factory.create_blob_storage("test/X", D)
    await storage.upload_async(data1)
    await storage.upload_async(data2)
    # And: Pipeline with StorageIterator
    pl = Pipeline(
        [
            BlobStorageIterator[str, D](lambda x: factory.create_blob_storage(f"test/{x}", D)),
        ]
    )
    # When: Run pipeline with argument "X"
    ret = await pl.run_and_return(["X"])
    ret.sort(key=lambda x: x.metadata.page_no)
    # Then: Returns all blobs from storage "X" directory
    assert ret == [data1, data2]

if __name__ == "__main__":
    pytest.main([__file__])
