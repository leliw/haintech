import pytest
from ampf.base import CollectionDef
from ampf.local import LocalFactory
from pydantic import BaseModel

from haintech.pipelines import Pipeline
from haintech.pipelines.ampf import StorageIterator
from haintech.pipelines.ampf.storage_writer import StorageWriter
from haintech.pipelines.progress_tracker import ProgressTracker


@pytest.fixture
def factory(tmp_path):
    return LocalFactory(tmp_path)


class D(BaseModel):
    page_no: str
    content: str


@pytest.fixture
def data1():
    return D(page_no="1", content="test")


@pytest.fixture
def data2():
    return D(page_no="2", content="test")


@pytest.mark.asyncio
async def test_iterator(factory, data1, data2):
    # Given: Storage with saved data
    storage = factory.create_storage("test", D)
    storage.save(data1)
    storage.save(data2)
    # And: Pipeline with StorageIterator
    pl = Pipeline(
        [
            StorageIterator(storage),
        ]
    )
    # When: Run pipeline without any data
    ret = await pl.run_and_return(None)
    ret.sort(key=lambda x: x.page_no)
    # Then: Returns all data from storage
    assert ret == [data1, data2]


@pytest.mark.asyncio
async def test_iterator_with_progress_tracker(factory, data1, data2):
    # Given: Storage with saved data
    storage = factory.create_storage("test", D)
    storage.save(data1)
    storage.save(data2)
    # And: Progress tracker
    pt = ProgressTracker()
    # And: Pipeline with StorageIterator and StorageWriter
    pl = Pipeline(
        [
            StorageIterator(storage, progress_tracker=pt),
            StorageWriter(storage, progress_tracker=pt),
        ]
    )
    # When: Run pipeline without any data
    ret = await pl.run_and_return(None)
    ret.sort(key=lambda x: x.page_no)
    # Then: Returns all data from storage
    assert ret == [data1, data2]
    # And: Progress tracker is completed
    assert pt.total_steps == 2
    assert pt.is_complete()
    # Clean up
    storage.drop()


class C(BaseModel):
    name: str


@pytest.mark.asyncio
async def test_lambda_storage(factory, data1, data2):
    # Given: Storage with saved data
    storage = factory.create_collection(
        CollectionDef("cs", C, subcollections=[CollectionDef("ds", D)])
    )
    ds = storage.get_collection("X", "ds")
    ds.save(data1)
    ds.save(data2)
    # And: Pipeline with StorageIterator
    pl = Pipeline(
        [
            StorageIterator[str, D](lambda x: storage.get_collection(x, "ds")),
        ]
    )
    # When: Run pipeline without any data
    ret = await pl.run_and_return(["X"])
    ret.sort(key=lambda x: x.page_no)
    # Then: Returns all data from storage
    assert ret == [data1, data2]


if __name__ == "__main__":
    pytest.main([__file__])
