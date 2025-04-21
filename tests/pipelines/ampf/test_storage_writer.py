import pytest
from ampf.in_memory import InMemoryFactory
from pydantic import BaseModel

from haintech.pipelines import Pipeline
from haintech.pipelines.ampf import StorageWriter
from haintech.pipelines.lambda_processor import LambdaProcessor


class D(BaseModel):
    page_no: int
    content: str


@pytest.fixture
def data():
    return D(page_no=1, content="test")


@pytest.fixture
def factory():
    return InMemoryFactory()


@pytest.mark.asyncio
async def test_pipe_key_name(factory, data):
    # Given: Storage
    storage = factory.create_storage("test", D)
    # And: Pipeline with BlobStorageWriter where key_name is set
    pl = Pipeline(
        [
            StorageWriter(storage, key_name="page_no"),
        ]
    )
    # When: Run pipeline with data
    await pl.run_and_return(data)
    # Then: Data is stored
    keys = list(storage.keys())
    assert len(keys) == 1
    assert keys[0] == 1
    assert storage.get(keys[0]) == data
    # Clean up
    storage.drop()


@pytest.mark.asyncio
async def test_pipe_key_name_expr(factory, data):
    # Given: Storage
    storage = factory.create_storage("test2", D)
    # And: Pipeline with BlobStorageWriter where key_name is set as lambda
    pl = Pipeline(
        [
            StorageWriter(storage, lambda x: f"xx{x.page_no}"),
        ]
    )
    # When: Run pipeline with data
    await pl.run_and_return(data)
    # Then: Data is stored
    keys = list(storage.keys())
    assert len(keys) == 1
    assert keys[0] == "xx1"
    assert storage.get(keys[0]) == data


@pytest.mark.asyncio
async def test_pipe_without_key_name(factory, data):
    # Given: Storage with key_name
    storage = factory.create_storage("test3", D, "content")
    # And: Pipeline with BlobStorageWriter without key_name
    pl = Pipeline(
        [
            StorageWriter[D](storage),
        ]
    )
    # When: Run pipeline with data
    await pl.run_and_return(data)
    # Then: Data is stored
    keys = list(storage.keys())
    assert len(keys) == 1
    assert keys[0] == "test"
    assert storage.get(keys[0]) == data


@pytest.mark.asyncio
async def test_pipe_divided_into_steps(factory, data):
    # Given: Storage with key_name
    storage = factory.create_storage("test3", D, "content")
    # And: Pipeline with StorageWriter
    pl = Pipeline(
        [
            LambdaProcessor[D, D](lambda d: setattr(d, "page_no", 2)),
            StorageWriter(storage),
            LambdaProcessor[D, D](lambda d: setattr(d, "content", "2")),
        ]
    )
    # And: Two steps are taken
    pl0 = pl.get_step(0)
    pl1 = pl.get_step(1)
    # When: Run first step with data
    await pl0.run_and_return(data)
    # And: Run second step with sotrage keys
    ret = await pl1.run_and_return(storage.keys())
    # Then: Both steps were executed
    assert ret[0].page_no == 2
    assert ret[0].content == "2"


if __name__ == "__main__":
    pytest.main([__file__])
