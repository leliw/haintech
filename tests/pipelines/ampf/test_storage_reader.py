import pytest
from ampf.in_memory import InMemoryFactory
from pydantic import BaseModel

from haintech.pipelines import Pipeline
from haintech.pipelines.ampf import StorageReader


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
    # Given: Storage with saved data
    storage = factory.create_storage("test", D)
    storage.save(data)
    # And: Pipeline with StorageReader
    pl = Pipeline(
        [
            StorageReader(storage),
        ]
    )
    # When: Run pipeline with key of saved data
    ret = await pl.run_and_return(1)
    # Then: Returns data
    assert ret == data


if __name__ == "__main__":
    pytest.main([__file__])
