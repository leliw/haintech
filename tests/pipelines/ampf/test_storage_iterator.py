import pytest
from pydantic import BaseModel

from haintech.pipelines import Pipeline
from haintech.pipelines.ampf import StorageIterator


class D(BaseModel):
    page_no: int
    content: str


@pytest.fixture
def data1():
    return D(page_no=1, content="test")


@pytest.fixture
def data2():
    return D(page_no=2, content="test")


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
    # Then: Returns all data from storage
    assert ret == [data1, data2]


if __name__ == "__main__":
    pytest.main([__file__])
