from typing import Self
import pytest
from ampf.base import CollectionDef
from ampf.in_memory import InMemoryFactory
from ampf.local import LocalFactory
from pydantic import BaseModel

from haintech.pipelines import Pipeline
from haintech.pipelines.ampf import StorageWriter
from haintech.pipelines.lambda_processor import LambdaProcessor
from haintech.pipelines.log_processor import LogProcessor


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
        ]  # type: ignore
    )
    # And: Two steps are taken
    pl0 = pl.get_step(0)
    pl1 = pl.get_step(1)
    # When: Run first step with data
    await pl0.run_and_return(data)
    # And: Run second step with sotrage keys
    ret = await pl1.run_and_return(storage.keys())
    # Then: Both steps were executed
    assert isinstance(ret, list)
    assert ret[0].page_no == 2
    assert ret[0].content == "2"


@pytest.fixture
def factory_l(tmp_path):
    return LocalFactory(tmp_path)


@pytest.mark.asyncio
async def test_lambda_storage(factory_l, data):
    # Given: Two classes parent - child
    class C(BaseModel):
        name: str

    class D(BaseModel):
        name: str
        _c: C = None  # type: ignore

        def set_c(self, c: C) -> Self:
            self._c = c
            return self

    # And: Storage collection for the classes
    storage = factory_l.create_collection(CollectionDef("cs", C, "name", subcollections=[CollectionDef("ds", D)]))
    # And: Pipeline with two writers, first "normal" and second - lambda expression
    pl = Pipeline(
        [
            LambdaProcessor[str, C](lambda x: C(name=x)),
            StorageWriter[C](storage),
            LambdaProcessor[C, D](lambda c: D(name=c.name).set_c(c)),
            LogProcessor(message="{name}"),
            StorageWriter[D](lambda d: storage.get_collection(d._c.name, "ds")),  # type: ignore
        ]  # type: ignore
    )
    # When: Run pipeline
    ret = await pl.run_and_return("xxx")
    # Then: Returns D object (with _c property)
    assert isinstance(ret, D)
    ret._c = None  # type: ignore
    assert ret == D(name="xxx")
    # And: Data is stored
    storage_d = storage.get_collection("xxx", "ds")
    keys = list(storage_d.keys())
    assert len(keys) == 1
    assert storage_d.get(keys[0]) == D(name="xxx")


@pytest.mark.asyncio
async def test_changed_only(factory, data):
    # STEP: 1
    # Given: Storage with key_name
    storage = factory.create_storage("test3", D, "content")
    # And: Pipeline with StorageWriter with changed_only set true
    pl = Pipeline([StorageWriter[D](storage, changed_only=True)])
    # When: Run pipeline with data
    ret = await pl.run_and_return(data)
    # Then: Data is returned
    assert ret == data
    # STEP: 2
    # Given: Second item
    data2 = D(page_no=2, content="2")
    # When: Run pipeline with data and data2
    ret = await pl.run_and_return([data, data2])
    # Then: Only data2 is returned
    assert ret == [data2]

if __name__ == "__main__":
    pytest.main([__file__])
