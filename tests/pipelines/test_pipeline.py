import asyncio
import logging
from typing import override

import pytest

from haintech.pipelines import BaseProcessor, LambdaProcessor, Pipeline
from haintech.pipelines.checkpoint_processor import CheckpointProcessor


@pytest.mark.asyncio
async def test_pipeline_sync_values():
    """Pipeline where all processors are synchronous and returns values."""
    pl = Pipeline()
    pl.add_processor(LambdaProcessor(lambda x: x + 1))
    pl.add_processor(LambdaProcessor(lambda x: x * 2))

    ret = await pl.run_and_return(5)
    assert ret == 12


@pytest.mark.asyncio
async def test_pipeline_init_with_processors():
    """Pipeline initialized with a list of processors."""
    pl = Pipeline(
        [
            LambdaProcessor(lambda x: x + 1),
            LambdaProcessor(lambda x: x * 2),
        ]
    )

    ret = await pl.run_and_return(5)
    assert ret == 12


class GenProcessor(BaseProcessor):
    async def process(self, x):
        for i in range(x):
            await asyncio.sleep(0.01)
            yield i


@pytest.mark.asyncio
async def test_pipeline_sync_generator():
    """Pipeline where processor is synchronous and returns generator."""
    pl = Pipeline()
    pl.add_processor(GenProcessor("Gen"))
    ret = await pl.run(5)
    assert list([r async for r in ret]) == [0, 1, 2, 3, 4]


class Mul2Processor(BaseProcessor):
    async def process_item(self, data):
        await asyncio.sleep(0.01)
        return data * 2


@pytest.mark.asyncio
async def test_pipeline_sync_generator_value():
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger("hanintech.pipelines").setLevel(logging.DEBUG)
    pl = Pipeline()
    pl.add_processor(GenProcessor("Gen"))
    pl.add_processor(Mul2Processor("Mul2"))
    ret = await pl.run_and_return(5)
    assert ret == [0, 2, 4, 6, 8]


@pytest.mark.asyncio
async def test_pipeline_run_and_return_list():
    pl = Pipeline()
    pl.add_processor(GenProcessor("Gen"))
    pl.add_processor(Mul2Processor("Mul2"))
    ret = await pl.run_and_return(5)
    assert ret == [0, 2, 4, 6, 8]


class FakeCheckpoint[T](CheckpointProcessor[T]):
    @override
    async def process_item(self, data, **kwargs):
        return data

    @override
    def create_generator(self) -> BaseProcessor[T, T]:
        return None


@pytest.mark.asyncio
async def test_get_step():
    # Given: A pipeline with checkpoint
    pl = Pipeline(
        [
            LambdaProcessor[int, int](lambda x: x + 1, name="L1"),
            FakeCheckpoint[int](),
            LambdaProcessor[int, int](lambda x: x + 2, name="L2"),
        ]
    )
    # When: Steps are taken
    pl0 = pl.get_step(0)
    pl1 = pl.get_step(1)
    # And : Run separately
    ret0 = await pl0.run_and_return(1)
    ret1 = await pl1.run_and_return(1)
    # Then: Results are independent
    assert ret0 == 2
    assert ret1 == 3
