import asyncio
import logging
import pytest

from haintech.pipelines import BaseProcessor, Pipeline, LambdaProcessor


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
