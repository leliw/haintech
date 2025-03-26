import asyncio
from typing import AsyncGenerator

import pytest

from haintech.pipelines.filter_processor import FilterProcessor


async def async_generator(count: int):
    # normal loop
    for i in range(count):
        # block to simulate doing work
        await asyncio.sleep(0.01)
        # yield the result
        yield i


@pytest.mark.asyncio
async def test_set_source_and_run():
    # Given: a processor with a source
    p = FilterProcessor(lambda x: x < 3)
    p.set_source(async_generator)
    # When: running the processor
    ret = p.process(5)
    # Then: the processor should return a generator
    assert isinstance(ret, AsyncGenerator)
    # And: the generator should return the expected values
    assert list([r async for r in ret]) == [0, 1, 2]
