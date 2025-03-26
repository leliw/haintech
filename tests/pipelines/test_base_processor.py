import asyncio
from typing import AsyncGenerator, Iterator, override
import pytest

from haintech.pipelines import Pipeline, BaseProcessor, LambdaProcessor


@pytest.mark.asyncio
async def test_not_implemented_error():
    """Test that pure BaseProcessor raises NotImplementedError."""
    p = BaseProcessor("Test")
    with pytest.raises(NotImplementedError):
        await p.process_item(5)


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
    p = LambdaProcessor(lambda x: x)
    p.set_source(async_generator)
    # When: running the processor
    ret = p.process(5)
    # Then: the processor should return a generator
    assert isinstance(ret, AsyncGenerator)
    # And: the generator should return the expected values
    assert list([r async for r in ret]) == [0, 1, 2, 3, 4]


@pytest.mark.asyncio
async def test_run_two_processors():
    # Given: a processor with a source
    p1 = LambdaProcessor(lambda x: x)
    p1.set_source(async_generator)
    p2 = LambdaProcessor(lambda x: x + 1)
    p2.set_source(p1)
    # When: running the processor
    ret = p2.process(5)
    # Then: the processor should return a generator
    assert isinstance(ret, AsyncGenerator)
    # And: the generator should return the expected values
    assert list([r async for r in ret]) == [1, 2, 3, 4, 5]


class GenProcessor(BaseProcessor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @override
    def generate(self, x) -> Iterator:
        for i in range(x):
            yield {"i": i}

    @override
    async def process_item(self, data):
        return data


@pytest.mark.asyncio
async def test_output_key_name_with_generator():
    """Test returning result in dict when result_key_name is set."""
    pl = Pipeline(
        [
            GenProcessor(),
            LambdaProcessor(lambda x: x["i"] + 1, output="ret"),
        ]
    )

    ret = await pl.run_and_return(6)

    assert isinstance(ret, list)
    assert ret[5]["ret"] == 6


@pytest.mark.asyncio
async def test_input_field_name():
    """Test using input field name."""
    p = LambdaProcessor(lambda x: x + 1, input="i")

    ret = await p.wrap_process_item({"i": 5})

    assert ret == 6


@pytest.mark.asyncio
async def test_input_lambda():
    """Test using input callable."""
    p = LambdaProcessor(lambda x: x + 1, input=lambda x: x["i"])

    ret = await p.wrap_process_item({"i": 5})

    assert ret == 6


@pytest.mark.asyncio
async def test_input_error():
    """Test wrong input type."""
    p = LambdaProcessor(lambda x: x + 1, input=5)

    with pytest.raises(TypeError):
        await p.wrap_process_item({"i": 5})


@pytest.mark.asyncio
async def test_output_lambda():
    """Test using output field name."""
    p = LambdaProcessor(lambda x: x + 1, output=lambda d, r: {"ret": r * r})

    ret = await p.wrap_process_item(5)

    assert ret["ret"] == 36


@pytest.mark.asyncio
async def test_output_lambda_with_generator():
    """Test returning result in dict when result_key_name is set."""
    pl = Pipeline(
        [
            GenProcessor(output=lambda d, r: dict(d=d, ret=r, l=len(r))),
        ]
    )

    ret = await pl.run_and_return(6)

    assert isinstance(ret[0], dict)
    assert ret[5]["ret"] == {"i": 5}


@pytest.mark.asyncio
async def test_output_error():
    """Test wrong output type."""
    p = LambdaProcessor(lambda x: x + 1, output=5)

    with pytest.raises(TypeError):
        await p.wrap_process_item(5)
