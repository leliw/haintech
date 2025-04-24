import pytest

from haintech.pipelines import Limit, Pipeline


@pytest.mark.asyncio
async def test_top_x():
    pl = Pipeline[str, str]([Limit(2)])
    ret = await pl.run_and_return(["a", "b", "c"])
    assert ret == ["a", "b"]


@pytest.mark.asyncio
async def test_range_x_y():
    pl = Pipeline[str, str]([Limit(1,3)])
    ret = await pl.run_and_return(["a", "b", "c", "d", "e", "f"])
    assert ret == ["b", "c", "d"]
