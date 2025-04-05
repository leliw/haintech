from pydantic import BaseModel
import pytest

from haintech.pipelines.lambda_processor import LambdaProcessor
from haintech.pipelines.pipeline import Pipeline


@pytest.mark.asyncio
async def test_ret_value():
    # Given: Pipeline with lambda returning value
    pl = Pipeline([LambdaProcessor[int, int](lambda x: x + 1)])
    # When: Run the pipeline
    ret = await pl.run_and_return(5)
    # Then: The pipeline should return expresion result
    assert ret == 6


@pytest.mark.asyncio
async def test_ret_none():
    # Given: Pipeline with lambda NOT returning value
    pl = Pipeline([LambdaProcessor[int, int](lambda x: print(x + 1))])
    # When: Run the pipeline
    ret = await pl.run_and_return(5)
    # Then: The pipeline should return input data
    assert ret == 5


@pytest.mark.asyncio
async def test_pydantic():
    # Given: Pydantic class
    class D(BaseModel):
        val: int
    # And: Pipeline with lambda changing attribute
    pl = Pipeline([LambdaProcessor[int, int](lambda d: setattr(d, "val", d.val + 1))])
    # When: Run the pipeline
    ret = await pl.run_and_return(D(val=5))
    # Then: The pipeline should return input data
    assert ret.val == 6
