from typing import Callable, List

import pytest
from pydantic import BaseModel

from haintech.pipelines import LambdaProcessor, Pipeline, PipelineProcessor


class D(BaseModel):
    page_no: int
    subitems: List[str]


@pytest.mark.asyncio
async def test_as_processor():
    # Given: A pipeline with subpipeline with str as input
    pl = Pipeline[D, List[str]](
        [
            LambdaProcessor[D, str](lambda d: d.subitems[0]),
            PipelineProcessor[str, str](
                [
                    LambdaProcessor[str, str](lambda s: f"{s}{s}"),
                ]
            ),
        ]
    )
    # When: Run the pipeline with object with list
    ret = await pl.run_and_return(
        D(
            page_no=1,
            subitems=["a", "b", "c"],
        )
    )
    # Then: Returns str
    assert ret == "aa"


@pytest.mark.asyncio
async def test_as_flatMap():
    # Given: A pipeline with subpipeline with list as input
    pl = Pipeline[D, List[str]](
        [
            LambdaProcessor[D, List[str]](lambda d: d.subitems),
            PipelineProcessor[List[str], List[str]](
                [
                    LambdaProcessor[str, str](lambda s: f"{s}{s}"),
                ]
            ),
        ]
    )
    # When: Run the pipeline with object with list
    ret = await pl.run_and_return(
        D(
            page_no=1,
            subitems=["a", "b", "c"],
        )
    )
    # Then: Returns list
    assert ret == ["aa", "bb", "cc"]


@pytest.mark.asyncio
async def test_as_flatMap_with_input():
    # Given: A pipeline with subpipeline where input parameter is set
    pl = Pipeline[D, List[str]](
        [
            PipelineProcessor[List[str], List[str]](
                input="subitems",
                processors=[
                    LambdaProcessor[str, str](lambda s: f"{s}{s}"),
                ],
            ),
        ]
    )
    # When: Run the pipeline with object with list
    ret = await pl.run_and_return(
        D(
            page_no=1,
            subitems=["a", "b", "c"],
        )
    )
    # Then: Returns list
    assert ret == ["aa", "bb", "cc"]


@pytest.mark.asyncio
async def test_as_flatMap_with_input_and_output():
    # Given: A pipeline with subpipeline where input and ouput parameter are set
    pl = Pipeline[D, D](
        [
            PipelineProcessor[List[str], List[str]](
                input="subitems",
                processors=[
                    LambdaProcessor[str, str](lambda s: f"{s}{s}"),
                ],
                output="subitems",
            ),
        ]
    )
    # When: Run the pipeline with object with list
    ret = await pl.run_and_return(
        D(
            page_no=1,
            subitems=["a", "b", "c"],
        )
    )
    # Then: Returns object
    assert isinstance(ret, D)
    # And: Attribute is changed
    assert ret.subitems == ["aa", "bb", "cc"]


@pytest.mark.asyncio
async def test_pipeline_as_object():
    # Given: Pipeline as function
    pipeline =  Pipeline[str, str]([LambdaProcessor[str, str](lambda s: f"{s}{s}")])

    # And: Parent pipeline with PipelineProcessor
    pl = Pipeline[D, List[str]](
        [
            PipelineProcessor[List[str], List[str]](
                input="subitems",
                pipeline=pipeline,
            ),
        ]
    )
    # When: Run the pipeline with object with list
    ret = await pl.run_and_return(
        D(
            page_no=1,
            subitems=["a", "b", "c"],
        )
    )
    # Then: Returns list
    assert ret == ["aa", "bb", "cc"]


@pytest.mark.asyncio
async def test_pipeline_as_function():
    # Given: Pipeline as function
    def create_pipeline(data) -> Pipeline:
        return Pipeline[str, str]([LambdaProcessor[str, str](lambda s: f"{s}{s}")])

    # And: Parent pipeline with PipelineProcessor
    pl = Pipeline[D, List[str]](
        [
            PipelineProcessor[List[str], List[str]](
                input="subitems",
                pipeline=create_pipeline,
            ),
        ]
    )
    # When: Run the pipeline with object with list
    ret = await pl.run_and_return(
        D(
            page_no=1,
            subitems=["a", "b", "c"],
        )
    )
    # Then: Returns list
    assert ret == ["aa", "bb", "cc"]

@pytest.mark.asyncio
async def test_pipeline_as_param_function():
    # Given: Pipeline as function returning function

    def create_pipeline(init_param: str) -> Callable:
        def create_pipeline(data) -> Pipeline:
            return Pipeline[str, str]([LambdaProcessor[str, str](lambda s: f"{s}{init_param}{s}")])
        return create_pipeline

    # And: Parent pipeline with PipelineProcessor
    pl = Pipeline[D, List[str]](
        [
            PipelineProcessor[List[str], List[str]](
                input="subitems",
                pipeline=create_pipeline("x"),
            ),
        ]
    )
    # When: Run the pipeline with object with list
    ret = await pl.run_and_return(
        D(
            page_no=1,
            subitems=["a", "b", "c"],
        )
    )
    # Then: Returns list
    assert ret == ["axa", "bxb", "cxc"]
