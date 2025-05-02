from typing import List

import pytest
from pydantic import BaseModel

from haintech.pipelines.flat_map_processor import FlatMapProcessor
from haintech.pipelines.lambda_processor import LambdaProcessor
from haintech.pipelines.pipeline import Pipeline


@pytest.mark.asyncio
async def test_flat_list_by_default():
    # Given: Pipeline with FlatMapProcessor without any argument
    pl = Pipeline([FlatMapProcessor[List[int], int]()])
    # When: Pipeline is run with list of lists
    ret = await pl.run_and_return([[1, 2], [3, 4]])
    # Then: The list is flattened
    assert ret == [1, 2, 3, 4]


class D(BaseModel):
    page_no: int
    subitems: List[str]


@pytest.mark.asyncio
async def test_pydantic_and_iterable_parameter():
    pl = Pipeline[D, str](
        [
            FlatMapProcessor[D, str](lambda d: d.subitems),
        ]
    )
    ret = await pl.run_and_return(D(page_no=1, subitems=["a", "b"]))
    assert ret == ["a", "b"]


@pytest.mark.asyncio
async def test_input_lambda_parameter():
    # Given: FlatMapProcessor with input as lambda
    pl = Pipeline[D, str](
        [
            FlatMapProcessor[D, str](input=lambda d: d.subitems),
        ]
    )
    # When: Run
    ret = await pl.run_and_return(D(page_no=1, subitems=["a", "b"]))
    # Then: Return subitems
    assert ret == ["a", "b"]


@pytest.mark.asyncio
async def test_input_field_name_parameter():
    # Given: FlatMapProcessor with input as lambda
    pl = Pipeline[D, str](
        [
            FlatMapProcessor[D, str](input="subitems"),
        ]
    )
    # When: Run
    ret = await pl.run_and_return(D(page_no=1, subitems=["a", "b"]))
    # Then: Return subitems
    assert ret == ["a", "b"]


@pytest.mark.asyncio
async def test_pydantic_and_extract_items_and_output_parameters():
    pl = Pipeline[D, str](
        [
            FlatMapProcessor[D, str](
                extract_items=lambda d: d.subitems,
                output=lambda d, r: r + "x",
            ),
        ]
    )
    ret = await pl.run_and_return(D(page_no=1, subitems=["a", "b"]))
    assert ret == ["ax", "bx"]


@pytest.mark.asyncio
async def test_flat_list_given_by_argument():
    # Given: A list of object with list inside each of object
    stream = [{"i": 0, "l": [1, 2, 3]}, {"i": 1, "l": [4, 5, 6]}]
    # Given: Pipeline with FlatMapProcessor
    pl = Pipeline([FlatMapProcessor(lambda x: x["l"])])
    # When: Pipeline is run with the list as argument
    ret = await pl.run_and_return(stream)
    # Then: The list is flattened
    assert ret == [1, 2, 3, 4, 5, 6]


@pytest.mark.asyncio
async def test_flat_list_given_by_source():
    # Given: A list of object with list inside each of object
    stream = [{"i": 0, "l": [1, 2, 3]}, {"i": 1, "l": [4, 5, 6]}]
    # Given: Pipeline with FlatMapProcessor and extra processor before
    pl = Pipeline([LambdaProcessor(lambda x: x), FlatMapProcessor(lambda x: x["l"])])
    # When: Pipeline is run with the list
    ret = await pl.run_and_return(stream)
    # Then: The list is flattened
    assert ret == [1, 2, 3, 4, 5, 6]


@pytest.mark.asyncio
async def test_flat_list_with_output():
    # Given: A list of object with list inside each of object
    stream = [{"i": 0, "l": [1, 2, 3]}, {"i": 1, "l": [4, 5, 6]}]
    # Given: Pipeline with FlatMapProcessor with output argument
    pl = Pipeline(
        [FlatMapProcessor(lambda x: x["l"], output=lambda d, r: {"i": d["i"], "l": r})]
    )
    # When: Pipeline is run with the list
    ret = await pl.run_and_return(stream)
    # Then: The list is flattened and converted to the output format
    assert ret == [
        {"i": 0, "l": 1},
        {"i": 0, "l": 2},
        {"i": 0, "l": 3},
        {"i": 1, "l": 4},
        {"i": 1, "l": 5},
        {"i": 1, "l": 6},
    ]
