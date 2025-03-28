import pytest

from haintech.pipelines.group_processor import GroupProcessor
from haintech.pipelines.lambda_processor import LambdaProcessor
from haintech.pipelines.pipeline import Pipeline


@pytest.mark.asyncio
async def test_list_given_by_argument():
    # Given: A flat list
    stream = [
        {"i": 0, "l": 1},
        {"i": 0, "l": 2},
        {"i": 0, "l": 3},
        {"i": 1, "l": 4},
        {"i": 1, "l": 5},
        {"i": 1, "l": 6},
    ]
    # Given: Pipeline with GroupProcessor
    pl = Pipeline(
        [
            GroupProcessor(
                group_by=lambda x: x["i"],
                init_group=lambda k, d: {"i": k, "l": []},
                aggregate=lambda g, k, d: g["l"].append(d["l"]),
            )
        ]
    )
    # When: Pipeline is run with the list as argument
    ret = await pl.run_and_return(stream)
    # Then: The list is grouped by "i"
    assert [{"i": 0, "l": [1, 2, 3]}, {"i": 1, "l": [4, 5, 6]}] == ret

@pytest.mark.asyncio
async def test_list_given_by_source():
    # Given: A flat list
    stream = [
        {"i": 0, "l": 1},
        {"i": 0, "l": 2},
        {"i": 0, "l": 3},
        {"i": 1, "l": 4},
        {"i": 1, "l": 5},
        {"i": 1, "l": 6},
    ]
    # Given: Pipeline with GroupProcessor
    pl = Pipeline(
        [
            LambdaProcessor(lambda x: x),
            GroupProcessor(
                group_by=lambda x: x["i"],
                init_group=lambda k, d: {"i": k, "l": []},
                aggregate=lambda g, k, d: g["l"].append(d["l"]),
            )
        ]
    )
    # When: Pipeline is run with the list as argument
    ret = await pl.run_and_return(stream)
    # Then: The list is grouped by "i"
    assert [{"i": 0, "l": [1, 2, 3]}, {"i": 1, "l": [4, 5, 6]}] == ret