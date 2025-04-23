from pydantic import BaseModel
import pytest

from haintech.pipelines import Pipeline, JsonlWriter


class D(BaseModel):
    page_no: int
    content: str


@pytest.mark.asyncio
async def test_jsonl_witer_with_path(tmp_path):
    # Given: A pipeline with JsonlWriter with path
    pl = Pipeline([JsonlWriter[D](path=tmp_path / "test.jsonl")])
    # When: The pipeline is run with two elements
    await pl.run_and_return(
        [
            D(page_no=1, content="test1"),
            D(page_no=2, content="test2"),
        ]
    )
    # Then: JSONL file is created
    out_path = tmp_path / "test.jsonl"
    assert out_path.exists()
    # And: Contains two passed elements in JSONL format
    with out_path.open() as f:
        assert f.read() == '{"page_no":1,"content":"test1"}\n{"page_no":2,"content":"test2"}\n'


@pytest.mark.asyncio
async def test_jsonl_witer_with_file_name(tmp_path):
    # Given: A pipeline with JsonlWriter with path and key_file_name
    pl = Pipeline([JsonlWriter[D](path=tmp_path, key_file_name="page_no")])
    # When: The pipeline is run with two elements   
    await pl.run_and_return(
        [
            D(page_no=1, content="test1"),
            D(page_no=2, content="test2"),
        ]
    )
    # Then: Two JSONL files are created with names dependent on element
    out_path1 = tmp_path / "1.jsonl"
    assert out_path1.exists()
    out_path2 = tmp_path / "2.jsonl"
    assert out_path2.exists()
    # And: They contain two passed elements in JSONL format
    with out_path1.open() as f:
        assert f.read() == '{"page_no":1,"content":"test1"}\n'
    with out_path2.open() as f:
        assert f.read() == '{"page_no":2,"content":"test2"}\n'