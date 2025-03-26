from pydantic import BaseModel
import pytest

from haintech.pipelines import Pipeline, JsonWriter


class D(BaseModel):
    page_no: int
    content: str


@pytest.mark.asyncio
async def test_pipe_file_name_key(tmp_path):
    pl = Pipeline(
        [JsonWriter(dir_path=tmp_path, file_name="page_no")]
    )

    await pl.run_and_return(D(page_no=1, content="test"))

    out_path = tmp_path / "1.json"
    assert out_path.exists()
    with out_path.open() as f:
        assert f.read() == '{\n  "page_no": 1,\n  "content": "test"\n}'


@pytest.mark.asyncio
async def test_pipe_file_name_expr(tmp_path):
    pl = Pipeline(
        [JsonWriter(dir_path=tmp_path, file_name=lambda x: f"{x.page_no:03}")]
    )

    await pl.run_and_return(D(page_no=1, content="test"))

    out_path = tmp_path / "001.json"
    assert out_path.exists()
    with out_path.open() as f:
        assert f.read() == '{\n  "page_no": 1,\n  "content": "test"\n}'

@pytest.mark.asyncio
async def test_dict(tmp_path):
    pl = Pipeline(
        [JsonWriter(dir_path=tmp_path, file_name="page_no")]
    )

    await pl.run_and_return({"page_no": 1, "content": "test"})

    out_path = tmp_path / "1.json"
    assert out_path.exists()
    with out_path.open() as f:
        assert f.read() == '{\n  "page_no": 1,\n  "content": "test"\n}'