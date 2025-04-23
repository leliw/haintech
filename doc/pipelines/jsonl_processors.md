# JSONL processors

Processors that operates on JSONL format.
JSONL is a list of JSON objects, one per line.

## JsonlWriter

Adds input data to JSONL file. 
Destination file name can be passed as `Path` to constructor
or as a key name of data object.

## Usage

One output file for all data.

```python
class D(BaseModel):
    page_no: int
    content: str

pl = Pipeline([JsonlWriter[D](path=tmp_path / "test.jsonl")])

await pl.run_and_return(
    [
        D(page_no=1, content="test1"),
        D(page_no=2, content="test2"),
    ]
)
```

Many output files.

```python
pl = Pipeline([JsonlWriter[D](path=tmp_path, key_file_name="page_no")])
await pl.run_and_return(
    [
        D(page_no=1, content="test1"),
        D(page_no=2, content="test2"),
    ]
)

tmp_path / "1.jsonl"
assert out_path1.exists()
out_path2 = tmp_path / "2.jsonl"
assert out_path2.exists()
```
