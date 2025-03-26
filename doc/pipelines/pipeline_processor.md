# PipelineProcessor

The processor that uses other pipeline to process items. It is useful when
a main pipeline processes items with list of another items (subitems)
and you want to process subitems whith another pipeline.

## Use cases

### Simple use

`PipelineProcessor` receive `List` and return `List`.

```python
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
```

### Use input and output parameters

`PipelineProcessor` can extract attribute for processing and return
result as (another) attribute.

```python
class D(BaseModel):
    page_no: int
    subitems: List[str]

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
```
