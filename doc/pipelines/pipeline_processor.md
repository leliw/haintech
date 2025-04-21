# PipelineProcessor

The processor that uses other pipeline to process items. It is useful when
a main pipeline processes items with list of another items (subitems)
and you want to process subitems whith another pipeline.

## Constructor

* processors: List[BaseProcessor] = None,
* pipeline: Pipeline[I, O] | Callable[[I], Pipeline[I, O]] = None,

There are three options to define internal pipeline:

### List of processors

The simples and easiest way is to pass list of processors.

```python
PipelineProcessor[str, str](
    [
        LambdaProcessor[str, str](lambda s: f"{s}{s}"),
    ]
),
```

### Pipeline object

There is also possible, to pass an already created pipeline object.

```python
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
```

### Pipeline as function

The most complex and usefull method is to define pipeline as function.
In this case, pipeline is created for each parent item.

```python
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
```

There is also possible to pass function with extra parameters returning the proper function.

```python
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
```

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
