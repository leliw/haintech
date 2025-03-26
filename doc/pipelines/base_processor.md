# BaseProcessor

```python
class BaseProcessor[I, O](ABC):
```

`BaseProcessor` is generic class which should be inherited by all processors.
All processors should inherit this class. Type parameters:

* I - type of input items
* O - type of output items

## Constructor parameters

```python
def __init__(
    self,
    name: str = None,
    input: FieldNameOrLambda = None,
    output: FieldNameOrLambda = None,
):
    self.name = name or self.__class__.__name__
    self.input = input
    self.output = output
    self.source = None
```

## Methods

### process_item()

```python
async def process_item(self, data: I, **kwargs) -> O:
```

The main method that processes data. It processes all items one by one In the pipeline.
Override this method in subclasses.

### process()

```python
async def process(self, data) -> AsyncIterator[O]:
    iterator = self.source(data) if self.source else self.generate(data)
    if isinstance(iterator, Iterator):
        for data in iterator:
            yield await self.wrap_process_item(data)
    else:
        async for data in iterator:
            yield await self.wrap_process_item(data)
```

The entry method that iterates over data. It calls previous processor in pipeline or creates
generator for entire pipeline input data. Override this method in subclasses if items numer
changes between input and output, e.g operations:

* filter()
* flatMap()
* group()

### generate()

```python
def generate(self, data: I | Iterator[I]) -> Iterator[I]:
    iterator = data if isinstance(data, Iterator) else [data]
    for item in iterator:
        yield item
```

It is called when current processor if the first in pipeline. It iterates
over pipeline input data or create iterator if the data is a single item.

## Type FieldNameOrLambda

```python
FieldNameOrLambda = Union[str, Callable[[], Any]]
```

`FieldNameOrLambda` is a type of parameters that appoint some value from bigger structure
(usually a Pydantic class). It can be:

* `str` - just name of attribute that is used
* `callable` - function (lambda) that returns some value

When it is used as `input` parameter it specify how to get value **processed** by current processor.

* `input="key"` - processes value from `key` attribute of processed item
* `input=lambda i: f"{i.key}.json"` - processes value returned by lambda expression

Whent it is used as `output` parameter it specify where to put value or just value **returned** by current processor.

* `output="key"` - result is stored in `key` attribute of processed item and item is returned
* `output=lambda r: f"{r.file_name}.json"` - returned is result of lambda expression where input parameter is result of processor
