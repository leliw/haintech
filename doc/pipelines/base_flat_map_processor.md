# FlatMapProcessor

The `BaseFlatMapProcessor` is a base class for all processors that flatten a list of lists 
into a single list. It is similar to the `flatMap` operation in functional programming.

## Usage

Just implement `process_flat_map()` method returning iteragtor.

```python
@override
def process_flat_map(self, data: I) -> Iterator[O]:
    for item in self.extract_items(data):
        yield self._put_output_data(data, item)
```
