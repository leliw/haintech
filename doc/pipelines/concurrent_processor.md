# ConcurrentProcessor

Processor that processes data concurrently.
The output order is not guaranteed to be the same as the input order.
The number of concurrent tasks is limited by `max_concurrent` constructor parameter.

## Usage

Just use it as base class.

```python
class TestProcessor[I, O](ConcurrentProcessor[I, O]):
    @override
    async def process_item(self, data: int) -> int:
        await asyncio.sleep(5 - data)
        return data

p = TestProcessor(max_concurrent=5)
ret = p.process([0, 1, 2, 3, 4])
assert list([r async for r in ret]) == [4, 3, 2, 1, 0]
```
