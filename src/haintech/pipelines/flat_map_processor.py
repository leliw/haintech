from typing import AsyncIterator, Callable, Iterable, Iterator, override

from .base_processor import BaseProcessor, FieldNameOrLambda


class FlatMapProcessor[I, O](BaseProcessor[I, O]):
    """Processor that iterates over a iterable in the data and yields the items"""

    def __init__(
        self,
        iterable: Callable[[I], Iterable[O]] = None,
        name: str = None,
        output: FieldNameOrLambda = None,
    ):
        """Processor that iterates over a iterable in the data and yields the items.

        Args:
            iterable: A lambda function that takes a pipeline item and returns an iterable.
        """
        super().__init__(name=name, output=output)
        self.iterable = iterable

    def process_flat_map(self, data: I) -> Iterator[O]:
        """Extra iteration based on the expression"""
        if not self.iterable:
            iterator = data
        else:
            iterator = self.iterable(data)
        for item in iterator:
            yield self._put_output_data(data, item)

    @override
    async def process(self, data) -> AsyncIterator[O]:
        """Add extra iteration based on the iterable expression"""
        iterator = self._get_iterator(data)
        if isinstance(iterator, Iterator):
            for data in iterator:
                for item in self.process_flat_map(data):
                    yield item
        else:
            async for data in iterator:
                for item in self.process_flat_map(data):
                    yield item
