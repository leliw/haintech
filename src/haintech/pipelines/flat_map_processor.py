from typing import AsyncGenerator, AsyncIterator, Callable, Iterator, override

from .base_processor import BaseProcessor


class FlatMapProcessor[I, O](BaseProcessor):
    """Processor that iterates over a iterable in the data and yields the items"""

    def __init__(self, iterator: Callable[[I], O], **kwargs):
        """Processor that iterates over a iterable in the data and yields the items.

        Args:
            iterator: A lambda function that takes an item from the data and returns an iterable.
        """
        super().__init__(**kwargs)
        self.iterator = iterator
        self.org_source = None

    @override
    def set_source(self, source: BaseProcessor | AsyncGenerator[I, None]):
        """Store the original source and wrap it with a source_wrapper"""
        super().set_source(source)
        self.org_source = self.source
        self.source = self.source_wrapper

    async def source_wrapper(self, data: I) -> AsyncIterator[O]:
        """Add extra iteration based on the expression to the original source"""
        async for data in self.org_source(data):
            for item in self.process_flat_map(data):
                yield item

    @override
    def generate(self, data) -> Iterator[I] | AsyncIterator[I]:
        """Add extra iteration based on the expression to the original source"""
        ret = super().generate(data)
        if isinstance(ret, Iterator):
            for data in ret:
                yield from self.process_flat_map(data)

    def process_flat_map(self, data: I) -> Iterator[O]:
        """Extra iteration based on the expression"""
        for item in self.iterator(data):
            yield self._put_output_data(data, item)

    @override
    async def process(self, data) -> AsyncIterator[O]:
        """Just override the process method because all the work is done in
        generate method or source_wrapper method"""
        if not self.source:
            source_iterator = self.generate(data)
        else:
            source_iterator = self.source(data)
        if isinstance(source_iterator, Iterator):
            for data in source_iterator:
                yield data
        else:
            async for data in source_iterator:
                yield data
