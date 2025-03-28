from typing import AsyncGenerator, AsyncIterator, Callable, Iterator, override

from pydantic import BaseModel

from .base_processor import BaseProcessor


class GroupProcessor[I, K, O](BaseProcessor):
    """GroupProcessor is a processor that groups the data based on the expression"""

    def __init__(
        self,
        group_by: str | Callable[[I], K],
        init_group: Callable[[K, I], O],
        aggregate: Callable[[O, K, I], O],
        **kwargs,
    ):
        """Initialize the GroupProcessor with the expression

        Args:
            group_by: The expression to group by, gets input data and returns the key
            init_group: The function to initialize the group, gets the key and the input data and returns the group (output data)
            aggregate: The function to aggregate the data to the group, gets the group (output data) and the input data and returns the group (output data) or None
        """
        super().__init__(**kwargs)
        self.group_by = group_by
        self.init_group = init_group
        self.aggregate = aggregate
        self.org_source = None

    @override
    def set_source(self, source: BaseProcessor | AsyncGenerator[I, None]):
        """Store the original source and wrap it with a source_wrapper"""
        super().set_source(source)
        self.org_source = self.source
        self.source = self.source_wrapper

    async def source_wrapper(self, data: I) -> AsyncIterator[O]:
        """Add extra iteration based on the expression to the original source"""
        async for item in self.aprocess_grouping(self.org_source(data)):
            yield item

    @override
    def generate(self, data) -> Iterator[I] | AsyncIterator[I]:
        """Add extra iteration based on the expression to the original source"""
        iterator = super().generate(data)
        if isinstance(iterator, Iterator):
            yield from self.process_grouping(iterator)

    async def aprocess_grouping(self, iterator: Iterator[I]) -> AsyncIterator[O]:
        """Extra iteration based on the expression"""
        prev_key = None
        group = None
        async for item in iterator:
            key = self._get_grouping_key(item)
            if prev_key is None:
                prev_key = key
            if key != prev_key:
                yield group
                group = None
                prev_key = key
            if group is None:
                group = self.init_group(key, item)
            group = self.aggregate(group, key, item) or group
        if group:
            yield group

    def process_grouping(self, iterator: Iterator[I]) -> Iterator[O]:
        """Iterate over the data and group them based on the expression"""
        prev_key = None
        group = None
        for item in iterator:
            key = self._get_grouping_key(item)
            if prev_key is None:
                prev_key = key
            if key != prev_key:
                yield group
                group = None
                prev_key = key
            if group is None:
                group = self.init_group(key, item)
            group = self.aggregate(group, key, item) or group
        if group:
            yield group

    def _get_grouping_key(self, data: I) -> str:
        """Get the grouping key based on the expression"""
        if isinstance(self.group_by, str):
            return (
                getattr(data, self.group_by)
                if isinstance(data, BaseModel)
                else data[self.group_by]
            )
        elif callable(self.group_by):
            return self.group_by(data)
        else:
            raise TypeError("Wrong input type")

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
