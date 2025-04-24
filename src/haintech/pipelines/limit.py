from typing import AsyncIterator, Iterator, override
from .base_processor import BaseProcessor, FieldNameOrLambda


class Limit[I, O](BaseProcessor[I, O]):
    """Simply limits numer of items floating over pipeline."""
    def __init__(
        self,
        min: int = 0,
        max: int = None,
        name: str = None,
        input: FieldNameOrLambda = None,
        output: FieldNameOrLambda = None,
    ):
        """Simply limits numer of items floating over pipeline.

        It can be one number (pass only first X items) or 
        range (pass items from X to Y),

        Args:
            min: Minimum number item to process.
            max: Maximum number item to process.
        """
        super().__init__(name=name, input=input, output=output)
        self.min = min if max is not None else 0
        self.max = max if max is not None else min-1

    @override
    async def process(self, data) -> AsyncIterator[O]:
        iterator = self._get_iterator(data)
        if isinstance(iterator, Iterator):
            for i, item in enumerate(iterator):
                if self.min <= i <= self.max:
                    yield item
        else:
            i = 0
            async for item in iterator:
                if self.min <= i <= self.max:
                    yield item
                i += 1
