from typing import AsyncIterator, Callable

from .base_processor import BaseProcessor


class FilterProcessor[I](BaseProcessor[I,I]):
    """Processor that filters data using a lambda function."""

    def __init__(self, expression: Callable[[I], bool], **kwargs):
        """Initialize the processor.

        Args:
            expression: The lambda function that processes data.
        """
        super().__init__(**kwargs)
        self.expression = expression

    async def process(self, source_data) -> AsyncIterator[I]:
        """
        Run the processor on the data.
        """
        async for data in self.source(source_data):
            if self.expression(data):
                yield data
