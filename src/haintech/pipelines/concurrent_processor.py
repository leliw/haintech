import asyncio
from typing import AsyncIterator, Iterator, override

from .base_processor import BaseProcessor, FieldNameOrLambda


class ConcurrentProcessor[I, O](BaseProcessor[I, O]):
    """Processor that processes data concurrently.
    The output order is not guaranteed to be the same as the input order.
    The number of concurrent tasks is limited by max_concurrent.

    Args:
        I: Input items data type
        O: Output items data type
    """

    def __init__(
        self,
        max_concurrent: int = 5,
        name: str = None,
        input: FieldNameOrLambda = None,
        output: FieldNameOrLambda = None,
    ):
        """Processor that processes data concurrently.

        Args:
            max_concurrent: The maximum number of concurrent tasks.
            name: The name of the processor.
            input: The name of the input field.
            output: The name of the output field.
        """
        super().__init__(name=name, input=input, output=output)
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(self.max_concurrent)

    @override
    async def process(self, data) -> AsyncIterator[O]:
        """Adds concurrency to the standard process method."""
        iterator = self._get_iterator(data)
        if isinstance(iterator, Iterator):
            rets = asyncio.as_completed(
                [self.wrap_process_item(item) for item in iterator]
            )
        else:
            rets = asyncio.as_completed(
                [self.wrap_process_item(item) async for item in iterator]
            )
        for ret in rets:
            yield await ret

    @override
    async def wrap_process_item(self, data):
        """Adds semaphore to the standard wrap_process_item method."""
        async with self.semaphore:
            return await super().wrap_process_item(data)
