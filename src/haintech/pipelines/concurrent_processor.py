import asyncio
from typing import AsyncIterator, Iterator, Set, override

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
        if max_concurrent <= 0:
            raise ValueError("max_concurrent must be greater than 0")
        self.max_concurrent = max_concurrent

    @override
    async def process(self, data) -> AsyncIterator[O]:
        """
        Processes items from the input iterator concurrently, yielding results
        as they become available, without loading the entire input into memory.
        Limits the number of concurrently running processing tasks.
        """
        iterator = self._get_iterator(data)
        pending_tasks: Set[asyncio.Task] = set()
        iterator_exhausted = False

        while True:
            while len(pending_tasks) < self.max_concurrent and not iterator_exhausted:
                try:
                    if isinstance(iterator, Iterator):
                        item = next(iterator)  # Can raise StopIteration
                    else:  # AsyncIterator
                        item = await anext(iterator)  # Can raise StopAsyncIteration
                except (StopIteration, StopAsyncIteration):
                    iterator_exhausted = True
                    break  # Stop trying to fetch new items
                except Exception as e:
                    # Handle potential errors during iteration itself
                    # Log or re-raise depending on desired behavior
                    print(f"Error fetching item from iterator: {e}")  # Or use logging
                    iterator_exhausted = True  # Assume iterator is broken
                    break
                task = asyncio.create_task(self.wrap_process_item(item))
                pending_tasks.add(task)

            # 2. If no tasks are running or waiting, and the iterator is done, exit.
            if not pending_tasks:
                break
            done, pending_tasks = await asyncio.wait(
                pending_tasks, return_when=asyncio.FIRST_COMPLETED
            )
            for task in done:
                try:
                    yield await task
                except Exception as e:
                    print(
                        f"Error processing item in task {task.get_name()}: {e}"
                    )  # Or use logging
                    raise e

    @override
    async def wrap_process_item(self, data):
        """
        Wraps the item processing with semaphore acquisition/release.
        This method remains unchanged as it correctly limits execution concurrency.
        """
        # Assuming the semaphore needs to be created if not already present
        # (though the original __init__ did create it)
        if not hasattr(self, "semaphore"):
            self.semaphore = asyncio.Semaphore(self.max_concurrent)

        async with self.semaphore:
            # The actual processing logic is called here, limited by the semaphore.
            return await super().wrap_process_item(data)
