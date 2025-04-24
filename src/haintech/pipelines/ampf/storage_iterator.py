from typing import AsyncIterator, Callable, Iterator, override

from ampf.base import BaseStorage
from pydantic import BaseModel

from ..base_processor import BaseProcessor
from ..progress_tracker import ProgressTracker


class StorageIterator[I, O: BaseModel](BaseProcessor[I, O]):
    """Iterate over data from storage.

    Args:
        I: Input items are ignored
        O: Output items data type (read from storage)
    """

    def __init__(
        self,
        storage: BaseStorage[O] | Callable[[I], BaseStorage[O]],
        progress_tracker: ProgressTracker = None,
        name=None,
        input=None,
        output=None,
    ):
        super().__init__(name, input, output)
        self.storage = storage
        self.progress_tracker = progress_tracker

    def process_flat_map(self, data: I) -> Iterator[O]:
        if isinstance(self.storage, Callable):
            storage = self.storage(data)
        else:
            storage = self.storage
        if self.progress_tracker:
            self.progress_tracker.set_total_steps(self.storage.count())
        for item in storage.get_all():
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
