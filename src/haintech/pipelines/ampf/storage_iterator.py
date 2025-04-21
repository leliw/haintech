from typing import Iterator, override

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
        storage: BaseStorage[O],
        progress_tracker: ProgressTracker = None,
        name=None,
        input=None,
        output=None,
    ):
        super().__init__(name, input, output)
        self.storage = storage
        self.progress_tracker = progress_tracker

    @override
    def _get_iterator(self, _: I | Iterator[I]) -> Iterator[O]:
        """Returns iterator for processing data."""
        if self.progress_tracker:
            self.progress_tracker.set_total_steps(self.storage.count())
        return self.storage.get_all()

    @override
    async def process_item(self, data: I) -> O:
        return data
