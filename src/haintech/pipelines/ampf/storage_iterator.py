from typing import Iterator, override

from ampf.base import BaseStorage
from pydantic import BaseModel

from ..base_processor import BaseProcessor


class StorageIterator[I, O: BaseModel](BaseProcessor[I, O]):
    """Iterate over data from storage.

    Args:
        I: Input items are ignored
        O: Output items data type (read from storage)
    """

    def __init__(
        self,
        storage: BaseStorage[O],
        name=None,
        input=None,
        output=None,
    ):
        super().__init__(name, input, output)
        self.storage = storage

    @override
    def _get_iterator(self, _: I | Iterator[I]) -> Iterator[O]:
        """Returns iterator for processing data."""
        return self.storage.get_all()

    @override
    async def process_item(self, data: I) -> O:
        return data
