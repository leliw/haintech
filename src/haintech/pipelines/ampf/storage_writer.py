from typing import Callable, Union, override

from ampf.base import BaseStorage
from pydantic import BaseModel

from ..base_processor import get_field_name_or_lambda
from ..checkpoint_processor import CheckpointProcessor
from ..progress_tracker import ProgressTracker
from .storage_reader import StorageReader


class StorageWriter[M: BaseModel](CheckpointProcessor[M]):
    def __init__(
        self,
        storage: BaseStorage,
        key_name: Union[str, Callable[[M], str]] = None,
        progress_tracker: ProgressTracker = None,
        name=None,
        input=None,
        output=None,
    ):
        super().__init__(name, input, output)
        self.storage = storage
        self.key_name = key_name
        self.progress_tracker = progress_tracker

    async def process_item(self, data: M) -> M:
        if self.key_name:
            key = get_field_name_or_lambda(self.key_name, data)
        else:
            key = self.storage.get_key(data)
        self.storage.put(key, data)
        if self.progress_tracker:
            self.progress_tracker.increment()
        return data

    @override
    def create_generator(self):
        return StorageReader[str, M](self.storage)
