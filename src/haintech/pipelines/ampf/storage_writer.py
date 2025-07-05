import logging
from typing import Callable, Optional, Union, override

from ampf.base import BaseStorage, KeyNotExistsException
from pydantic import BaseModel

from haintech.pipelines.base_processor import get_field_name_or_lambda
from haintech.pipelines.checkpoint_processor import CheckpointProcessor
from haintech.pipelines.progress_tracker import ProgressTracker
from haintech.pipelines.ampf.storage_reader import StorageReader


class StorageWriter[M: BaseModel](CheckpointProcessor[M]):
    """Writes Pydantic object into storage"""
    _log = logging.getLogger(__name__)

    def __init__(
        self,
        storage: BaseStorage[M] | Callable[[M], BaseStorage[M]],
        key_name: Optional[Union[str, Callable[[M], str]]] = None,
        changed_only: bool = False,
        progress_tracker: Optional[ProgressTracker] = None,
        name=None,
        input=None,
        output=None,
    ):
        super().__init__(name, input, output)
        self.storage = storage
        self.key_name = key_name
        self.changed_only = changed_only
        self.progress_tracker = progress_tracker

    async def process_item(self, data: M) -> M | None:
        if isinstance(self.storage, Callable):
            storage = self.storage(data)
        else:
            storage = self.storage

        if self.key_name:
            key = get_field_name_or_lambda(self.key_name, data)
        else:
            key = storage.get_key(data)

        try:
            if self.changed_only:
                old_val = storage.get(key)
                if old_val == data:
                    self._log.debug("Skip: %s", key)
                    return None
            self._log.debug("Update: %s", key)
            storage.put(key, data)
            return data
        except KeyNotExistsException:
            self._log.debug("Create: %s", key)
            storage.create(data)
            return data
        finally:
            if self.progress_tracker:
                self.progress_tracker.increment()

    @override
    def create_generator(self):
        return StorageReader[str, M](self.storage) # type: ignore

