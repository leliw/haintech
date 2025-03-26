from pydantic import BaseModel

from ..base_processor import BaseProcessor
from ampf.base import BaseStorage


class StorageReader[str, M: BaseModel](BaseProcessor[str, M]):
    def __init__(
        self,
        storage: BaseStorage,
        name=None,
        input=None,
        output=None,
    ):
        super().__init__(name, input, output)
        self.storage = storage

    async def process_item(self, key_name: str) -> M:
        return self.storage.get(key_name)
