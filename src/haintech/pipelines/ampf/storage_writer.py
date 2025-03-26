from typing import Callable, Union
from pydantic import BaseModel

from ..base_processor import BaseProcessor, get_field_name_or_lambda
from ampf.base import BaseStorage


class StorageWriter[M: BaseModel](BaseProcessor[M, M]):
    def __init__(
        self,
        storage: BaseStorage,
        key_name: Union[str, Callable[[M], str]] = None,
        name=None,
        input=None,
        output=None,
    ):
        super().__init__(name, input, output)
        self.key_name = key_name
        self.storage = storage

    async def process_item(self, data: M) -> M:
        if self.key_name:
            key = get_field_name_or_lambda(self.key_name, data)
        else:
            key = self.storage.get_key(data)
        self.storage.put(key, data)
        return data
