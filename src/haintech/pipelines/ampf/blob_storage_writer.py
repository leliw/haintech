from typing import Callable, Tuple, Union
from pydantic import BaseModel

from ..base_processor import BaseProcessor, get_field_name_or_lambda
from ampf.base import BaseBlobStorage


class BlobStorageWriter[M: BaseModel](BaseProcessor[Tuple[bytes, M], Tuple[bytes, M]]):
    """Write to storage blob and metadata."""

    def __init__(
        self,
        storage: BaseBlobStorage,
        key_name: Union[str, Callable[[M], str]],
        name=None,
        input=None,
        output=None,
    ):
        super().__init__(name, input, output)
        self.key_name = key_name
        self.storage = storage

    async def process_item(self, data: Tuple[bytes, M]) -> Tuple[bytes, M]:
        file_name = get_field_name_or_lambda(self.key_name, data[1])
        content = data[0] if isinstance(data[0], bytes) else data[0].encode()
        self.storage.upload_blob(file_name, content, data[1])
        return data
