from typing import Tuple
from pydantic import BaseModel

from ..base_processor import BaseProcessor
from ampf.base import BaseBlobStorage


class BlobStorageReader[str, M: BaseModel](BaseProcessor[str, Tuple[bytes | str, M]]):
    """Read from storage blob and metadata."""

    def __init__(
        self,
        storage: BaseBlobStorage,
        name=None,
        input=None,
        output=None,
    ):
        super().__init__(name, input, output)
        self.storage = storage

    async def process_item(self, file_name: str) -> Tuple[bytes | str, M]:
        blob = self.storage.download_blob(file_name)
        if isinstance(blob, bytes) and any(x in self.storage.content_type for x in ("text", "json", "xml")):
            blob = blob.decode("utf-8")
        metadata = self.storage.get_metadata(file_name)
        return (blob, metadata)
