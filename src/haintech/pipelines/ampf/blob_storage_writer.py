from typing import Callable, Optional, override

from ampf.base import BaseBlobAsyncStorage, Blob
from pydantic import BaseModel

from ..base_processor import BaseProcessor, FieldNameOrLambda
from ..progress_tracker import ProgressTracker


class BlobStorageWriter[M: BaseModel](BaseProcessor[Blob[M], Blob[M]]):
    """Write to storage blob and metadata."""

    def __init__(
        self,
        storage: BaseBlobAsyncStorage[M] | Callable[[Blob[M]], BaseBlobAsyncStorage[M]],
        progress_tracker: Optional[ProgressTracker] = None,
        name: Optional[str] = None,
        input: Optional[FieldNameOrLambda] = None,
        output: Optional[FieldNameOrLambda] = None,
    ):
        super().__init__(name, input, output)
        self.storage = storage
        self.progress_tracker = progress_tracker

    @override
    async def process_item(self, item: Blob[M]) -> Blob[M]:
        await self.storage.upload_async(item)
        return item
