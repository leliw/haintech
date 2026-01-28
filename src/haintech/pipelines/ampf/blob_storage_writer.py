from typing import Callable, Optional, override

from ampf.base import Blob, BaseAsyncBlobStorage, BaseBlobMetadata

from ..base_processor import BaseProcessor, FieldNameOrLambda, FieldNameOrLambda2
from ..progress_tracker import ProgressTracker


class BlobStorageWriter[M: BaseBlobMetadata](BaseProcessor[Blob[M], Blob[M]]):
    """Write to storage blob and metadata."""

    def __init__(
        self,
        storage: BaseAsyncBlobStorage[M] | Callable[[Blob[M]], BaseAsyncBlobStorage[M]],
        progress_tracker: Optional[ProgressTracker] = None,
        name: Optional[str] = None,
        input: Optional[FieldNameOrLambda] = None,
        output: Optional[FieldNameOrLambda2] = None,
    ):
        super().__init__(name, input, output)
        self.storage = storage
        self.progress_tracker = progress_tracker

    @override
    async def process_item(self, item: Blob[M]) -> Blob[M]:
        await self.storage.upload_async(item)
        return item
