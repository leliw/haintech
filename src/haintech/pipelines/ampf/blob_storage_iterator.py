from typing import AsyncIterator, Callable, Iterator, Optional, override

from ampf.base import BaseAsyncBlobStorage, Blob, BaseBlobMetadata

from ..base_flat_map_processor import BaseProcessor
from ..progress_tracker import ProgressTracker


class BlobStorageIterator[I, M: BaseBlobMetadata](BaseProcessor[I, Blob[M]]):
    def __init__(
        self,
        storage: BaseAsyncBlobStorage[M] | Callable[[I], BaseAsyncBlobStorage[M]],
        progress_tracker: Optional[ProgressTracker] = None,
        name=None,
        input=None,
        output=None,
    ):
        super().__init__(name, input, output)
        self.storage = storage
        self.progress_tracker = progress_tracker

    @override
    async def process(self, data: I) -> AsyncIterator[Blob[M]]:
        iterator = self._get_iterator(data)
        if isinstance(iterator, Iterator):
            for data in iterator:
                async for item in self.wrap_process_flat_map(data):
                    yield item
        else:
            async for data in iterator:
                async for item in self.wrap_process_flat_map(data):
                    yield item

    async def wrap_process_flat_map(self, data):
        """
        Run process method.
        If result_key_name is set, store result in input data and return it.
        """
        if self.input:
            input_data = self._get_input_data(data)
        else:
            input_data = data
        async for ret in self.process_flat_map(input_data):
            yield self._put_output_data(data, ret)

    async def process_flat_map(self, data: I) -> AsyncIterator[Blob[M]]:
        if isinstance(self.storage, Callable):
            storage = self.storage(data)
        else:
            storage = self.storage
        blob_headers = list([bh async for bh in storage.list_blobs()])
        if self.progress_tracker:
            self.progress_tracker.set_total_steps(len(blob_headers))
        for header in blob_headers:
            item = await storage.download_async(header.name)
            if self.progress_tracker:
                self.progress_tracker.increment()
            yield self._put_output_data(data, item)

    @override
    async def process_item(self, data: Blob[M]) -> Blob[M]:
        return data
