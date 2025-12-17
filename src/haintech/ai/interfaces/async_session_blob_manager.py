from typing import Protocol
from ampf.base import BlobLocation, Blob


class AsyncSessionBlobManager(Protocol):
    """Interface for managing blobs in a session."""
    async def download_blob(self, blob_location: BlobLocation) -> Blob: ...
