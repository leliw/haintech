from typing import Protocol
from ampf.base import BlobLocation, Blob


class SessionBlobManager(Protocol):
    """Interface for managing blobs in a session."""
    def download_blob(self, blob_location: BlobLocation) -> Blob: ...
