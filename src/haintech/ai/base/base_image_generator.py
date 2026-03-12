from abc import ABC
from typing import AsyncGenerator, Generator

from ampf.base import BlobCreate


class BaseImageGenerator(ABC):

    def generate(self, prompt: str) -> Generator[BlobCreate]:
        raise NotImplementedError()

    async def generate_async(self, prompt: str) -> AsyncGenerator[BlobCreate]:
        for blob_create in self.generate(prompt):
            yield blob_create
