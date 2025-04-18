import asyncio
from abc import ABC, abstractmethod
from typing import List


class BaseAITextEmbeddingModel(ABC):
    """Base class for text embedding models."""

    @abstractmethod
    def get_embedding(self, text: str) -> List[float]:
        """Embeds text with a pre-trained model."""
        pass

    async def get_embedding_async(self, text: str) -> List[float]:
        """Embeds text with a pre-trained model."""
        return await asyncio.to_thread(self.get_embedding, text)
