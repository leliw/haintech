from typing import List, override

from ...ai.base import BaseAITextEmbeddingModel
from ..concurrent_processor import ConcurrentProcessor

Vector = List[float]


class TextEmbedder[I: str, O: Vector](ConcurrentProcessor[I, O]):
    def __init__(
        self,
        ai_model: BaseAITextEmbeddingModel,
        max_concurrent: int = 5,
        name=None,
        input=None,
        output=None,
    ):
        super().__init__(
            max_concurrent=max_concurrent, name=name, input=input, output=output
        )
        self.ai_model = ai_model

    @override
    async def process_item(self, data: I) -> O:
        return await self.ai_model.get_embedding_async(data)
