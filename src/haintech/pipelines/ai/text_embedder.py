import asyncio
import logging
from random import random
from typing import List, override

import httpx

from ...ai.base import BaseAITextEmbeddingModel
from ..concurrent_processor import ConcurrentProcessor

Vector = List[float]


class TextEmbedder[I: str, O: Vector](ConcurrentProcessor[I, O]):
    _log = logging.getLogger(__name__)

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
        self.max_retries = 3
        self.wait_time_seconds = random() + 0.5

    @override
    async def process_item(self, data: I) -> O:
        for attempt in range(self.max_retries + 1):
            try:
                result = await self.ai_model.get_embedding_async(data)
                return result
            except httpx.ReadTimeout as e:
                if attempt < self.max_retries:
                    wait_time_seconds = self.wait_time_seconds * pow(2, attempt)
                    self._log.warning(
                        f"Attempt {attempt + 1} of {self.max_retries + 1} failed with timeout: {e}. "
                        f"Retrying in {wait_time_seconds:.2f} seconds..."
                    )

                    await asyncio.sleep(wait_time_seconds)
                else:
                    self._log.error(
                        f"All {self.max_retries + 1} attempts to get embedding failed with timeout."
                    )
                    raise e
