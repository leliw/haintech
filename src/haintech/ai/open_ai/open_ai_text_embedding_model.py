from typing import List, override

import httpx
from haintech.ai import BaseAITextEmbeddingModel
from openai import AsyncOpenAI, OpenAI


class OpenAITextEmbeddingModel(BaseAITextEmbeddingModel):
    client: OpenAI = None

    def __init__(
        self,
        base_url: str | httpx.URL | None = None,
        api_key: str | None = None,
        ai_model_name: str = "text-embedding-3-small",
        dimensions=1536,
    ):
        self.ai_model_name = ai_model_name
        self.dimensions = dimensions
        if base_url:
            self.client = OpenAI(base_url=base_url, api_key=api_key)
            self.client_async = AsyncOpenAI(base_url=base_url, api_key=api_key)
        elif not self.client:
            self.setup()

    @classmethod
    def setup(cls):
        cls.client = OpenAI()
        cls.client_async = AsyncOpenAI()

    @override
    def get_embedding(self, text: str) -> List[float]:
        response = self.client.embeddings.create(
            input=text,
            model=self.ai_model_name,
            dimensions=self.dimensions,
        )
        return response.data[0].embedding

    @override
    async def get_embedding_async(self, text: str) -> List[float]:
        response = await self.client_async.embeddings.create(
            input=text,
            model=self.ai_model_name,
            dimensions=self.dimensions,
        )
        return response.data[0].embedding
