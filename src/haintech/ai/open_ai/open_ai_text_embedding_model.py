from typing import List, override

from openai import OpenAI

from ..base import BaseAITextEmbeddingModel


class OpenAITextEmbeddingModel(BaseAITextEmbeddingModel):
    client: OpenAI = None

    def __init__(
        self,
        ai_model_name: str = "text-embedding-3-small",
        dimensions=1536,
    ):
        self.ai_model_name = ai_model_name
        self.dimensions = dimensions
        if not self.client:
            self.setup()

    @classmethod
    def setup(cls):
        cls.client = OpenAI()

    @override
    def get_embedding(self, text: str) -> List[float]:
        response = self.client.embeddings.create(
            input=text,
            model=self.ai_model_name,
            dimensions=self.dimensions,
        )
        return response.data[0].embedding
