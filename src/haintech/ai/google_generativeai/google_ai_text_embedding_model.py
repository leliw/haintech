from typing import List, override

from google import genai
from google.genai.types import EmbedContentConfig

from ..base import BaseAITextEmbeddingModel


class GoogleAITextEmbeddingModel(BaseAITextEmbeddingModel):
    client: genai.Client = None

    def __init__(
        self,
        ai_model_name: str = "text-multilingual-embedding-002",
        dimensions=768,
    ):
        self.ai_model_name = ai_model_name
        self.dimensions = dimensions
        if not self.client:
            self.setup()

    @classmethod
    def setup(cls, vertexai: bool = True, location: str = "us-central1"):
        cls.client = genai.Client(vertexai=vertexai, location=location)

    @override
    def get_embedding(self, text: str) -> List[float]:
        response = self.client.models.embed_content(
            model=self.ai_model_name,
            contents=[text],
            config=EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
                output_dimensionality=self.dimensions,
            ),
        )
        print(response)
        return response.embeddings[0].values
