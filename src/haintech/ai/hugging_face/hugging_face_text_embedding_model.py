from typing import List, override


from ..base import BaseAITextEmbeddingModel


class HuggingFaceTextEmbeddingModel(BaseAITextEmbeddingModel):
    def __init__(
        self,
        ai_model_name: str = "all-mpnet-base-v2",
        dimensions: int = None,
    ):
        self.ai_model_name = ai_model_name
        self.dimensions = dimensions
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(ai_model_name)

    @override
    def get_embedding(self, text: str) -> List[float]:
        return self.model.encode(text)
