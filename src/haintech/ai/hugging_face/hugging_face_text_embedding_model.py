import logging
from typing import List, override


from ..base import BaseAITextEmbeddingModel


class HuggingFaceTextEmbeddingModel(BaseAITextEmbeddingModel):

    _log = logging.getLogger(__name__)
    def __init__(
        self,
        ai_model_name: str = "all-mpnet-base-v2",
    ):
        self.ai_model_name = ai_model_name
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(ai_model_name)
        except ImportError:
            self._log.error("The package `sentence_transformers` is not installed ")
            self._log.error("Try: pip install haintech[huggingface]")


    @override
    def get_embedding(self, text: str) -> List[float]:
        return self.model.encode(text)
