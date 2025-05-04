from typing import List
import pytest

from haintech.ai.hugging_face.hugging_face_text_embedding_model import (
    HuggingFaceTextEmbeddingModel,
)
from haintech.pipelines.ai.text_embedder import TextEmbedder, Vector
from haintech.pipelines.pipeline import Pipeline


@pytest.mark.asyncio
async def test_set_source_and_run(log):
    # Given: Pipeline with TextEmbedder with model
    pl = Pipeline([TextEmbedder[str, Vector](ai_model=HuggingFaceTextEmbeddingModel())])
    # When: Pipeline is run with texts
    ret = await pl.run_and_return(
        [
            "Who was the first US president?",
            "How many coutries are there in Europe?",
        ]
    )
    # Then: Embeddings are returned
    assert len(ret) == 2
    assert all([isinstance(i, List)  for i in ret])
