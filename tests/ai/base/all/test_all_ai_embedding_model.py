import pytest
from haintech.ai import BaseAITextEmbeddingModel


def test_get_embedding(ai_embedding_model: BaseAITextEmbeddingModel):
    # Given: Text
    t = "Kim był król Jan III Sobieski?"
    # When: Calculate embeddings
    ret = ai_embedding_model.get_embedding(t)
    # Then: Embedding vector is returned
    assert ret is not None
    assert len(ret) > 0


@pytest.mark.asyncio
async def test_get_embedding_async(ai_embedding_model: BaseAITextEmbeddingModel):
    # Given: Text
    t = "Kim był król Jan III Sobieski?"
    # When: Calculate embeddings
    ret = await ai_embedding_model.get_embedding_async(t)
    # Then: Embedding vector is returned
    assert ret is not None
    assert len(ret) > 0


if __name__ == "__main__":
    pytest.main([__file__])
