from typing import List, override
import pytest
import requests

from haintech.ai.base.base_ai_text_embedding_model import BaseAITextEmbeddingModel
from haintech.ai.hugging_face.hugging_face_text_embedding_model import (
    HuggingFaceTextEmbeddingModel,
)
from haintech.ai.open_ai.open_ai_text_embedding_model import OpenAITextEmbeddingModel
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
    assert all([isinstance(i, List) for i in ret])


async def get_embeddings(
    ai_model: BaseAITextEmbeddingModel, texts: List[str]
) -> List[List[float]]:
    # Given: Pipeline with TextEmbedder with model
    pl_hf = Pipeline([TextEmbedder[str, Vector](ai_model=ai_model)])
    # When: Pipeline is run with texts
    ret = await pl_hf.run_and_return(texts)
    # Then: Embeddings are returned
    assert len(ret) == 2
    assert all([isinstance(i, List) for i in ret])
    return ret


class TextEmbeddingsInferenceModel(BaseAITextEmbeddingModel):
    def __init__(self):
        self.base_url = "http://localhost:8081"

    @override
    def get_embedding(self, text: str) -> List[float]:
        data = {
            "inputs": text,
            "normalize": True,
            "prompt_name": None,
            "truncate": False,
            "truncation_direction": "Right",
        }
        ret = requests.post(f"{self.base_url}/embed", json=data)
        r = ret.json()
        return r[0]


@pytest.mark.asyncio
async def test_silver_retriever(log):
    texts = [
        "Pytanie: Kto był pierwszym królem Polski?",
        "Bolesław Chrobry</s>Bolesław I Chrobry – władca Polski z dynastii Piastów w latach 992–1025, książę Polski od 992 i pierwszy koronowany król Polski, w latach 1003–1004 także książę Czech jako Bolesław IV. Był synem Mieszka I, księcia Polski i Dobrawy, czeskiej księżniczki. Ani miejsce, ani dokładna data urodzenia Bolesława nie są znane.",
    ]
    ai_model_hf = HuggingFaceTextEmbeddingModel("ipipan/silver-retriever-base-v1.1")
    ret_hf1 = await get_embeddings(ai_model_hf, texts)
    ret_hf2 = await get_embeddings(ai_model_hf, texts)
    ret_hf3 = await get_embeddings(ai_model_hf, texts)
    ai_model_oai = OpenAITextEmbeddingModel(
        "http://localhost:8081", "xxx", "data/local"
    )
    ret_oai1 = await get_embeddings(ai_model_oai, texts)
    ret_oai2 = await get_embeddings(ai_model_oai, texts)
    ret_oai3 = await get_embeddings(ai_model_oai, texts)
    ai_model_tei = TextEmbeddingsInferenceModel()
    ret_tei1 = await get_embeddings(ai_model_tei, texts)
    ret_tei2 = await get_embeddings(ai_model_tei, texts)
    ret_tei3 = await get_embeddings(ai_model_tei, texts)
    assert ret_hf1 == ret_hf2
    assert ret_hf2 == ret_hf3
    assert ret_oai1 == ret_oai2
    assert ret_oai2 == ret_oai3
    assert ret_tei1 == ret_tei2
    assert ret_tei2 == ret_tei3

