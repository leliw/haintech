# TextEmbeddingModel

`BaseAITextEmbeddingModel` is a base class for text embedding models.
Implementations:

* OpenAITextEmbeddingModel
* GoogleAITextEmbeddingModel
* HuggingFaceTextEmbeddingModel

## Methods

* get_embedding(self, text: str) -> List[float]:
* get_embedding_async(self, text: str) -> List[float]:

Both returns embedding for given text. The second one is async version.

## HuggingFaceTextEmbeddingModel

`HuggingFaceTextEmbeddingModel` is a text embedding model that uses Hugging Face's `sentence_transformers` library to generate embeddings for text. It is initialized with a model name. The class provides methods to get embeddings for a given text (synchronously).

**Caution.**

To avoid loading (very big) `cuda` packages by `uv`, add to `pyproject.toml` below sections:

```toml
[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[tool.uv.sources]
torch = [
  { index = "pytorch-cpu" },
]
```
