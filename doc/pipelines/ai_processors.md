# AI Processors

A group of processors using AI models.

* TextEmbedder - returns embeddings for given text

## TextEmbedder

It inherits from `ConcurrentProcessor`.

### Constructor

Parameters:

* ai_model: BaseAITextEmbeddingModel - ai model
* max_concurrent: int - number of concurrent calculations 