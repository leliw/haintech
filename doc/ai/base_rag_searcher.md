# BaseRAGSearcher

An abstract base class for retrieval-augmented generation (RAG) searchers. It defines methods for searching and agent-based searching.

## Abstract methods

* `search_sync(self, query: RAGQuery) -> Iterable[RAGItem]:` - Abstract method to perform a synchronous search based on the provided RAG query.
* `search_async(self, query: RAGQuery) -> Iterable[RAGItem]:` - Asynchronous method that calls the synchronous search method.

## Implemented methods

These methods are called by `BaseAIAgent` if `searcher` is defined while creating the agent. They simply call the above methods with message content.

* `agent_search_sync(self, system_prompt: AIPrompt, history: List[AIModelInteractionMessage], message: Optional[AIModelInteractionMessage] = None) -> Optional[AIContext]:` - Synchronous method for agent-based searching.
* `agent_search_async(self, system_prompt: AIPrompt, history: List[AIModelInteractionMessage], message: Optional[AIModelInteractionMessage] = None) -> Optional[AIContext]:` - Asynchronous method that calls the synchronous method.

## Use cases

Below is an example of how to implement a RAG searcher that connects to an external knowledge base service.

```python
class KnowledgeBaseService(BaseRAGSearcher):
    def __init__(self, base_url: str, timeout: int = 60, limit: int = 10):
        self.base_url = base_url
        self.timeout = timeout
        self.limit = limit

    def semantic_search(self, req: SemanticSearchRequest) -> List[ChunkDetails]:
        with httpx.Client() as client:
            response = client.post(
                url=f"{self.base_url}/api/search/semantic",
                json=req.model_dump(),
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
            return [ChunkDetails(**chunk) for chunk in data]

    @override
    def search_sync(self, query: RAGQuery) -> Iterable[RAGItem]:
        req = SemanticSearchRequest(query=query.text, limit=self.limit)
        for chunk in self.semantic_search(req):
            yield RAGItem(item_id=str(chunk.chunk_id), content=chunk.text)
```
