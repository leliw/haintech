# BaseAgentSearcher

Extends `BaseRAGSearcher()` with an AI model that is used to process the query before searching.

## Constructor

Arguments:

* ai_model: [BaseAIModel](ai_base_ai_model.md) - AI model class implementing `BaseAIModel()`. It shuld be cheap model like `gpt-4.1-nano` or `gemini-2.5-flash-lite`
* system_instructions: str = "Task: Based on the system description, conversation, and user question, generate one short search query for document retrieval."
* prompt: str = "Input:\nSystem: {system_prompt}\nHistory: {conversation_history}\nQuestion: {user_question}\n\nOutput:\nOnly one query between 20 and 100 tokens, no comments or markdown."

## Methods

* agent_search_sync / agent_search_async - These methods are called by `BaseAIAgent` if `searcher` is defined while creating the agent. They process the message content with the AI model to generate a search query, and then call the search methods from `BaseRAGSearcher`.

## Use cases

Below is an example of how to implement a agent searcher that connects to an external knowledge base service.

```python
class KnowledgeBaseService(BaseAgentSearcher):
    _log = logging.getLogger(__name__)

    def __init__(self, ai_model: BaseAIModel, base_url: str, timeout: int = 60, limit: int = 10):
        super().__init__(ai_model)
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
        self._log.info("Searching for: %s", query.text)
        req = SemanticSearchRequest(query=query.text, limit=self.limit)
        for chunk in self.semantic_search(req):
            yield RAGItem(item_id=str(chunk.chunk_id), content=chunk.text)

    async def ping(self) -> None:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/api/ping",
                # headers=self._get_headers(),
                timeout=self.timeout,
            )
            response.raise_for_status()
```
