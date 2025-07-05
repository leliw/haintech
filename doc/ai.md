# AI Package

Implementations of `BaseAIModel` include:

* OpenAIModel()
* GoogleAIModel()
* DeepSeekAIModel()

## Model Classes

### AIPrompt

Structured prompt for AI model. Each AI model implementation can process this structure differently.

For more information, refer to: <https://cloud.google.com/vertex-ai/generative-ai/docs/learn/prompts/prompt-design-strategies#components-of-a-prompt>

* persona
* objective
* instructions
* constraints
* context
* documents
* output_format
* examples
* recap

### AIModelInteraction

Represents one whole interaction with AIModel. It can be used for debugging and cost calculations.

### AIModelSession

It is an abstract class used by `BaseAIChat` to store chat session history. It contains only abstract methods.

* `add_interaction(self, interaction: AIModelInteraction) -> None:` adds the next interaction with the AI model (usually with a response).
* `messages_iterator(self) -> Iterator[AIModelInteractionMessage]:` returns the chat session history as an iterator of messages.
* `get_last_response(self) -> Optional[AIChatResponse]:` returns the last response from the AI model.

## Utility Classes

* BaseAIModel
* BaseAIChat / BaseAIChatAsync
* BaseAIAgent / BaseAIAgentAsync
* [AITaskExecutor](ai/ai_task_executor.md) - wraps an AI model so it can be used like a function.
* [BaseAITextEmbeddingModel](ai/text_embedding_model.md) - wraps an AI model that can be used for text embedding.
* [MCPAIAgent](ai/mcp_ai_agent.md) - BaseAIAgentAsync that allows the use of MCP servers.
