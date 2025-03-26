# AI package

## Model classes

### AIPrompt

Structured prompt for AI model.
Each AI model implementation can process this structure
in its own way.

 According to: <https://cloud.google.com/vertex-ai/generative-ai/docs/learn/prompts/prompt-design-strategies#components-of-a-prompt>

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

Represents one interaction with AIModel.
It can be used fro debugging and costs calculations.

### AIModelSession

It is abstract class used by `BaseAIChat` to store
chat session history. There are only mabstract methods.

* add_interaction(self, interaction: AIModelInteraction) -> None:
  adds next iteraction with AI model (usually with a response)
* messages_iterator(self) -> Iterator[AIModelInteractionMessage]:
  returns chat session history as iterator of messages
* get_last_response(self) -> Optional[AIChatResponse]:
  returns last response from AI model

## Utility classes

* BaseAIModel
* BaseAIChat
* BaseAIAgent
