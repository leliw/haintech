# BaseAIChat

Extends `BaseAIModel()` with chat session functionality.

## Constructor

Arguments:

* ai_model: [BaseAIModel](ai_base_ai_model.md) -
  AI model class implementing `BaseAIModel()`
* context: str | [AIPrompt](ai.md#aiprompt) -
  prompt for model
* session: [AIModelSession](ai.md#aimodelsession) -
  current session object
