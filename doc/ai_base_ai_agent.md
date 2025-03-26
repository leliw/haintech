# BaseAIAgent

Extends `BaseAIChat()` with tools:

* searching documents relative to conversation
* function calling

## Constructor

Arguments:

* name: name of the agent
* description: description of the agent
* ai_model: AI model
* prompt: Context prompt for model
* context: Context prompt for model
* session: current session object
* searcher: RAG searcher
* functions: list of functions to add
