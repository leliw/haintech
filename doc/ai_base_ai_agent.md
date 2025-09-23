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

## Methods

* add_function(self, function: Callable, name: Optional[str] = None, definition: Any = None) -> None - add function to agent
* add_rag_searcher(self, searcher: BaseRAGSearcher) -> None - add RAG searcher to agent
* add_ai_task(self, ai_task: AITask) -> None - add AI task to agent

* get_response(self, message: Optional[AIModelInteractionMessage | str] = None) -> AIChatResponse - get only one response from LLM (tools are not called)
* get_text_response(self, message: Optional[str] = None) -> str - get text response from LLM (all tools are called automatically)
* accept_tools(self, tool_call_ids: str | List[str]) -> AIChatResponse   accept tool calls, call them, send results to LLM and get response from LLM

## Use case

Functions to be used by agent have to have docstrings.

```python
def get_remaining_vacation_days(year: int):
    """Returns the number of remaining vacation days for the given year

    Args:
        year: Year for which to calculate the remaining vacation days
    Returns:
        Number of remaining vacation days
    """
    return 26


def get_remaining_home_office_days(year: int):
    """Returns the number of remaining home office days for the given year

    Args:
        year: Year for which to calculate the remaining home office days
    Returns:
        Number of remaining home office days
    """
    return 4
```

Then they can be used in agent:

```python
ai_model = OpenAIModel("gpt-4o-mini", { "temperatoure": 0.7 })
session = AIChatSession()
ai_agent = BaseAIAgent(
    ai_model=ai_model,
    ession=session,
                functions=[
                get_remaining_vacation_days,
                get_remaining_home_office_days,
            ],
    )
```

You can get response from agent (with tools called automatically):

```python
response = ai_agent.get_text_response("How many vacation days do I have left this year?")
assert "26" in response
```

Or you can accept tool calls manually:

```python
response = ai_agent.get_response("How many vacation days do I have left in 2025?")
assert response.tool_calls
assert 1 == len(response.tool_calls)
assert response.tool_calls[0].id

response = ai_agent.accept_tools(response.tool_calls[0].id)
assert response.content
assert "26" in response.content
```
