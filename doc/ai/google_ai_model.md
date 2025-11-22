# GoogleAIModel

Google AI implementation of BaseAIModel.

## Class methods

### Setup

This method calls `google.generativeai.client.configure()` function. It should be called only once at application startup.

## Constructor

* model_name
* parameters
* api_key

## Tests

There is a problem with async testing, so always use **function** fixture and call configure() manuually.

```python
@pytest.fixture(
    params=[OpenAIModel, GoogleAIModel, DeepSeekAIModel],
    # params=[GoogleAIModel],
    scope="function",
)
def ai_model(request: pytest.FixtureRequest) -> BaseAIModel:
    if request.param == GoogleAIModel:
        from google.generativeai.client import configure as genai_configure
        genai_configure()
    return request.param(parameters={"temperature": 0})
```

For tests mock `_get_chat_response()` or `_get_chat_response_async()` respectivly.

```python
@pytest.mark.asyncio
async def test_get(service: ChatSessionMessageService, ai_agent: MCPAIAgent, mock_method: MockMethod):
    gemini_mock = mock_method(
        GoogleAIModel._get_chat_response_async,
        return_value=AIChatResponse(content="George Washington was the first president of the USA."),
    )
```

or if it is called more than once

```python
@pytest.mark.asyncio
async def test_post(service: ChatSessionMessageService, ai_agent: MCPAIAgent, mock_method: MockMethod):
    responses = [
        AIChatResponse(content="George Washington was the first president of the USA."),
        AIChatResponse(content="The second president was John Adams."),
    ]
    gemini_mock = mock_method(
        GoogleAIModel._get_chat_response_async,
        side_effect=lambda *args, **kwargs: responses.pop(0),
    )
```
