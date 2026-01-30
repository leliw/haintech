# Testing package

## mocker_ai_model fixture / MockerAIModel

`mocker_ai_model` fixture provide object of MockerAIModel class. It mocks calls to LLM classes:

* `GoogleAIModel`
* `OpenAIModel`
* `DeepSeekAIModel`
* `AnthropicAIModel`

Both sync and async methods are supported.

### Methods

* add() - adds a mocked AI response.
  * response (str | AIChatResponse): The response to return.
  * message (Optional[str], optional): The exact message string to match. Defaults to None.
  * message_containing (Optional[str], optional): A substring that should be contained in the message. Defaults to None.
  * tool_result (Optional[str], optional): The tool result to match. Defaults to None.
  * blob_contents (Optional[List[bytes]], optional): The blob contents to match. Defaults to None.
* record() - records all AI responses and prints them to console.

### Use case

Recording calls and responses to AI models for later mocking.

```python
def test_recording(mocker_ai_model: MockerAIModel):  # noqa: F811
    # Given: A real AI model
    ai_model = OpenAIModel(parameters={"temperature": 0})
    with pytest.raises(AssertionError):
        # When: AI model calls are recorded
        with mocker_ai_model.record():
            response = ai_model.get_chat_response(
                message=AIModelInteractionMessage(role="user", content="Who was the first US president?")
            )
            # Then: A real answer is returned
            assert response.content
            assert "George Washington" in response.content
        # And: AssertionError is raised by record() method
        assert False
        # And: Calls are printed in console
```

Mocking responses from AI models.

```python
def test_get_chat_response(ai_model: BaseAIModel, mocker_ai_model: MockerAIModel):  # noqa: F811
    # Given: A mock with an interaction defined
    mocker_ai_model.add(
        message="Who was the first US president?",
        response="George Washington.",
    )
    # When: get_chat_response() for AI model is called
    response = ai_model.get_chat_response(
        message=AIModelInteractionMessage(role="user", content="Who was the first US president?")
    )
    # Then: The mock returns a defined response
    assert response.content == "George Washington."
```
