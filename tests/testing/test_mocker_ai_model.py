from typing import Iterable

import pytest

from haintech.ai import AIChatResponse, AIModelInteractionMessage, BaseAIChat, BaseAIChatAsync, BaseAIModel
from haintech.ai.anthropic import AnthropicAIModel
from haintech.ai.deep_seek import DeepSeekAIModel
from haintech.ai.google_genai import GoogleAIModel
from haintech.ai.open_ai import ResponsesAIModel
from haintech.testing.mocker_ai_model import MockerAIModel, mocker_ai_model  # noqa: F401


@pytest.fixture(
    params=[ResponsesAIModel, GoogleAIModel, DeepSeekAIModel, AnthropicAIModel],
    # params=[GoogleAIModel],
    scope="function",
)
def ai_model(request: pytest.FixtureRequest) -> BaseAIModel:
    return request.param(parameters={"temperature": 0})


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


@pytest.mark.asyncio
async def test_get_chat_response_async(ai_model: BaseAIModel, mocker_ai_model: MockerAIModel):  # noqa: F811
    # Given: A mock with an interaction defined
    mocker_ai_model.add(
        message="Who was the first US president?",
        response="George Washington.",
    )
    # When: get_chat_response() for AI model is called
    response = await ai_model.get_chat_response_async(
        message=AIModelInteractionMessage(role="user", content="Who was the first US president?")
    )
    # Then: The mock returns a defined response
    assert response.content == "George Washington."


def test_recording(mocker_ai_model: MockerAIModel):  # noqa: F811
    # Given: A real AI model
    ai_model = ResponsesAIModel(parameters={"temperature": 0})
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


def test_mock_with_system_prompt(mocker_ai_model: MockerAIModel):  # noqa: F811
    system_prompt = "Always answer `Yes sir!`"
    mocker_ai_model.add(system_prompt=system_prompt, response="Yes sir!")
    ai_model = ResponsesAIModel(parameters={"temperature": 0})
    ai_chat = BaseAIChat(ai_model, system_prompt=system_prompt)
    response = ai_chat.get_text_response("Who was the first US president?")
    assert response == "Yes sir!"


@pytest.mark.asyncio
async def test_mock_with_system_prompt_async(mocker_ai_model: MockerAIModel):  # noqa: F811
    system_prompt = "Always answer `Yes sir!`"
    mocker_ai_model.add(system_prompt=system_prompt, response="Yes sir!")
    ai_model = ResponsesAIModel(parameters={"temperature": 0})
    ai_chat = BaseAIChatAsync(ai_model, system_prompt=system_prompt)
    response = await ai_chat.get_text_response("Who was the first US president?")
    assert response == "Yes sir!"


def test_add_calls(ai_model: BaseAIModel, mocker_ai_model: MockerAIModel):  # noqa: F811
    # Given: A call list returned by record() method
    call_list = [
        {
            "message_str": "Who was the first US president?",
            "response": {"content": "The first US president was **George Washington**."},
        }
    ]
    # When: Add this list
    mocker_ai_model.add_calls(call_list)
    # And: get_chat_response() for AI model is called
    response = ai_model.get_chat_response(
        message=AIModelInteractionMessage(role="user", content="Who was the first US president?")
    )
    # Then: The mock returns a defined response in the list
    assert response.content == "The first US president was **George Washington**."


def test_mock_with_history_ok(ai_model: BaseAIModel, mocker_ai_model: MockerAIModel):  # noqa: F811
    # Given: A mock with two interactions, where the second checks history - it is correct
    mocker_ai_model.add(
        message="Who was the first US president?", response="The first US president was **George Washington**."
    )

    def check_history(history: Iterable[AIModelInteractionMessage] | None = None, **kwargs) -> AIChatResponse:
        for m in history or []:
            if m.role == "user" and m.content == "Who was the first US president?":
                return AIChatResponse(content="John Adams")
        raise AssertionError("History is wrong!")

    mocker_ai_model.add_call(check_history)

    # When: Get responses twice
    ai_model.get_chat_response(
        message=AIModelInteractionMessage(role="user", content="Who was the first US president?")
    )
    response = ai_model.get_chat_response(message=AIModelInteractionMessage(role="user", content="Who was the second?"))
    # Then: The second response is returned
    assert response.content
    assert "John Adams" in response.content


def test_mock_with_history_err(ai_model: BaseAIModel, mocker_ai_model: MockerAIModel):  # noqa: F811
    # Given: A mock with two interactions, where the second checks history - it is wrong
    mocker_ai_model.add(
        message="Who was the first US president?", response="The first US president was **George Washington**."
    )

    def check_history(history: Iterable[AIModelInteractionMessage] | None = None, **kwargs) -> AIChatResponse:
        for m in history or []:
            if m.role == "user" and m.content == "Who was the first Polish president?":
                return AIChatResponse(content="John Adams")
        raise AssertionError("History is wrong!")

    mocker_ai_model.add_call(check_history)

    # When: Get responses twice
    ai_model.get_chat_response(
        message=AIModelInteractionMessage(role="user", content="Who was the first US president?")
    )
    # Then: The second response raises error
    with pytest.raises(AssertionError) as e:
        ai_model.get_chat_response(message=AIModelInteractionMessage(role="user", content="Who was the second?"))
    assert "History is wrong!" in str(e.value)
