import pytest

from haintech.ai.anthropic.anthropic_ai_model import AnthropicAIModel
from haintech.ai.base.base_ai_chat import BaseAIChat
from haintech.ai.base.base_ai_chat_async import BaseAIChatAsync
from haintech.ai.base.base_ai_model import BaseAIModel
from haintech.ai.deep_seek.deep_seek_ai_model import DeepSeekAIModel
from haintech.ai.google_genai.google_ai_model import GoogleAIModel
from haintech.ai.model import AIModelInteractionMessage
from haintech.ai.open_ai.open_ai_model import OpenAIModel
from haintech.testing.mocker_ai_model import MockerAIModel, mocker_ai_model  # noqa: F401


@pytest.fixture(
    params=[OpenAIModel, GoogleAIModel, DeepSeekAIModel, AnthropicAIModel],
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

def test_mock_with_system_prompt(mocker_ai_model: MockerAIModel):  # noqa: F811
    system_prompt = "Always answer `Yes sir!`"
    mocker_ai_model.add(system_prompt=system_prompt, response="Yes sir!")
    ai_model = OpenAIModel(parameters={"temperature": 0})
    ai_chat = BaseAIChat(ai_model, system_prompt=system_prompt)
    response = ai_chat.get_text_response("Who was the first US president?")
    assert response == "Yes sir!"

@pytest.mark.asyncio
async def test_mock_with_system_prompt_async(mocker_ai_model: MockerAIModel):  # noqa: F811
    system_prompt = "Always answer `Yes sir!`"
    mocker_ai_model.add(system_prompt=system_prompt, response="Yes sir!")
    ai_model = OpenAIModel(parameters={"temperature": 0})
    ai_chat = BaseAIChatAsync(ai_model, system_prompt=system_prompt)
    response = await ai_chat.get_text_response("Who was the first US president?")
    assert response == "Yes sir!"
