import pytest

from haintech.ai.model import AIModelInteractionMessage
from haintech.ai.open_ai.model import OpenAIParameters
from haintech.ai.open_ai.open_ai_model import OpenAIModel


@pytest.fixture(params=["gpt-4o-mini", "gpt-4.1-mini", "gpt-5-mini", "gpt-5.1"])
def ai_model(request: pytest.FixtureRequest) -> OpenAIModel:
    return OpenAIModel(request.param, parameters=OpenAIParameters(temperature=0))


def test_get_chat_response(ai_model: OpenAIModel):
    response = ai_model.get_chat_response(
        system_prompt="You are a helpful assistant.",
        message=AIModelInteractionMessage(role="user", content="What is the capital of France?"),
    )
    assert response.content == "The capital of France is Paris."


def test_get_model_names():
    ai_model = OpenAIModel("gpt-4o-mini")
    model_names = ai_model.get_model_names()
    assert "gpt-4o-mini" in model_names
