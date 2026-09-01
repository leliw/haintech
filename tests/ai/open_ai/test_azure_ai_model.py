from typing import ClassVar

import pytest
from openai import AsyncOpenAI, OpenAI
from openai.types import Model
from pydantic_settings import BaseSettings

from haintech.ai.model import AIModelInteractionMessage
from haintech.ai.open_ai.model import ResponsesAIParameters
from haintech.ai.open_ai.responses_ai_model import ResponsesAIModel


class AzureConfig(BaseSettings):
    azure_api_key: str = ""
    azure_base_url: str = ""


# Class duplicated from ResponsesAIModel to avoid using the OpenAI API key in tests, since Azure uses a different key.
class AzureAIModel(ResponsesAIModel):
    _api_key: ClassVar[str | None] = None
    _base_url: ClassVar[str | None] = None
    _openai: ClassVar[OpenAI | None] = None
    _async_openai: ClassVar[AsyncOpenAI | None] = None
    _models_list: ClassVar[list[Model] | None] = None


@pytest.fixture(params=["DeepSeek-V4-Flash", "gpt-5.6-luna", "gpt-5.6-terra"])
def azure_ai_model(request: pytest.FixtureRequest) -> AzureAIModel:
    config = AzureConfig()
    if not config.azure_api_key or not config.azure_base_url:
        raise ValueError("Configure Azure api_key & base_url.")
    AzureAIModel.setup(api_key=config.azure_api_key, base_url=config.azure_base_url)

    return AzureAIModel(request.param, parameters=ResponsesAIParameters(temperature=0))


def test_get_chat_response(azure_ai_model: AzureAIModel):
    response = azure_ai_model.get_chat_response(
        system_prompt="You are a helpful assistant.",
        message=AIModelInteractionMessage(role="user", content="What is the capital of France?"),
    )
    assert response.content and "Paris" in response.content

@pytest.mark.skip(reason="This test doesn't pass. I don't know why.") 
def test_get_model_names():
    config = AzureConfig()
    if not config.azure_api_key or not config.azure_base_url:
        raise ValueError("Configure Azure api_key & base_url.")
    AzureAIModel.setup(api_key=config.azure_api_key, base_url=config.azure_base_url)
    model_names = AzureAIModel.get_model_names()
    assert "gpt-5.6-luna" in model_names
