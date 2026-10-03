import pytest
from pydantic_settings import BaseSettings, SettingsConfigDict

from haintech.ai.azure_foundry import AzureFoundryAIModel, AzureFoundryConfig
from haintech.ai.model import AIModelInteractionMessage


class AzureConfig(BaseSettings):
    model_config = SettingsConfigDict(env_nested_delimiter="__")

    azure_foundry: AzureFoundryConfig


@pytest.fixture
def azure_ai_model() -> AzureFoundryAIModel:
    config = AzureConfig()  # type: ignore
    if not config.azure_foundry.api_key or not config.azure_foundry.base_url:
        raise ValueError("Configure Azure api_key & base_url.")
    AzureFoundryAIModel.setup(config.azure_foundry)
    return AzureFoundryAIModel("gpt-5.6-luna")


def test_get_chat_response(azure_ai_model: AzureFoundryAIModel):
    response = azure_ai_model.get_chat_response(
        system_prompt="You are a helpful assistant.",
        message=AIModelInteractionMessage(role="user", content="What is the capital of France?"),
    )
    assert response.content and "Paris" in response.content


def test_get_model_names(azure_ai_model: AzureFoundryAIModel):
    model_names = azure_ai_model.get_model_names()
    assert "gpt-6-luna" in model_names
