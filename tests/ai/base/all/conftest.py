from typing import Iterator, Optional

import pytest
from ampf.local import LocalFactory

from haintech.ai import (
    AIChatSession,
    BaseAIAgent,
    BaseAIChat,
    BaseAIChatAsync,
    BaseAIModel,
    BaseAITextEmbeddingModel,
)
from haintech.ai.anthropic import AnthropicAIModel
from haintech.ai.deep_seek.deep_seek_ai_model import DeepSeekAIModel
from haintech.ai.google_generativeai import GoogleAIModel, GoogleAITextEmbeddingModel
from haintech.ai.hugging_face import HuggingFaceTextEmbeddingModel
from haintech.ai.open_ai import OpenAIModel, OpenAITextEmbeddingModel


@pytest.fixture(
    params=[OpenAIModel, GoogleAIModel, DeepSeekAIModel, AnthropicAIModel],
    scope="session",
)
def ai_model(request: pytest.FixtureRequest) -> BaseAIModel:
    return request.param(parameters={"temperature": 0})


@pytest.fixture(
    params=[
        OpenAITextEmbeddingModel,
        GoogleAITextEmbeddingModel,
        HuggingFaceTextEmbeddingModel,
    ],
    scope="session",
)
def ai_embedding_model(request: pytest.FixtureRequest) -> BaseAITextEmbeddingModel:
    return request.param()


@pytest.fixture
def ai_chat(ai_model: BaseAIModel) -> BaseAIChat:
    return BaseAIChat(ai_model=ai_model)


@pytest.fixture
def ai_chat_async(ai_model: BaseAIModel) -> BaseAIChatAsync:
    return BaseAIChatAsync(ai_model=ai_model)


@pytest.fixture
def session() -> Iterator[AIChatSession]:
    """Session object which is stored on disk."""
    factory = LocalFactory("./tests/data/")
    storage = factory.create_storage("test_sessions", AIChatSession, "datetime")
    session = AIChatSession()
    yield session
    storage.create(session)


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


@pytest.fixture
def ai_chat_with_session(ai_model: GoogleAIModel, session: AIChatSession) -> BaseAIChat:
    return BaseAIChat(ai_model=ai_model, session=session)


@pytest.fixture
def ai_chat_async_with_session(ai_model: GoogleAIModel, session: AIChatSession) -> BaseAIChatAsync:
    return BaseAIChatAsync(ai_model=ai_model, session=session)


class HRAgent(BaseAIAgent):
    def __init__(self, ai_model: BaseAIModel, session: Optional[AIChatSession] = None):
        super().__init__(
            description="HR assistant. Answer prersonal information about HR, days off and vacation related question.",
            context="You are a helpful HR assistant.",
            ai_model=ai_model,
            functions=[
                get_remaining_vacation_days,
                get_remaining_home_office_days,
            ],
            session=session,
        )


@pytest.fixture
def hr_agent(ai_model: BaseAIModel, session: AIChatSession) -> HRAgent:
    return HRAgent(ai_model=ai_model, session=session)
