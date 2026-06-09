from typing import Iterator

import pytest
from ampf.base import BaseStorage
from ampf.in_memory import InMemoryFactory
from ampf.local import LocalFactory

from haintech.ai import AIChatSession, AIModelInteraction, BaseAIAgent, BaseAIChat
from haintech.ai.open_ai import ResponsesAIModel, ResponsesAIParameters


@pytest.fixture
def ai_model() -> ResponsesAIModel:
    return ResponsesAIModel()


@pytest.fixture
def storage() -> Iterator[BaseStorage[AIModelInteraction]]:
    factory = InMemoryFactory()
    storage = factory.create_storage("ai_log", AIModelInteraction)
    yield storage
    storage.drop()


@pytest.fixture
def ai_chat(ai_model: ResponsesAIModel) -> BaseAIChat:
    return BaseAIChat(ai_model)


@pytest.fixture
def session() -> Iterator[AIChatSession]:
    """Session object which is stored on disk."""
    factory = LocalFactory("./tests/data/")
    storage = factory.create_storage("test_sessions", AIChatSession, "datetime")
    session = AIChatSession()
    yield session
    storage.create(session)


@pytest.fixture
def ai_chat_with_session(ai_model: ResponsesAIModel, session: AIChatSession) -> BaseAIChat:
    return BaseAIChat(ai_model, session=session)


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


class HRAgent(BaseAIAgent):
    def __init__(self, session: AIChatSession | None = None):
        super().__init__(
            description="HR Assistant",
            system_prompt="You are a helpful HR assistant.",
            ai_model=ResponsesAIModel(parameters=ResponsesAIParameters(temperature=0)),
            functions=[
                get_remaining_vacation_days,
                get_remaining_home_office_days,
            ],
            session=session,
        )


@pytest.fixture
def hr_agent(session: AIChatSession) -> HRAgent:
    return HRAgent(session=session)
