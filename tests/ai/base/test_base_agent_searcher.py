from typing import Iterable
import pytest
from pytest_mock import MockerFixture

from haintech.ai.base.base_agent_searcher import BaseAgentSearcher
from haintech.ai.base.base_ai_model import BaseAIModel
from haintech.ai.model import AIModelInteractionMessage, RAGItem, RAGQuery

class AgentSearcher(BaseAgentSearcher):
    def search_sync(self, query: RAGQuery) -> Iterable[RAGItem]:
        raise NotImplementedError


@pytest.fixture(scope="session")
def agent_searcher(ai_model_lite: BaseAIModel):
    return AgentSearcher(ai_model_lite)


def test_only_message(agent_searcher: BaseAgentSearcher, mocker: MockerFixture):
    # Given: A user question
    question = "Who was the first US president?"

    # When: Search for documents
    # We mock the search_sync method to avoid actual search operations.
    mock_search = mocker.patch.object(
        agent_searcher, "search_sync", return_value=[RAGItem(content="George Washington was the first US president.")]
    )
    ret = agent_searcher.agent_search_sync(None, [], AIModelInteractionMessage(role="user", content=question))

    # Then: The search result is not None
    assert ret is not None
    # And: search_sync is called once with the given question.
    mock_search.assert_called_once()
    mock_search.assert_called_once_with(RAGQuery(text=question))

@pytest.mark.asyncio
async def test_only_message_async(agent_searcher: BaseAgentSearcher, mocker: MockerFixture):
    # Given: A user question
    question = "Who was the first US president?"

    # When: Search for documents
    # We mock the search_sync method to avoid actual search operations.
    mock_search = mocker.patch.object(
        agent_searcher, "search_sync", return_value=[RAGItem(content="George Washington was the first US president.")]
    )
    ret = await agent_searcher.agent_search_async(None, [], AIModelInteractionMessage(role="user", content=question))

    # Then: The search result is not None
    assert ret is not None
    # And: search_sync is called once with the given question.
    mock_search.assert_called_once()
    mock_search.assert_called_once_with(RAGQuery(text=question))

def test_history_and_message(agent_searcher: BaseAgentSearcher, mocker: MockerFixture):
    # Given: A history
    history = [
        AIModelInteractionMessage(role="user", content="Who was the first US president?"),
        AIModelInteractionMessage(role="assistant", content="George Washington was the first US president."),
    ]
    # And: A question
    question = "Who was the next?"
    # When: Search for documents
    # We mock the search_sync method to avoid actual search operations.
    mock_search = mocker.patch.object(
        agent_searcher, "search_sync", return_value=[RAGItem(content="George Washington was the first US president.")]
    )
    ret = agent_searcher.agent_search_sync(None, history, AIModelInteractionMessage(role="user", content=question))

    # Then: The search result is not None
    assert ret is not None
    # And: search_sync is called once with the given question.
    mock_search.assert_called_once()
    assert "second US president" in mock_search.call_args_list[0].args[0].text
    # mock_search.assert_called_once_with(RAGQuery(text="second US president"))


@pytest.mark.asyncio
async def test_history_and_message_async(agent_searcher: BaseAgentSearcher, mocker: MockerFixture):
    # Given: A history
    history = [
        AIModelInteractionMessage(role="user", content="Who was the first US president?"),
        AIModelInteractionMessage(role="assistant", content="George Washington was the first US president."),
    ]
    # And: A question
    question = "Who was the next?"
    # When: Search for documents
    # We mock the search_sync method to avoid actual search operations.
    mock_search = mocker.patch.object(
        agent_searcher, "search_sync", return_value=[RAGItem(content="George Washington was the first US president.")]
    )
    ret = await agent_searcher.agent_search_async(
        None, history, AIModelInteractionMessage(role="user", content=question)
    )

    # Then: The search result is not None
    assert ret is not None
    # And: search_sync is called once with the given question.
    mock_search.assert_called_once()
    assert "second US president" in mock_search.call_args_list[0].args[0].text
    # mock_search.assert_called_once_with(RAGQuery(text="second US president"))
