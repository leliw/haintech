from typing import Iterable
from unittest.mock import AsyncMock
import pytest
from pytest_mock import MockerFixture
from haintech.ai.base.base_ai_agent_async import BaseAIAgentAsync
from haintech.ai.base.base_ai_model import BaseAIModel
from haintech.ai.base.base_rag_searcher import BaseRAGSearcher
from haintech.ai.model import AIContext, AIChatSession, RAGItem, RAGQuery

class MockSearcher(BaseRAGSearcher):
    def search_sync(self, query: RAGQuery) -> Iterable[RAGItem]:
        # This is a mock implementation, replace with actual search logic if needed
        return []


@pytest.mark.asyncio
async def test_agent_one_question(ai_model: BaseAIModel, mocker: MockerFixture):
    # Given: A mock for the RAG searcher
    mock_searcher = MockSearcher()
    mock_agent_search = mocker.patch.object(
        mock_searcher,
        "agent_search_async",
        new_callable=AsyncMock,
        return_value=AIContext(
            documents=[
                RAGItem(content="Remaining vacation days in 2024 is 0."),
                RAGItem(content="Remaining vacation days in 2025 are 26."),
                RAGItem(content="Remaining home office days in 2025 are 4."),
            ]
        ),
    )
    # Given: An agent with session and searcher
    session = AIChatSession()
    ai_agent = BaseAIAgentAsync(ai_model=ai_model, session=session, searcher=mock_searcher)
    question = "How many vacation days do I have left in 2025?"

    # When: I ask agent
    response = await ai_agent.get_text_response(question)

    # Then: I should get answer
    assert "26" in response
    # And: The searcher should be called with the correct query
    mock_agent_search.assert_called_once()


@pytest.mark.asyncio
async def test_agent_two_questions(ai_model: BaseAIModel, mocker: MockerFixture):
    # Given: A mock for the RAG searcher
    mock_searcher = MockSearcher()
    mock_agent_search = mocker.patch.object(
        mock_searcher, "agent_search_async", new_callable=AsyncMock
    )
    mock_agent_search.return_value = AIContext(
        documents=[
            RAGItem(content="Remaining vacation days in 2024 is 0."),
            RAGItem(content="Remaining vacation days in 2025 are 26."),
            RAGItem(content="Remaining home office days in 2025 are 4."),
        ]
    )

    # Given: An agent with session and searcher
    session = AIChatSession()
    ai_agent = BaseAIAgentAsync(ai_model=ai_model, session=session, searcher=mock_searcher)
    # When: I ask agent
    question = "How many vacation days do I have left in 2025?"
    response = await ai_agent.get_text_response(question)
    # Then: I should get answer
    assert "26" in response
    # And: The searcher should be called with the correct query
    mock_agent_search.assert_called_once()

    # When: I ask agent
    question = "And how many in 2024?"
    response = await ai_agent.get_text_response(question)
    # Then: I should get answer
    assert "0" in response
    # And: The searcher should be called again
    assert mock_agent_search.call_count == 2
