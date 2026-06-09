import pytest

from haintech.ai import AIChatSession, BaseAIChat
from haintech.ai.open_ai import ResponsesAIModel


@pytest.fixture
def tm_session() -> AIChatSession:
    return AIChatSession()


@pytest.fixture
def ai_chat(ai_model: ResponsesAIModel, tm_session: AIChatSession) -> BaseAIChat:
    return BaseAIChat(ai_model=ai_model, session=tm_session)


@pytest.mark.skip(reason="Next tests covers it.")
def test_chat_one_question(ai_chat: BaseAIChat, tm_session: AIChatSession):
    # Given: Simple question
    q = "Who was the first polish king?"
    # When: I ask model
    response = ai_chat.get_text_response(q)
    # Then: I should get answer
    assert "Bolesław" in response
    # And: There is one interaction
    assert 1 == len(tm_session.interactions)


def test_dialog_with_two_chats(
    ai_model: ResponsesAIModel,
    ai_chat: BaseAIChat,
    tm_session: AIChatSession,
):
    # Given: Session with one interaction
    test_chat_one_question(ai_chat, tm_session)
    # When: I create a new chat with this session
    c2 = BaseAIChat(ai_model=ai_model, session=tm_session)
    # And: A question refers to the previous one
    q = "Who was his father?"
    # And: I ask the second model
    response = c2.get_text_response(q)
    # Then: I should get answer
    assert "Mieszko" in response
    # And: There are two interactions
    assert 2 == len(tm_session.interactions)
