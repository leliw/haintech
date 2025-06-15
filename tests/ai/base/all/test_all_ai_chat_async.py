import pytest

from haintech.ai import AIChatSession, BaseAIChatAsync


# @pytest.mark.skip(reason="Next tests covers it.")
@pytest.mark.asyncio
async def test_chat_one_question(ai_chat_async: BaseAIChatAsync):
    # Given: Simple question
    q = "Who was the first US president?"
    # When: I ask model
    response = await ai_chat_async.get_text_response(q)
    # Then: I should get answer
    assert "George" in response


@pytest.mark.asyncio
async def test_chat_dialog_with_two_questions(ai_chat_async: BaseAIChatAsync):
    # Given: A chat with one question
    await test_chat_one_question(ai_chat_async)
    # And: A second question refers to the previous one
    q = "Who was his father?"
    # When: I ask model
    response = await ai_chat_async.get_text_response(q)
    # Then: I should get answer
    assert "Augustine" in response


@pytest.mark.asyncio
async def test_chat_dialog_with_two_questions_and_session(
    ai_chat_async_with_session: BaseAIChatAsync,
    session: AIChatSession,
):
    # Given: Simple question
    q = "Who was the first US president?"
    # And: A response
    response = await ai_chat_async_with_session.get_text_response(q)
    assert "George" in response
    # And: A second question refers to the previous one
    q = "Who was his father?"
    # When: I ask model
    response = await ai_chat_async_with_session.get_text_response(q)
    # Then: I should get answer
    assert "Augustine" in response
    # And: Both interactions are stored
    keys = list(session.interactions)
    assert len(keys) == 2
    # And: Second interaction has a history with two messages
    assert len(session.interactions[1].history) == 2
