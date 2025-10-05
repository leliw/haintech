from typing import override

import pytest

from haintech.ai import AIChatSession, AIPrompt, BaseAIChat, BaseAIModel


@pytest.mark.skip(reason="Next tests covers it.")
def test_chat_one_question(ai_chat: BaseAIChat):
    # Given: Simple question
    q = "Who was the first US president?"
    # When: I ask model
    response = ai_chat.get_text_response(q)
    # Then: I should get answer
    assert "George" in response


# @pytest.mark.skip(reason="Next tests covers it.")
def test_chat_dialog_with_two_questions(ai_chat: BaseAIChat):
    # Given: A chat with one question
    test_chat_one_question(ai_chat)
    # And: A second question refers to the previous one
    q = "Who was his father?"
    # When: I ask model
    response = ai_chat.get_text_response(q)
    # Then: I should get answer
    assert "Augustine" in response


def test_chat_dialog_with_two_questions_and_session(
    ai_chat_with_session: BaseAIChat,
    session: AIChatSession,
):
    # Given: Simple question
    q = "Who was the first US president?"
    # And: A response
    response = ai_chat_with_session.get_text_response(q)
    assert "George" in response
    # And: A second question refers to the previous one
    q = "Who was his father?"
    # When: I ask model
    response = ai_chat_with_session.get_text_response(q)
    # Then: I should get answer
    assert "Augustine" in response
    # And: Both interactions are stored
    keys = list(session.interactions)
    assert len(keys) == 2
    # And: Second interaction has a history with two messages
    assert len(session.interactions[1].history) == 2


class Pl2EnTranslator(BaseAIChat):
    def __init__(self, ai_model: BaseAIModel):
        super().__init__(ai_model=ai_model)

    @override
    def _get_prompt(self, message=None):
        return AIPrompt(
            persona="You are an English to Polish translator.",
            constraints="Answer always in polish.",
        )


def test_get_chat_response_with_context(ai_model: BaseAIModel):
    # Given: An AI Chat with get_context() method
    ai_chat = Pl2EnTranslator(ai_model=ai_model)
    # When: I get response from chat
    response = ai_chat.get_text_response("The day after Sunday is?")
    # Then: A response is returned
    assert "oniedziałek" in response
