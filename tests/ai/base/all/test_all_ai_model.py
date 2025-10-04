import json
from haintech.ai import (
    AIChatResponse,
    AIModelInteractionMessage,
    AIPrompt,
    BaseAIModel,
)


def test_get_chat_response(ai_model: BaseAIModel):
    # Given: An AI Model
    # When: I get response from chat
    resp = ai_model.get_chat_response(
        history=[
            AIModelInteractionMessage(role="user", content="The day after Sunday is?")
        ]
    )
    # Then: A response is returned
    assert isinstance(resp, AIChatResponse)
    assert resp.content
    assert "Monday" in resp.content


def test_get_chat_response_with_context(ai_model: BaseAIModel):
    # Given: An AI Model
    # When: I get response from chat
    resp = ai_model.get_chat_response(
        system_prompt=AIPrompt(
            persona="You are an English to Polish translator.",
            constraints="Answer always in polish.",
        ),
        message=AIModelInteractionMessage(role="user", content="The day after Sunday is?")
    )
    # Then: A response is returned
    assert isinstance(resp, AIChatResponse)
    assert resp.content
    assert "oniedziałek" in resp.content

def test_get_chat_response_json(ai_model: BaseAIModel):
    # Given: An AI Model
    # When: I get response from chat
    resp = ai_model.get_chat_response(
        history=[
            AIModelInteractionMessage(role="user", content="The day after Sunday is? Retrun JSON. Example: {\"answer\": \"Sunday\"}")
        ],
        response_format="json",
    )
    # Then: A response is returned
    assert isinstance(resp, AIChatResponse)
    assert resp.content
    r = json.loads(resp.content)
    assert "Monday" in r["answer"]