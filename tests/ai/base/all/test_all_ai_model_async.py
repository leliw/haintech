import pytest

from haintech.ai import (
    AIChatResponse,
    AIModelInteractionMessage,
    BaseAIModel,
)


@pytest.mark.asyncio
async def test_get_chat_response(ai_model: BaseAIModel):
    # Given: An AI Model
    # When: I get response from chat
    resp = await ai_model.get_chat_response_async(
        history=[
            AIModelInteractionMessage(role="user", content="The day after Sunday is?")
        ]
    )
    # Then: A response is returned
    assert isinstance(resp, AIChatResponse)
    assert "Monday" in resp.content
