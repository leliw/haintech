import pytest

from haintech.ai import (
    AIChatResponse,
    AIModelInteractionMessage,
    BaseAIModel,
)
from haintech.ai.model import AIContext, RAGItem


@pytest.mark.asyncio
async def test_get_chat_response(ai_model: BaseAIModel):
    # Given: An AI Model
    # When: I get response from chat
    resp = await ai_model.get_chat_response_async(
        history=[AIModelInteractionMessage(role="user", content="The day after Sunday is?")]
    )
    # Then: A response is returned
    assert isinstance(resp, AIChatResponse)
    assert resp.content
    assert "Monday" in resp.content


@pytest.mark.asyncio
async def test_get_chat_response_with_prompt(ai_model: BaseAIModel):
    # Given: A system prompt
    system_prompt = "You are a helpful assistant. Always answer in Polish."
    # When: I get response from chat
    resp = await ai_model.get_chat_response_async(
        system_prompt=system_prompt,
        history=[AIModelInteractionMessage(role="user", content="The day after Sunday is?")],
    )
    # Then: A response is returned
    assert isinstance(resp, AIChatResponse)
    assert resp.content
    assert "poniedziałek" in resp.content.lower()


@pytest.mark.asyncio
async def test_get_chat_response_with_context(ai_model: BaseAIModel):
    # Given: A context
    context = AIContext(context="Today is Sunday.")
    # When: I get response from chat
    resp = await ai_model.get_chat_response_async(
        history=[AIModelInteractionMessage(role="user", content="The day after Sunday is?")],
        context=context,
    )
    # Then: A response is returned
    assert isinstance(resp, AIChatResponse)
    assert resp.content
    assert "Monday" in resp.content


@pytest.mark.asyncio
async def test_get_chat_response_with_context_documents(ai_model: BaseAIModel):
    # Given: A context with documents
    context = AIContext(documents=[
        RAGItem(content="My daughter's name is Victoria."),
        RAGItem(content="My wife's name is Dorothy."),
    ])
    # When: I get response from chat
    resp = await ai_model.get_chat_response_async(
        history=[AIModelInteractionMessage(role="user", content="What is my wife's name?")],
        context=context,
    )
    # Then: A response is returned
    assert isinstance(resp, AIChatResponse)
    assert resp.content
    assert "Dorothy" in resp.content