import pytest
from ampf.local import LocalFactory

from haintech.ai.ai_agent_factory import AIAgentFactory
from haintech.ai.base import BaseAIAgent
from haintech.ai.google_genai import GoogleAIModel, GoogleAIParameters
from haintech.ai.model import AIChatSession, AIModelInteractionMessage


@pytest.mark.parametrize("file_name", ["answer.txt", "answer.pdf", "answer.png"])
def test_get_response_with_blob_location(file_name: str):
    # Given: The Google AI Model with factory
    factory = LocalFactory("./tests/data")
    ai_model = GoogleAIModel(parameters=GoogleAIParameters(temperature=0))
    session = AIChatSession()
    ai_agent = BaseAIAgent(
        ai_model=ai_model, system_prompt="You are a helpful assistant.", session=session, session_blob_manager=factory
    )
    # And: A blob with answer
    blob_location = factory.create_blob_location(file_name)
    # When: Ask for a response
    response = ai_agent.get_response(
        message=AIModelInteractionMessage(
            role="user", content="What is my dog's breed?", blob_locations=[blob_location]
        ),
    )
    # Then: The response contains the dog's breed from an answer blob
    assert response.content and "labrador" in response.content.lower()
    iteration = session.get_last_interaction()
    assert iteration and iteration.message
    assert iteration.message.blob_locations is not None
    assert iteration.message.blobs is None


@pytest.fixture
def ai_agent_factory():
    return AIAgentFactory.create([GoogleAIModel])


async def test_vendor_model_name(ai_agent_factory: AIAgentFactory):
    # Given: The AI Agent with session
    vendor_model_name = "google/gemini-3.5-flash-lite"
    session = AIChatSession[AIModelInteractionMessage]()
    ai_agent = ai_agent_factory.create_ai_agent_async(vendor_model_name, session=session)
    # When: Get response
    response = await ai_agent.get_response("Who was the first US president?")
    # Then: The same vendor model name is returned
    assert vendor_model_name == response.vendor_model_name
    # And: It is stored in session
    assert session.interactions[-1].response
    assert vendor_model_name == session.interactions[-1].response.vendor_model_name
    messages = [m for m in session.messages_iterator()]
    assert vendor_model_name == messages[-1].vendor_model_name
