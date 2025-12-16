import pytest
from ampf.local_async import LocalAsyncFactory
from google.generativeai.client import configure as genai_configure

from haintech.ai.base import BaseAIAgentAsync
from haintech.ai.google_generativeai import GoogleAIModel, GoogleAIParameters
from haintech.ai.model import AIChatSession, AIModelInteractionMessage


@pytest.fixture(scope="function")
def ai_model() -> GoogleAIModel:
    genai_configure()
    return GoogleAIModel("gemini-2.5-flash-lite", parameters=GoogleAIParameters(temperature=0))


@pytest.mark.parametrize("file_name", ["answer.txt", "answer.pdf", "answer.png"])
@pytest.mark.asyncio
async def test_get_response_with_blob_location(ai_model: GoogleAIModel, file_name: str):
    # Given: The Google AI Model with factory
    factory = LocalAsyncFactory("./tests/data")
    session = AIChatSession()
    ai_agent = BaseAIAgentAsync(
        ai_model=ai_model, system_prompt="You are a helpful assistant.", session=session, factory=factory
    )
    # And: A blob with answer
    blob_location = factory.create_blob_location(file_name)
    # When: Ask for a response
    response = await ai_agent.get_response(
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
