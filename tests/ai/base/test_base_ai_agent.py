from ampf.local import LocalFactory
from haintech.ai.base import BaseAIAgent
from haintech.ai.google_generativeai import GoogleAIModel, GoogleAIParameters
from haintech.ai.model import AIChatSession, AIModelInteractionMessage
import pytest


@pytest.mark.parametrize("file_name", ["answer.txt", "answer.pdf", "answer.png"])
def test_get_response_with_blob_location(file_name: str):
    # Given: The Google AI Model with factory
    factory = LocalFactory("./tests/data")
    ai_model = GoogleAIModel("gemini-2.5-flash-lite", parameters=GoogleAIParameters(temperature=0))
    session = AIChatSession()
    ai_agent = BaseAIAgent(ai_model=ai_model, system_prompt="You are a helpful assistant.", session=session, session_blob_manager=factory)
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
