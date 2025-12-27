import pytest
from ampf.local import LocalFactory

from haintech.ai.exceptions import UnsupportedMimeTypeError
from haintech.ai.google_generativeai import GoogleAIModel, GoogleAIParameters
from haintech.ai.model import AIModelInteractionMessage

from google.generativeai.client import configure as genai_configure

@pytest.fixture(
    params=[
        "gemini-2.5-flash-lite",
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        #
        "nano-banana-pro-preview",
        "gemini-3-pro-preview",
    ]
)
def ai_model(request: pytest.FixtureRequest) -> GoogleAIModel:
    genai_configure()
    return GoogleAIModel(request.param, parameters=GoogleAIParameters(temperature=0))


def test_list_models():
    # Given: Google AI Model
    ai_model = GoogleAIModel()
    # When: Ask for model names
    ret = ai_model.get_model_names()
    # Then: At least one gemini model name is returned
    assert any("gemini" in n for n in ret)


def test_get_chat_response(ai_model: GoogleAIModel):
    response = ai_model.get_chat_response(
        system_prompt="You are a helpful assistant.",
        message=AIModelInteractionMessage(role="user", content="What is the capital of France?"),
    )
    assert response.content and "Paris" in response.content


@pytest.mark.parametrize("file_name", ["answer.txt", "answer.pdf", "answer.png"])
def test_get_chat_response_with_blob(file_name: str):
    # Given: Google AI Model
    ai_model = GoogleAIModel("gemini-2.5-flash-lite", parameters=GoogleAIParameters(temperature=0))
    # And: A blob with answer
    blob_storage = LocalFactory("./tests/data").create_blob_storage("")
    blob_storage.default_ext = None
    blob = blob_storage.download(file_name)
    # When: Ask for a response
    response = ai_model.get_chat_response(
        system_prompt="You are a helpful assistant.",
        message=AIModelInteractionMessage(role="user", content="What is my dog's breed?", blobs=[blob]),
    )
    # Then: The response contains the dog's breed from an answer blob
    assert response.content and "labrador" in response.content.lower()


@pytest.mark.parametrize("file_name", ["answer.docx"])
def test_get_chat_response_with_unsuported_blob(file_name: str):
    # Given: Google AI Model
    ai_model = GoogleAIModel("gemini-2.5-flash-lite", parameters=GoogleAIParameters(temperature=0))
    # And: A blob with answer
    blob_storage = LocalFactory("./tests/data").create_blob_storage("")
    blob_storage.default_ext = None
    blob = blob_storage.download(file_name)
    # When: Ask for a response
    # Then: The response contains the dog's breed from an answer blob
    with pytest.raises(UnsupportedMimeTypeError):
        ai_model.get_chat_response(
            system_prompt="You are a helpful assistant.",
            message=AIModelInteractionMessage(role="user", content="What is my dog's breed?", blobs=[blob]),
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("file_name", ["answer.docx"])
async def test_get_chat_response_async_with_unsuported_blob(file_name: str):
    # Given: Google AI Model
    ai_model = GoogleAIModel("gemini-2.5-flash-lite", parameters=GoogleAIParameters(temperature=0))
    # And: A blob with answer
    blob_storage = LocalFactory("./tests/data").create_blob_storage("")
    blob_storage.default_ext = None
    blob = blob_storage.download(file_name)
    # When: Ask for a response
    # Then: The response contains the dog's breed from an answer blob
    with pytest.raises(UnsupportedMimeTypeError):
        await ai_model.get_chat_response_async(
            system_prompt="You are a helpful assistant.",
            message=AIModelInteractionMessage(role="user", content="What is my dog's breed?", blobs=[blob]),
        )
