from pydantic import BaseModel
import pytest
from ampf.local import LocalFactory

from haintech.ai.exceptions import UnsupportedMimeTypeError
from haintech.ai.model import AIModelInteractionMessage
from haintech.ai.google_genai import GoogleAIModel, GoogleAIParameters


@pytest.fixture(
    params=[
        "gemini-2.5-flash-lite",
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        #
        "gemini-3-flash-preview",
        "gemini-3.1-flash-lite-preview",
        "nano-banana-pro-preview",
    ]
)
def ai_model(request: pytest.FixtureRequest) -> GoogleAIModel:
    return GoogleAIModel(request.param, parameters=GoogleAIParameters(temperature=0))


def test_list_models():
    # Given: Google AI Model
    ai_model = GoogleAIModel()
    # When: Ask for model names
    ret = ai_model.get_model_names()
    # Then: At least one gemini model name is returned
    print(list(ret))
    assert any("gemini" in n for n in ret)
    # assert False

def test_get_chat_response(ai_model: GoogleAIModel):
    response = ai_model.get_chat_response(
        system_prompt="You are a helpful assistant.",
        message=AIModelInteractionMessage(role="user", content="What is the capital of France?"),
    )
    assert response.content and "Paris" in response.content


@pytest.mark.parametrize("file_name", ["answer.pdf", "answer.png"])
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

def test_get_chat_response_with_text_blob():
    # Given: Google AI Model
    ai_model = GoogleAIModel("gemini-2.5-flash-lite", parameters=GoogleAIParameters(temperature=0))
    # And: A text blob with answer
    blob_storage = LocalFactory("./tests/data").create_blob_storage("")
    blob_storage.default_ext = None
    blob = blob_storage.download("answer.txt")
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


class Book(BaseModel):
    title: str
    author: str
    year: int
    genre: str


def test_get_response(ai_model: GoogleAIModel):
    # When: Get text response from ai model
    ret = ai_model.get_response("Return first Harry Potter book. ")
    # Then: Text is returned
    assert "Harry Potter" in ret


def test_get_response_typed(ai_model: GoogleAIModel):
    # When: Get typed response from ai model
    ret = ai_model.get_response_typed("Return first Harry Potter book. (in json format)", Book)
    # Then: Object is returned with the response
    assert isinstance(ret, Book)
    assert "Harry Potter" in ret.title


def test_get_response_list_typed(ai_model: GoogleAIModel):
    # When: Get typed list response from ai model
    ret = ai_model.get_response_list_typed("Return list of Harry Potter books. (in json format)", Book)
    # Then: List of objects is returned with the response
    assert all("Harry Potter" in b.title for b in ret)


def test_get_response_list_str(ai_model: GoogleAIModel):
    # When: Get list response from ai model
    ret = ai_model.get_response_list("Return list of Harry Potter books. (in json format)")
    # Then: List of strings is returned with the response
    assert all("Harry Potter" in b for b in ret)

def test_get_response_list_ints(ai_model: GoogleAIModel):
    # When: Get list response from ai model
    ret = ai_model.get_response_list("Return a list of Harry Potter book release years. (in json format)", int)
    # Then: List of strings is returned with the response
    assert any(1998 == b for b in ret)



@pytest.mark.asyncio
async def test_get_response_async(ai_model: GoogleAIModel):
    # When: Get text response from ai model
    ret = await ai_model.get_response_async("Return the first Harry Potter book title. ")
    # Then: Text is returned
    assert "Harry Potter" in ret


@pytest.mark.asyncio
async def test_get_response_typed_async(ai_model: GoogleAIModel):
    # When: Get typed response from ai model
    ret = await ai_model.get_response_typed_async("Return the first Harry Potter book. (in json format)", Book)
    # Then: Object is returned with the response
    assert isinstance(ret, Book)
    assert "Harry Potter" in ret.title


@pytest.mark.asyncio
async def test_get_response_list_typed_async(ai_model: GoogleAIModel):
    # When: Get typed list response from ai model
    ret = await ai_model.get_response_list_typed_async("Return list of Harry Potter books. (in json format)", Book)
    # Then: List of objects is returned with the response
    assert all("Harry Potter" in b.title for b in ret)


@pytest.mark.asyncio
async def test_get_response_list_str_async(ai_model: GoogleAIModel):
    # When: Get list response from ai model
    ret = await ai_model.get_response_list_async("Return list of Harry Potter books. (in json format)")
    # Then: List of strings is returned with the response
    assert all("Harry Potter" in b for b in ret)

@pytest.mark.asyncio
async def test_get_response_list_ints_async(ai_model: GoogleAIModel):
    # When: Get list response from ai model
    ret = await ai_model.get_response_list_async("Return a list of Harry Potter book release years. (in json format)", int)
    # Then: List of strings is returned with the response
    assert any(1998 == b for b in ret)
