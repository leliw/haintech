import pytest
from ampf.local import LocalFactory
from pydantic import BaseModel

from haintech.ai.model import AIModelInteractionMessage
from haintech.ai.open_ai.model import ResponsesAIParameters
from haintech.ai.open_ai.responses_ai_model import ResponsesAIModel


@pytest.fixture(params=["gpt-4o-mini", "gpt-4.1-mini", "gpt-5-mini", "gpt-5.1", "gpt-5.4-nano"])
# @pytest.fixture(params=["gpt-5.4-nano"])
def ai_model(request: pytest.FixtureRequest) -> ResponsesAIModel:
    return ResponsesAIModel(request.param, parameters=ResponsesAIParameters(temperature=0.1))


def test_get_chat_response(ai_model: ResponsesAIModel):
    response = ai_model.get_chat_response(
        system_prompt="You are a helpful assistant.",
        message=AIModelInteractionMessage(role="user", content="What is the capital of France?"),
    )
    assert response.content and "Paris" in response.content


def test_get_model_names():
    ai_model = ResponsesAIModel("gpt-4o-mini")
    model_names = ai_model.get_model_names()
    assert "gpt-4o-mini" in model_names


class Book(BaseModel):
    title: str
    author: str
    year: int
    genre: str


def test_get_response(ai_model: ResponsesAIModel):
    # When: Get text response from ai model
    ret = ai_model.get_response("Return the title of the first Harry Potter book.")
    # Then: Text is returned
    assert "Harry Potter" in ret


def test_get_response_typed(ai_model: ResponsesAIModel):
    # When: Get typed response from ai model
    ret = ai_model.get_response_typed("Return the title of the first Harry Potter book. (in json format)", Book)
    # Then: Object is returned with the response
    assert isinstance(ret, Book)
    assert "Harry Potter" in ret.title


def test_get_response_list_typed(ai_model: ResponsesAIModel):
    # When: Get typed list response from ai model
    ret = ai_model.get_response_list_typed("Return list of Harry Potter book titles. (in json format)", Book)
    # Then: List of objects is returned with the response
    assert all("Harry Potter" in b.title for b in ret)


def test_get_response_list_str(ai_model: ResponsesAIModel):
    # When: Get list response from ai model
    ret = ai_model.get_response_list("Return list of Harry Potter book titles. (in json format)")
    # Then: List of strings is returned with the response
    assert all("Harry Potter" in b for b in ret)


def test_get_response_list_ints(ai_model: ResponsesAIModel):
    # When: Get list response from ai model
    ret = ai_model.get_response_list("Return list of release years of Harry Potter books. (in json format)", int)
    # Then: List of strings is returned with the response
    assert any(1998 == b for b in ret)


@pytest.mark.asyncio
async def test_get_response_async(ai_model: ResponsesAIModel):
    # When: Get text response from ai model
    ret = await ai_model.get_response_async("Return the title of the first Harry Potter book.")
    # Then: Text is returned
    assert "Harry Potter" in ret


@pytest.mark.asyncio
async def test_get_response_typed_async(ai_model: ResponsesAIModel):
    # When: Get typed response from ai model
    ret = await ai_model.get_response_typed_async(
        "Return the title of the first Harry Potter book. (in json format)", Book
    )
    # Then: Object is returned with the response
    assert isinstance(ret, Book)
    assert "Harry Potter" in ret.title


@pytest.mark.asyncio
async def test_get_response_list_typed_async(ai_model: ResponsesAIModel):
    # When: Get typed list response from ai model
    ret = await ai_model.get_response_list_typed_async("Return list of Harry Potter books. (in json format)", Book)
    # Then: List of objects is returned with the response
    assert all("Harry Potter" in b.title for b in ret)


@pytest.mark.asyncio
async def test_get_response_list_str_async(ai_model: ResponsesAIModel):
    # When: Get list response from ai model
    ret = await ai_model.get_response_list_async("Return list of Harry Potter books. (in json format)")
    # Then: List of strings is returned with the response
    assert all("Harry Potter" in b for b in ret)


@pytest.mark.asyncio
async def test_get_response_list_ints_async(ai_model: ResponsesAIModel):
    # When: Get list response from ai model
    ret = await ai_model.get_response_list_async(
        "Return list of release years of Harry Potter books. (in json format)", int
    )
    # Then: List of strings is returned with the response
    assert any(1998 == b for b in ret)


def test_get_chat_response_with_text_blob():
    # Given: Google AI Model
    ai_model = ResponsesAIModel("gpt-5.4-nano")
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
