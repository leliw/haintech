from pathlib import Path

from pydantic import BaseModel
import pytest
from ampf.local import LocalFactory

from haintech.ai import AIChatResponse
from haintech.ai.google_genai import GoogleAIModel
from haintech.ai.prompts.prompt_executor import PromptExecutor
from haintech.ai.prompts.prompt_service import PromptService
from haintech.testing import MockerAIModel
from tests.ai.prompts.model import InfoPageCreate


@pytest.fixture
def prompt_service() -> PromptService:
    root_path = Path("./tests/data/prompts")
    return PromptService(root_path)


@pytest.fixture
def prompt_executor(prompt_service: PromptService) -> PromptExecutor:
    ai_model = GoogleAIModel("gemini-2.5-flash-lite")
    return PromptExecutor(ai_model, prompt_service)


class Book(BaseModel):
    title: str
    author: str
    year: int
    genre: str


def test_execute_typed(prompt_executor: PromptExecutor, mocker_ai_model: MockerAIModel):
    mocker_ai_model.add(
        message_containing="{'properties': {'content': {'title': 'Content', 'type': 'string'}}, 'required': ['content'], 'title': 'Output', 'type': 'object'}",
        response='{"content": "Welcome to the fascinating world of English verbs! Today, we\'re diving into one of the most fundamental and frequently used verbs: \'to be\'." }',
    )
    # When:
    ret = prompt_executor.execute_typed(
        "test_output", InfoPageCreate, title="Verb to be - Introduction", language="en", level="A1"
    )
    # Then: A InfoPageCreate objects is returned after conversion
    assert isinstance(ret, InfoPageCreate)


def test_execute_typed_list(prompt_executor: PromptExecutor, mocker_ai_model: MockerAIModel):
    mocker_ai_model.add_calls(
        [
            {
                "system_prompt": "",
                "message_str": "Return list of Harry Potter books.",
                "response": {
                    "content": '{"list":[{"title":"Harry Potter and the Sorcerer\'s Stone","author":"J.K. Rowling","year":1997,"genre":"Fantasy"},{"title":"Harry Potter and the Chamber of Secrets","author":"J.K. Rowling","year":1998,"genre":"Fantasy"},{"title":"Harry Potter and the Prisoner of Azkaban","author":"J.K. Rowling","year":1999,"genre":"Fantasy"},{"title":"Harry Potter and the Goblet of Fire","author":"J.K. Rowling","year":2000,"genre":"Fantasy"},{"title":"Harry Potter and the Order of the Phoenix","author":"J.K. Rowling","year":2003,"genre":"Fantasy"},{"title":"Harry Potter and the Half-Blood Prince","author":"J.K. Rowling","year":2005,"genre":"Fantasy"},{"title":"Harry Potter and the Deathly Hallows","author":"J.K. Rowling","year":2007,"genre":"Fantasy"}]}'
                },
            }
        ]
    )
    # When:
    ret = prompt_executor.execute_typed_list("book_list", Book)
    # Then: A InfoPageCreate objects is returned after conversion
    assert all(isinstance(book, Book) for book in ret)


def test_execute_list(prompt_executor: PromptExecutor, mocker_ai_model: MockerAIModel):
    mocker_ai_model.add_calls(
        [
            {
                "system_prompt": "",
                "message_str": "Return a list of Harry Potter book release years.",
                "response": {"content": '{"list":[1997,1998,1999,2000,2003,2005,2007]}'},
            }
        ]
    )
    # When:
    ret = prompt_executor.execute_list("book_years", int)
    # Then: A InfoPageCreate objects is returned after conversion
    assert all(isinstance(year, int) for year in ret)


def test_execute_with_blob(prompt_executor: PromptExecutor, mocker_ai_model: MockerAIModel):
    mocker_ai_model.add_calls(
        [
            {
                "system_prompt": "",
                "message_str": "Describe what is in the attached picture.",
                "blob_contents": [
                    "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"
                ],
                "response": {
                    "content": 'The attached image displays a short sentence against a plain white background.\n\nThe sentence reads: **"My dog is a labrador."**\n\nKey details:\n*   The text is dark (likely black or dark gray) and appears to be in a standard serif font.\n*   To the upper left of the first letter "M", there is a small, light gray, L-shaped symbol, which typically represents a text cursor or insertion point in a word processing application.\n*   The image appears to be a cropped screenshot or a section of a digital document.'
                },
            }
        ]
    )
    file_name = "answer.png"
    factory = LocalFactory("./tests/data")
    blob_location = factory.create_blob_location(file_name)
    blob = factory.download_blob(blob_location)
    # When:
    ret = prompt_executor.execute("picture_descriptor", blobs=[blob])
    # Then: A InfoPageCreate objects is returned after conversion
    assert "dog" in ret.lower()
