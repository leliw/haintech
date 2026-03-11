from pathlib import Path


from haintech.ai import AIChatResponse
import pytest

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




def test_prepare_response_typed(prompt_executor: PromptExecutor):
    m_resp = AIChatResponse(
        content="""
    {
        "language": "en",
        "level": "A1",
        "order": 1,
        "type": "info",
        "title": "Verb to be - Introduction",
        "content": "The verb 'to be' is very important in English. It means 'am', 'is', or 'are'. We use it to talk about people, things, and places."
    }
"""
    )
    ret = prompt_executor._prepare_response_typed(InfoPageCreate, m_resp)
    # Then: A InfoPageCreate objects is returned
    assert isinstance(ret, InfoPageCreate)


def test_prepare_response_typed_output(prompt_service: PromptService, prompt_executor: PromptExecutor):
    m_resp = AIChatResponse(
        content="""{"content": "The verb 'to be' is very important in English. It means 'am', 'is', or 'are'. We use it to talk about people, things, and places."}"""
    )
    ret = prompt_executor._prepare_response_typed(prompt_service.get_output_class("test_output"), m_resp)  # type: ignore
    # Then: A InfoPageCreate objects is returned after conversion
    out = ret.convert(language="en", level="A1", title="Verb to be - Introduction")

    assert isinstance(out, InfoPageCreate)


def test_execute_typed_output(prompt_executor: PromptExecutor, mocker_ai_model: MockerAIModel):
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


def test_prepare_response_typed_list(prompt_executor: PromptExecutor):
    m_resp = AIChatResponse(
        content="""
{
  "type": "array",
  "items": [
    {
        "language": "en",
        "level": "A1",
        "order": 1,
        "type": "info",
        "title": "Verb to be - Introduction",
        "content": "The verb 'to be' is very important in English. It means 'am', 'is', or 'are'. We use it to talk about people, things, and places."
    },
    {
        "language": "en",
        "level": "A1",
        "order": 1,
        "type": "info",
        "title": "Using 'am'",
        "content": "'Am' is used with 'I'. For example: I am happy. I am a student."
    }
  ]
}
"""
    )
    ret = prompt_executor._prepare_response_typed_list(InfoPageCreate, m_resp)
    # Then: A list of InfoPageCreate objects is returned
    assert len(ret) > 0
    assert isinstance(ret, list)
    assert all(isinstance(page, InfoPageCreate) for page in ret)
