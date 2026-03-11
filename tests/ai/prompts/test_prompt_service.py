from pathlib import Path

from pydantic import BaseModel
import pytest

from haintech.ai.prompts.prompt_service import PromptService

@pytest.fixture
def prompt_service() -> PromptService:
    root_path = Path("./tests/data/prompts")
    return PromptService(root_path)

def test_load_all(prompt_service: PromptService):
    # Given:A prompt service
    assert prompt_service is not None
    # When: Load all prompts
    prompt_service.load_all_prompts()
    # Then: Prompts are loaded
    assert len(prompt_service.registry) > 0


def test_render_ok(prompt_service: PromptService):
    # Given:A prompt service
    assert prompt_service is not None
    # When: Render test prompt
    s, u = prompt_service.render("test", a="a", b="b")
    # Then: System and user prompts are returned
    assert s == "System prompt (a,b)"
    assert u == "User prompt (a,b)"


def test_render_error(prompt_service: PromptService):
    # Given:A prompt service
    assert prompt_service is not None
    # When: Render not existing prompt
    with pytest.raises(ValueError) as e:
        s, u = prompt_service.render("not_existing", a="a", b="b")
        # Then: Error is raised
        assert e


def test_get_output_class(prompt_service: PromptService):
    # Given:A prompt service
    assert prompt_service is not None
    # When: Render test prompt
    oc = prompt_service.get_output_class("test_output")
    # Then: Output class is returned
    assert oc
    assert oc.__name__ == "Output"
    assert issubclass(oc, BaseModel)