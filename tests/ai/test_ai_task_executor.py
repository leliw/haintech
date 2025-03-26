import pytest

from haintech.ai.ai_task_executor import AITaskExecutor
from haintech.ai.base.base_ai_model import BaseAIModel
from haintech.ai.model import AIFunctionParameter, AIPrompt, AITask
from haintech.ai.google_generativeai.google_ai_model import GoogleAIModel
from haintech.ai.open_ai.open_ai_model import OpenAIModel


@pytest.fixture(params=[OpenAIModel, GoogleAIModel])
def ai_model(request: pytest.FixtureRequest) -> BaseAIModel:
    ai_model = request.param(parameters={"temperature": 0})
    return ai_model


def test_return_str(ai_model):
    # Given: AITaskExecutor
    te = AITaskExecutor(
        ai_model=ai_model,
        system_instructions=AIPrompt(
            persona="You are experienced developer",
            objective="Prepare the name for the feature",
            instructions="Use snake_case.",
            context="It will be used as a folder name contais code.",
            constraints="Use english language and plural form.",
            examples=[
                "Q: Zarządzanie użytkownikami\nA: users",
                "Q: CRUD dla książek\nA: books",
                "Q: Obsługa projektów\nA: projects",
            ],
        ),
        prompt="{prompt}",
    )
    # When: Execute
    ret = te.execute(prompt="Zarządzanie użytkownikami")
    # Then: Return string
    assert isinstance(ret, str)
    assert "users" == ret


def test_create_from_definition(ai_model):
    # Given: AITask definition
    td = AITask(
        name="get_feature_name",
        description="Get feature name",
        parameters=[
            AIFunctionParameter(name="feature", description="Feature description", type="str")
        ],
        return_type="str",
        return_description="Feature name",
        system_instructions=AIPrompt(
            persona="You are experienced developer",
            objective="Prepare the name for the feature",
            instructions="Use snake_case.",
            context="It will be used as a folder name contais code.",
            constraints="Use english language and plural form.",
            examples=[
                "Q: Zarządzanie użytkownikami\nA: users",
                "Q: CRUD dla książek\nA: books",
                "Q: Obsługa projektów\nA: projects",
            ],
        ),
        prompt="{feature}",
    )
    # When: Create AITaskExecutor from definition
    te = AITaskExecutor.create_from_definition(ai_model=ai_model, task_definition=td)
    # And: Execute
    ret = te.execute(feature="Zarządzanie użytkownikami")
    # Then: Return string
    assert isinstance(ret, str)
    assert "users" == ret
