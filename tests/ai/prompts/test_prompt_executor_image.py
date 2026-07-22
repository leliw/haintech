from pathlib import Path

import pytest

from haintech.ai.google_genai import GoogleAIModel
from haintech.ai.prompts.prompt_executor import PromptExecutor
from haintech.ai.prompts.prompt_executor_image import PromptExecutorImage
from haintech.ai.prompts.prompt_service import PromptService
from haintech.testing import MockerAIModel
from haintech.testing.mocker_image_generator import MockerImageGenerator


@pytest.fixture
def prompt_service() -> PromptService:
    root_path = Path("./tests/data/prompts")
    return PromptService(root_path)


@pytest.fixture
def prompt_executor(prompt_service: PromptService) -> PromptExecutor:
    ai_model = GoogleAIModel("gemini-3.1-flash-lite")
    return PromptExecutor(ai_model, prompt_service)


def test_from_prompt_executor(prompt_executor: PromptExecutor):
    # Given: A prompt executor
    assert prompt_executor
    # When: A prompt executor image is created
    pei = PromptExecutorImage.from_prompt_executor(prompt_executor)
    # Then: It is returned
    assert pei


@pytest.mark.asyncio
async def test_execute_image_prompt_async(
    prompt_executor: PromptExecutor, mocker_ai_model: MockerAIModel, mocker_image_generator: MockerImageGenerator
):
    mocker_ai_model.add_calls(
        [
            {
                "system_prompt": "You are an expert Educational Illustrator and AI Image Prompt Engineer",
                "message_str": "Please generate an image prompt of dog",
                "response": {
                    "content": '{"image_prompt": "A friendly golden retriever, sitting attentively, with a wagging tail, in a vibrant green park on a sunny day, digital illustration, high detail, warm colors, soft lighting"}'
                },
            }
        ]
    )
    mocker_image_generator.add(
        prompt="A friendly golden retriever, sitting attentively, with a wagging tail, in a vibrant green park on a sunny day, digital illustration, high detail, warm colors, soft lighting",
        response=b"Mocked image bytes",
    )
    # Given: A prompt executor image
    pei = PromptExecutorImage.from_prompt_executor(prompt_executor)
    # When: An image is generated
    bc = await pei.execute_image_prompt_async("generate_image", object="dog")
    # Then: It is generated
    assert bc
    assert bc.content == b"Mocked image bytes"
    assert bc.metadata.content_type.startswith("image/")
