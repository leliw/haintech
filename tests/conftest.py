import logging

import pytest

from haintech.ai.base.base_ai_model import BaseAIModel
from haintech.ai.google_generativeai.google_ai_model import GoogleAIModel
from haintech.ai.open_ai.open_ai_model import OpenAIModel


@pytest.fixture
def log():
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger("").setLevel(logging.INFO)
    logging.getLogger("haintech.pipeline").setLevel(logging.DEBUG)
    logging.getLogger("haintech.base_processor").setLevel(logging.DEBUG)
    logging.getLogger("markdown_pages_pre_processor").setLevel(logging.DEBUG)


@pytest.fixture(
    params=[
        OpenAIModel(model_name="gpt-4.1-nano", parameters={"temperature": 0}),
        GoogleAIModel(model_name="gemini-2.5-flash-lite", parameters={"temperature": 0}),
    ],
    ids=["OpenAI-nano", "GoogleAI-flash-lite"],
    scope="session",
)
def ai_model_lite(request: pytest.FixtureRequest) -> BaseAIModel:
    return request.param
