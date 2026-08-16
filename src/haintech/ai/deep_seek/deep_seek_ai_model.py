import logging
import os
from typing import Dict, Optional

from openai import AsyncOpenAI, OpenAI

from haintech.ai.open_ai import OpenAIModel, OpenAIParameters


class DeepSeekAIModel(OpenAIModel):
    _log = logging.getLogger(__name__)

    @classmethod
    def get_vendor_name(cls) -> str:
        return "deepseek"

    def __init__(
        self,
        model_name: str = "deepseek-chat",
        parameters: Optional[OpenAIParameters | Dict[str, str | int | float]] = None,
    ):
        self.openai = OpenAI(
            api_key=os.getenv("DEEP_SEEK_API_KEY"),
            base_url="https://api.deepseek.com",
        )
        self.async_openai = AsyncOpenAI(
            api_key=os.getenv("DEEP_SEEK_API_KEY"),
            base_url="https://api.deepseek.com",
        )
        super().__init__(model_name, parameters)
