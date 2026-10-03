import logging
from typing import ClassVar

from openai.types import Model
from pydantic import BaseModel, Field

from haintech.ai.open_ai import ResponsesAIModel

_log = logging.getLogger(__name__)


class AzureFoundryConfig(BaseModel):
    api_key: str = Field(min_length=1)
    base_url: str = Field(min_length=1)
    model_names: list[str] = Field(min_length=1)


class AzureFoundryAIModel(ResponsesAIModel):
    config: ClassVar[AzureFoundryConfig | None] = None

    @classmethod
    def get_vendor_name(cls) -> str:
        return "azure"

    @classmethod
    def setup(cls, config: AzureFoundryConfig):
        cls.config = config
        super().setup(api_key=config.api_key, base_url=config.base_url)
        cls._models_list = [Model(id=n, object="model", created=0, owned_by="") for n in config.model_names]
