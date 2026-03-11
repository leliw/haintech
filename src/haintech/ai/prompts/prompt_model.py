from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Type

from jinja2 import Template
from pydantic import BaseModel

class BaseOutput[T: BaseModel](BaseModel, ABC):
    @abstractmethod
    def convert(self, **kwargs) -> T:
        raise NotImplementedError

@dataclass
class PromptSet:
    name: str
    system: Template
    user: Template
    output_class: Type[BaseOutput] | None = None
    version: str = "1.0.0"

    def render(self, **kwargs) -> tuple[str, str]:
        """Returns (system_prompt, user_prompt) ready for OpenAI/VertexAI messages"""
        system = self.system.render(**kwargs)
        user = self.user.render(**kwargs)
        return system.strip(), user.strip()


