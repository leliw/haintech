from abc import ABC, abstractmethod
from .base_processor import BaseProcessor


class CheckpointProcessor[T](BaseProcessor[T, T], ABC):
    """Base class for processors that can split pipeline into steps."""

    def __init__(self, name=None, input=None, output=None):
        super().__init__(name, input, output)

    @abstractmethod
    def create_generator(self) -> BaseProcessor[str, T]:
        pass
