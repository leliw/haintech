from typing import Callable

from .base_processor import BaseProcessor


class LambdaProcessor[I, O](BaseProcessor):
    """Processor that processes data using a lambda function."""

    def __init__(self, expression: Callable[[I], O], **kwargs):
        """Initialize the processor.

        Args:
            expression: The lambda function that processes data.
        """
        super().__init__(**kwargs)
        self.expression = expression

    async def process_item(self, data: I, *kwargs) -> O:
        return self.expression(data, *kwargs)
