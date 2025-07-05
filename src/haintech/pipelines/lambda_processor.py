from typing import Callable

from .base_processor import BaseProcessor, FieldNameOrLambda


class LambdaProcessor[I, O](BaseProcessor):
    """Processor that processes data using a lambda function."""

    def __init__(
        self,
        expression: Callable[[I], O | None],
        name: str = None,
        input: FieldNameOrLambda = None,
        output: FieldNameOrLambda = None,
    ):
        """Initialize the processor.

        Args:
            expression: The lambda function that processes data.
        """
        super().__init__(name=name, input=input, output=output)
        self.expression = expression

    async def process_item(self, data: I) -> O:
        ret = self.expression(data)
        return ret if ret is not None else data
