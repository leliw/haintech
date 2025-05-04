from typing import Callable, Iterable, Iterator, override

from .base_flat_map_processor import BaseFlatMapProcessor
from .base_processor import FieldNameOrLambda


class FlatMapProcessor[I, O](BaseFlatMapProcessor[I, O]):
    """Processor that iterates over a iterable in the data and yields the items"""

    def __init__(
        self,
        extract_items: Callable[[I], Iterable[O]] = None,
        name: str = None,
        input: FieldNameOrLambda = None,
        output: FieldNameOrLambda = None,
    ):
        """Processor that iterates over a iterable in the data and yields the items.

        Args:
            extract_items: A lambda function that takes a pipeline item and returns an iterable.
        """
        super().__init__(name=name, input=input, output=output)
        self.extract_items = extract_items

    @override
    def process_flat_map(self, data: I) -> Iterator[O]:
        """Extra iteration based on the expression"""
        if not self.extract_items:
            iterator = data
        else:
            iterator = self.extract_items(data)
        for item in iterator:
            yield item
