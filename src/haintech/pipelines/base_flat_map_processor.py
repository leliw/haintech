from abc import ABC, abstractmethod
from pathlib import Path
from typing import AsyncIterator, Iterator, override
from haintech.pipelines import BaseProcessor, FieldNameOrLambda


class BaseFlatMapProcessor[I: Path, O: Path](BaseProcessor[I, O], ABC):
    """Base class for all processors that iterate over a iterable in the data and yields the items."""
    def __init__(
        self,
        name: str = None,
        input: FieldNameOrLambda = None,
        output: FieldNameOrLambda = None,
    ):
        super().__init__(name=name, input=input, output=output)

    @override
    async def process(self, data: I) -> AsyncIterator[O]:
        iterator = self._get_iterator(data)
        if isinstance(iterator, Iterator):
            for data in iterator:
                if self.input:
                    input_data = self._get_input_data(data)
                else:
                    input_data = data
                for item in self.process_flat_map(input_data):
                    yield item
        else:
            async for data in iterator:
                if self.input:
                    input_data = self._get_input_data(data)
                else:
                    input_data = data                
                for item in self.process_flat_map(input_data):
                    yield item

    @abstractmethod
    def process_flat_map(self, data: I) -> Iterator[O]:
        """Returns iterator based on input data.

        >>>
        def process_flat_map(self, data: I) -> Iterator[O]:
            for item in self.extract_items(data):
                yield self._put_output_data(data, item)
        """
        raise NotImplementedError

    @override
    async def process_item(self, data: O) -> O:
        return data
