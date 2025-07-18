from abc import ABC, abstractmethod
from typing import AsyncIterator, Iterator, Optional, override
from haintech.pipelines import BaseProcessor, FieldNameOrLambda


class BaseFlatMapProcessor[I, O](BaseProcessor[I, O], ABC):
    """Base class for all processors that iterate over a iterable in the data and yields the items."""
    def __init__(
        self,
        name: Optional[str] = None,
        input: Optional[FieldNameOrLambda] = None,
        output: Optional[FieldNameOrLambda] = None,
    ):
        super().__init__(name=name, input=input, output=output)

    @override
    async def process(self, data: I) -> AsyncIterator[O]:
        iterator = self._get_iterator(data)
        if isinstance(iterator, Iterator):
            for data in iterator:
                for item in self.wrap_process_flat_map(data):
                    yield item
        else:
            async for data in iterator:
                for item in self.wrap_process_flat_map(data):
                    yield item

    def wrap_process_flat_map(self, data):
        """
        Run process method.
        If result_key_name is set, store result in input data and return it.
        """
        if self.input:
            input_data = self._get_input_data(data)
        else:
            input_data = data
        for ret in self.process_flat_map(input_data):
            yield self._put_output_data(data, ret)
    
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
