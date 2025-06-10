from __future__ import annotations

from abc import ABC
from typing import Any, AsyncGenerator, AsyncIterator, Callable, Iterator, List, Optional, Union

from pydantic import BaseModel

FieldNameOrLambda = Union[str, Callable[[Any], Any]]
ListOrIterator = Union[List, Iterator]


def get_field_name_or_lambda(
    field_name_or_lambda: FieldNameOrLambda, data: BaseModel
) -> Any:
    if isinstance(field_name_or_lambda, str):
        return getattr(data, field_name_or_lambda)

    elif callable(field_name_or_lambda):
        return field_name_or_lambda(data)
    else:
        raise TypeError("Wrong field_name_or_lambda type")


class BaseProcessor[I, O](ABC):
    """Base class for all processors.

    Args:
        I: Input items data type
        O: Output items data type
    """

    def __init__(
        self,
        name: Optional[str] = None,
        input: Optional[FieldNameOrLambda] = None,
        output: Optional[FieldNameOrLambda] = None,
    ):
        self.name = name or self.__class__.__name__
        self.input = input
        self.output = output
        self.source = None

    async def process_item(self, data: I) -> O:
        """The main method that processes data.
        Override this method in subclasses.
        """
        raise NotImplementedError

    def set_source(self, source: BaseProcessor[Any, Any] | AsyncGenerator[I, None]):
        """Set the source of the processor.
        It is called by Pipeline._build() method.
        """
        if isinstance(source, BaseProcessor):
            self.source = source.process
        else:
            self.source = source

    def _get_iterator(self, data: I | Iterator[I]) -> Iterator[I] | AsyncIterator[I]:
        """Returns iterator for processing data.
        It can be overriden to output more or less data than input."""
        return self.source(data) if self.source else self.generate(data)

    async def process(self, data) -> AsyncIterator[O]:
        """Run the processor on the data. If previous processor is set,
        process the data from it.
        """
        iterator = self._get_iterator(data)
        if isinstance(iterator, Iterator):
            for item in iterator:
                ret =await self.wrap_process_item(item)
                if ret is not None:
                    yield ret
        else:
            async for item in iterator:
                ret = await self.wrap_process_item(item)
                if ret is not None:
                    yield ret

    def generate(self, data: I | Iterator[I]) -> Iterator[I]:
        """It is called when current processor if the first in pipeline. It iterates
        over pipeline input data or create iterator if the data is a single item.
        """
        iterator = data if isinstance(data, ListOrIterator) else [data]
        for item in iterator:
            yield item

    def get_value(self, exp: FieldNameOrLambda, data: I | O) -> Any:
        """Get or evaluate the value of the expression from the data."""
        if callable(exp):
            return exp(data)
        return data.model_dump()[exp] if isinstance(data, BaseModel) else data[exp]

    async def wrap_process_item(self, data):
        """
        Run process method.
        If result_key_name is set, store result in input data and return it.
        """
        if self.input:
            input_data = self._get_input_data(data)
        else:
            input_data = data
        ret = await self.process_item(input_data)
        return self._put_output_data(data, ret)

    def _get_input_data(self, data) -> Any:
        """Returns data sent to the processor based on input type."""
        if isinstance(self.input, str):
            return (
                getattr(data, self.input)
                if isinstance(data, BaseModel)
                else data[self.input]
            )
        elif callable(self.input):
            return self.input(data)
        else:
            raise TypeError("Wrong input type")

    def _put_output_data(self, data, ret) -> Any:
        """Returns data sent to the processor based on ouptut type."""
        if self.output:
            if isinstance(self.output, str):
                if isinstance(data, BaseModel):
                    setattr(data, self.output, ret)
                else:
                    data[self.output] = ret
                return data
            elif callable(self.output):
                return self.output(data, ret) or data
            else:
                raise TypeError("Wrong input type")
        else:
            return ret
