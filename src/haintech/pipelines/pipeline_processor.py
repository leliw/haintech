from typing import override
from .base_processor import BaseProcessor, FieldNameOrLambda
from .pipeline import Pipeline


class PipelineProcessor[I, O](BaseProcessor[I, O]):
    """The processor that uses other pipeline to process items."""

    def __init__(
        self,
        processors: list[BaseProcessor] = None,
        pipeline: Pipeline[I, O] = None,
        name: str = None,
        input: FieldNameOrLambda = None,
        output: FieldNameOrLambda = None,
    ):
        super().__init__(name=name, input=input, output=output)
        self.pipeline = pipeline if pipeline else Pipeline(processors)

    @override
    async def process_item(self, data: I) -> O:
        return await self.pipeline.run_and_return(data)
