from typing import Callable, List, override

from .base_processor import BaseProcessor, FieldNameOrLambda
from .pipeline import Pipeline
import logging

class PipelineProcessor[I, O](BaseProcessor[I, O]):
    """The processor that uses other pipeline to process items."""
    _log = logging.getLogger(__name__)

    def __init__(
        self,
        processors: List[BaseProcessor] = None,
        pipeline: Pipeline[I, O] | Callable[[I], Pipeline[I, O]] = None,
        name: str = None,
        input: FieldNameOrLambda = None,
        output: FieldNameOrLambda = None,
    ):
        super().__init__(name=name, input=input, output=output)
        self.pipeline = pipeline if pipeline else Pipeline(processors)

    @override
    async def process_item(self, data: I) -> O:
        if isinstance(self.pipeline, Callable):
            pipeline = self.pipeline(data)
        else:
            pipeline = self.pipeline
        try:
            return await pipeline.run_and_return(data)
        except IndexError as e:
            self._log.warning("%s doesn't return anything.", self.name)
            self._log.info(e)
            return None