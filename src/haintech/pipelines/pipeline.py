import logging
from typing import Iterator, List, Self

from .base_processor import BaseProcessor, ListOrIterator
from .checkpoint_processor import CheckpointProcessor


class Pipeline[I, O]:
    """Pipeline class that runs a series of processors on data."""

    def __init__(self, processors: list[BaseProcessor] = None):
        self.processors = processors or []
        self._log = logging.getLogger(__name__)

    def add_processor(self, processor: BaseProcessor):
        """
        Add processor to pipeline.
        """
        self.processors.append(processor)

    def _build(self) -> BaseProcessor:
        """
        Build pipeline from processors.
        """
        pipeline = None
        for i, processor in enumerate(self.processors):
            if i != 0:
                processor.set_source(pipeline)
            pipeline = processor
        return pipeline

    async def run(self, data: I | Iterator[I] = None):
        """Run pipeline with data. Return async generator of results.

        Args:
            data: Data to process which is sent to the first processor.
        """
        self._log.debug(f"Running pipeline with data: {data}")
        pipeline = self._build()
        return pipeline.process(data)

    async def run_and_return(self, data: I | Iterator[I] = None) -> O | List[O]:
        """Run pipeline and return result.
        If the last processor returns a generator, return a list of results.
        """
        ret_list = data is None or isinstance(data, ListOrIterator)
        ret = list([r async for r in await self.run(data)])
        if ret is not None:
            return ret if ret_list or len(ret) > 1 else ret[0]
        else:
            return None

    def get_step(self, no: int) -> Self:
        """Return part (step) of pipeline.

        Pipeline is divided into steps by CheckpointProcessor.
        This method returns a new pipeline with processors from the step.

        Args:
            no: Number of step.
        Returns:
            Pipeline: New pipeline with processors from the step.
        """
        steps = []
        step = []
        for pr in self.processors:
            step.append(pr)
            if isinstance(pr, CheckpointProcessor):
                steps.append(step)
                gen = pr.create_generator()
                if gen:
                    step = [gen]
                else:
                    step = []
        if len(step) > 0:
            steps.append(step)
        return Pipeline(steps[no])
