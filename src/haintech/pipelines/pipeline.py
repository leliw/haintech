import logging
from typing import List

from .base_processor import BaseProcessor, ListOrIterator


class Pipeline[I,O]:
    """Pipeline class that runs a series of processors on data.
    """

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

    async def run(self, data=None):
        """Run pipeline with data. Return async generator of results.

        Args:
            data: Data to process which is sent to the first processor.
        """
        self._log.debug(f"Running pipeline with data: {data}")
        pipeline = self._build()
        return pipeline.process(data)

    async def run_and_return(self, data=None) -> O | List[O]:
        """Run pipeline and return result.
        If the last processor returns a generator, return a list of results.
        """
        ret_list = data is None or isinstance(data, ListOrIterator)
        ret = list([r async for r in await self.run(data)])
        return ret if ret_list or len(ret) > 1 else ret[0]
