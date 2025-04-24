import asyncio
import logging
from typing import override

import pytest

from haintech.pipelines import ConcurrentProcessor


@pytest.mark.asyncio
async def test_set_source_and_run(log):
    # Given: Processor where sleep time is in reverse to input data
    class TestProcessor[I, O](ConcurrentProcessor[I, O]):
        _log = logging.getLogger(__name__)

        @override
        async def process_item(self, data: int) -> int:
            self._log.info(f"Processing {data}")
            await asyncio.sleep((5 - data)/10)
            self._log.info(f"Processed {data}")
            return data
    # When: All data can be processed concurrently
    p = TestProcessor(max_concurrent=5)
    # And: Run the processor
    ret = p.process([0, 1, 2, 3, 4])
    # Then: Reversed data is returned because of processing time
    assert list([r async for r in ret]) == [4, 3, 2, 1, 0]
