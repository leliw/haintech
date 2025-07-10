import logging
from pydantic import BaseModel
from typing import Optional, Callable, Dict

from .base_processor import BaseProcessor


class LogProcessor[I](BaseProcessor[I,I]):
    """Log the data using the logging module."""

    def __init__(
        self,
        message: str = "{item}",
        extra: Optional[Callable[[I], Dict]] = None, 
        level: int = logging.INFO,
        name: Optional[str] = None,
    ):
        """Log the flowing data using the logging module.

        Args:
            name: Name of the logger.
            level: Logging level.
            message: F-string message to log.

        Arguments for f-string message depend on the type of the data.
        If the data is a dictionary or Pydantic class, the arguments 
        are the keys of the dictionary. Otherwise, the argument is 'item'.
        """
        super().__init__(name=name)
        self.message = message
        self.extra = extra
        self.level = level
        self._log = logging.getLogger(name or __name__)

    async def process_item(self, data: I, **kwargs) -> I:
        """Log the data."""
        extra = self.extra(data) if self.extra else None
        if isinstance(data, dict):
            self._log.log(self.level, self.message.format(**data), extra=extra)
        elif isinstance(data, BaseModel):
            self._log.log(self.level, self.message.format(**data.model_dump()), extra=extra)
        else:
            self._log.log(self.level, self.message.format(item=data), extra=extra)
        return data
