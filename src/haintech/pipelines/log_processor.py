import logging
from pydantic import BaseModel

from .base_processor import BaseProcessor


class LogProcessor[I](BaseProcessor[I,I]):
    """Log the data using the logging module."""

    def __init__(
        self,
        name: str = None,
        level: int = logging.INFO,
        message: str = "{item}",
        **kwargs,
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
        super().__init__(name=name, **kwargs)
        self.level = level
        self.message = message
        self._log = logging.getLogger(name or __name__)

    async def process_item(self, data: dict, **kwargs) -> dict:
        """Log the data."""
        if isinstance(data, dict):
            self._log.log(self.level, self.message.format(**data))
        elif isinstance(data, BaseModel):
            self._log.log(self.level, self.message.format(**data.model_dump()))
        else:
            self._log.log(self.level, self.message.format(item=data))
        return data
