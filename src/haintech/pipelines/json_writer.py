import json
import logging
import os
from pathlib import Path
from typing import Callable, Union
from pydantic import BaseModel

from .base_processor import BaseProcessor


class JsonWriter[T: dict | BaseModel](BaseProcessor[T, T]):
    """Write data to a JSON file."""

    def __init__(self, dir_path: Path, file_name: Union[str, Callable[[T], str]], **kwargs):
        super().__init__(**kwargs)
        self.dir_path = dir_path
        os.makedirs(self.dir_path, exist_ok=True)
        self.file_name = file_name
        self._log = logging.getLogger(__name__)

    async def process_item(self, data: T) -> T:
        file_name = self.get_value(self.file_name, data)
        if not isinstance(file_name, str):
            file_name = str(file_name)
        file_path = self.dir_path / f"{file_name}.json"
        if isinstance(data, BaseModel):
            jstr = data.model_dump_json(indent=2)
        else:
            jstr = json.dumps(data, ensure_ascii=False, indent=2)
        self._log.debug(f"Processor {self.name} result: {jstr}")
        with file_path.open("w") as f:
            f.write(jstr)
        return data
