from pathlib import Path
from typing import Callable, Union, override

from pydantic import BaseModel

from .base_processor import BaseProcessor, FieldNameOrLambda


class JsonlWriter[T: BaseModel](BaseProcessor[T, T]):
    """Adds data to a JSONL file."""

    def __init__(
        self,
        path: Path,
        key_file_name: Union[str, Callable[[T], str]] = None,
        name: str = None,
        input: FieldNameOrLambda = None,
        output: FieldNameOrLambda = None,
    ):
        """Adds data to a JSONL file.

        Args:
            path: Path to the file or parent direcotry.
            key_file_name: Name of data attribute where file name is stored.
        """
        super().__init__(name=name, input=input, output=output)
        self.path = path
        # os.makedirs(self.path, exist_ok=True)
        self.file_name = key_file_name

    @override
    async def process_item(self, data: T) -> T:
        if self.file_name:
            file_name = self.get_value(self.file_name, data)
            if not isinstance(file_name, str):
                file_name = str(file_name)
            file_path = self.path / f"{file_name}.jsonl"
        else:
            file_path = (
                self.path if self.path.suffix == ".jsonl" else f"{self.path}.jsonl"
            )
        jstr = data.model_dump_json()
        with file_path.open("a") as f:
            f.write(jstr)
            f.write("\n")
        return data
