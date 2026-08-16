import logging
from typing import Literal

from haintech.ai.base.base_ai_model import BaseAIModel

_log = logging.getLogger(__name__)


class AIModelFactory:
    def __init__(self, ai_model_classes: dict[str, type[BaseAIModel]] | None = None):
        self.ai_model_classes = ai_model_classes or {}
        self.session_blob_manager = None

    def add_ai_model_class(self, vendor_name: str, clazz: type[BaseAIModel]) -> None:
        self.ai_model_classes[vendor_name] = clazz

    def get_ai_model_names(self, task: Literal["chat", "image", "embedding"] = "chat") -> list[str]:
        """Returns available models formatted as vendor/model_name filtered by task."""
        ret = []
        for provider_name, clazz in self.ai_model_classes.items():
            # Pass task down to model classes
            for model_name in clazz.get_model_names(task=task):
                ret.append(f"{provider_name}/{model_name}")
        return ret

    async def get_ai_model_names_async(self, task: Literal["chat", "image", "embedding"] = "chat") -> list[str]:
        """Returns available models formatted as vendor/model_name filtered by task."""
        ret = []
        for provider_name, clazz in self.ai_model_classes.items():
            # Pass task down to model classes
            for model_name in await clazz.get_model_names_async(task=task):
                ret.append(f"{provider_name}/{model_name}")
        return ret
    
    def create_ai_model(
        self, vendor_model_name: str, parameters: dict[str, str | int | float] | None = None
    ) -> BaseAIModel:
        """
        Creates a new AI model instance based on the specified vendor and model name.

        Args:
            vendor_model_name: A string in the format "vendor/model_name" or "vendor".
            parameters: Optional dictionary of parameters to pass to the AI model constructor.

        Returns:
            An instance of BaseAIModel.
        Raises:
            ValueError: If the vendor is unknown.
        """
        if "/" not in vendor_model_name:
            raise ValueError(f"Invalid vendor_model_name format: {vendor_model_name}. Expected 'vendor/model_name'.")

        vendor, model_name = vendor_model_name.split("/", 1)
        _log.debug("Creating AI model: %s -> %s", vendor, model_name)
        parameters = parameters or {"temperature": 0}
        clazz = self.ai_model_classes.get(vendor)
        if not clazz:
            raise ValueError(f"Unknown vendor: {vendor}. Please add the AI model class using add_ai_model_class().")
        ai_model = clazz(model_name, parameters)
        _log.debug("AI model created: %s", vendor_model_name)
        return ai_model
