import logging
from collections.abc import Callable
from typing import Literal

from haintech.ai.ai_model_factory import AIModelFactory
from haintech.ai.base.base_agent_searcher import BaseAgentSearcher
from haintech.ai.base.base_ai_agent_async import BaseAIAgentAsync
from haintech.ai.base.base_ai_model import BaseAIModel
from haintech.ai.interfaces.async_session_blob_manager import AsyncSessionBlobManager
from haintech.ai.model import AIModelSession

_log = logging.getLogger(__name__)


class AIAgentFactory:
    def __init__(
        self,
        ai_model_factory: AIModelFactory,
        searcher: BaseAgentSearcher | None = None,
        session_blob_manager: AsyncSessionBlobManager | None = None,
        functions: list[Callable] | None = None,
    ):
        self.ai_model_factory = ai_model_factory
        self.searcher = searcher
        self.session_blob_manager = session_blob_manager
        self.functions: dict[str, Callable] = {}
        if functions:
            for f in functions:
                name = getattr(f, "__name__", type(f).__name__)
                self.functions[name] = f

    @classmethod
    def create(
        cls,
        ai_model_classes: list[type[BaseAIModel]],
        searcher: BaseAgentSearcher | None = None,
        session_blob_manager: AsyncSessionBlobManager | None = None,
        functions: list[Callable] | None = None,
    ) -> "AIAgentFactory":
        ai_model_factory = AIModelFactory(ai_model_classes)
        return cls(ai_model_factory, searcher, session_blob_manager, functions)

    def get_ai_model_names(self, task: Literal["chat", "image", "embedding"] = "chat") -> list[str]:
        return self.ai_model_factory.get_ai_model_names(task)

    async def get_ai_model_names_async(self, task: Literal["chat", "image", "embedding"] = "chat") -> list[str]:
        return await self.ai_model_factory.get_ai_model_names_async(task)

    def create_ai_model(
        self, vendor_model_name: str, parameters: dict[str, str | int | float] | None = None
    ) -> BaseAIModel:
        return self.ai_model_factory.create_ai_model(vendor_model_name, parameters)

    def create_ai_agent_async(
        self,
        vendor_model_name: str,
        system_prompt: str | None = None,
        session: AIModelSession | None = None,
        searcher: BaseAgentSearcher | None = None,
        functions: list[Callable] | list[str] | None = None,
        session_blob_manager: AsyncSessionBlobManager | None = None,
        ai_model_parameters: dict[str, str | int | float] | None = None,
    ) -> BaseAIAgentAsync:
        if functions and isinstance(functions[0], Callable):
            fns: list[Callable] | None = functions  # pyright: ignore[reportAssignmentType]
        else:
            fns = self.get_functions(functions)  # pyright: ignore[reportArgumentType]
        return BaseAIAgentAsync(
            ai_model=self.create_ai_model(vendor_model_name, ai_model_parameters),
            system_prompt=system_prompt,
            session=session,
            searcher=searcher or self.searcher,
            functions=fns,
            session_blob_manager=session_blob_manager or self.session_blob_manager,
        )

    def get_function_names(self) -> list[str]:
        return list(self.functions.keys())

    def get_functions(self, names: list[str] | None = None) -> list[Callable]:
        if names is None:
            return list(self.functions.values())

        selected_functions = []
        for name in set(names):
            if name not in self.functions:
                raise ValueError(
                    f"Function '{name}' is not registered in the factory. "
                    f"Available functions: {self.get_function_names()}"
                )
            selected_functions.append(self.functions[name])

        return selected_functions
