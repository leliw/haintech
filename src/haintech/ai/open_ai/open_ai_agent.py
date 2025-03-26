import logging
from typing import Callable, List

from haintech.ai.model import AIChatSession

from ..base import BaseAIAgent, BaseRAGSearcher
from .open_ai_model import OpenAIModel


class OpenAIAgent(BaseAIAgent):
    """OpenAI agent"""

    _log = logging.getLogger(__name__)

    def __init__(
        self,
        name: str = None,
        description: str = None,
        ai_model: OpenAIModel = None,
        context: str = None,
        session: AIChatSession = None,
        searcher: BaseRAGSearcher = None,
        functions: List[Callable] = None,
    ) -> None:
        super().__init__(
            name=name or self.__class__.__name__,
            description=description,
            ai_model=ai_model or OpenAIModel(),
            context=context,
            session=session,
            searcher=searcher,
            functions=functions,
        )
