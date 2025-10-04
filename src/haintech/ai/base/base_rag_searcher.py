from abc import ABC, abstractmethod
import logging
from typing import Iterable, List, Optional

from ..model import AIContext, AIModelInteractionMessage, AIPrompt, RAGItem, RAGQuery


class BaseRAGSearcher(ABC):
    _log = logging.getLogger(__name__)

    @abstractmethod
    def search_sync(self, query: RAGQuery) -> Iterable[RAGItem]:
        """Search for items in RAG.

        Args:
            query: RAG query
        Returns:
            iterable: iterable of RAG items
        """  
        pass

    async def search_async(self, query: RAGQuery) -> Iterable[RAGItem]:
        """Search for items in RAG.

        Args:
            query: RAG query
        Returns:
            iterable: iterable of RAG items
        """  
        return self.search_sync(query)

    def agent_search_sync(
        self,
        system_prompt: str | AIPrompt | None,
        history: List[AIModelInteractionMessage],
        message: Optional[AIModelInteractionMessage] = None,
    ) -> Optional[AIContext]:
        """Search for items in RAG.

        Args:
            system_prompt: system prompt
            history: history
            message: message
        Returns:
            AIContext: AI context
        """
        if message and message.content:
            msg = message.content
            if len(msg) > 15:
                self._log.debug("Searching for: %s", msg)
                ret = AIContext(documents= list(self.search_sync(RAGQuery(text=msg))))
                self._log.debug("Found: %d items", len(ret.documents))
                return ret
            else:
                self._log.debug("Message too short for search: %s", msg)
        return None 

    async def agent_search_async(
        self,
        system_prompt: str | AIPrompt | None,
        history: List[AIModelInteractionMessage],
        message: Optional[AIModelInteractionMessage] = None,
    ) -> Optional[AIContext]:
        """Search for items in RAG.

        Args:
            system_prompt: system prompt
            history: history
            message: message
        Returns:
            AIContext: AI context
        """       
        if message and message.content:
            msg = message.content
            if len(msg) > 15:
                self._log.debug("Searching for: %s", msg)
                ret = AIContext(documents= list(await self.search_async(RAGQuery(text=msg))))
                self._log.debug("Found: %d items", len(ret.documents))
                return ret
            else:
                self._log.debug("Message too short for search: %s", msg)
        return None 