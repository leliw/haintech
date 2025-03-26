from abc import ABC, abstractmethod
from typing import Iterable

from ..model import RAGItem, RAGQuery


class BaseRAGSearcher(ABC):
    @abstractmethod
    def search_sync(self, query: RAGQuery) -> Iterable[RAGItem]:
        pass

    async def search_async(self, query: RAGQuery) -> Iterable[RAGItem]:
        return self.search_sync(query)
