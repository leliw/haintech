import logging
from typing import List

from haintech.ai.ai_task_executor import AITaskExecutor
from haintech.ai.base.base_ai_model import BaseAIModel
from haintech.ai.base.base_rag_searcher import BaseRAGSearcher
from haintech.ai.model import AIContext, AIModelInteractionMessage, AIPrompt, RAGQuery


class BaseAgentSearcher(BaseRAGSearcher):
    """Agent-based RAG searcher. It uses AI model to generate search query."""

    _log = logging.getLogger(__name__)

    def __init__(
        self,
        ai_model: BaseAIModel,
        system_instructions: str = "Task: Based on the system description, conversation, and user question, generate one short search query for document retrieval.",
        prompt: str = "Input:\nSystem: {system_prompt}\nHistory: {conversation_history}\nQuestion: {user_question}\n\nOutput:\nOnly one query between 20 and 100 tokens, no comments or markdown.",
    ):
        self.ai_task_executor = AITaskExecutor(
            ai_model=ai_model, system_instructions=system_instructions, prompt=prompt, response_format="text"
        )

    def agent_search_sync(
        self,
        system_prompt: str | AIPrompt | None,
        history: List[AIModelInteractionMessage],
        message: AIModelInteractionMessage,
    ) -> AIContext | None:
        if not system_prompt and not history:
            return super().agent_search_sync(system_prompt, history, message)
        elif message and message.content:
            conversation_history = [f"{m.role}: {m.content}" for m in history]
            ret = self.ai_task_executor.execute(
                system_prompt=system_prompt,
                conversation_history="\n".join(conversation_history),
                user_question=message.content,
            )
            return AIContext(documents=list(self.search_sync(RAGQuery(text=ret))))  # type: ignore
        else:
            return None

    async def agent_search_async(
        self,
        system_prompt: str | AIPrompt | None,
        history: List[AIModelInteractionMessage],
        message: AIModelInteractionMessage,
    ) -> AIContext | None:
        if not system_prompt and not history:
            return await super().agent_search_async(system_prompt, history, message)
        elif message and message.content:
            conversation_history = [f"{m.role}: {m.content}" for m in history]
            ret = await self.ai_task_executor.execute_async(
                system_prompt=system_prompt,
                conversation_history="\n".join(conversation_history),
                user_question=message.content,
            )
            return AIContext(documents=[r async for r in self.search_async(RAGQuery(text=ret))])  # type: ignore
        else:
            return None
