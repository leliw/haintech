import json
import logging
from typing import Dict, List, Literal

from .base.base_ai_model import BaseAIModel
from .model import AIModelInteractionMessage, AIPrompt, AITask


class AITaskExecutor:
    """AI task executor.

    Executes one task using AI model.
    """

    _log = logging.getLogger(__name__)

    def __init__(
        self,
        ai_model: BaseAIModel,
        system_instructions: AIPrompt | str,
        prompt: str,
        response_format: Literal["text", "json"] = "text",
    ):
        """Initialize AI task executor.

        Args:
            ai_model: AI model.
            system_instructions: System instructions.
            prompt: Prompt (f-string).
        """
        self.ai_model = ai_model
        self.system_instructions = system_instructions
        self.prompt = prompt
        self.response_format = response_format

    def execute(self, **kwargs) -> str | Dict | List:
        """Execute task.

        Args:
            kwargs: Prompt arguments.
        Returns:
            str: AI Model response.
        """
        m_resp = self.ai_model.get_chat_response(
            system_prompt=self.system_instructions,
            message=self._prepare_message(**kwargs),
            response_format=self.response_format, # type: ignore
        )
        return self._prepare_response(m_resp)

    async def execute_async(self, **kwargs) -> str | Dict | List:
        """Execute task.

        Args:
            kwargs: Prompt arguments.
        Returns:
            str: AI Model response.
        """
        m_resp = await self.ai_model.get_chat_response_async(
            system_prompt=self.system_instructions,
            message=self._prepare_message(**kwargs),
            response_format=self.response_format, # type: ignore
        )
        return self._prepare_response(m_resp)

    def _prepare_message(self, **kwargs) -> AIModelInteractionMessage:
        message = self.prompt.format(**kwargs)
        if message:
            return AIModelInteractionMessage(role="user", content=message)
        else:
            raise ValueError("Prompt is empty")
    
    def _prepare_response(self, m_resp) -> str | Dict | List:
        if self.response_format == "json":
            try:
                return json.loads(m_resp.content)
            except json.JSONDecodeError as e:
                self._log.warning(e)
                self._log.warning("JSON content: %s", m_resp.content)
                raise e
        else:
            return m_resp.content.strip()


    @classmethod
    def create_from_definition(
        cls, ai_model: BaseAIModel, task_definition: AITask
    ) -> "AITaskExecutor":
        """Create AI task executor from task definition.

        Args:
            ai_model: AI model.
            task_definition: Task definition.
        Returns:
            AITaskExecutor: AI task executor.
        """
        return cls(
            ai_model=ai_model,
            system_instructions=task_definition.system_instructions,
            prompt=task_definition.prompt,
        )
