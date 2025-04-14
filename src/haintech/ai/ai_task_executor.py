from typing import Literal

from .base.base_ai_chat import BaseAIChat
from .base.base_ai_model import BaseAIModel
from .model import AIPrompt, AITask


class AITaskExecutor:
    """AI task executor.

    Executes one task using AI model.
    """

    def __init__(
        self,
        ai_model: BaseAIModel,
        system_instructions: AIPrompt,
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

    def execute(self, **kwargs) -> str:
        """Execute task.

        Args:
            kwargs: Prompt arguments.
        Returns:
            str: AI Model response.
        """
        prompt = self.prompt.format(**kwargs)
        ai_chat = BaseAIChat(ai_model=self.ai_model, context=self.system_instructions)
        if self.response_format == "json":
            return ai_chat.get_json_response(prompt)
        else:
            return ai_chat.get_text_response(prompt).strip()

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
