from haintech.ai.model import AIChatSession

from ..base import BaseAIChat
from .open_ai_model import OpenAIModel


class OpenAIChat(BaseAIChat):
    """OpenAI chat session"""

    def __init__(
        self,
        ai_model: OpenAIModel = None,
        context: str = None,
        session: AIChatSession = None,
    ):
        """Initialize OpenAiChat with OpenAI object and model name

        Args:
            openai: OpenAI object
            context: system instructions
            session: chat session
        """
        BaseAIChat.__init__(self, ai_model or OpenAIModel(), context, session)
