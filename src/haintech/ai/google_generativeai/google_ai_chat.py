from haintech.ai.model import AIChatSession

from ..base import BaseAIChat
from .google_ai_model import GoogleAIModel


class GoogleAIChat(BaseAIChat):
    """OpenAI chat session"""

    def __init__(
        self,
        ai_model: GoogleAIModel = None,
        context: str = None,
        session: AIChatSession = None,
    ):
        """Initialize OpenAiChat with OpenAI object and model name

        Args:
            ai_model: AI model object
            context: system instructions
            session: chat session
        """
        BaseAIChat.__init__(self, ai_model or GoogleAIModel(), context, session)
