import logging
from abc import ABC
from pathlib import Path
from typing import Callable, Iterator, List, TypeAlias

from haintech.ai.base.base_ai_model import BaseAIModel
from haintech.ai.model import (
    AIChatResponse,
    AIModelInteraction,
    AIModelInteractionMessage,
    AIModelSession,
    AIPrompt,
)

StrPath: TypeAlias = str | Path


class BaseAIChat(ABC):
    """Base AIChat"""

    _log = logging.getLogger(__name__)

    def __init__(
        self,
        ai_model: BaseAIModel = None,
        context: str | AIPrompt = None,
        session: AIModelSession = None,
    ) -> None:
        """Base AIChat

        If session is passed, all messages (expect system)
        and last response are copied to current chat object.

        Args:
            ai_model: AI model
            context: Context prompt for model
            session: current session object,
        """
        self.ai_model = ai_model
        self.context = context
        self.session = session
        self.history: List[AIModelInteractionMessage] = []
        self._interaction_logger = None
        if session:
            self.set_session(session)

    def set_session(self, session: AIModelSession):
        self.session = session
        self._interaction_logger = session.add_interaction if session else None

        self.history = []
        if self.session:
            for message in self.session.messages_iterator():
                if message.role != "system":
                    self.add_message(message)
            if self.session.get_last_response():
                self.add_response_message(self.session.get_last_response())

    def _get_response(
        self, message: AIModelInteractionMessage = None
    ) -> AIChatResponse:
        """Get response from LLM

        All necessary messages and tools should be added before calling this function.

        Returns:
            response: LLM response
        """
        response = self.ai_model.get_chat_response(
            message=message,
            context=self._get_context(message),
            history=self.iter_messages(),
            interaction_logger=self._interaction_logger,
        )
        return response

    def add_message(self, message: AIModelInteractionMessage = None) -> None:
        """Add message to AI chat session

        Args:
            message: message
        """
        # self._log.debug("Adding message: %9s: %s", message.role, (message.content or message.tool_calls)[:50])
        self.history.append(message)

    def add_system_message(self, content: str) -> None:
        self.add_message(AIModelInteractionMessage(role="system", content=content))

    def add_user_message(self, content: str) -> None:
        self.add_message(AIModelInteractionMessage(role="user", content=content))

    def add_response_message(self, response: AIChatResponse) -> None:
        """Add response message to AI chat session

        Args:
            response: response
        """
        self.add_message(response.toMessage())

    def iter_messages(self) -> Iterator[AIModelInteractionMessage]:
        """Iterates over all messages

        Returns:
            iterator: iterator of messages
        """
        for message in self.history:
            yield message

    def _get_context(self, message: AIModelInteractionMessage = None) -> str | AIPrompt:
        """Returns context for LLM (self.context).

        It can be overriden to return dynamic context.

        Args:
            message: message
        Returns:
            context: context
        """
        if self.context and isinstance(self.context, AIPrompt):
            return self.context
        return AIPrompt(context=self.context) if self.context else None

    def set_interaction_logger(self, logger: Callable[[AIModelInteraction], None]):
        self._interaction_logger = logger

    def get_response(self, message: str = None) -> AIChatResponse:
        i_msg = (
            AIModelInteractionMessage(role="user", content=message) if message else None
        )
        # Call LLM
        m_resp = self._get_response(message=i_msg)
        # Add message and response to history
        if i_msg:
            self.add_message(i_msg)
        self.add_response_message(m_resp)
        return m_resp

    def get_text_response(self, message: str = None) -> str:
        return self.get_response(message).content
