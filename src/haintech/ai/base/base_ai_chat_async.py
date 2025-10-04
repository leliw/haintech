import json
import logging
from abc import ABC
from pathlib import Path
from typing import Callable, Iterator, List, Literal, Optional, TypeAlias

from haintech.ai.base.base_ai_model import BaseAIModel
from haintech.ai.model import (
    AIChatResponse,
    AIChatSession,
    AIModelInteraction,
    AIModelInteractionMessage,
    AIModelSession,
    AIPrompt,
)

StrPath: TypeAlias = str | Path


class BaseAIChatAsync(ABC):
    """Base AIChat async version"""

    _log = logging.getLogger(__name__)

    def __init__(
        self,
        ai_model: BaseAIModel,
        system_prompt: Optional[str | AIPrompt] = None,
        session: Optional[AIModelSession] = None,
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
        self.system_prompt = system_prompt
        self.session = session or AIChatSession()
        self.history: List[AIModelInteractionMessage] = []
        self._interaction_logger = None
        self.set_session(self.session)

    def set_session(self, session: AIModelSession):
        self.session = session
        self._interaction_logger = session.add_interaction if session else None

        self.history = []
        if self.session:
            for message in self.session.messages_iterator():
                if message.role != "system":
                    self.add_message(message)

    async def _get_response(
        self,
        message: Optional[AIModelInteractionMessage] = None,
        response_format: Literal["text", "json"] = "text",
    ) -> AIChatResponse:
        """Get response from LLM

        All necessary messages and tools should be added before calling this function.

        Returns:
            response: LLM response
        """
        response = await self.ai_model.get_chat_response_async(
            message=message,
            prompt=self._get_prompt(),
            history=self.iter_messages(),
            interaction_logger=self._interaction_logger,
            response_format=response_format,
        )
        return response

    def add_message(self, message: AIModelInteractionMessage) -> None:
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
        self.add_message(self.session.create_message_from_response(response))

    def iter_messages(self) -> Iterator[AIModelInteractionMessage]:
        """Iterates over all messages

        Returns:
            iterator: iterator of messages
        """
        for message in self.history:
            yield message

    def _get_prompt(self,) -> str | AIPrompt | None:
        """Returns system prompt for LLM (self.system_prompt).

        It can be overriden to return dynamic prompt.
        """
        if self.system_prompt and isinstance(self.system_prompt, AIPrompt):
            return self.system_prompt
        return AIPrompt(context=self.system_prompt) if self.system_prompt else None

    def set_interaction_logger(self, logger: Callable[[AIModelInteraction], None]):
        self._interaction_logger = logger

    async def get_response(self, message: Optional[str] = None) -> AIChatResponse:
        i_msg = AIModelInteractionMessage(role="user", content=message) if message else None
        # Call LLM
        m_resp = await self._get_response(message=i_msg)
        # Add message and response to history
        if i_msg:
            self.add_message(i_msg)
        self.add_response_message(m_resp)
        return m_resp

    async def get_text_response(self, message: Optional[str] = None) -> str:
        ret = (await self.get_response(message)).content
        if not ret:
            raise ValueError("No content in response")
        return ret

    async def get_json_response(self, message: Optional[str] = None) -> str:
        i_msg = AIModelInteractionMessage(role="user", content=message) if message else None
        # Call LLM
        m_resp = await self._get_response(message=i_msg, response_format="json")
        # Add message and response to history
        if i_msg:
            self.add_message(i_msg)
        self.add_response_message(m_resp)
        try:
            ret_json = m_resp.content
            if not ret_json:
                raise ValueError("No content in response")
            return json.loads(ret_json.strip())
        except json.JSONDecodeError as e:
            self._log.warning(e)
            self._log.warning("JSON content: %s", m_resp.content)
            raise e

