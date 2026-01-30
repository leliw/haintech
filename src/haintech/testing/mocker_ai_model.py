import json
import logging
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional

import pytest
from haintech.ai import AIChatResponse, AIContext, AIModelInteraction, AIModelInteractionMessage, AIPrompt
from pydantic import BaseModel, field_serializer
from pytest_mock.plugin import MockerFixture

try:
    from haintech.ai.google_generativeai import GoogleAIModel  # noqa: F401

    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False

try:
    from haintech.ai.open_ai import OpenAIModel  # noqa: F401

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from haintech.ai.anthropic import AnthropicAIModel  # noqa: F401

    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    from haintech.ai.deep_seek import DeepSeekAIModel  # noqa: F401

    DEEPSEEK_AVAILABLE = True
except ImportError:
    DEEPSEEK_AVAILABLE = False

_log = logging.getLogger(__name__)


class AICall(BaseModel):
    message_str: Optional[str] = None
    message_containing: Optional[str] = None
    blob_contents: Optional[List[bytes]] = None
    tool_result: Optional[str] = None
    response: AIChatResponse

    @field_serializer("blob_contents")
    def serialize_blobs(self, value: Optional[List[bytes]]) -> Optional[List[str]]:
        if value is None:
            return None
        return [repr(b) for b in value]


class MockerAIModel:
    _mocked_methods: List[str] = []
    
    if GOOGLE_AVAILABLE:
        _mocked_methods.append("haintech.ai.google_generativeai.GoogleAIModel.get_chat_response")
        _mocked_methods.append("haintech.ai.google_generativeai.GoogleAIModel.get_chat_response_async")
    if OPENAI_AVAILABLE:
        _mocked_methods.append("haintech.ai.open_ai.OpenAIModel.get_chat_response")
        _mocked_methods.append("haintech.ai.open_ai.OpenAIModel.get_chat_response_async")
    if ANTHROPIC_AVAILABLE:
        _mocked_methods.append("haintech.ai.anthropic.AnthropicAIModel.get_chat_response")
        _mocked_methods.append("haintech.ai.anthropic.AnthropicAIModel.get_chat_response_async")
    if DEEPSEEK_AVAILABLE:
        _mocked_methods.append("haintech.ai.deep_seek.DeepSeekAIModel.get_chat_response")
        _mocked_methods.append("haintech.ai.deep_seek.DeepSeekAIModel.get_chat_response_async")

    def __init__(self, mocker: MockerFixture):
        self.mocker = mocker
        self.org_ai_model = None
        self.responses: List[AICall] = []
        self.setup()

    def setup(self) -> None:
        for method in self._mocked_methods:
            self.mocker.patch(method, side_effect=self.get_chat_response)

    @contextmanager
    def record(self):
        """Records all AI responses and prints them to console."""
        from haintech.ai.google_generativeai import GoogleAIModel, GoogleAIParameters

        self.org_ai_model = GoogleAIModel(parameters=GoogleAIParameters(temperature=0))
        yield
        print(
            json.dumps(
                [call.model_dump(mode="python", exclude_none=True, exclude_unset=True) for call in self.responses],
                indent=2,
            )
        )
        assert False

    def add(
        self,
        response: str | AIChatResponse,
        message: Optional[str] = None,
        message_containing: Optional[str] = None,
        tool_result: Optional[str] = None,
        blob_contents: Optional[List[bytes]] = None,
    ) -> None:
        """Adds a mocked AI response.

        Args:
            response (str | AIChatResponse): The response to return.
            message (Optional[str], optional): The exact message string to match. Defaults to None.
            message_containing (Optional[str], optional): A substring that should be contained in the message. Defaults to None.
            tool_result (Optional[str], optional): The tool result to match. Defaults to None.
            blob_contents (Optional[List[bytes]], optional): The blob contents to match. Defaults to None.
        """
        if isinstance(response, str):
            response_obj = AIChatResponse(content=response)
        else:
            response_obj = response
        self.responses.append(
            AICall(
                response=response_obj,
                message_str=message,
                message_containing=message_containing,
                blob_contents=blob_contents,
                tool_result=tool_result,
            )
        )

    def get_chat_response(
        self,
        system_prompt: Optional[str | AIPrompt] = None,
        history: Optional[Iterable[AIModelInteractionMessage]] = None,
        context: Optional[AIContext] = None,
        message: Optional[AIModelInteractionMessage] = None,
        functions: Optional[Dict[Callable, Any]] = None,
        interaction_logger: Optional[Callable[[AIModelInteraction], None]] = None,
        response_format: Literal["text", "json"] = "text",
    ) -> AIChatResponse:
        history = list(history or [])
        if history and history[-1].role == "tool":
            tool_result = history[-1].content
        else:
            tool_result = None

        #####################

        if self.org_ai_model:
            self.mocker.stopall()
            response = self.org_ai_model.get_chat_response(
                system_prompt=system_prompt,
                history=history,
                context=context,
                message=message,
                functions=functions,
                interaction_logger=interaction_logger,
                response_format=response_format,
            )

            call = AICall(
                message_str=message.content if message else None,
                blob_contents=[blob.content for blob in message.blobs] if message and message.blobs else None,
                tool_result=tool_result,
                response=response,
            )
            _log.info("AI response: %s", response)
            self.responses.append(call)
            self.setup()
            return response
        elif not self.responses:
            raise RuntimeError("No mocked AI responses available.")
        else:
            call = self.responses.pop(0)
            if call.message_str:
                assert message and message.content
                assert message.content == call.message_str, (
                    f"Expected message content '{call.message_str}', got '{message.content}'"
                )
            if call.message_containing:
                assert message and message.content
                assert call.message_containing in message.content
            if call.blob_contents:
                assert message and message.blobs
                assert len(call.blob_contents) == len(message.blobs), (
                    f"Expected {len(call.blob_contents)} blobs, got {len(message.blobs)}"
                )
                assert all([blob.content in call.blob_contents for blob in message.blobs]), (
                    f"Expected blobs {call.blob_contents}, got {[blob.content for blob in message.blobs]}"
                )
            if call.tool_result:
                assert tool_result
                assert tool_result == call.tool_result, (
                    f"Expected tool result '{call.tool_result}', got '{tool_result}'"
                )
            response = call.response

        ####################
        if interaction_logger:
            interaction_logger(
                AIModelInteraction(
                    model="MockerAIModel",
                    message=message,
                    context=context,
                    history=history,
                    response=response,
                )
            )
        return response


@pytest.fixture
def mocker_ai_model(mocker: MockerFixture) -> MockerAIModel:
    return MockerAIModel(mocker)
