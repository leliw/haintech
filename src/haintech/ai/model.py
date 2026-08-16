import uuid
from abc import ABC, abstractmethod
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any, override
from warnings import deprecated

from ampf.base import Blob, BlobLocation
from pydantic import BaseModel, ConfigDict, Field, model_validator


class RAGQuery(BaseModel):
    """
    RAG query model.
    """

    text: str
    keywords: list[str] | None = None
    limit: int = 5


class RAGItem(BaseModel):
    item_id: str | None = None
    title: str | None = None
    url: str | None = None
    description: str | None = None
    keywords: list[str] = Field(
        default_factory=list,
        description="List of keywords associated with the item.",
    )
    content: str
    metadata: dict[str, str] = Field(
        default_factory=dict,
        description="Additional metadata for the item.",
    )


class AIModelToolCall(BaseModel):
    """Tool call request returned by AIModel"""

    id: str
    function_name: str
    arguments: dict[str, Any]
    thought_signature: str | None = Field(default=None, description="Thought signature required by Google API")

    def __str__(self):
        return f"{self.id}: {self.function_name}({', '.join([f'{k}="{v}"' for k, v in self.arguments.items()])})"


class AIChatResponse(BaseModel):
    """Chat response model."""

    content: str | None = None
    tool_calls: list[AIModelToolCall] | None = None
    vendor_model_name: str | None = None
    input_tokens: int | None = None
    input_tokens_cached: int | None = None
    reasoning_tokens: int | None = None
    output_tokens: int | None = None

    def __str__(self) -> str:
        ret = []
        if self.content:
            ret.append(f"Assistant: {self.content}")
        if self.tool_calls:
            for tc in self.tool_calls:
                ret.append(str(tc))
        return "\n".join(ret)


class AIModelInteractionMessage(BaseModel):
    """One message within AIModelInteraction"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    role: str
    name: str | None = None
    tool_call_id: str | None = None  # Only for role=tool
    content: str | None = None
    blob_locations: list[BlobLocation] = Field(default_factory=list)
    blobs: list[Blob] | None = Field(exclude=True, repr=False, default_factory=list)
    tool_calls: list[AIModelToolCall] | None = None

    @classmethod
    def create_from_response(cls, response: AIChatResponse):
        return cls(
            role="assistant",
            content=response.content,
            tool_calls=response.tool_calls,
        )

    def __str__(self) -> str:
        ret = f"{self.role:10}:" + (f" {self.tool_call_id} => " if self.tool_call_id else "")
        if self.content:
            if len(self.content) < 1024:
                ret += f" {self.content}"
            else:
                ret += f" {self.content[:1000]}..."
        if self.tool_calls:
            for i, tc in enumerate(self.tool_calls):
                ret += f"{' ' if i == 0 else '\n'}{tc}"

        return ret


class AIModelInteractionTool(BaseModel):
    type: str
    function: Any


@deprecated("Use just string")
class AIPrompt(BaseModel):
    """Structured AI prompt model.

    According to: <https://cloud.google.com/vertex-ai/generative-ai/docs/learn/prompts/prompt-design-strategies#components-of-a-prompt>
    """

    persona: str | None = None
    objective: str | None = None
    instructions: str | None = None
    constraints: str | None = None
    context: str | None = None
    documents: list[str | RAGItem] = Field(default_factory=list)
    output_format: str | None = None
    examples: list[str] = Field(default_factory=list)
    recap: str | None = None


class AIContext(BaseModel):
    context: str | None = None
    documents: list[str | RAGItem] = Field(default_factory=list)


class AIModelInteraction[T: AIModelInteractionMessage](BaseModel):
    """One interaction with AIModel"""

    uid: str = Field(default_factory=lambda: uuid.uuid4().hex)
    model: str
    tools: list[AIModelInteractionTool] | None = None
    parallel_tool_calls: bool | None = None
    response_format: dict | None = None
    context: AIContext | None = None
    prompt: str | AIPrompt | None = None
    history: list[T]
    message: T | None = None
    response: AIChatResponse | None = Field(default=None, exclude=True)
    response_message: T | None = None

    model_config = ConfigDict(validate_assignment=True)

    @model_validator(mode="after")
    def populate_response_message(self):
        if not self.response_message and self.response:
            if self.message:
                self.response_message = self.message.create_from_response(self.response)
            elif self.history:
                self.response_message = self.history[-1].create_from_response(self.response)
            else:
                raise ValueError("Neither message nor history provided")
        return self


class AIModelSession[T: AIModelInteractionMessage](ABC):
    """AIModel session model."""

    @abstractmethod
    def add_interaction(self, interaction: AIModelInteraction[T]) -> None:
        """Add interaction to session."""

    @abstractmethod
    def get_last_interaction(self) -> AIModelInteraction[T] | None:
        """Get last response (from last interaction)."""

    @abstractmethod
    def messages_iterator(self) -> Iterator[T]:
        """Iterates  over all messages (from last interaction)."""

    def get_last_response(self) -> T | None:
        """Get last response."""
        last_interaction = self.get_last_interaction()
        if last_interaction:
            return last_interaction.response_message
        return None

    # TODO: Remove this method
    @classmethod
    def create_message_from_response(cls, response: AIChatResponse) -> T:
        return AIModelInteractionMessage.create_from_response(response)  # type: ignore


class AIChatSession[T: AIModelInteractionMessage](BaseModel, AIModelSession[T]):
    """Chat session model."""

    uid: str = Field(default_factory=lambda: uuid.uuid4().hex)
    datetime: str = Field(default_factory=lambda: str(datetime.now(UTC)))
    interactions: list[AIModelInteraction[T]] = Field(default_factory=list)

    @override
    def add_interaction(self, interaction: AIModelInteraction[T]):
        self.interactions.append(interaction)

    @override
    def messages_iterator(self) -> Iterator[T]:
        """Iterates  over all messages (from last interaction)"""
        last_interaction = self.get_last_interaction()
        if last_interaction:
            yield from last_interaction.history
            if last_interaction.message:
                yield last_interaction.message
                last_response = self.get_last_response()
                if last_response:
                    yield last_response

    def get_last_interaction(self) -> AIModelInteraction[T] | None:
        """Get last interaction."""
        if self.interactions:
            return self.interactions[-1]
        return None

    def add_message(self, message: T):
        """Add message to last interaction."""
        last_interaction = self.get_last_interaction()
        if last_interaction:
            last_interaction.history.append(message)

    def __str__(self) -> str:
        ret = ""
        for m in self.messages_iterator():
            ret += str(m) + "\n"
        last_response = self.get_last_response()
        if last_response:
            ret += str(last_response) + "\n"
        return ret


class AIAgentInteraction[T: AIModelInteractionMessage](BaseModel):
    agent_name: str | None = None
    interaction: AIModelInteraction[T]


class AIAgentSession[T: AIModelInteractionMessage](AIModelSession[T]):
    """Agent session model.

    It links to supervisor session.
    All methods operate on supervisor session interactions
    with the agent name differentiator.
    """

    def __init__(self, agent_name: str, interactions: list[AIAgentInteraction]):
        self.agent_name = agent_name
        self.interactions = interactions

    @override
    def add_interaction(self, interaction: AIModelInteraction[T]):
        self.interactions.append(AIAgentInteraction(agent_name=self.agent_name, interaction=interaction))

    @override
    def get_last_interaction(self) -> AIModelInteraction[T] | None:
        """Get last response."""
        if self.interactions:
            for i in reversed(self.interactions):
                if i.agent_name == self.agent_name:
                    return i.interaction
        return None

    @override
    def messages_iterator(self) -> Iterator[T]:
        """Iterates  over all messages (from last interaction)"""
        if self.interactions:
            for i in reversed(self.interactions):
                if i.agent_name == self.agent_name:
                    yield from i.interaction.history
                    if i.interaction.message:
                        yield i.interaction.message
                    if i.interaction.response_message:
                        yield i.interaction.response_message
                    # I've found the last interation for the agent
                    return


class AIMultiagentSession[T: AIModelInteractionMessage](BaseModel, AIModelSession[T]):
    """Supervisor session model. It can create agent sessions."""

    interactions: list[AIAgentInteraction[T]] = Field(default_factory=list)
    agent_name: str | None = None

    def create_agent_session(self, agent_name: str) -> AIModelSession[T]:
        """Create agent session."""
        return AIAgentSession[T](agent_name=agent_name, interactions=self.interactions)

    @override
    def add_interaction(self, interaction: AIModelInteraction[T]):
        self.interactions.append(AIAgentInteraction[T](agent_name=self.agent_name, interaction=interaction))

    def get_last_interaction(self) -> AIModelInteraction[T] | None:
        """Get last interaction."""
        if self.interactions:
            for i in reversed(self.interactions):
                if i.agent_name == self.agent_name:
                    return i.interaction
        return None

    @override
    def get_last_response(self) -> T | None:
        """Get last response."""
        if self.interactions:
            for i in reversed(self.interactions):
                if i.agent_name == self.agent_name:
                    return i.interaction.response_message
        return None

    @override
    def messages_iterator(self) -> Iterator[T]:
        """Iterates  over all messages (from last interaction)"""
        if self.interactions:
            for i in reversed(self.interactions):
                if i.agent_name == self.agent_name:
                    yield from i.interaction.history
                    if i.interaction.message:
                        yield i.interaction.message
                    if i.interaction.response_message:
                        yield i.interaction.response_message
                    # I've found the last interation for the agent
                    return

    def __str__(self) -> str:
        ret = ""
        for m in self.messages_iterator():
            ret += str(m) + "\n"
        last_response = self.get_last_response()
        if last_response:
            ret += str(last_response) + "\n"
        return ret


class AIFunctionParameter(BaseModel):
    """AI function parameter model."""

    name: str
    description: str
    type: str
    required: bool = False


class AIFunction(BaseModel):
    """AI function model."""

    name: str
    description: str | None = None
    parameters: list[AIFunctionParameter]
    return_type: str | None = None
    return_description: str | None = None


class AITask(AIFunction):
    """AI task model."""

    system_instructions: AIPrompt
    prompt: str
