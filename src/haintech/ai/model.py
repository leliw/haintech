import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, Iterator, List, Optional, Tuple, override

from pydantic import BaseModel, Field


class RAGQuery(BaseModel):
    """
    RAG query model.
    """

    text: str
    keywords: Optional[List[str]] = None
    limit: int = 5


class RAGItem(BaseModel):
    item_id: Optional[str] = None
    title: Optional[str] = None
    keywords: Optional[List[str]] = Field(
        default_factory=list,
        description="List of keywords associated with the item.",
    )
    content: str
    metadata: Optional[Dict[str, str]] = Field(
        default_factory=dict,
        description="Additional metadata for the item.",
    )


class AIModelToolCall(BaseModel):
    """Tool call request returned by AIModel"""

    id: Optional[str] = None
    function_name: str
    arguments: Dict[str, Any]

    def __str__(self):
        return f"{self.id}: {self.function_name}({', '.join([f'{k}="{v}"' for k, v in self.arguments.items()])})"


class AIChatResponse(BaseModel):
    """Chat response model."""

    content: Optional[str] = None
    tool_calls: Optional[List[AIModelToolCall]] = None

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

    role: str
    name: Optional[str] = None
    tool_call_id: Optional[str] = None  # Only for role=tool
    content: Optional[str] = None
    tool_calls: Optional[List[AIModelToolCall]] = None

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


class AIPrompt(BaseModel):
    """Structured AI prompt model.

    According to: <https://cloud.google.com/vertex-ai/generative-ai/docs/learn/prompts/prompt-design-strategies#components-of-a-prompt>
    """

    persona: Optional[str] = None
    objective: Optional[str] = None
    instructions: Optional[str] = None
    constraints: Optional[str] = None
    context: Optional[str] = None
    documents: Optional[List[str | RAGItem]] = Field(default_factory=list)
    output_format: Optional[str] = None
    examples: Optional[List[str]] = Field(default_factory=list)
    recap: Optional[str] = None


class AIContext(BaseModel):
    context: Optional[str] = None
    documents: List[str | RAGItem]


class AIModelInteraction[T: AIModelInteractionMessage](BaseModel):
    """One interaction with AIModel"""

    uid: str = Field(default_factory=lambda: uuid.uuid4().hex)
    model: str
    tools: Optional[List[AIModelInteractionTool]] = None
    parallel_tool_calls: Optional[bool] = None
    response_format: Optional[Dict[str, str]] = None
    context: Optional[str] = None
    prompt: Optional[AIPrompt] = None
    history: List[T]
    message: Optional[T] = None
    response: Optional[AIChatResponse] = None


class AIModelSession[T: AIModelInteractionMessage](ABC):
    """AIModel session model."""

    @abstractmethod
    def add_interaction(self, interaction: AIModelInteraction[T]) -> None:
        """Add interaction to session."""
        pass

    @abstractmethod
    def messages_iterator(self) -> Iterator[T]:
        """Itrerates over all messages (from last interaction)."""
        pass

    @abstractmethod
    def get_last_response(self) -> Optional[AIChatResponse]:
        """Get last response (from last interaction)."""
        pass

    @classmethod
    def create_message_from_response(cls, response: AIChatResponse) -> T:
        return AIModelInteractionMessage.create_from_response(response)  # type: ignore


class AIChatSession[T: AIModelInteractionMessage](BaseModel, AIModelSession[T]):
    """Chat session model."""

    uid: str = Field(default_factory=lambda: uuid.uuid4().hex)
    datetime: str = Field(default_factory=lambda: str(datetime.now()))
    interactions: List[AIModelInteraction[T]] = Field(default_factory=list)

    @override
    def add_interaction(self, interaction: AIModelInteraction[T]):
        self.interactions.append(interaction)

    @override
    def get_last_response(self) -> Optional[AIChatResponse]:
        """Get last response."""
        last_interaction = self.get_last_interaction()
        if last_interaction:
            return last_interaction.response
        return None

    @override
    def messages_iterator(self) -> Iterator[T]:
        """Itrerates over all messages (from last interaction)"""
        last_interaction = self.get_last_interaction()
        if last_interaction:
            for message in last_interaction.history:
                yield message
            if last_interaction.message:
                yield last_interaction.message
                clazz = last_interaction.message.__class__
                last_response = self.get_last_response()
                if last_response:
                    yield clazz.create_from_response(last_response)

    def get_last_interaction(self) -> Optional[AIModelInteraction[T]]:
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
        if m:
            clazz = m.__class__
            last_response = self.get_last_response()
            if last_response:
                ret += str(clazz.create_from_response(last_response)) + "\n"
        return ret


class AISupervisorSession[T: AIModelInteractionMessage](BaseModel, AIModelSession[T]):
    """Supervisor session model. It can create agent sessions."""

    uid: str = Field(default_factory=lambda: uuid.uuid4().hex)
    datetime: str = Field(default_factory=lambda: str(datetime.now()))
    interactions: List[Tuple[str | None, AIModelInteraction[T]]] = Field(default_factory=list)
    agent_name: Optional[str] = None

    def create_agent_session(self, agent_name: str) -> AIModelSession[T]:
        """Create agent session."""
        return AIAgentSession(agent_name=agent_name, interactions=self.interactions)

    @override
    def add_interaction(self, interaction: AIModelInteraction[T]):
        self.interactions.append((self.agent_name, interaction))

    @override
    def get_last_response(self) -> Optional[AIChatResponse]:
        """Get last response."""
        if self.interactions:
            for agent_name, interaction in reversed(self.interactions):
                if agent_name == self.agent_name:
                    return interaction.response
        return None

    @override
    def messages_iterator(self) -> Iterator[T]:
        """Itrerates over all messages (from last interaction)"""
        if self.interactions:
            for agent_name, interaction in reversed(self.interactions):
                if agent_name == self.agent_name:
                    for message in interaction.history:
                        yield message
                    if interaction.message:
                        yield interaction.message
                    # I've found the last iteration for the agent
                    return

    def __str__(self) -> str:
        ret = ""
        for m in self.messages_iterator():
            ret += str(m) + "\n"
        if m:
            clazz = m.__class__
            last_response = self.get_last_response()
            if last_response:
                ret += str(clazz.create_from_response(last_response)) + "\n"
        return ret


class AIAgentSession[T: AIModelInteractionMessage](AIModelSession[T]):
    """Agent session model.

    It links to supervisor session.
    All methods operate on supervisor session interactions
    with the agent name differentiator.
    """

    def __init__(self, agent_name: str, interactions: List[Tuple[str | None, AIModelInteraction[T]]]):
        self.agent_name = agent_name
        self.interactions = interactions

    @override
    def add_interaction(self, interaction: AIModelInteraction[T]):
        self.interactions.append((self.agent_name, interaction))

    @override
    def get_last_response(self) -> Optional[AIChatResponse]:
        """Get last response."""
        if self.interactions:
            for agent_name, interaction in reversed(self.interactions):
                if agent_name == self.agent_name:
                    return interaction.response
        return None

    @override
    def messages_iterator(self) -> Iterator[T]:
        """Itrerates over all messages (from last interaction)"""
        if self.interactions:
            for agent_name, interaction in reversed(self.interactions):
                if agent_name == self.agent_name:
                    for message in interaction.history:
                        yield message
                    if interaction.message:
                        yield interaction.message
                    # I've found the last iteration for the agent
                    return


class AIFunctionParameter(BaseModel):
    """AI function parameter model."""

    name: str
    description: str
    type: str
    required: bool = False


class AIFunction(BaseModel):
    """AI function model."""

    name: str
    description: Optional[str] = None
    parameters: List[AIFunctionParameter]
    return_type: Optional[str] = None
    return_description: Optional[str] = None


class AITask(AIFunction):
    """AI task model."""

    system_instructions: AIPrompt
    prompt: str
