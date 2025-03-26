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
    keywords: List[str] = None
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


class AIModelInteractionMessageToolCall(BaseModel):
    """Tool call request returned by AIModel"""

    id: Optional[str] = None
    function_name: str
    arguments: Dict[str, Any]

    def __str__(self) -> str:
        return f"{self.id}: {self.function_name}({self.arguments})"


class AIModelInteractionMessage(BaseModel):
    """One message within AIModelInteraction"""

    role: str
    name: Optional[str] = None
    tool_call_id: Optional[str] = None  # Only for role=tool
    content: Optional[str] = None
    tool_calls: Optional[List[AIModelInteractionMessageToolCall]] = None

    def __str__(self) -> str:
        ret = f"{self.role:10}:" + (
            f" {self.tool_call_id} => " if self.tool_call_id else ""
        )
        if self.content:
            if len(self.content) < 1024:
                ret += f" {self.content}"
            else:
                ret += f" {self.content[:1000]}..."
        if self.tool_calls:
            for i, tc in enumerate(self.tool_calls):
                ret += f"{' ' if i == 0 else '\n'}{tc}"

        return ret


class AIChatResponseToolCall(BaseModel):
    """Chat response tool call model."""

    id: Optional[str] = None
    function_name: str
    arguments: Dict[str, Any]

    def __str__(self):
        return f"{self.id}: {self.function_name}({', '.join([f'{k}="{v}"' for k, v in self.arguments.items()])})"


class AIChatResponse(BaseModel):
    """Chat response model."""

    content: Optional[str] = None
    tool_calls: Optional[List[AIChatResponseToolCall]] = None

    def toMessage(self) -> AIModelInteractionMessage:
        tool_calls = []
        if self.tool_calls:
            for t in self.tool_calls:
                tool_calls.append(AIModelInteractionMessageToolCall(**t.model_dump()))
        return AIModelInteractionMessage(
            role="assistant", content=self.content, tool_calls=tool_calls
        )

    def __str__(self) -> str:
        ret = []
        if self.content:
            ret.append(f"Assistant: {self.content}")
        if self.tool_calls:
            for tc in self.tool_calls:
                ret.append(str(tc))
        return "\n".join(ret)


class AIModelInteractionTool(BaseModel):
    type: str
    function: Any


class AIModelInteraction(BaseModel):
    """One interaction with AIModel"""

    uid: str = Field(default_factory=lambda: uuid.uuid4().hex)
    model: str
    message: AIModelInteractionMessage = None
    context: Optional[str] = None
    history: List[AIModelInteractionMessage]
    tools: Optional[List[AIModelInteractionTool]] = None
    parallel_tool_calls: Optional[bool] = None
    response_format: Optional[Dict[str, str]] = None
    response: AIChatResponse


class AIModelSession(ABC):
    """AIModel session model."""

    @abstractmethod
    def add_interaction(self, interaction: AIModelInteraction):
        """Add interaction to session."""
        pass

    @abstractmethod
    def messages_iterator(self) -> Iterator[AIModelInteractionMessage]:
        """Itrerates over all messages (from last interaction)."""
        pass

    @abstractmethod
    def get_last_response(self) -> Optional[AIChatResponse]:
        """Get last response (from last interaction)."""
        pass


class AIChatSession(BaseModel, AIModelSession):
    """Chat session model."""

    uid: str = Field(default_factory=lambda: uuid.uuid4().hex)
    datetime: str = Field(default_factory=lambda: str(datetime.now()))
    interactions: List[AIModelInteraction] = Field(default_factory=list)

    @override
    def add_interaction(self, interaction: AIModelInteraction):
        self.interactions.append(interaction)

    @override
    def get_last_response(self) -> Optional[AIChatResponse]:
        """Get last response."""
        if self.interactions:
            return self.get_last_interaction().response
        return None

    @override
    def messages_iterator(self) -> Iterator[AIModelInteractionMessage]:
        """Itrerates over all messages (from last interaction)"""
        if self.interactions:
            for message in self.get_last_interaction().history:
                yield message
            yield self.get_last_interaction().message

    def get_last_interaction(self) -> Optional[AIModelInteraction]:
        """Get last interaction."""
        if self.interactions:
            return self.interactions[-1]
        return None

    def add_message(self, message: AIModelInteractionMessage):
        """Add message to last interaction."""
        if self.interactions:
            self.get_last_interaction().history.append(message)

    def __str__(self) -> str:
        ret = ""
        for m in self.messages_iterator():
            ret += str(m) + "\n"
        ret += str(self.get_last_response().toMessage()) + "\n"
        return ret


class AISupervisorSession(BaseModel, AIModelSession):
    """Supervisor session model. It can create agent sessions."""

    uid: str = Field(default_factory=lambda: uuid.uuid4().hex)
    datetime: str = Field(default_factory=lambda: str(datetime.now()))
    interactions: List[Tuple[str, AIModelInteraction]] = Field(default_factory=list)
    agent_name: str = None

    def create_agent_session(self, agent_name: str) -> AIModelSession:
        """Create agent session."""
        return AIAgentSession(agent_name=agent_name, interactions=self.interactions)

    @override
    def add_interaction(self, interaction: AIModelInteraction):
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
    def messages_iterator(self) -> Iterator[AIModelInteractionMessage]:
        """Itrerates over all messages (from last interaction)"""
        if self.interactions:
            for agent_name, interaction in reversed(self.interactions):
                if agent_name == self.agent_name:
                    for message in interaction.history:
                        yield message
                    yield interaction.message
                    # I've found the last iteration for the agent
                    return

    def __str__(self) -> str:
        ret = ""
        for m in self.messages_iterator():
            ret += str(m) + "\n"
        ret += str(self.get_last_response().toMessage()) + "\n"
        return ret


class AIAgentSession(AIModelSession):
    """Agent session model.

    It links to supervisor session.
    All methods operate on supervisor session interactions
    with the agent name differentiator.
    """

    def __init__(
        self, agent_name: str, interactions: List[Tuple[str, AIModelInteraction]]
    ):
        self.agent_name = agent_name
        self.interactions = interactions

    @override
    def add_interaction(self, interaction: AIModelInteraction):
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
    def messages_iterator(self) -> Iterator[AIModelInteractionMessage]:
        """Itrerates over all messages (from last interaction)"""
        if self.interactions:
            for agent_name, interaction in reversed(self.interactions):
                if agent_name == self.agent_name:
                    for message in interaction.history:
                        yield message
                    yield interaction.message
                    # I've found the last iteration for the agent
                    return


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


class AIFunctionParameter(BaseModel):
    """AI function parameter model."""

    name: str
    description: str
    type: str
    required: bool = False


class AIFunction(BaseModel):
    """AI function model."""

    name: str
    description: str
    parameters: List[AIFunctionParameter]
    return_type: Optional[str] = None
    return_description: Optional[str] = None


class AITask(AIFunction):
    """AI task model."""

    system_instructions: AIPrompt
    prompt: str
