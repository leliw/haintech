import asyncio
import logging
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

from haintech.ai.ai_task_executor import AITaskExecutor

from ..model import (
    AIChatResponse,
    AIContext,
    AIModelInteractionMessage,
    AIModelSession,
    AIPrompt,
    AITask,
    RAGQuery,
)
from .base_ai_chat_async import BaseAIChatAsync
from .base_ai_model import BaseAIModel
from .base_rag_searcher import BaseRAGSearcher


class BaseAIAgentAsync(BaseAIChatAsync):
    """Base AI Agent async version. It is AI Chat with tools."""

    _log = logging.getLogger(__name__)

    def __init__(
        self,
        ai_model: BaseAIModel,
        name: Optional[str] = None,
        description: Optional[str] = None,
        system_prompt: Optional[AIPrompt] = None,
        session: Optional[AIModelSession] = None,
        searcher: Optional[BaseRAGSearcher] = None,
        functions: Optional[List[Callable]] = None,
    ):
        """Base AI Agent.

        Args:
            name: name of the agent
            description: description of the agent
            ai_model: AI model
            prompt: Context prompt for model
            context: Context prompt for model
            session: current session object
            searcher: RAG searcher
            functions: list of functions to add
        """
        super().__init__(ai_model, system_prompt, session)
        self.name = name or self.__class__.__name__
        self.description = description
        self.searcher = searcher
        self.functions: Dict[Callable, Any] = {}
        self.function_names: Dict[str, Callable] = {}
        if functions:
            for f in functions:
                self.add_function(f)
        self.rag_items = []

    def add_function(self, function: Callable, name: Optional[str] = None, definition: Any = None) -> None:
        name = name or function.__name__
        definition = definition or self.ai_model.prepare_function_definition(function, name=name)
        self.functions[function] = definition
        self.function_names[name] = function

    def add_rag_searcher(self, searcher: BaseRAGSearcher):
        self.searcher = searcher

    def add_ai_task(self, ai_task: AITask) -> None:
        definition = self.ai_model.model_function_definition(ai_task)
        function = AITaskExecutor.create_from_definition(self.ai_model, ai_task).execute
        self.functions[function] = definition
        self.function_names[ai_task.name] = function

    def get_name(self) -> str:
        return f"Agent__{self.name}"

    async def _get_response(self, message: Optional[AIModelInteractionMessage] = None) -> AIChatResponse:
        """Get response from LLM

        All necessary messages and tools should be added before calling this function.

        Returns:
            response: LLM response
        """
        history = list(self.iter_messages())
        response = await self.ai_model.get_chat_response_async(
            message=message,
            system_prompt=self._get_prompt(),
            history=history,
            functions=self.functions,
            interaction_logger=self._interaction_logger,
        )
        return response

    async def get_response(self, message: Optional[AIModelInteractionMessage | str] = None) -> AIChatResponse:
        """Get response from LLM

        Args:
            message: message to send to LLM
        Returns:
            response: LLM response
        """
        if isinstance(message, str):
            message = AIModelInteractionMessage(role="user", content=message)
        # Call LLM
        m_resp = await self._get_response(message=message)
        # Add message and response to history
        if message:
            self.add_message(message)
        self.add_response_message(m_resp)
        return m_resp

    async def get_text_response(self, message: Optional[str] = None) -> str:
        response = await self.get_response(message)
        while response.tool_calls:
            response = await self.accept_tools([tool_call.id for tool_call in response.tool_calls if tool_call.id])
        if response.content:
            return response.content
        else:
            raise ValueError("No content in response")

    async def accept_tools(self, tool_call_ids: str | List[str]) -> AIChatResponse:
        """Accept calling tools, call them and return response

        Args:
            tool_call_ids: list of accepted tool call ids
        Returns:
            response: OpenAI response
        """
        if isinstance(tool_call_ids, str):
            tool_call_ids = [tool_call_ids]
        for id, name, arguments in self.iter_tool_calls():
            if id in tool_call_ids:
                self._log.debug("Calling tool: %s : %s with arguments: %s", id, name, arguments)
                await self.call_function(id, name, **arguments)
            else:
                self._log.debug("Refused calling: %s : %s with arguments: %s", id, name, arguments)
                # self.add_message(
                #     AIModelInteractionMessage(
                #         tool_call_id=id,
                #         role="tool",
                #         content="The user refused execution.",
                #     )
                # )
        resp = await self.get_response()
        return resp

    def iter_tool_calls(self) -> Iterator[Tuple[str | None, str, Dict[str, Any]]]:
        """Iterate over tool calls from last response

        Returns:
            iterator: iterator of tool calls
        """
        last_response = self.get_last_response()
        if last_response and last_response.tool_calls:
            for tool_call in last_response.tool_calls:
                yield (tool_call.id, tool_call.function_name, tool_call.arguments)

    def get_last_response(self) -> AIChatResponse | None:
        return self.session.get_last_response()

    async def call_function(self, tool_call_id: str, name: str, **arguments) -> Any:
        function = self.function_names[name]
        if not function:
            self._log.error("Function %s not found", name)
            for f in self.functions:
                self._log.debug("Function: %s", f.__name__)
            raise ValueError(f"Function {name} not found")
        try:
            if asyncio.iscoroutinefunction(function):
                ret = await function(**arguments)
            else:
                ret = function(**arguments)
        except Exception as e:
            ret = f"Error calling function: {e}"
            self._log.error(ret)
        content = str(ret)
        self.add_tool_message(tool_call_id, content)
        return ret

    def add_tool_message(self, tool_call_id: str, content: str):
        self.add_message(AIModelInteractionMessage(role="tool", tool_call_id=tool_call_id, content=content))

    async def _get_context(self, message: Optional[AIModelInteractionMessage] = None) -> AIContext:
        ret = AIContext(documents=[])
        if message and message.content:
            msg = message.content
            if self.searcher:
                if len(msg) > 15:
                    self._log.debug("Searching for: %s", msg)
                    self.rag_items = list(await self.searcher.search_async(RAGQuery(text=msg)))
                    self._log.debug("Found: %d items", len(self.rag_items))
                # When message is too short, do not search, just use last search results
                if isinstance(ret, AIPrompt):
                    for item in self.rag_items:
                        self._log.debug("Item: %s", item.title)
                    if ret.documents is None:
                        ret.documents = []
                    ret.documents += self.rag_items
        return ret
