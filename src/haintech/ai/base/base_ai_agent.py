import logging
from typing import Any, Callable, Dict, Iterator, List, Tuple, override

from haintech.ai.ai_task_executor import AITaskExecutor

from ..model import (
    AIChatResponse,
    AIModelInteractionMessage,
    AIModelSession,
    AIPrompt,
    AITask,
)
from .base_ai_chat import BaseAIChat
from .base_ai_model import BaseAIModel
from .base_rag_searcher import BaseRAGSearcher


class BaseAIAgent(BaseAIChat):
    """Base AI Agent. It is AI Chat with tools."""

    _log = logging.getLogger(__name__)

    def __init__(
        self,
        name: str = None,
        description: str = None,
        ai_model: BaseAIModel = None,
        prompt: AIPrompt = None,
        context: str = None,
        session: AIModelSession = None,
        searcher: BaseRAGSearcher = None,
        functions: List[Callable] = None,
    ):
        super().__init__(ai_model, prompt or context, session)
        self.name = name or self.__class__.__name__
        self.description = description
        self.searcher = searcher
        self.functions: Dict[Callable, Any] = {}
        self.function_names: Dict[str, Callable] = {}
        if functions:
            for f in functions:
                self.add_function(f)
        self.rag_items = []

    def add_function(
        self, function: Callable, name: str = None, definition: Any = None
    ) -> None:
        name = name or function.__name__
        definition = definition or self.ai_model.prepare_function_definition(
            function, name=name
        )
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
            functions=self.functions,
            interaction_logger=self._interaction_logger,
        )
        return response

    def get_response(self, message: str = None) -> AIChatResponse:
        """Get response from LLM

        Args:
            message: message to send to LLM
        Returns:
            response: LLM response
        """
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
        response = self.get_response(message)
        while response.tool_calls:
            response = self.accept_tools(
                [tool_call.id for tool_call in response.tool_calls]
            )
        return response.content

    def accept_tools(self, tool_call_ids: str | List[str]) -> AIChatResponse:
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
                self._log.debug(
                    "Calling tool: %s : %s with arguments: %s", id, name, arguments
                )
                self.call_function(id, name, **arguments)
            else:
                self._log.debug(
                    "Refused calling: %s : %s with arguments: %s", id, name, arguments
                )
                # self.add_message(
                #     AIModelInteractionMessage(
                #         tool_call_id=id,
                #         role="tool",
                #         content="The user refused execution.",
                #     )
                # )
        resp = self.get_response()
        self.add_response_message(resp)
        return resp

    def iter_tool_calls(self) -> Iterator[Tuple[str, str, Dict[str, Any]]]:
        """Iterate over tool calls from last response

        Returns:
            iterator: iterator of tool calls
        """
        last_response = self.get_last_response()
        if last_response and last_response.tool_calls:
            for tool_call in last_response.tool_calls:
                yield (tool_call.id, tool_call.function_name, tool_call.arguments)

    def get_last_response(self) -> AIChatResponse:
        return self.session.get_last_response()

    def call_function(self, tool_call_id: str, name: str, **arguments) -> Any:
        function = self.function_names[name]
        if not function:
            self._log.error("Function %s not found", name)
            for f in self.functions:
                self._log.debug("Function: %s", f.__name__)
            raise ValueError(f"Function {name} not found")
        ret = function(**arguments)
        content = str(ret)
        self.add_tool_message(tool_call_id, content)
        return ret

    def add_tool_message(self, tool_call_id: str, content: str):
        self.add_message(
            AIModelInteractionMessage(
                role="tool", tool_call_id=tool_call_id, content=content
            )
        )

    @override
    def _get_context(self, message: AIModelInteractionMessage = None) -> str | AIPrompt:
        ret = super()._get_context(message) or AIPrompt()
        if message and message.content:
            msg = message.content
            if self.searcher:
                if len(msg) > 15:
                    self._log.debug("Searching for: %s", msg)
                    self.rag_items = self.searcher.search_sync(msg)
                    self._log.debug("Found: %d items", len(self.rag_items))
                # When message is too short, do not search, just use last search results
                for item in self.rag_items:
                    self._log.debug("Item: %s", item.title)
                ret.documents += self.rag_items
        return ret
