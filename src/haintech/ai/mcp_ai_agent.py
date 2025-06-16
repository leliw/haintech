import logging
from typing import Callable, List, Optional

from .base import BaseAIAgentAsync, BaseAIModel, BaseRAGSearcher
from .model import (
    AIModelSession,
    AIPrompt,
)

try:
    from agents.mcp import MCPServer

    class MCPAIAgent(BaseAIAgentAsync):
        """Base AI Agent. It is AI Chat with tools."""

        _log = logging.getLogger(__name__)

        def __init__(
            self,
            ai_model: BaseAIModel,
            mcp_servers: List[MCPServer],
            name: Optional[str] = None,
            description: Optional[str] = None,
            prompt: Optional[AIPrompt] = None,
            context: Optional[str] = None,
            session: Optional[AIModelSession] = None,
            searcher: Optional[BaseRAGSearcher] = None,
            functions: Optional[List[Callable]] = None,
        ):
            super().__init__(
                ai_model=ai_model,
                name=name,
                description=description,
                prompt=prompt,
                context=context,
                session=session,
                searcher=searcher,
                functions=functions,
            )
            self.mcp_servers = mcp_servers or []

        async def __aenter__(self):
            await self.mcp_servers_connect()
            return self

        async def __aexit__(self, exc_type, exc_value, traceback):
            await self.mcp_servers_cleanup()

        async def mcp_servers_connect(self):
            for server in self.mcp_servers:
                await server.connect()
            await self.define_mcp_servers()

        async def mcp_servers_cleanup(self):
            for server in self.mcp_servers:
                await server.cleanup()

        async def define_mcp_servers(self,):
            for server in self.mcp_servers:
                tools = await server.list_tools()
                for tool in tools:
                    # Capture tool_name by making it a default argument
                    # to avoid closure issues in the loop.
                    async def mcp_call_tool(tool_name=tool.name, **kwargs):
                        return await server.call_tool(tool_name, kwargs)

                    self.function_names[tool.name] = mcp_call_tool
                    self.functions[self.function_names[tool.name]] = self.ai_model.prepare_mcp_tool_definition(tool)


except ImportError:
    pass
