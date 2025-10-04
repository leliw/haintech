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
        """AI Agent that can interact with tools provided via Model Context Protocol (MCP).

        This agent extends the `BaseAIAgentAsync` to connect to one or more `MCPServer`
        instances, discover their available tools, and make them usable by the AI model.
        It is designed to be used as an asynchronous context manager.
        """
        _log = logging.getLogger(__name__)

        def __init__(
            self,
            ai_model: BaseAIModel,
            mcp_servers: List[MCPServer],
            name: Optional[str] = None,
            description: Optional[str] = None,
            prompt: Optional[AIPrompt] = None,
            session: Optional[AIModelSession] = None,
            searcher: Optional[BaseRAGSearcher] = None,
            functions: Optional[List[Callable]] = None,
        ):
            """Initializes the MCPAIAgent.

            Args:
                ai_model: The AI model to use.
                mcp_servers: A list of MCPServer instances to connect to.
                name: The name of the agent.
                description: A description of the agent's purpose.
                prompt: A structured prompt for the agent.
                context: A string providing context for the agent's conversations.
                session: The session object to store conversation history.
                searcher: A RAG searcher for retrieving documents.
                functions: A list of additional callable functions to be used as tools.
            """
            super().__init__(
                ai_model=ai_model,
                name=name,
                description=description,
                system_prompt=prompt,
                session=session,
                searcher=searcher,
                functions=functions,
            )
            self.mcp_servers = mcp_servers or []

        async def __aenter__(self):
            """Asynchronous context manager entry point. Connects to MCP servers."""
            await self.mcp_servers_connect()
            return self

        async def __aexit__(self, exc_type, exc_value, traceback):
            """Asynchronous context manager exit point. Cleans up MCP server connections."""
            await self.mcp_servers_cleanup()

        async def mcp_servers_connect(self):
            """Connects to all configured MCP servers and defines their tools."""
            for server in self.mcp_servers:
                await server.connect()
            await self.define_mcp_servers()

        async def mcp_servers_cleanup(self):
            """Disconnects and cleans up all configured MCP servers."""
            for server in self.mcp_servers:
                await server.cleanup()

        async def define_mcp_servers(self):
            """Discovers tools from all connected MCP servers and registers them with the agent.

            For each tool found on a server, it creates a corresponding asynchronous
            function that can be called by the agent to execute the tool.
            """
            for server in self.mcp_servers:
                tools = await server.list_tools(None, None) # type: ignore
                for tool in tools:
                    # Capture tool_name by making it a default argument
                    # to avoid closure issues in the loop.
                    async def mcp_call_tool(tool_name=tool.name, **kwargs):
                        """Dynamically created async function to call an MCP tool."""
                        return await server.call_tool(tool_name, kwargs)

                    # Register the dynamically created function as a tool for the agent.
                    self.function_names[tool.name] = mcp_call_tool
                    self.functions[self.function_names[tool.name]] = self.ai_model.prepare_mcp_tool_definition(tool)


except ImportError:
    # The 'agents-mcp' package is optional. If it's not installed,
    # the MCPAIAgent class will not be available, and importing it will raise an ImportError.
    pass
