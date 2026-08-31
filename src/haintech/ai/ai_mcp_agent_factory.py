import logging
from collections.abc import Callable

from agents.mcp import MCPServer

from haintech.ai.ai_agent_factory import AIAgentFactory
from haintech.ai.ai_mcp_agent import AIMCPAgent
from haintech.ai.ai_model_factory import AIModelFactory
from haintech.ai.base.base_agent_searcher import BaseAgentSearcher
from haintech.ai.base.base_ai_model import BaseAIModel
from haintech.ai.interfaces.async_session_blob_manager import AsyncSessionBlobManager
from haintech.ai.model import AIModelSession

_log = logging.getLogger(__name__)


class AIMCPAgentFactory(AIAgentFactory):
    def __init__(
        self,
        ai_model_factory: AIModelFactory,
        searcher: BaseAgentSearcher | None = None,
        session_blob_manager: AsyncSessionBlobManager | None = None,
        functions: list[Callable] | None = None,
        mcp_servers: list[MCPServer] | None = None,
    ):
        super().__init__(ai_model_factory, searcher, session_blob_manager, functions)
        self.mcp_servers: dict[str, MCPServer] = {}
        if mcp_servers:
            for s in mcp_servers:
                self.mcp_servers[s.name] = s

    @classmethod
    def create(
        cls,
        ai_model_classes: list[type[BaseAIModel]],
        searcher: BaseAgentSearcher | None = None,
        session_blob_manager: AsyncSessionBlobManager | None = None,
        functions: list[Callable] | None = None,
        mcp_servers: list[MCPServer] | None = None,
    ) -> "AIMCPAgentFactory":
        ai_model_factory = AIModelFactory(ai_model_classes)
        return cls(ai_model_factory, searcher, session_blob_manager, functions, mcp_servers)

    def create_ai_mcp_agent(
        self,
        vendor_model_name: str,
        system_prompt: str | None = None,
        session: AIModelSession | None = None,
        searcher: BaseAgentSearcher | None = None,
        functions: list[Callable] | list[str] | None = None,
        mcp_servers: list[MCPServer] | list[str] | None = None,
        session_blob_manager: AsyncSessionBlobManager | None = None,
        ai_model_parameters: dict[str, str | int | float] | None = None,
    ) -> AIMCPAgent:
        if functions and isinstance(functions[0], Callable):
            fns: list[Callable] | None = functions  # pyright: ignore[reportAssignmentType]
        else:
            fns = self.get_functions(functions)  # pyright: ignore[reportArgumentType]
        if mcp_servers and isinstance(mcp_servers[0], MCPServer):
            mcps: list[MCPServer] = mcp_servers  # pyright: ignore[reportAssignmentType]
        else:
            mcps = self.get_mcp_servers(mcp_servers)  # pyright: ignore[reportArgumentType]
        return AIMCPAgent(
            ai_model=self.create_ai_model(vendor_model_name, ai_model_parameters),
            mcp_servers=mcps,
            system_prompt=system_prompt,
            session=session,
            searcher=searcher or self.searcher,
            functions=fns,
            session_blob_manager=session_blob_manager or self.session_blob_manager,
        )

    def get_mcp_server_names(self) -> list[str]:
        return list(self.mcp_servers.keys())

    def get_mcp_servers(self, names: list[str] | None = None) -> list[MCPServer]:
        if names is None:
            return list(self.mcp_servers.values())

        ret = []
        for name in set(names):
            if name not in self.mcp_servers:
                raise ValueError(
                    f"MCP server '{name}' is not registered in the factory. "
                    f"Available servers: {self.get_mcp_server_names()}"
                )
            ret.append(self.mcp_servers[name])

        return ret
