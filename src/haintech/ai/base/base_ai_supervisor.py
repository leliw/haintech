from typing import Any, Callable, Dict, List, Optional

from haintech.ai.base.base_ai_agent import BaseAIAgent
from haintech.ai.base.base_ai_model import BaseAIModel
from haintech.ai.base.base_rag_searcher import BaseRAGSearcher
from haintech.ai.model import AIChatResponse, AISupervisorSession


class BaseAISupervisor(BaseAIAgent):
    """Base AI Supervisor. It is AI Agent with subagents."""

    def __init__(
        self,
        ai_model: BaseAIModel,
        name: Optional[str] = None,
        description: Optional[str] = None,
        system_prompt: Optional[str] = None,
        session: Optional[AISupervisorSession] = None,
        searcher: Optional[BaseRAGSearcher] = None,
        functions: Optional[List[Callable]] = None,
        agents: Optional[List[BaseAIAgent]] = None,
    ):
        super().__init__(
            name=name,
            description=description,
            ai_model=ai_model,
            system_prompt=system_prompt,
            session=session,
            searcher=searcher,
            functions=functions,
        )
        self.session: Optional[AISupervisorSession] = session
        self.agents: Dict[str, BaseAIAgent] = {}
        if agents:
            for agent in agents:
                self.add_agent(agent)
        self.agent_tool_calls: Dict[str, List[str]] = {}

    def add_agent(self, agent: BaseAIAgent):
        """Add agent to supervisor

        * Agent is added as a function.
        * Agents' sessions are reflections to
          Supervisor's session.

        Args:
            agent: agent to add
        """
        self.add_function(
            agent.get_response,
            name=agent.get_name(),
            definition=self.get_agent_definition(agent),
        )
        if self.session:
            agent.set_session(self.session.create_agent_session(agent.name))
        self.agents[agent.name] = agent

    def get_agent_definition(self, agent: BaseAIAgent) -> Any:
        """Return agent definition

        Args:
            agent: agent
        Returns:
            definition: agent definition
        """
        return self.ai_model.prepare_function_definition(
            agent.get_response,
            name=agent.get_name(),
            description=agent.description,
        )

    def accept_tools(self, tool_call_ids: str | List[str]) -> AIChatResponse:
        """Accept calling tools, call them and return response

        Args:
            tool_call_ids: list of accepted tool call ids
        Returns:
            response: AI response
        """
        if isinstance(tool_call_ids, str):
            tool_call_ids = [tool_call_ids]

        agent_responded_ids = []
        for supervisor_tc_id, agent_tc_ids in self.agent_tool_calls.items():
            if set(tool_call_ids).intersection(agent_tc_ids):
                agent = self.agents[agent_tc_ids[0].split("__")[0]]
                agent_response = self.accept_agent_tools(agent, tool_call_ids)
                if agent_response.content:
                    self.add_tool_message(supervisor_tc_id, agent_response.content)
                    agent_responded_ids.append(supervisor_tc_id)

        for id, name, arguments in self.iter_tool_calls():
            if id in tool_call_ids:
                if name.startswith("Agent__"):
                    self._log.debug(
                        "Calling agent: %s : %s with arguments: %s", id, name, arguments
                    )
                    agent_resp = self.call_agent(id, name[7:], **arguments)
                    if agent_resp and agent_resp.tool_calls:
                        return agent_resp
                    # Agent returns response without calling tools
                    if agent_resp and agent_resp.content:
                        self.add_tool_message(id, agent_resp.content)

                else:
                    self.call_function(id, name, **arguments)
            elif id and id not in agent_responded_ids:
                self.add_tool_message(id, "User refused execution.")

        return self.get_response()

    def accept_agent_tools(
        self, agent: BaseAIAgent, tool_call_ids: str | List[str]
    ) -> AIChatResponse:
        """Accept calling tools, call them and return response

        Args:
            agent: agent to call
            tool_call_ids: list of accepted tool call ids
        Returns:
            response: AI response
        """
        prefix = f"{agent.name}__"
        agent_tool_call_ids = [
            id.removeprefix(prefix) for id in tool_call_ids if id.startswith(prefix)
        ]
        return agent.accept_tools(agent_tool_call_ids)

    def call_agent(
        self, tool_call_id: str, agent_name: str, **arguments
    ) -> AIChatResponse | None:
        """Call agent and return response

        Args:
            tool_call_id: tool call id
            agent_name: agent name
            arguments: arguments for agent
        Returns:
            response: AI response
        """
        agent = self.agents[agent_name]
        self._log.debug("Agent: %s", agent_name)
        response = agent.get_response(**arguments)
        self._log.debug("Agent response: %s", response)

        if response.tool_calls:
            # Agent requires calling tools
            # I store link between supervisor call and agent calls
            self.agent_tool_calls[tool_call_id] = []
            # And return copy of agent response with renamed call_ids
            response = AIChatResponse(**response.model_dump())
            for tool_call in response.tool_calls or []:
                tool_call.id = f"{agent_name}__{tool_call.id}"
                self.agent_tool_calls[tool_call_id].append(tool_call.id)
        else:
            self._log.info("%s: %s", agent_name, response.content)
        return response
