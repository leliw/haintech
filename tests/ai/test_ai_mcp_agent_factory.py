from pathlib import Path

from agents.mcp import MCPServerStdio

from haintech.ai.ai_mcp_agent import AIMCPAgent
from haintech.ai.ai_mcp_agent_factory import AIMCPAgentFactory
from haintech.ai.google_genai.google_ai_model import GoogleAIModel


def test_create_agent():
    # Given: A factory
    factory = AIMCPAgentFactory.create([GoogleAIModel])
    # When: Create agent
    agent = factory.create_ai_mcp_agent("google/gemini-3.5-flash-lite")
    # Then: An agent is created
    assert isinstance(agent, AIMCPAgent)


curr_dir = Path(__file__).resolve().parent
add = MCPServerStdio(
    name="add", params={"command": "uv", "args": ["--directory", str(curr_dir), "run", "mcp_server_add.py"]}
)
sub = MCPServerStdio(
    name="sub", params={"command": "uv", "args": ["--directory", str(curr_dir), "run", "mcp_server_sub.py"]}
)


def test_get_mcp_server_names():
    # Given: A factory with mcp server
    factory = AIMCPAgentFactory.create([GoogleAIModel], mcp_servers=[add, sub])
    # When: Get mcp server names
    names = factory.get_mcp_server_names()
    # Then: They are as passed
    assert "add" in names
    assert "sub" in names


def test_create_agent_with_factory_mcp():
    # Given: A factory with a mcp server
    factory = AIMCPAgentFactory.create([GoogleAIModel], mcp_servers=[add])
    # When: Create agent
    agent = factory.create_ai_mcp_agent("google/gemini-3.5-flash-lite")
    # Then: An agent is created
    assert isinstance(agent, AIMCPAgent)
    # And: The agent can use a mcp server
    assert add in agent.mcp_servers


def test_create_agent_without_factory_mcp():
    # Given: A factory with a mcp server
    factory = AIMCPAgentFactory.create([GoogleAIModel], mcp_servers=[add])
    # When: Create agent with an empty mcp server list
    agent = factory.create_ai_mcp_agent("google/gemini-3.5-flash-lite", mcp_servers=[])
    # Then: An agent is created
    assert isinstance(agent, AIMCPAgent)
    # And: The agent can't use a mcp server
    assert add not in agent.mcp_servers


def test_create_agent_with_mcp():
    # Given: A factory without mcp servers
    factory = AIMCPAgentFactory.create([GoogleAIModel])
    # When: Create agent with a mcp server
    agent = factory.create_ai_mcp_agent("google/gemini-3.5-flash-lite", mcp_servers=[add])
    # Then: An agent is created
    assert isinstance(agent, AIMCPAgent)
    # And: The agent can use a mcp server
    assert add in agent.mcp_servers


def test_create_agent_with_selected_mcp():
    # Given: A factory with mcp servers
    factory = AIMCPAgentFactory.create([GoogleAIModel], mcp_servers=[add, sub])
    # When: Create agent with a selected mcp server
    agent = factory.create_ai_mcp_agent("google/gemini-3.5-flash-lite", mcp_servers=["sub"])
    # Then: An agent is created
    assert isinstance(agent, AIMCPAgent)
    # And: The agent can use a mcp server
    assert add not in agent.mcp_servers
    assert sub in agent.mcp_servers
