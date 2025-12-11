import logging
import os
from pathlib import Path
from typing import List, Optional

import pytest
from agents.mcp import MCPServer, MCPServerStdio

from haintech.ai import (
    AIChatSession,
    BaseAIModel,
)
from haintech.ai.mcp_ai_agent import MCPAIAgent


@pytest.fixture(scope="session")
def mcp_server():
    samples_dir = Path(os.getcwd()) / "tests/data/samples"
    return MCPServerStdio(
        params={
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", str(samples_dir)],
        }
    )


@pytest.fixture(scope="session")
def hr_mcp_server():
    curr_dir = Path(__file__).resolve().parent
    return MCPServerStdio(params={"command": "uv", "args": ["--directory", str(curr_dir), "run", "mcp_server.py"]})


class HRAgent(MCPAIAgent):
    def __init__(
        self,
        ai_model: BaseAIModel,
        mcp_servers: List[MCPServer],
        session: Optional[AIChatSession] = None,
    ):
        super().__init__(
            ai_model=ai_model,
            mcp_servers=mcp_servers,
            description="HR Assistant",
            session=session,
        )


@pytest.mark.asyncio
async def test(mcp_server, ai_model: BaseAIModel):
    async with MCPAIAgent(ai_model=ai_model, mcp_servers=[mcp_server]) as agent:
        tools = await mcp_server.list_tools(None, None)
        assert len(tools) == 14
        response = await agent.get_text_response("Ile plików jest w katalogu tests/data/samples? Podaj liczbę.")
        assert "1" in response


@pytest.mark.asyncio
async def test_agent_one_question(ai_model: BaseAIModel, hr_mcp_server):
    logging.getLogger("haintech").setLevel(logging.DEBUG)
    # Given: An agent with session and tools
    session = AIChatSession()
    async with HRAgent(ai_model=ai_model, session=session, mcp_servers=[hr_mcp_server]) as ai_agent:
        # When: I ask agent
        response = await ai_agent.get_text_response("How many vacation days do I have left in 2025?")
    # Then: I should get answer
    assert "26" in response


@pytest.mark.asyncio
async def test_agent_with_acceptance(ai_model: BaseAIModel, hr_mcp_server):
    # STEP: 1
    # =======
    # Given: An agent with session and tools
    session = AIChatSession()
    async with HRAgent(ai_model=ai_model, session=session, mcp_servers=[hr_mcp_server]) as ai_agent:
        # When: I ask agent
        response = await ai_agent.get_response("How many vacation days do I have left in 2025?")
        # Then: I shult get function call
        assert response.tool_calls and 1 == len(response.tool_calls)
        # STEP: 2
        # =======
        # When: I accept function call
        response = await ai_agent.accept_tools(response.tool_calls[0].id)  # type: ignore
    # Then: I should get answer
    assert response.content and "26" in response.content

@pytest.mark.asyncio
async def test_agent_without_context(ai_model: BaseAIModel, hr_mcp_server):
    logging.getLogger("haintech").setLevel(logging.DEBUG)
    # Given: A mcp server context
    async with hr_mcp_server:
        # And: An agent with mcp server
        ai_agent = HRAgent(ai_model=ai_model, mcp_servers=[hr_mcp_server])
        await ai_agent.define_mcp_servers()
        # When: I ask agent
        response = await ai_agent.get_text_response("How many vacation days do I have left in 2025?")
    # Then: I should get answer
    assert "26" in response


@pytest.mark.asyncio
async def test_agent_and_mcp_server_without_context(ai_model: BaseAIModel, hr_mcp_server):
    logging.getLogger("haintech").setLevel(logging.DEBUG)
    # Given: A mcp server context
    await hr_mcp_server.connect()
    # And: An agent with mcp server
    ai_agent = HRAgent(ai_model=ai_model, mcp_servers=[hr_mcp_server])
    await ai_agent.define_mcp_servers()
    # When: I ask agent
    response = await ai_agent.get_text_response("How many vacation days do I have left in 2025?")
    await hr_mcp_server.cleanup()
    # Then: I should get answer
    assert "26" in response