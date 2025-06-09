from typing import Dict

import pytest

from haintech.ai import AIChatSession, BaseAIAgent, BaseAIModel, BaseAISupervisor
from haintech.ai.open_ai import OpenAIAgent


def test_agent_with_session(ai_model: BaseAIModel):
    # Given: An agent with session
    session = AIChatSession()
    ai_agent = OpenAIAgent(ai_model=ai_model, session=session)
    # And: The session is in progress
    response = ai_agent.get_text_response("Who was the first polish king?")
    assert "first" in response
    assert 1 == len(session.interactions)
    assert 2 == len(ai_agent.history)
    # When: A new agent is created with previous session
    ai_agent = OpenAIAgent(session=session)
    # And: The agent is asked for answer
    response = ai_agent.get_text_response("Who was his father?")
    # Then: The answer is connected with previous session
    assert "Mieszko" in response
    assert 2 == len(session.interactions)
    for m in ai_agent.history:
        print(m)
    assert 4 == len(ai_agent.history)


@pytest.mark.skip(reason="#TODO: Split this test into AI Model tests")
def test_agent_definition(ai_model: BaseAIModel):
    # Given: An agent
    ai_agent = BaseAIAgent(ai_model=ai_model, name="007", description="Agent 007")
    # And: A Supervisor
    ai_supervisor = BaseAISupervisor(ai_model=ai_model)
    # When: I get agent definition
    definition = ai_supervisor.get_agent_definition(ai_agent)
    if not isinstance(definition, Dict):
        definition = type(definition).to_dict(definition)
    # Then: I should get definition
    assert definition["name"] == "Agent__007"
    assert definition["description"] == "Agent 007"
    assert "parameters" in definition
    assert "properties" in definition["parameters"]
    assert "message" in definition["parameters"]["properties"]
    # assert "type" in definition["parameters"]["properties"]["message"]
    assert "description" in definition["parameters"]["properties"]["message"]
    assert "required" in definition["parameters"]
    # assert "message" in definition["parameters"]["required"]
