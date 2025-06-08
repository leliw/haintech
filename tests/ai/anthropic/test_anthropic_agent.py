from typing import Dict

from haintech.ai import BaseAIAgent, BaseAISupervisor
from haintech.ai.anthropic import AnthropicAIModel


def test_agent_definition():
    # Given: An agent
    ai_model = AnthropicAIModel()
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
    assert "input_schema" in definition
    assert "properties" in definition["input_schema"]
    assert "message" in definition["input_schema"]["properties"]
    assert "description" in definition["input_schema"]["properties"]["message"]
    assert "required" in definition["input_schema"]
