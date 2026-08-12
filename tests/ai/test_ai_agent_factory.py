from haintech.ai.ai_agent_factory import AIAgentFactory
from haintech.ai.base.base_ai_agent_async import BaseAIAgentAsync
from haintech.ai.google_genai.google_ai_model import GoogleAIModel


def test_create_agent():
    # Given: A factory
    factory = AIAgentFactory.create({"google": GoogleAIModel})
    # When: Create agent
    agent = factory.create_ai_agent_async("google/gemini-3.5-flash-lite")
    # Then: An agent is created
    assert isinstance(agent, BaseAIAgentAsync)

def add(a: int, b: int) -> int:
    return a + b

def sub(a: int, b: int) -> int:
    return a - b

def test_get_function_names():
    # Given: A factory with functions
    factory = AIAgentFactory.create({"google": GoogleAIModel}, functions=[add, sub])
    # When: Get function names
    names = factory.get_function_names()
    # Then: They are as passed
    assert "add" in names
    assert "sub" in names

def test_create_agent_with_factory_func():
    # Given: A factory with a function
    factory = AIAgentFactory.create({"google": GoogleAIModel}, functions=[add])
    # When: Create agent
    agent = factory.create_ai_agent_async("google/gemini-3.5-flash-lite")
    # Then: An agent is created
    assert isinstance(agent, BaseAIAgentAsync)
    # And: The agent can use a function
    assert add in agent.functions

def test_create_agent_without_factory_func():
    # Given: A factory with a function
    factory = AIAgentFactory.create({"google": GoogleAIModel}, functions=[add])
    # When: Create agent with an empty function list
    agent = factory.create_ai_agent_async("google/gemini-3.5-flash-lite", functions=[])
    # Then: An agent is created
    assert isinstance(agent, BaseAIAgentAsync)
    # And: The agent can't use a function
    assert add not in agent.functions

def test_create_agent_with_func():
    # Given: A factory without functions
    factory = AIAgentFactory.create({"google": GoogleAIModel})
    # When: Create agent with a function
    agent = factory.create_ai_agent_async("google/gemini-3.5-flash-lite", functions=[add])
    # Then: An agent is created
    assert isinstance(agent, BaseAIAgentAsync)
    # And: The agent can use a function
    assert add in agent.functions

def test_create_agent_with_selected_func():
    # Given: A factory with functions
    factory = AIAgentFactory.create({"google": GoogleAIModel}, functions=[add, sub])
    # When: Create agent with a selected function
    agent = factory.create_ai_agent_async("google/gemini-3.5-flash-lite", functions=["sub"])
    # Then: An agent is created
    assert isinstance(agent, BaseAIAgentAsync)
    # And: The agent can use a function
    assert add not in agent.functions
    assert sub in agent.functions
