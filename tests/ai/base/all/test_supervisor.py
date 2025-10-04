import pytest

from haintech.ai import (
    AISupervisorSession,
    BaseAIAgent,
    BaseAIModel,
    BaseAISupervisor,
)
from haintech.ai.google_generativeai.google_ai_model import GoogleAIModel
from haintech.ai.open_ai.open_ai_model import OpenAIModel


def test_agents_collaboration(ai_model: BaseAIModel):
    # Given : Two specialised agents and one manager agent
    hist_agent = BaseAIAgent(
        ai_model=ai_model,
        name="Historian",
        description="Historian assistant. Answer any question about history.",
        context="You are a helpful historian assistant.",
    )
    geo_agent = BaseAIAgent(
        ai_model=ai_model,
        name="Geographer",
        description="Geographer assistant. Answer any question about geography.",
        context="You are a helpful geographer assistant.",
    )
    man_session = AISupervisorSession()
    man_agent = BaseAISupervisor(
        ai_model=ai_model,
        context="You are a helpful assistant.",
        agents=[hist_agent, geo_agent],
        session=man_session,
    )
    # When: The manager is asked for something specific
    response = man_agent.get_text_response("Who was the first US president?")
    # Then: The answer is correct
    assert "George" in response
    # And: There were 3 interactions (manager - historian - manager)
    assert 3 == len(man_session.interactions)
    # And: The manager session contains 2 history item + message + response (user - assistant - tool - assistant )
    history = man_session.interactions[-1][1].history
    print(str(man_session))
    assert 2 == len(history)


def test_supervisor_with_acceptance(
    ai_model: BaseAIModel,
    hr_agent: BaseAIAgent,
    tm_session: AISupervisorSession,
):
    # STEP: 1
    # =======
    # Given: A supervisor with HR agent
    ai_supervisor = BaseAISupervisor(
        ai_model=ai_model,
        description="Company Assistant",
        context="You are a helpful company assistant. Ask other agents if needed.",
        agents=[hr_agent],
        session=tm_session,
    )
    # When: I ask agent
    response = ai_supervisor.get_response(
        "How many vacation days do I have left in 2025?"
    )
    # Then: I should get agent call
    print(response)
    assert response.tool_calls
    assert 1 == len(response.tool_calls)
    assert "Agent__HRAgent" in response.tool_calls[0].function_name
    # STEP: 2
    # =======
    # When: I accept agent call
    response = ai_supervisor.accept_tools([tc.id for tc in response.tool_calls if tc.id])
    # Then: I should get function call
    assert response.tool_calls
    assert 1 <= len(response.tool_calls)
    # STEP: 2
    # =======
    # When: I accept function call
    response = ai_supervisor.accept_tools([tc.id for tc in response.tool_calls if tc.id])
    # Then: I should get answer
    assert response.content
    assert "26" in response.content


def test_supervisor_with_without_acceptance_agent_call(
    ai_model: BaseAIModel,
    hr_agent: BaseAIAgent,
    tm_session: AISupervisorSession,
):
    # STEP: 1
    # =======
    # Given: A supervisor with HR agent
    ai_supervisor = BaseAISupervisor(
        ai_model=ai_model,
        description="Company Assistant",
        context="You are a helpful company assistant. Ask other agents if needed.",
        agents=[hr_agent],
        session=tm_session,
    )
    # When: I ask agent
    response = ai_supervisor.get_response(
        "How many vacation days do I have left in 2025?"
    )
    # Then: I should get agent call
    assert response.tool_calls
    assert 1 == len(response.tool_calls)
    assert "Agent__HRAgent" in response.tool_calls[0].function_name
    # STEP: 2
    # =======
    # When: I don't accept agent call
    response = ai_supervisor.accept_tools([])
    # Then: I should get answer
    assert not response.tool_calls
    # And: Assistant "cannot" answer or "need" some information, or similar negative phrasing
    negative_keywords = ["cannot", "need", "don't have", "can't", "couldn't", "unable"]
    assert response.content and any(keyword in response.content for keyword in negative_keywords)


@pytest.fixture
def agent_ai_model(ai_model: BaseAIModel) -> BaseAIModel:
    """Return agent with different model"""
    if isinstance(ai_model, OpenAIModel):
        return GoogleAIModel(parameters={"temperature": 0})
    else:
        return OpenAIModel(parameters={"temperature": 0})


def test_agent_with_different_model(ai_model: BaseAIModel, agent_ai_model: BaseAIModel):
    # Given : Two specialised agents and one manager agent with different model
    hist_agent = BaseAIAgent(
        ai_model=agent_ai_model,
        name="Historian",
        description="Historian assistant. Answer any question about history.",
        context="You are a helpful historian assistant.",
    )
    geo_agent = BaseAIAgent(
        ai_model=agent_ai_model,
        name="Geographer",
        description="Geographer assistant. Answer any question about geography.",
        context="You are a helpful geographer assistant.",
    )
    man_session = AISupervisorSession()
    man_agent = BaseAISupervisor(
        ai_model=ai_model,
        context="You are a helpful assistant.",
        agents=[hist_agent, geo_agent],
        session=man_session,
    )
    # When: The manager is asked for something specific
    response = man_agent.get_text_response("Who was the first US president?")
    # Then: The answer is correct
    assert "George" in response
    # And: There were 3 interactions (manager - historian - manager)
    assert 3 == len(man_session.interactions)
    # And: The manager session contains 2 history item + message + response (user - assistant - tool - assistant )
    history = man_session.interactions[-1][1].history
    print(str(man_session))
    assert 2 == len(history)
