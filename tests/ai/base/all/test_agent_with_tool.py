import logging

from haintech.ai import (
    AIChatSession,
    BaseAIAgent,
    BaseAIModel,
)


def get_remaining_vacation_days(year: int):
    """Returns the number of remaining vacation days for the given year

    Args:
        year: Year for which to calculate the remaining vacation days
    Returns:
        Number of remaining vacation days
    """
    return 26


def get_remaining_home_office_days(year: int):
    """Returns the number of remaining home office days for the given year

    Args:
        year: Year for which to calculate the remaining home office days
    Returns:
        Number of remaining home office days
    """
    return 4


class HRAgent(BaseAIAgent):
    def __init__(self, ai_model: BaseAIModel, session: AIChatSession = None):
        super().__init__(
            ai_model=ai_model,
            description="HR Assistant",
            # context="You are a helpful HR assistant.",
            functions=[
                get_remaining_vacation_days,
                get_remaining_home_office_days,
            ],
            session=session,
        )


def test_agent_one_question(ai_model: BaseAIModel):
    logging.getLogger("haintech").setLevel(logging.DEBUG)
    # Given: An agent with session and tools
    session = AIChatSession()
    ai_agent = HRAgent(ai_model=ai_model, session=session)
    # When: I ask agent
    response = ai_agent.get_text_response(
        "How many vacation days do I have left in 2025?"
    )
    # Then: I should get answer
    assert "26" in response


def test_agent_with_acceptance(ai_model: BaseAIModel, session):
    # STEP: 1
    # =======
    # Given: An agent with session and tools
    ai_agent = HRAgent(ai_model=ai_model, session=session)
    # When: I ask agent
    response = ai_agent.get_response("How many vacation days do I have left in 2025?")
    # Then: I shult get function call
    assert 1 == len(response.tool_calls)
    # STEP: 2
    # =======
    # When: I accept function call
    response = ai_agent.accept_tools(response.tool_calls[0].id)
    # Then: I should get answer
    assert "26" in response.content
