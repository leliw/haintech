from haintech.ai import (
    AIFunctionParameter,
    AIPrompt,
    AITask,
    BaseAIAgent,
    BaseAIModel,
)


def test_agent_with_ai_task(ai_model: BaseAIModel, session):
    # Given: AI Agent with AI Task
    ai_agent = BaseAIAgent(
        ai_model=ai_model,
        context="You are a helpful python developer. Use available tools.",
        session=session,
    )
    td = AITask(
        name="create_feature_name",
        description="Creates feature name from its description it can be used as folder name and so on.",
        parameters=[
            AIFunctionParameter(
                name="feature", description="Feature description", type="str"
            )
        ],
        return_type="str",
        return_description="Feature name",
        system_instructions=AIPrompt(
            persona="You are experienced developer",
            objective="Prepare the name for the feature",
            instructions="Use snake_case.",
            context="It will be used as a folder name contais code.",
            constraints="Use english language and plural form.",
            examples=[
                "Q: Zarządzanie użytkownikami\nA: users",
                "Q: User Management\nA: users",
                "Q: Zarządzanie projektami\nA: projects",
                "Q: Project Management\nA: projectsQ: CRUD dla książek\nA: books",
                "Q: Obsługa projektów\nA: projects",
            ],
        ),
        prompt="{feature}",
    )
    ai_agent.add_ai_task(td)

    response = ai_agent.get_response(
        "Zapropounuj nazwę katalogu dla kodu zarządzania użytkownikami"
    )
    # Then: I shult get function call
    print(response)
    assert 1 == len(response.tool_calls)
    # STEP: 2
    # =======
    # When: I accept function call
    response = ai_agent.accept_tools(response.tool_calls[0].id)
    # Then: I should get answer
    assert "users" in response.content
