# AITaskExecutor

Wraps an AI model that it can be used like a function.

## Constructor

Arguments:

* ai_model: AI model.
* system_instructions: System instructions.
* prompt: Prompt (f-string)
* interaction_logger: Optional[Callable[[AIModelInteraction], None]] = None

## Methods

### create_from_definition()

Class method - creates an AI task executor from a task definition.

Arguments:

* ai_model: AI model.
* task_definition: Task definition.

Returns:

* AITaskExecutor: AI task executor

### execute()

Execute task.

Argument:

* kwargs: Prompt arguments.

Returns:

* str: AI Model response.

## Use cases

### Prepare the name for the feature

```python
te = AITaskExecutor(
    ai_model=ai_model,
    system_instructions=AIPrompt(
        persona="You are experienced developer",
        objective="Prepare the name for the feature",
        instructions="Use snake_case.",
        context="It will be used as a folder name contais code.",
        constraints="Use english language and plural form.",
        examples=[
            "Q: Zarządzanie użytkownikami\nA: users",
            "Q: CRUD dla książek\nA: books",
            "Q: Obsługa projektów\nA: projects",
        ],
    ),
    prompt="{prompt}",
)
ret = te.execute(prompt="Zarządzanie użytkownikami")
assert isinstance(ret, str)
assert "users" == ret
```

### Store interaction in a session

```python
    session = AISupervisorSession()
    te = AITaskExecutor(
        ai_model=GoogleAIModel(parameters={"temperature": 0}),
        system_instructions="You are helpful assistant.",
        prompt="{prompt}",
        interaction_logger=session.create_agent_session("TaskExecutor").add_interaction
    )
```
