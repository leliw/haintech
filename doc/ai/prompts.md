# Prompt Management & Execution System

## Overview

A structured system for managing AI prompts using Jinja2 templates and a robust execution layer that supports type-safe responses via Pydantic.

## 1. Prompt Structure

Prompts are organized in a directory-based registry. Each prompt set must be a directory containing:

- `system.jinja`: Jinja2 template for the system prompt.
- `user.jinja`: Jinja2 template for the user prompt.
- `output.py` (Optional): Defines a `BaseOutput` subclass for structured response transformation.

## 2. Core Components

### `PromptService`

Responsible for loading and rendering templates from the filesystem.

- **Initialization**: Requires a `root_path` to the prompts directory (usually `./app/prompts/`)
- **Dynamic Loading**: Automatically discovers prompt folders and optionally loads transformation logic from `output.py`.
- **Rendering**: Uses Jinja2 to inject variables into templates.

### `PromptExecutor`

Handles interactions with the `BaseAIModel` and parses responses into various formats.

- **`execute()`**: Returns a raw `AIChatResponse`.
- **`execute_list()`**: Parses JSON responses into a `list[str]`.
- **`execute_typed(prompt_name, clazz, **kwargs)`**: Returns an instance of the specified Pydantic model.
- **`execute_typed_list()`**: Returns a list of Pydantic models.
- **Retries**: Implements up to 3 attempts for typed execution to handle transient JSON formatting issues.

### `BaseOutput[T]`

An abstract base class used in `output.py` to define how an AI's raw JSON response should be transformed into a final Pydantic model `T`.

- **`convert(**kwargs)`**: Implement this method to map AI output fields to the target model, allowing for the injection of runtime variables.

```python
from haintech.ai.prompts.prompt_model import BaseOutput
from tests.ai.prompts.model import InfoPageCreate


class Output(BaseOutput[InfoPageCreate]):
    content: str

    def convert(self, **kwargs) -> InfoPageCreate:
        return InfoPageCreate(
            order=0,
            type="info",
            **kwargs,
            **self.model_dump(),
        )
```

## 3. Usage Example

```python
# 1. Setup
service = PromptService(root_path="./app/prompts")
executor = PromptExecutor(ai_model, service)

# 2. Execution with Pydantic validation
result = executor.execute_typed(
    prompt_name="generate_article",
    clazz=ArticleModel,
    topic="AI Trends"
)
```
