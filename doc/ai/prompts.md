# Prompt Management & Execution System

## Overview

A structured system for managing AI prompts using Jinja2 templates and a robust execution layer that supports type-safe responses via Pydantic, including support for AI-driven image generation.

## Key Features

- **Template-Based Prompts**: Manage system and user prompts using Jinja2 templates.
- **Type-Safe Responses**: Parse and validate AI responses directly into Pydantic models.
- **Structured Output Transformation**: Map raw AI outputs to custom domain models using `BaseOutput`.
- **Image Generation Pipeline**: Generate images using an LLM to construct detailed prompts, which are then executed by an image generator.
- **Asynchronous Support**: Full async execution for both text and image generation tasks.
- **Robust Error Handling & Retries**: Built-in mechanisms to handle transient JSON formatting issues.

## Installation/Integration

To use this prompt management system in your project, ensure you have the required dependencies installed:

```bash
pip install pydantic jinja2
```

Include the prompt files in your project structure (e.g., under `./app/prompts/`) and import the classes from `haintech.ai.prompts`.

## 1. Prompt Structure

Prompts are organized in a directory-based registry. Each prompt set must be a directory containing:

- `system.jinja`: Jinja2 template for the system prompt.
- `user.jinja`: Jinja2 template for the user prompt.
- `output.py` (Optional): Defines a `BaseOutput` subclass for structured response transformation.

## 2. Core Components

### `PromptService`

Responsible for loading and rendering templates from the filesystem.

- **Initialization**: Requires a `root_path` to the prompts directory (usually `./app/prompts/`).
- **Dynamic Loading**: Automatically discovers prompt folders and optionally loads transformation logic from `output.py`.
- **Rendering**: Uses Jinja2 to inject variables into templates.

### `PromptExecutor`

Handles interactions with the `BaseAIModel` and parses responses into various formats.

- `execute()`: Returns a raw `AIChatResponse`.
- `execute_list()`: Parses JSON responses into a `list[str]`.
- `execute_typed(prompt_name, clazz, **kwargs)`: Returns an instance of the specified Pydantic model.
- `execute_typed_list()`: Returns a list of Pydantic models.
- **Retries**: Implements up to 3 attempts for typed execution to handle transient JSON formatting issues.

### `PromptExecutorImage`

An extension of `PromptExecutor` designed specifically for generating images. It uses an LLM to generate a detailed image prompt first, and then passes that prompt to an image generator to produce the final image.

- **Initialization**: Requires an `ai_model`, an optional `image_generator` (defaults to `GenAIImageGenerator` if not provided), a `prompt_service`, and an optional `interaction_logger`.
- `from_prompt_executor(prompt_executor, image_generator=None)`: A class method to easily instantiate `PromptExecutorImage` from an existing `PromptExecutor` instance.
- `execute_image_prompt_async(prompt_name, **kwargs)`: An asynchronous method that executes the specified prompt to obtain an image description, and then uses the image generator to generate the image. Returns a `BlobCreate[ImageGeneratedMetadata]` or `None`.

### `BaseOutput[T]`

An abstract base class used in `output.py` to define how an AI's raw JSON response should be transformed into a final Pydantic model `T`.

- `convert(**kwargs)`: Implement this method to map AI output fields to the target model, allowing for the injection of runtime variables.

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

## 3. Usage Examples

### Text Generation with Pydantic Validation

```python
from haintech.ai.prompts.prompt_service import PromptService
from haintech.ai.prompts.prompt_executor import PromptExecutor
from pydantic import BaseModel

class ArticleModel(BaseModel):
    title: str
    summary: str

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

### Image Generation Pipeline

The `PromptExecutorImage` allows you to generate images by first using an LLM to generate a detailed image prompt, and then passing that prompt to an image generator.

```python
import asyncio
from haintech.ai.prompts.prompt_service import PromptService
from haintech.ai.prompts.prompt_executor import PromptExecutor
from haintech.ai.prompts.prompt_executor_image import PromptExecutorImage

async def generate_image_example():
    # 1. Setup
    service = PromptService(root_path="./app/prompts")
    executor = PromptExecutor(ai_model, service)
    
    # 2. Create PromptExecutorImage from existing PromptExecutor
    image_executor = PromptExecutorImage.from_prompt_executor(
        prompt_executor=executor,
        image_generator=image_generator # Optional, defaults to GenAIImageGenerator
    )
    
    # 3. Execute image prompt asynchronously
    blob_create = await image_executor.execute_image_prompt_async(
        prompt_name="generate_image",
        object="dog"
    )
    
    if blob_create:
        print(f"Generated image content size: {len(blob_create.content)} bytes")
        print(f"Generated prompt: {blob_create.metadata.prompt}")
        print(f"Content type: {blob_create.metadata.content_type}")

asyncio.run(generate_image_example())
```

## 4. API Reference

### Classes & Interfaces

#### `PromptExecutorImage`
Inherits from `PromptExecutor`. Manages the orchestration of generating an image prompt via an LLM and executing it via an image generator.

##### Methods

*   `__init__(self, ai_model: BaseAIModel, image_generator: BaseImageGenerator | None, prompt_service: PromptService, interaction_logger: Callable[[AIModelInteraction], None] | None = None)`
    *   **Description**: Initializes the image prompt executor.
    *   **Parameters**:
        *   `ai_model` (`BaseAIModel`): The AI model used to generate the image prompt.
        *   `image_generator` (`BaseImageGenerator | None`): The image generator. If `None`, defaults to `GenAIImageGenerator`.
        *   `prompt_service` (`PromptService`): The prompt service to load and render templates.
        *   `interaction_logger` (`Callable[[AIModelInteraction], None] | None`): Optional callback for logging interactions.
    *   **Returns**: `None`

*   `from_prompt_executor(cls, prompt_executor: PromptExecutor, image_generator: BaseImageGenerator | None = None) -> Self`
    *   **Description**: Factory method to create a `PromptExecutorImage` from an existing `PromptExecutor`.
    *   **Parameters**:
        *   `prompt_executor` (`PromptExecutor`): An existing prompt executor instance.
        *   `image_generator` (`BaseImageGenerator | None`): Optional image generator.
    *   **Returns**: `PromptExecutorImage`

*   `async def execute_image_prompt_async(self, prompt_name: str, **kwargs) -> BlobCreate[ImageGeneratedMetadata] | None`
    *   **Description**: Executes the specified prompt to generate an image prompt, then generates the image.
    *   **Parameters**:
        *   `prompt_name` (`str`): The name of the prompt directory.
        *   `kwargs`: Variables to render in the prompt templates.
    *   **Returns**: `BlobCreate[ImageGeneratedMetadata] | None` - The generated image blob with metadata, or `None` if no prompt was generated.

#### `ImageGeneratorPrompt`
A Pydantic model representing the structured output expected from the LLM when generating an image prompt.

##### Fields
*   `image_prompt` (`str | None`): The detailed prompt text to be passed to the image generator.

#### `ImageGeneratedMetadata`
Inherits from `BaseBlobMetadata`. Metadata attached to the generated image blob.

##### Fields
*   `prompt` (`str | None`): The prompt used to generate the image.

## 5. Error Handling

- **Invalid Response Formats**: If the AI model returns a response that does not match the expected Pydantic model or schema, `execute_typed` and `execute_typed_async` will raise a `ValueError`.
- **Missing Prompts**: If a requested prompt directory or its required templates (`user.jinja`) do not exist, `PromptService.load_prompt` raises a `ValueError`.
- **Empty Image Prompts**: In `execute_image_prompt_async`, if the LLM fails to generate an image prompt (i.e., `image_prompt` is empty or `None`), the method gracefully returns `None` without invoking the image generator.
