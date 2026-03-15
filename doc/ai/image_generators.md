# Image Generators

the components for generating images using Google GenAI and OpenAI, returning `BlobCreate` objects for consistent data handling.

## BaseImageGenerator

An abstract base class defining the interface for all image generators.

### Methods

* `generate(prompt: str)` - Synchronous generator yielding `BlobCreate` objects.
* `generate_async(prompt: str)` - Asynchronous generator yielding `BlobCreate` objects.

## Implementations

### GenAIImageGenerator (Google)

Uses the Google GenAI SDK. Default model: `gemini-2.5-flash-image`.

### OpenAIImageGenerator (OpenAI)

Uses the OpenAI SDK. Default model: `gpt-image-1-mini`. Automatically handles `b64_json` response format for DALL-E models.

## Usage Examples

### Synchronous Generation (Google GenAI)

```python
from haintech.ai.google_genai import GenAIImageGenerator

# Initialize with API key
generator = GenAIImageGenerator(api_key="your_google_api_key")

# Generate images (returns a generator of BlobCreate)
for blob in generator.generate("A futuristic city in the clouds, digital art"):
    print(f"Generated image type: {blob.metadata.content_type}")
    # Access raw bytes via blob.content
    with open("output.png", "wb") as f:
        f.write(blob.content)
```

### Asynchronous Generation (OpenAI)

```python
import asyncio
from haintech.ai.open_ai import OpenAIImageGenerator

async def main():
    # Initialize with specific model
    generator = OpenAIImageGenerator(model_name="dall-e-3", api_key="your_openai_api_key")

    # Use asynchronous generator
    async for blob in generator.generate_async("A cute robot drinking coffee"):
        print(f"Received image: {blob.metadata.content_type}")
        # blob.content contains the image data

asyncio.run(main())
```

### Class-level Configuration

Both generators support a class-level `setup()` method to configure the client once for the entire application.

```python
from haintech.ai.open_ai import OpenAIImageGenerator

# Configure once
OpenAIImageGenerator.setup(api_key="your_api_key")

# Create instances without passing the key again
generator = OpenAIImageGenerator(model_name="dall-e-2")
```

## Integration with ampf

The generators return `ampf.base.BlobCreate` objects, making it easy to upload results directly to storage:

```python
from ampf.base import Blob
# ... generator setup ...

blob_create = next(generator.generate("A black dog"))
blob_create.name = "dog_image.png"

# Upload using ampf storage
blob_storage.upload(Blob.create(blob_create))
```
