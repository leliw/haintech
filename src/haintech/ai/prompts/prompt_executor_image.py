from typing import Callable, Self

from ampf.base import BaseBlobMetadata, BlobCreate
from haintech.ai import AIModelInteraction, BaseAIModel, BaseImageGenerator
from pydantic import BaseModel

from .prompt_executor import PromptExecutor
from .prompt_service import PromptService


class ImageGeneratorPrompt(BaseModel):
    image_prompt: str | None


class ImageGeneratedMetadata(BaseBlobMetadata):
    prompt: str | None = None



class PromptExecutorImage(PromptExecutor):
    def __init__(
        self,
        ai_model: BaseAIModel,
        image_generator: BaseImageGenerator | None,
        prompt_service: PromptService,
        interaction_logger: Callable[[AIModelInteraction], None] | None = None,
    ):
        super().__init__(ai_model, prompt_service, interaction_logger)
        if image_generator is None:
            from haintech.ai.google_genai import GenAIImageGenerator
            self.image_generator = GenAIImageGenerator()
        else:
            self.image_generator = image_generator

    @classmethod
    def from_prompt_executor(
        cls, prompt_executor: PromptExecutor, image_generator: BaseImageGenerator | None = None
    ) -> Self:
        return cls(
            ai_model=prompt_executor.ai_model,
            image_generator=image_generator,
            prompt_service=prompt_executor.prompt_service,
            interaction_logger=prompt_executor.interaction_logger,
        )

    async def execute_image_prompt_async(self, prompt_name: str, **kwargs) -> BlobCreate[ImageGeneratedMetadata] | None:
        result = await self.execute_typed_async(prompt_name, ImageGeneratorPrompt, **kwargs)
        image_prompt = result.image_prompt if result else None
        if not image_prompt:
            return None
        async for blob_create in self.image_generator.generate_async(image_prompt):
            metadata_dict = blob_create.metadata.model_dump() if blob_create.metadata else {}
            return BlobCreate(
                content=blob_create.content,
                metadata=ImageGeneratedMetadata(**metadata_dict, prompt=image_prompt),
            )
        return None
