from pydantic import BaseModel


class OpenAIParameters(BaseModel):
    temperature: float | None  = 1
    max_tokens: int | None = 2048
    top_p: float | None = 1
    frequency_penalty: float | None = 0
    presence_penalty: float | None = 0

    def get_for_model(self, model_name: str) -> dict:
        if model_name.startswith("gpt-5"):
            return self.model_dump(exclude_none=True, exclude={"max_tokens", "temperature"})
        else:
            return self.model_dump(exclude_none=True)

class ResponsesAIParameters(BaseModel):
    temperature: float | None  = 1
    top_p: float | None = 1

    def get_for_model(self, model_name: str) -> dict:
        if model_name.startswith("gpt-5-") or model_name == "gpt-5":
            return self.model_dump(exclude_none=True, exclude={"temperature"})
        else:
            return self.model_dump(exclude_none=True)
