from typing import Optional
from pydantic import BaseModel


class OpenAIParameters(BaseModel):
    temperature: Optional[float] = 1
    max_tokens: Optional[int] = 2048
    top_p: Optional[float] = 1
    frequency_penalty: Optional[float] = 0
    presence_penalty: Optional[float] = 0
