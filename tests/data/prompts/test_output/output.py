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
