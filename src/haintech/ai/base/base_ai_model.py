from abc import ABC, abstractmethod
from inspect import Parameter, signature
from types import UnionType
from typing import Any, Callable, Dict, Iterator, Literal

from openai.types.shared_params import FunctionDefinition

from ..model import (
    AIChatResponse,
    AIFunction,
    AIFunctionParameter,
    AIModelInteraction,
    AIModelInteractionMessage,
    AIPrompt,
    RAGItem,
)


class BaseAIModel(ABC):
    @abstractmethod
    def get_chat_response(
        self,
        message: AIModelInteractionMessage = None,
        prompt: AIPrompt = None,
        context: str | AIPrompt = None,
        history: Iterator[AIModelInteractionMessage] = None,
        functions: Dict[Callable, Any] = None,
        interaction_logger: Callable[[AIModelInteraction], None] = None,
        response_format: Literal["text", "json"] = "text",
    ) -> AIChatResponse:
        """Return chat response from LLM

        Args:
            message: message to send to LLM,
            context: context to send to LLM,
            history: history of previous messages
            functions: functions to call
            interaction_logger: function to log interaction
        Returns:
            LLM response
        """
        pass

    @classmethod
    def _prompt_to_str(cls, prompt: str | AIPrompt) -> str:
        """Creates a string representation of AIPrompt object.

        It can be overriden if other models expect different representation.
        Args:
            prompt: AIPrompt object
        Returns:
            string representation of AIPrompt object
        """
        if isinstance(prompt, str):
            return prompt
        ret = ""
        if prompt.persona:
            ret += f"Persona: {prompt.persona}\n"
        if prompt.objective:
            ret += f"Objective: {prompt.objective}\n"
        if prompt.instructions:
            ret += f"Instructions: {prompt.instructions}\n"
        if prompt.constraints:
            ret += f"Constraints: {prompt.constraints}\n"
        if prompt.context:
            ret += f"Context: {prompt.context}\n"
        if prompt.documents:
            ret += "Documents:\n"
            for d in prompt.documents:
                if isinstance(d, RAGItem):
                    r: RAGItem = d
                    ret += f"\nDocument: {r.title}\n{r.content}\n"
                else:
                    ret += f"\nDocument:\n{d}\n"
        if prompt.examples:
            ret += f"Examples: {prompt.examples}\n"
        if prompt.recap:
            ret += f"Recap: {prompt.recap}\n"
        return ret

    @classmethod
    def model_function_definition(cls, ai_function: AIFunction) -> FunctionDefinition:
        """Creates an OpenAI FunctionDefinition from a Python callable.

        It can be overriden if other models expect different definition
        Args:
            ai_function: AIFunction object
        Returns:
            A FunctionDefinition object representing the callable.  Returns None if input is invalid.
        Raises:
            TypeError: If input is not a callable.
            ValueError: If the function signature is invalid or missing required information.
        """
        parameters: Dict[str, Any] = {
            "type": "object",
            "properties": {},
            "required": [],
        }
        for param in ai_function.parameters:
            param_name = param.name
            param_description = param.description
            parameters["properties"][param_name] = {
                "type": (
                    # Work around for not required parameters - union with null
                    "string" if param.required else ["string", "null"]
                ),
                "description": param_description,
            }  # Default type is string, could be improved.
            parameters["required"].append(param_name)

        parameters["additionalProperties"] = (
            False  # Ensure no extra properties are allowed
        )

        return FunctionDefinition(
            name=ai_function.name,
            description=ai_function.description,
            parameters=parameters,
            strict=True,
        )

    @classmethod
    def prepare_function_definition(
        cls,
        func: Callable[..., Any],
        name: str = None,
        description: str = None,
    ) -> FunctionDefinition:
        """Creates an OpenAI FunctionDefinition from a Python callable.

        It can be overriden if other models expect different definition
        Args:
            func: The Python callable to create the FunctionDefinition from.
            name: the name of the function
            description: description of the function
        Returns:
            A FunctionDefinition object representing the callable.  Returns None if input is invalid.
        Raises:
            TypeError: If input is not a callable.
            ValueError: If the function signature is invalid or missing required information.
        """
        ai_function = cls.create_ai_function(func)
        if name:
            ai_function.name = name
        if description:
            ai_function.description = description
        return cls.model_function_definition(ai_function)

    @classmethod
    def create_ai_function(cls, func: Callable[..., Any]) -> AIFunction:
        """Creates an AIFunction from a Python callable.

        Args:
            func: The Python callable to create the FunctionDefinition from.
        Returns:
            A AIFunction object representing the callable.  Returns None if input is invalid.
        Raises:
            TypeError: If input is not a callable.
            ValueError: If the function signature is invalid or missing required information.
        """
        if not callable(func):
            raise TypeError("Input must be a callable.")

        sig = signature(func)
        docstring = func.__doc__ or ""  # Handle cases where docstring is None

        name = func.__name__
        description = docstring.strip().split("\n")[0]
        try:
            return_type = (
                sig.return_annotation.__name__
                if sig.return_annotation != Parameter.empty
                else "str"
            )
        except AttributeError:
            return_type = None

        parameters = []
        for param in sig.parameters.values():
            if (
                param.kind == Parameter.VAR_KEYWORD
                or param.kind == Parameter.VAR_POSITIONAL
            ):
                raise ValueError(
                    "Function cannot have variable keyword or positional arguments."
                )

            param_name = param.name
            param_description = ""  # Default description

            # Attempt to extract description from docstring (this is very basic and could be improved).
            docstring_lines = docstring.strip().split("\n")
            for line in docstring_lines[
                1:
            ]:  # Skip the first line (already in description)
                if line.strip().startswith(param_name + ":"):
                    param_description = line.strip().split(":", 1)[1].strip()
                    break
            parameters.append(
                AIFunctionParameter(
                    name=param_name,
                    description=param_description,
                    type=str(param.annotation)
                    if isinstance(param.annotation, UnionType)
                    else param.annotation.__name__
                    if param.annotation != Parameter.empty
                    else "str",
                    required=param.default == Parameter.empty,
                )
            )

        return AIFunction(
            name=name,
            description=description,
            parameters=parameters,
            return_type=return_type,
        )
