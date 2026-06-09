import json
from abc import ABC, abstractmethod
from inspect import Parameter, signature
from types import UnionType
from typing import Any, Callable, Dict, Iterable, Literal, Optional, Sequence, Type

from ampf.base import Blob, BlobLocation
from openai.types.shared_params import FunctionDefinition
from pydantic import BaseModel

from haintech.helpers import get_inner_type, is_list_type

from ..model import (
    AIChatResponse,
    AIContext,
    AIFunction,
    AIFunctionParameter,
    AIModelInteraction,
    AIModelInteractionMessage,
    AIPrompt,
    RAGItem,
)


class BaseAIModel(ABC):
    def get_response(
        self,
        message: str,
        response_format: Literal["text", "json"] | dict = "text",
        system_prompt: str | None = None,
        blob_locations: list[BlobLocation] | None = None,
        blobs: list[Blob] | None = None,
    ) -> str:
        response = self.get_chat_response(
            system_prompt=system_prompt,
            message=AIModelInteractionMessage(
                role="user", content=message, blob_locations=blob_locations or [], blobs=blobs
            ),
            response_format=response_format,
        )
        if response.content is None:
            raise ValueError("AI model returned empty content")
        return response.content

    async def get_response_async(
        self,
        message: str,
        response_format: Literal["text", "json"] | dict = "text",
        system_prompt: str | None = None,
        blob_locations: list[BlobLocation] | None = None,
        blobs: list[Blob] | None = None,
    ) -> str:
        response = await self.get_chat_response_async(
            system_prompt=system_prompt,
            message=AIModelInteractionMessage(
                role="user",
                content=message,
                blob_locations=blob_locations or [],
                blobs=blobs,
            ),
            response_format=response_format,
        )
        if response.content is None:
            raise ValueError("AI model returned empty content")
        return response.content

    def _prepare_response_format(
        self,
        response_format: Literal["text", "json"]
        | Type[Sequence[BaseModel | str | int | float | bool]]
        | Type[BaseModel] = "text",
    ) -> dict:
        if response_format == "text":
            ret = {"type": "text"}
        elif response_format == "json":
            ret = {"type": "json_object"}
        elif isinstance(response_format, type) and issubclass(response_format, BaseModel):
            ret = {
                "type": "json_schema",
                "json_schema": {
                    "name": response_format.__name__,
                    "schema": response_format.model_json_schema(),
                },
            }
        elif is_list_type(response_format):
            inner_type = get_inner_type(response_format)
            if inner_type is str:
                inner_ret = {"type": "array", "items": {"type": "string"}}
            elif inner_type is int:
                inner_ret = {"type": "array", "items": {"type": "integer"}}
            elif inner_type is float:
                inner_ret = {"type": "array", "items": {"type": "number"}}
            elif inner_type is bool:
                inner_ret = {"type": "array", "items": {"type": "boolean"}}
            elif issubclass(inner_type, BaseModel):
                inner_ret = {"type": "array", "items": inner_type.model_json_schema()}
            else:
                raise ValueError(f"Unsupported inner response format: {inner_type}")
            ret = {
                "type": "json_schema",
                "json_schema": {
                    "name": "list_wrapper",
                    "schema": {
                        "type": "object",
                        "properties": {"list": inner_ret},
                        "additionalProperties": False,
                    },
                },
            }
        else:
            raise ValueError(f"Unsupported response format: {response_format}")
        return ret

    def get_response_typed[T: BaseModel](
        self,
        message: str,
        response_format: Type[T],
        system_prompt: str | None = None,
        blob_locations: list[BlobLocation] | None = None,
        blobs: list[Blob] | None = None,
    ) -> T:
        response = self.get_response(
            message,
            self._prepare_response_format(response_format),
            system_prompt,
            blob_locations=blob_locations,
            blobs=blobs,
        )
        return response_format.model_validate_json(response)

    def get_response_list_typed[T: BaseModel](
        self,
        message: str,
        response_format: Type[T],
        system_prompt: str | None = None,
        blob_locations: list[BlobLocation] | None = None,
        blobs: list[Blob] | None = None,
    ) -> list[T]:
        response = self.get_response(
            message,
            self._prepare_response_format(list[response_format]),
            system_prompt,
            blob_locations=blob_locations,
            blobs=blobs,
        )
        list_wrapper = json.loads(response)
        return [response_format.model_validate(j) for j in list_wrapper["list"]]

    def get_response_list[T: str | int | float | bool](
        self,
        message: str,
        type: Type[T] = str,
        system_prompt: str | None = None,
        blob_locations: list[BlobLocation] | None = None,
        blobs: list[Blob] | None = None,
    ) -> list[T]:
        response = self.get_response(
            message,
            self._prepare_response_format(list[type]),
            system_prompt,
            blob_locations=blob_locations,
            blobs=blobs,
        )
        list_wrapper = json.loads(response)
        return list_wrapper["list"]

    async def get_response_typed_async[T: BaseModel](
        self,
        message: str,
        response_format: Type[T],
        system_prompt: str | None = None,
        blob_locations: list[BlobLocation] | None = None,
        blobs: list[Blob] | None = None,
    ) -> T:
        response = await self.get_response_async(
            message,
            self._prepare_response_format(response_format),
            system_prompt,
            blob_locations=blob_locations,
            blobs=blobs,
        )
        return response_format.model_validate_json(response)

    async def get_response_list_typed_async[T: BaseModel](
        self,
        message: str,
        response_format: Type[T],
        system_prompt: str | None = None,
        blob_locations: list[BlobLocation] | None = None,
        blobs: list[Blob] | None = None,
    ) -> list[T]:
        response = await self.get_response_async(
            message,
            self._prepare_response_format(list[response_format]),
            system_prompt,
            blob_locations=blob_locations,
            blobs=blobs,
        )
        list_wrapper = json.loads(response)
        return [response_format.model_validate(j) for j in list_wrapper["list"]]

    async def get_response_list_async[T: str | int | float | bool](
        self,
        message: str,
        type: Type[T] = str,
        system_prompt: str | None = None,
        blob_locations: list[BlobLocation] | None = None,
        blobs: list[Blob] | None = None,
    ) -> list[T]:
        response = await self.get_response_async(
            message,
            self._prepare_response_format(list[type]),
            system_prompt,
            blob_locations=blob_locations,
            blobs=blobs,
        )
        list_wrapper = json.loads(response)
        return list_wrapper["list"]

    @abstractmethod
    def get_chat_response(
        self,
        system_prompt: Optional[str | AIPrompt] = None,
        history: Optional[Iterable[AIModelInteractionMessage]] = None,
        context: Optional[AIContext] = None,
        message: Optional[AIModelInteractionMessage] = None,
        functions: Optional[Dict[Callable, Any]] = None,
        interaction_logger: Optional[Callable[[AIModelInteraction], None]] = None,
        response_format: Literal["text", "json"] | dict = "text",
    ) -> AIChatResponse:
        """Return chat response from LLM

        Args:
            system_prompt: System prompt for LLM
            history: Chat history
            context: Context for LLM
            message: Current message
            functions: List of functions to use
            interaction_logger: Callback for logging interactions
            response_format: Response format, "text" or "json"
        Returns:
            LLM response
        """
        pass

    async def get_chat_response_async(
        self,
        system_prompt: Optional[str | AIPrompt] = None,
        history: Optional[Iterable[AIModelInteractionMessage]] = None,
        context: Optional[AIContext] = None,
        message: Optional[AIModelInteractionMessage] = None,
        functions: Optional[Dict[Callable, Any]] = None,
        interaction_logger: Optional[Callable[[AIModelInteraction], None]] = None,
        response_format: Literal["text", "json"] | dict = "text",
    ) -> AIChatResponse:
        return self.get_chat_response(
            message=message,
            system_prompt=system_prompt,
            history=history,
            context=context,
            functions=functions,
            interaction_logger=interaction_logger,
            response_format=response_format,
        )

    @classmethod
    def _prompt_to_str(cls, prompt: str | AIPrompt) -> str:
        """Creates a string representation of AIPrompt object.

        It can be overriden if other models expect different representation.
        Args:
            prompt: AIPrompt object
        Returns:
            string representation of AIPrompt object
        """
        if not prompt or isinstance(prompt, str):
            return prompt
        ret = ""
        if prompt.persona:
            ret += f"Persona: {prompt.persona}\n\n"
        if prompt.objective:
            ret += f"Objective: {prompt.objective}\n\n"
        if prompt.instructions:
            ret += f"Instructions: {prompt.instructions}\n\n"
        if prompt.constraints:
            ret += f"Constraints: {prompt.constraints}\n\n"
        if prompt.context:
            ret += f"Context: {prompt.context}\n\n"
        if prompt.documents:
            ret += "Documents:\n"
            for d in prompt.documents:
                if isinstance(d, RAGItem):
                    r: RAGItem = d
                    ret += f"\nDocument: {r.title}\n{r.content}\n"
                else:
                    ret += f"\nDocument:\n{d}\n"
            ret += "\n"
        if prompt.output_format:
            ret += f"Output format: {prompt.output_format}\n\n"
        if prompt.examples:
            ret += "Examples:\n\n"
            for e in prompt.examples:
                ret += f"{e}\n\n"
            ret += "\n"
        if prompt.recap:
            ret += f"Recap: {prompt.recap}\n\n"
        return ret

    @classmethod
    def _context_to_str(cls, context: AIContext) -> str:
        ret = f"{context.context}\n" if context.context else "Document context:\n"
        for i, doc in enumerate(context.documents):
            if isinstance(doc, RAGItem):
                ret += f"Document {i}:\n"
                if doc.title:
                    ret += f"Title: {doc.title}\n"
                if doc.url:
                    ret += f"URL: {doc.url}\n"
                if doc.description:
                    ret += f"Description: {doc.description}\n"
                if doc.keywords:
                    ret += f"Keywords: {', '.join(doc.keywords)}\n"
                if doc.metadata:
                    for k, v in doc.metadata.items():
                        ret += f"{k}: {v}\n"
                ret += f"{doc.content}\n"
            else:
                ret += f"Document {i}:\n{doc}\n"
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

        parameters["additionalProperties"] = False  # Ensure no extra properties are allowed

        return FunctionDefinition(
            name=ai_function.name,
            description=ai_function.description or "",
            parameters=parameters,
            strict=True,
        )

    @classmethod
    def prepare_function_definition(
        cls,
        func: Callable[..., Any],
        name: Optional[str] = None,
        description: Optional[str] = None,
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
            return_type = sig.return_annotation.__name__ if sig.return_annotation != Parameter.empty else "str"
        except AttributeError:
            return_type = None

        parameters = []
        for param in sig.parameters.values():
            if param.kind == Parameter.VAR_KEYWORD or param.kind == Parameter.VAR_POSITIONAL:
                raise ValueError("Function cannot have variable keyword or positional arguments.")

            param_name = param.name
            param_description = ""  # Default description

            # Attempt to extract description from docstring (this is very basic and could be improved).
            docstring_lines = docstring.strip().split("\n")
            for line in docstring_lines[1:]:  # Skip the first line (already in description)
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

    try:
        from agents.mcp import MCPServer
        from mcp import Tool as MCPTool

        def prepare_mcp_tool_definition(self, tool: MCPTool) -> Dict[str, Any]:
            """Creates a FunctionDefinition from an MCP Tool.

            It can be overriden if other models expect different definition

            Args:
                tool: The MCP Tool to create the FunctionDefinition from.
            Returns:
                A FunctionDefinition object representing the tool.
            """
            try:
                ret = {
                    "name": tool.name,
                    "description": tool.description,
                }
                if tool.inputSchema["properties"]:
                    ret["parameters"] = {
                        "type": "object",
                        "properties": tool.inputSchema["properties"],
                        "required": tool.inputSchema["required"],
                    }
                return ret
            except Exception as e:
                print(f"Error preparing MCP tool definition: {e}")
                raise e

    except ImportError:
        pass
