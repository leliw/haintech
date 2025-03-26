from haintech.ai.base.base_ai_model import BaseAIModel
from haintech.ai.model import AIFunction
from haintech.ai.tools import load_text_from_file

def my_function1(param1: str, param2: int) -> str:
    """My function does something.

    Args:
        param1: First parameter.
        param2: Second parameter (default is 10).
    Returns:
        The result of the function.
    """
    pass

def test_prepare_function_definition_ok():
    definition = BaseAIModel.prepare_function_definition(my_function1)
    # assert isinstance(definition, FunctionDefinition)
    assert definition["name"] == "my_function1"
    assert definition["description"] == "My function does something."
    assert definition["parameters"] == {
        "type": "object",
        "properties": {
            "param1": {"type": "string", "description": "First parameter."},
            "param2": {
                "type": "string",
                "description": "Second parameter (default is 10).",
            },
        },
        "required": ["param1", "param2"],
        "additionalProperties": False,
    }
    assert definition["strict"] is True

def test_create_function_definition_ok():
    definition = BaseAIModel.create_ai_function(my_function1)
    assert isinstance(definition, AIFunction)
    assert definition.name == "my_function1"
    assert definition.description == "My function does something."
    assert definition.return_type == "str"
    assert definition.parameters[0].name == "param1"
    assert definition.parameters[0].description == "First parameter."
    assert definition.parameters[0].type == "str"
    assert definition.parameters[1].name == "param2"
    assert definition.parameters[1].description == "Second parameter (default is 10)."
    assert definition.parameters[1].type == "int"




def test_prepare_function_definition_no_docstring():
    def my_function(param1: str, param2: int):
        pass

    definition = BaseAIModel.prepare_function_definition(my_function)
    assert definition["name"] == "my_function"
    assert definition["description"] == ""
    assert definition["parameters"] == {
        "type": "object",
        "properties": {
            "param1": {"type": "string", "description": ""},
            "param2": {"type": "string", "description": ""},
        },
        "required": ["param1", "param2"],
        "additionalProperties": False,
    }
    assert definition["strict"] is True


def test_prepare_function_definition_no_args():
    def my_function():
        """My function does something."""
        pass

    definition = BaseAIModel.prepare_function_definition(my_function)
    assert definition["name"] == "my_function"
    assert definition["description"]


def test_load_text_from_file():

    definition = BaseAIModel.prepare_function_definition(load_text_from_file)
    assert definition["name"] == "load_text_from_file"
    assert definition["description"] == "Load text content from a file"
    assert definition["parameters"] == {
        "type": "object",
        "required": ["file_path", "parent_dir"],
        "properties": {
            "file_path": {
                "type": "string",
                "description": "The path to the file",
            },
            "parent_dir": {
                "description": "The parent directory of the file",
                "type": ["string", "null"],
            },
        },
        "additionalProperties": False,
    }
    assert definition["strict"] is True
