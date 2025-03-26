# BaseAIModel

Abstract class for AI models.

## Methods

### get_chat_response()

Return chat response from LLM

Args:

* message: message to send to LLM,
* context: context to send to LLM,
* history: history of previous messages
* functions: functions to call
* interaction_logger: function to log interaction

Returns:

* LLM response

### _prompt_to_str()

Creates a string representation of AIPrompt object.
It can be overriden if other models expect different representation.

Args:

* prompt: AIPrompt object

Returns:

* string representation of AIPrompt object

### prepare_function_definition()

Creates an OpenAI FunctionDefinition from a Python callable.
It can be overriden if other models expect different definition

Args:

* func: The Python callable to create the FunctionDefinition from.
* name: the name of the function
* description: description of the function

Returns:

* A FunctionDefinition object representing the callable.  Returns None if input is invalid.

Raises:

* TypeError: If input is not a callable.
* ValueError: If the function signature is invalid or missing required information.

### create_ai_function()

Creates an AIFunction from a Python callable.

Args:

* func: The Python callable to create the FunctionDefinition from.

Returns:

* A AIFunction object representing the callable.  Returns None if input is invalid.

Raises:

* TypeError: If input is not a callable.
* ValueError: If the function signature is invalid or missing required information.
