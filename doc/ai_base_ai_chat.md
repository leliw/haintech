# BaseAIChat

Extends `BaseAIModel()` with chat session functionality.

## Constructor

Arguments:

* ai_model: [BaseAIModel](ai_base_ai_model.md) - AI model class implementing `BaseAIModel()`
* context: str | [AIPrompt](ai.md#aiprompt) - prompt for model
* session: [AIModelSession](ai.md#aimodelsession) - current session object

## Methods

* get_response(self, message: Optional[str] = None) -> AIChatResponse: - get response from LLM
* get_text_response(self, message: Optional[str] = None) -> str: - get text response from LLM
* get_json_response(self, message: Optional[str] = None) -> Any: - get JSON response from LLM

## Use case

```python
ai_model = OpenAIModel("gpt-4o-mini", { "temperatoure": 0.7 })
session = AIChatSession()
ai_chat = OpenAIChat(ai_model=ai_model, session=session)
response = ai_chat.get_text_response("Who was the first polish king?")
assert "Bolesław" in response
# You can continue the conversation with the same or a new chat object
ai_chat = OpenAIChat(ai_model=ai_model, session=session)
response = ai_chat.get_text_response("Who was his father?")
assert "Mieszko" in response
```

### Streamlit

Simplest use `BaseAIChat` with `Streamlit` library.

```python
import streamlit as st
from haintech.ai import BaseAIChat
from haintech.ai.open_ai import OpenAIModel

st.title("HaInTech ChatBot")

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.llm = BaseAIChat(ai_model=OpenAIModel())

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    response = st.session_state.llm.get_text_response(prompt)
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
```
