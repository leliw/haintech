# BaseAIChat

Extends `BaseAIModel()` with chat session functionality.

## Constructor

Arguments:

* ai_model: [BaseAIModel](ai_base_ai_model.md) -
  AI model class implementing `BaseAIModel()`
* context: str | [AIPrompt](ai.md#aiprompt) -
  prompt for model
* session: [AIModelSession](ai.md#aimodelsession) -
  current session object

## Use case

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
