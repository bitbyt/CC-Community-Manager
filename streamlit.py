from langchain.callbacks.base import BaseCallbackHandler
from langchain.memory import StreamlitChatMessageHistory
from langchain.memory import ConversationBufferMemory

import streamlit as st

from utilities.chat_agents import create_agent

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text + "▌")

def main():
    st.set_page_config(page_title="CC Community Manager")

    st.header("CC Community Manager")

    msgs = StreamlitChatMessageHistory(key="langchain_messages")
    conversational_memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=msgs, return_messages=True)

    if len(msgs.messages) == 0:
        msgs.add_ai_message("How can I help you?")

    agent = create_agent(conversational_memory)

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("How can I help you today?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            stream_response = StreamHandler(message_placeholder)
            response = agent.run(prompt, callbacks=[stream_response])
            message_placeholder.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == '__main__':
    main()
