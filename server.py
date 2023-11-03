import os
from dotenv import load_dotenv
from langchain.callbacks.base import BaseCallbackHandler
from langchain.memory import ConversationBufferWindowMemory

from pydantic import BaseModel, Field
from typing import Type
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from utilities.chat_agents import knowledge_retrieval

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text + "▌")

app = FastAPI()

class Query(BaseModel):
    query: str

# Make the agent accessable through an endpoint
@app.post("/knowledge_retrieval")
def chat(query: Query):
    query = query.query
    response = knowledge_retrieval(query)
    return response