import os
from dotenv import load_dotenv

import chromadb
from chromadb.utils import embedding_functions
from chromadb.config import Settings
from langchain.embeddings import OpenAIEmbeddings

load_dotenv()

CHROMA_SERVER_HOST = os.getenv("CHROMA_SERVER_HOST")
CHROMA_SERVER_HTTP_PORT = os.getenv("CHROMA_SERVER_HTTP_PORT")

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=os.getenv("OPENAI_API_KEY"),
                model_name="text-embedding-ada-002"
            )

client = chromadb.PersistentClient(path="./chroma")
chroma_client = chromadb.HttpClient(host=CHROMA_SERVER_HOST, port=CHROMA_SERVER_HTTP_PORT)

local_collection = client.get_collection(name="langchain", embedding_function=openai_ef)
# remote_collection = client.get_collection(name="langchain", embedding_function=openai_ef)

embeddings = local_collection.get()['embeddings']
documents = local_collection.get()['documents']
metadatas = local_collection.get()['metadatas']
ids = local_collection.get()['ids']

# remote_collection.add(
#     embeddings=embeddings,
#     ids=ids
# )

print(embeddings)