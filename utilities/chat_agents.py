import os
from dotenv import load_dotenv

import requests
from bs4 import BeautifulSoup

import json

from langchain.prompts import (
    PromptTemplate, MessagesPlaceholder
    )
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.chains import LLMMathChain, create_qa_with_sources_chain, ConversationalRetrievalChain, LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory, ConversationSummaryBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import SystemMessage
from langchain.vectorstores import Chroma
from langchain.agents.agent_toolkits import create_retriever_tool

load_dotenv()
browserless_api_key = os.getenv("BROWSERLESS_API_KEY")
serper_api_key = os.getenv("SERP_API_KEY")

# Load default config
PRO_GPT_MODEL = os.getenv("PRO_GPT_MODEL")
BASE_GPT_MODEL = os.getenv("BASE_GPT_MODEL")
SUB_GPT_MODEL = os.getenv("SUB_GPT_MODEL")

# Retrieve private info
def knowledge_retrieval(query):
    print('Retrieving private knowledge..')
    llm = ChatOpenAI(temperature=0, model=SUB_GPT_MODEL)

    condense_question_prompt = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.\
    Make sure to avoid using any unclear pronouns.

    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:"""

    condense_question_prompt = PromptTemplate.from_template(condense_question_prompt)

    condense_question_chain = LLMChain(
        llm=llm,
        prompt=condense_question_prompt,
    )

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    qa_chain = create_qa_with_sources_chain(llm)

    doc_prompt = PromptTemplate(
        template="Content: {page_content}\nSource: {source}",
        input_variables=["page_content", "source"],
    )

    final_qa_chain = StuffDocumentsChain(
        llm_chain=qa_chain,
        document_variable_name="context",
        document_prompt=doc_prompt,
    )

    db = Chroma(
        persist_directory="./chroma",
        embedding_function=OpenAIEmbeddings(model="text-embedding-ada-002"),
    )

    retrieval_qa = ConversationalRetrievalChain(
        question_generator=condense_question_chain,
        retriever=db.as_retriever(),
        memory=memory,
        combine_docs_chain=final_qa_chain,
        verbose=True,
    )

    results = retrieval_qa.run(query)
    return results

# Search Function
def search(query):
    url = "https://google.serper.dev/search"

    payload = json.dumps({
        "q": query
    })

    headers = {
        'X-API-KEY': serper_api_key,
        'Content-Type': 'application/json'
    }

    print('Searching for... ', query)

    response = requests.request("POST", url, headers=headers, data=payload)

    # print(response.text)
    return response.text

# Website Scraper
def scrape(url: str):
    print('Scraping website... ', url)

    headers = {
        'Cache-Control': 'no-cache',
        'Content-Type': 'application/json',
    }

    # Parse the data
    data = json.dumps({"url": url})

    # Send the POST request
    browserless_url = f"https://chrome.browserless.io/content?token={browserless_api_key}"
    response = requests.post(browserless_url, headers=headers, data=data)

    if response.status_code == 200:
         # Parse the HTML
        soup = BeautifulSoup(response.content, 'html.parser')
        text = soup.get_text()
        print("Scraped content:", text)

        # Content might be really long and hit the token limit, we should summarize the text
        if len(text) > 10000:
            output = summary(text)
            return output
        else:
            return text
    else:
        print("HTTP request failed with status code {response.status_code}")


# Summarise Function
def summary(content):
    # invoke chatGPT
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")

    # Use LangChain text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"], chunk_size=10000, chunk_overlap=500)
    
    documents = text_splitter.create_documents([content])

    # Reusable prompt for each content on the split chain
    map_prompt = """
    Summarize of the following text for research purpose:
    "{text}"
    SUMMARY:
    """

    map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])
    
    summary_chain = load_summarize_chain(
        llm = llm,
        chain_type = 'map_reduce',
        map_prompt = map_prompt_template,
        combine_prompt = map_prompt_template,
        verbose = True
    )

    # Run the summary chain
    output = summary_chain.run(input_documents=documents)

    return output

# Researcher
def research(query):
    tools = [    
        Tool(
            name = "search",
            func = search,
            description = "Use this to answer questions about current events, data, or terms that you don't really understand. You should ask targeted questions"
        ),          
        Tool(
            name = "scrape",
            func = scrape,
            description = "Use this to load content from a website url"
        ),   
    ]

    llm = ChatOpenAI(temperature=0, model=SUB_GPT_MODEL)
    llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)

    system_message = SystemMessage(
        content="""You are a world-class researcher dedicated to factual accuracy and thorough data gathering. You do not make things up, you will try as hard as possible to gather facts & data to back up the research.
            
            Please make sure you complete the objective above with the following rules:
            1/ You should do enough research to gather as much information as possible about the query
            2/ If there are url of relevant links & articles, you will scrape it to gather more information
            3/ After searching and scraping, you should think "can I increase the research quality by searching and scraping for something new?" If answer is yes, continue; But don't do this more than 3 iterations
            4/ You should not make things up, you should only write facts & data that you have gathered. 
            5/ You should not make things up, you should only write facts & data that you have gathered.
            6/ In the final output, You should include all reference data & links to back up your research."""
    )

    agent_kwargs = {
        "system_message": system_message,
    }
   

    agent = initialize_agent(
        tools, 
        llm, 
        agent=AgentType.OPENAI_FUNCTIONS,
        verbose=True,
        agent_kwargs=agent_kwargs,
    )

    results = agent.run(query)
    return results


agents = {}

# Inititalise agent
def create_agent(id, user_name, ai_name, instructions):
    db = Chroma(
        persist_directory="./chroma",
        embedding_function=OpenAIEmbeddings(model="text-embedding-ada-002"),
    )

    retriever = db.as_retriever()

    tools = [
        create_retriever_tool(
            retriever,
            "knowledge_retrieval",
            "Use this to get our internal knowledge base data of curated information on Calm Collective and mental health topics. Use this first before other tools.",
        ), 
        Tool(
            name="research",
            func=research,
            description="Always use this to answer questions about current events and expand your knowledge on mental health. Ask targeted questions"
        ),
    ]

    db = Chroma(
        persist_directory="./chroma",
        embedding_function=OpenAIEmbeddings(model="text-embedding-ada-002"),
    )

    llm = ChatOpenAI(temperature=0.1, model=BASE_GPT_MODEL, streaming=True, verbose=True)

    system_message = SystemMessage(
        content = instructions
    )

    agent_kwargs = {
        "extra_prompt_messages": [MessagesPlaceholder(variable_name="chat_history")],
        "system_message": system_message,
    }

    conversational_memory = ConversationSummaryBufferMemory(llm=llm, memory_key="chat_history", return_messages=True, user_prefix=f"<@{user_name}>", ai_prefix=ai_name, max_token_limit=1000)

    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.OPENAI_FUNCTIONS,
        verbose=True,
        agent_kwargs=agent_kwargs,
        memory=conversational_memory,
    )

    agents[id] = agent
    
    return agent


def general_response(instructions, user_input, ai_name):   
    id = user_input["user"]   
    message = user_input["text"]

    if id not in agents:
        user_name = user_input["user"]
        agent = create_agent(id, user_name, ai_name, instructions)
    else:
        agent = agents[id]
    
    response = agent.run(message)

    return response