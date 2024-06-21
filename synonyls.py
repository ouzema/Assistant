import streamlit as st
from pathlib import Path
from langchain.llms.openai import OpenAI
from langchain.agents import create_sql_agent
from langchain.sql_database import SQLDatabase
from langchain.agents.agent_types import AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from sqlalchemy import create_engine
import sqlite3
import base64
import os
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_openai import ChatOpenAI
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.vectorstores import FAISS
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
    SystemMessagePromptTemplate,
)

from dataclasses import dataclass
from typing import Literal

from langchain.callbacks import get_openai_callback
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationSummaryMemory
import streamlit.components.v1 as components

from examples import examples

from dotenv import load_dotenv

from sqlalchemy.engine.url import URL
from sqlalchemy import create_engine
from urllib.parse import urlparse, parse_qs

# Load environment variables
load_dotenv()

# Set Streamlit page configuration with the uploaded logo
st.set_page_config(page_title="Leanios_core Assistant", page_icon="ln.png")

# Custom CSS for a modern and minimalistic design
st.markdown("""
    <style>
        body {
            font-family: 'Arial', sans-serif;
            background-color: #f5f5f5;
            color: #333;
        }
        .stApp {
            padding: 2rem;
        }
        .stTextInput input {
            border: 1px solid #ddd;
            padding: 0.5rem;
            border-radius: 5px;
        }
        .stButton button {
            background-color: #007bff;
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 5px;
            cursor: pointer;
        }
        .stButton button:hover {
            background-color: #0056b3;
        }
        .stAlert {
            border: 1px solid #ddd;
            padding: 1rem;
            border-radius: 5px;
            background-color: #fff;
        }
    </style>
""", unsafe_allow_html=True)



st.title("🤖 Leanios_core Assistant")

# Database configuration
database_name = os.getenv('DATABASE_NAME')
database_user = os.getenv('DATABASE_USER')
database_password = os.getenv('DATABASE_PASSWORD')
database_host = os.getenv('DATABASE_HOST')
database_port = os.getenv('DATABASE_PORT')

# Function to get database connection with dynamic schema based on URL parameter
@st.cache_resource(ttl="2h")
def get_db(subdomain):
    db_url = URL(
        drivername='postgresql+psycopg2',
        username=database_user,
        password=database_password,
        host=database_host,
        port=database_port,
        database=database_name,
        query={'options': f'-csearch_path={subdomain}'}
    )
    engine = create_engine(db_url)
    return SQLDatabase(engine)

# Parse the URL to get the subdomain parameter
query_params = st.experimental_get_query_params()
subdomain = query_params.get('subdomain', ['dummy'])[0]

# Ensure the subdomain is visible in the URL
if 'subdomain' not in query_params:
    st.experimental_set_query_params(subdomain='dummy')
    st.experimental_rerun()

# Get the database connection
db = get_db(subdomain)

# Synonym replacement function
synonyms = {
    "highest": ["top", "most"],
    "quantity": ["amount", "number"],
    "revenue": ["income", "earnings"],
    "orders": ["purchases", "requests"],
    "customers": ["clients", "buyers"],
    "products": ["items", "goods"],
    "total": ["sum", "entire"],
    "last": ["previous", "past"],
    "month": ["30 days", "four weeks"],
    "year": ["12 months", "52 weeks"],
    "category": ["type", "group"],
    "price": ["cost", "value"],
    "name": ["title", "designation"],
    "order": ["purchase", "request"],
    "quantity": ["amount", "number"],
    "product": ["item", "good"],
    "customer": ["client", "buyer"],
    "created_at": ["created", "made"],
    "sales": ["transactions", "deals"],
    "value": ["worth", "price"],
    "popular": ["frequent", "common"]
    # Add more synonyms as needed
}

def replace_synonyms(query):
    for word, synonym_list in synonyms.items():
        for synonym in synonym_list:
            query = query.replace(synonym, word)
    return query

example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,
    OpenAIEmbeddings(),
    FAISS,
    k=5,
    input_keys=["input"],
)

system_prefix = """You are an AI agent designed to interact with a SQL database.
Given an input question, I will create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
Unless you specify a specific number of examples you wish to obtain, I will always limit the query to at most {top_k} results.
I can order the results by a relevant column to return the most interesting examples in the database.
I have access to tools for interacting with the database, and I will only use the given tools. I will only use the information returned by the tools to construct my final answer.
I will double check my query before executing it. If I encounter an error while executing a query, I will rewrite the query and try again.

I will not make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

If your question is not well-formed or the answer isn't available for that specific question, I will respond with an appropriate message such as "I am sorry, I couldn't find any relevant information regarding your question."

Here are some examples of user inputs and their corresponding SQL queries:"""

few_shot_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=PromptTemplate.from_template(
        "User input: {input}\nSQL query: {query}"
    ),
    input_variables=["input", "dialect", "top_k"],
    prefix=system_prefix,
    suffix="",
)

full_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate(prompt=few_shot_prompt),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)

llm = OpenAI(model="gpt-3.5-turbo", temperature=0)

agent = create_sql_agent(
    llm=llm,
    db=db,
    prompt=full_prompt,
    verbose=True,
    agent_type="openai-tools",
    handle_parsing_errors=True,
)
if "messages" not in st.session_state or st.sidebar.button("Clear message history", key="clear_history"):
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

# Load the assistant logo and encode it to base64
with open("ln.png", "rb") as image_file:
    encoded_logo = base64.b64encode(image_file.read()).decode()

# Display chat messages
for msg in st.session_state.messages:
    if msg["role"] == "assistant":
        icon = f'<img src="data:image/png;base64,{encoded_logo}" width="40" height="40">'
    else:
        icon = "👤"
    st.markdown(f"""
        <div class="chat-message {msg['role']}">
            <div class="icon">{icon}</div>
            <div class="content">{msg['content']}</div>
        </div>
    """, unsafe_allow_html=True)

user_query = st.text_input("Ask me anything!", key="user_query_input")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    if "messages" not in st.session_state or st.sidebar.button("Clear message history", key="clear_history_2"):
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

    for msg in st.session_state["messages"]:
        if msg["role"] == "assistant":
            icon = f'<img src="data:image/png;base64,{encoded_logo}" width="40" height="40">'
        else:
            icon = "👤"
        st.markdown(f"""
            <div class="chat-message {msg['role']}">
                <div class="icon">{icon}</div>
                <div class="content">{msg['content']}</div>
            </div>
        """, unsafe_allow_html=True)

    response = agent.run(replace_synonyms(user_query))

    # Enhanced response handling
    if isinstance(response, list) and len(response) < 3:
        response_text = f"I found {len(response)} results:\n" + "\n".join([f"{i+1}. {row[0]}: {row[1]}" for i, row in enumerate(response)])
    elif not response:
        response_text = "I am sorry, I couldn't find any relevant information regarding your question."
    else:
        response_text = "\n".join([f"{i+1}. {row[0]}: {row[1]}" for i, row in enumerate(response)])

    st.session_state["messages"].append({"role": "assistant", "content": response_text})
    if "messages" not in st.session_state or st.sidebar.button("Clear message history", key="clear_history_3"):
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

    for msg in st.session_state["messages"]:
        if msg["role"] == "assistant":
            icon = f'<img src="data:image/png;base64,{encoded_logo}" width="40" height="40">'
        else:
            icon = "👤"
        st.markdown(f"""
            <div class="chat-message {msg['role']}">
                <div class="icon">{icon}</div>
                <div class="content">{msg['content']}</div>
            </div>
        """, unsafe_allow_html=True)