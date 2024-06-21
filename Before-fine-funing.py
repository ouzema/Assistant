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

st.set_page_config(page_title="Leanios_core Assistant", page_icon="🤖")
st.title("🤖 Leanios_core Assistant")



# Setup agent
llm = OpenAI(openai_api_key="sk-proj-5HpTAByJImynOhTp9CpJT3BlbkFJgWQndJVGOYYCQel1BXun", temperature=0, streaming=True)


@st.cache_resource(ttl="2h")
def get_db():
    return SQLDatabase.from_uri(
        'postgresql+psycopg2://postgres:postgres@localhost:5432/Leanios_development?options=-csearch_path=dummy'
    )

db = get_db()

toolkit = SQLDatabaseToolkit(db=db, llm=llm)

agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
)

if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

user_query = st.chat_input(placeholder="Ask me anything!")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        response = agent.run(user_query)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)