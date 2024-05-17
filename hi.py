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

st.set_page_config(page_title="LangChain: Chat with SQL DB", page_icon="🦜")
st.title("🦜 LangChain: Chat with SQL DB")

INJECTION_WARNING = """
                    SQL agent can be vulnerable to prompt injection. Use a DB role with limited permissions.
                    Read more [here](https://python.langchain.com/docs/security).
                    """
LOCALDB = "USE_LOCALDB"

# User inputs
radio_opt = ["Use sample database - Chinook.db", "Connect to your SQL database"]
selected_opt = st.sidebar.radio(label="Choose suitable option", options=radio_opt)
if radio_opt.index(selected_opt) == 1:
    st.sidebar.warning(INJECTION_WARNING, icon="⚠️")
    db_uri = st.sidebar.text_input(
        label="Database URI", placeholder="mysql://user:pass@hostname:port/db"
    )
else:
    db_uri = LOCALDB

openai_api_key = st.sidebar.text_input(
    label="OpenAI API Key",
    type="password",
)

# Check user inputs
if not db_uri:
    st.info("Please enter database URI to connect to your database.")
    st.stop()

if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.")
    st.stop()

# Setup agent
llm = OpenAI(openai_api_key=openai_api_key, temperature=0, streaming=True)


@st.cache_resource(ttl="2h")
def configure_db(db_uri):
    try:
        engine = create_engine(db_uri)
        connection = engine.connect()
        connection.close()
        return SQLDatabase(engine)
    except OperationalError as e:
        st.error("Could not connect to the database. Please check the database URI and ensure the PostgreSQL server is running.")
        st.error(str(e))
        st.stop()

db = configure_db(db_uri)


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
        st_cb = StreamlitCallbackHandler(st.container())
        response = agent.run(user_query, callbacks=[st_cb])
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)