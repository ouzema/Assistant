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


st.set_page_config(page_title="Leanios_core Assistant", page_icon="🤖")
st.title("🤖 Leanios_core Assistant")


# User inputs
radio_opt = ["Connect to your SQL database"]
if "Connect to your SQL database" in radio_opt:
    db_uri = st.sidebar.text_input(
        label="Database URI", placeholder="mysql://user:pass@hostname:port/db"
    )




OPENAI_API_KEY = st.sidebar.text_input(
    label="OpenAI API Key",
    type="password",
)
# Check user inputs
if not db_uri:
    st.info("Please enter database URI to connect to your database.")
    st.stop()

if not OPENAI_API_KEY:
    st.info("Please add your OpenAI API key to continue.")
    st.stop()


examples = [
    {"input": "What is the total quantity of product X?",
      "query": "SELECT SUM(quantity) FROM products WHERE name = 'X';"},
    {
        "input": "How many orders were placed by customer Y?",
        "query": "SELECT COUNT(*) FROM orders WHERE customer_id = (SELECT id FROM customers WHERE name = 'Y');",
    },
    {
        "input": "Who are the customers who have made a purchase in the last week.",
        "query": "SELECT DISTINCT customers.name FROM customers JOIN orders ON customers.id = orders.customer_id WHERE orders.created_at >= CURRENT_DATE - INTERVAL '1 week';",
    },
    {
        "input": "Show me the average price of products in category 'Electronics'.",
        "query": "SELECT AVG(price) FROM products WHERE category = 'Electronics';",
    },
    {
        "input": "Find the most popular payment method used by customers.",
        "query": "SELECT payment_method FROM ( SELECT payment_method, COUNT(*) AS method_count FROM orders WHERE payment_method IS NOT NULL GROUP BY payment_method) AS popular_payment_method ORDER BY COUNT(*) DESC LIMIT 1;",
    },
    {
        "input": "Show me the order details 'X'.",
        "query": "SELECT products.name, workorders.created_at FROM workorders JOIN orders ON workorders.order_id = orders.id JOIN customers ON orders.customer_id = customers.id JOIN products ON workorders.product_id = products.id WHERE customers.name = 'X';",
    },
    {
        "input": "Which orders contain products with a quantity less than 10?",
        "query": "SELECT DISTINCT orders.id FROM orders JOIN order_items ON orders.id = order_items.order_id WHERE order_items.quantity < 10;",
    },
    {
        "input": "How many customers have made a purchase in the last month?",
        "query": "SELECT COUNT(DISTINCT customer_id) FROM orders WHERE DATE_PART('month', CURRENT_DATE - INTERVAL '1 month') = DATE_PART('month', created_at);",
    },
    {
        "input": "Who are the top 5 customers by total purchase?",
        "query": "SELECT CustomerId, SUM(Total) AS TotalPurchase FROM Invoice GROUP BY CustomerId ORDER BY TotalPurchase DESC LIMIT 5;",
    },
    {
        "input": "Show me the total revenue for each product category.",
        "query": "SELECT category, SUM(total_amount) AS revenue FROM orders JOIN order_items ON orders.id = order_items.order_id JOIN products ON order_items.product_id = products.id GROUP BY category;",
    },
    {
        "input" : "What's in productions?",
        "query" : "SELECT products.name, customers.name, workorders.created_at FROM workorders INNER JOIN products ON workorders.product_id = products.id INNER JOIN orders ON workorders.order_id = orders.id INNER JOIN customers ON orders.customer_id = customers.id WHERE workorders.status = 1;"
    },
    
    {
        "input" : "What's your name?",
        "query" : "I am Leanios_Core Assistant"
    },
    {
        "input" : "Who are you?",
        "query" : "I am your Assistant"
    },
    {
        "input" : "What is the stock level per warehouse?",
        "query" : "SELECT warehouses.name, warehouses_products.quantity FROM warehouses JOIN warehouses_products ON warehouses.id = warehouses_products.warehouse_id ORDER BY warehouses.name;"
    },
    {
        "input" : "What can you do?",
        "query" : "I can answer your questions or inquiries about the Workflow, the Production or any other related process"
    },
    {
        "input" : "Hi",
        "query" : "Hello there, how may I assist you today?"
    },
    {
        "input" : "Hello",
        "query" : "Greetings! How may I be at your service you today?"
    },
    {
        "input" : "What are the OF that are pending",
        "query" : "SELECT products.name FROM workorders JOIN products ON workorders.product_id = products.id WHERE workorders.status = 0;"
    },
        {
        "input" : "What are the OF that are stopping",
        "query" : "SELECT products.name FROM workorders JOIN products ON workorders.product_id = products.id WHERE workorders.status = 2;"
    },
    {
        "input" : "What are the OF that are completed",
        "query" : "SELECT products.name FROM workorders JOIN products ON workorders.product_id = products.id WHERE workorders.status = 4;"
    },
    {
        "input" : "What is the number of client orders for this month?",
        "query" : "SELECT customer_id, COUNT(*) AS number_of_orders FROM orders WHERE EXTRACT(MONTH FROM created_at) = EXTRACT(MONTH FROM CURRENT_DATE) GROUP BY customer_id;"
    },
    {
        "input" : "Who works on the machine 'X'",
        "query" : "SELECT operators.full_name FROM operators JOIN machines_operators ON operators.id = machines_operators.operator_id JOIN machines ON machines.id = machines_operators.machine_id WHERE machines.name = 'X';"
    },
    {
        "input" : "what is the production rate by item?",
        "query" : "SELECT products.name AS 'Product Name', COUNT(productions.id) AS 'Number of Productions',SUM(productions.quantity) AS 'Total Quantity Produced', AVG(productions.performance) AS 'Average Production Rate' FROM productions JOIN products ON productions.product_id = products.id GROUP BY products.name;"
    },
    {
        "input" : "what is the rate of non-compliance",
        "query" : "SELECT downtime_issues.name, COUNT(downtimes.id), SUM(CASE WHEN downtimes.status = 1 THEN 1 ELSE 0 END), (SUM(CASE WHEN downtimes.status = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(downtimes.id)) FROM downtimes JOIN downtime_issues ON downtimes.downtime_issue_id = downtime_issues.id GROUP BY downtime_issues.name;"
    },
    {
        "input" : "what raw material stocks will expire this month?",
        "query" : "SELECT products.name, warehouses.name, inventory_items.quantity, inventory_items.created_at, inventory_items.created_at + INTERVAL '1 month' FROM inventory_items JOIN products ON inventory_items.product_id = products.id JOIN warehouses ON inventory_items.warehouse_id = warehouses.id WHERE products.is_final_product = FALSE AND products.expiration_date IS NOT NULL AND EXTRACT(MONTH FROM (inventory_items.created_at + INTERVAL '1 month')) = EXTRACT(MONTH FROM CURRENT_DATE) AND EXTRACT(YEAR FROM (inventory_items.created_at + INTERVAL '1 month')) = EXTRACT(YEAR FROM CURRENT_DATE);"
    },
    {
        "input" : "What are the products categories?",
        "query" : "SELECT products.name AS product_name, product_categories.name AS category_name FROM products JOIN product_categories ON products.product_category_id = product_categories.id;"
    },
    {
        "input" : "What are the machine areas?",
        "query" : "SELECT machines.name AS machine_name, areas.name AS area_name FROM machines JOIN areas ON machines.area_id = areas.id;"
    },
    {
        "input" : "what are the Internal OF?",
        "query" : "SELECT products.name FROM products JOIN workorders ON products.id = workorders.product_id WHERE workorders.internal = true;"
    },
    {
        "input" : "what are the external OF?",
        "query" : "SELECT products.name FROM products JOIN workorders ON products.id = workorders.product_id WHERE workorders.internal = false;"
    },
    {
        "input" : "what raw material stocks will expire this month?",
        "query" : "SELECT products.name, batches.expiration_date FROM products JOIN batches ON products.id = batches.product_id JOIN product_categories ON products.product_category_id = product_categories.id WHERE product_categories.raw_material = true AND DATE_TRUNC('month', batches.expiration_date) = DATE_TRUNC('month', CURRENT_DATE) ORDER BY batches.expiration_date;"
    },
    {
        "input" : "How many semi finished products do I have in stock?",
        "query" : "SELECT name, quantity FROM dummy.products WHERE is_semi_finished = true;"
    },
    {
        "input" : "Who are the amdin users?",
        "query" : "SELECT firstname, lastname FROM users WHERE admin = TRUE;"
    },
    {
        "input" : "Who are the amdin users?",
        "query" : "SELECT firstname, lastname FROM users WHERE admin = TRUE;"
    },
    {
        "input" : "Show me the products that have been delivered",
        "query" : "SELECT products.name, customers.name, delivery_notes.created_at FROM delivery_notes JOIN workorders ON workorders.order_id = delivery_notes.order_id JOIN products ON products.id = workorders.product_id JOIN orders ON orders.id = workorders.order_id JOIN customers ON customers.id = orders.customer_id;"
    }
]

openai_embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,
    openai_embeddings,
    FAISS,
    k=5,
    input_keys=["input"],
)
system_prefix = """You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most {top_k} results.
You can order the results by a relevant column to return the most interesting examples in the database.
Never query for all the columns from a specific table, only ask for the relevant columns given the question.
You have access to tools for interacting with the database.
Only use the given tools. Only use the information returned by the tools to construct your final answer.
You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

If the question does not seem related to the database, just return "I am sorry I cannot find any related data to your question" or if in french return "Je suis désolé mais je ne trouve pas des informations liées à votre question" as the answer.

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




# Setup agent
@st.cache_resource(ttl="2h")
def configure_db(db_uri):
    return SQLDatabase.from_uri(database_uri=db_uri)



db = configure_db(db_uri)

llm = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-3.5-turbo", temperature=0)

agent = create_sql_agent(
    llm=llm,
    db=db,
    prompt=full_prompt,
    verbose=True,
    agent_type="openai-tools",
    handle_parsing_errors=True,
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