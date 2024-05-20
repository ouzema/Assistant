import streamlit as st
import requests

st.set_page_config(page_title="Leanios_core Assistant", page_icon="🤖")
st.title("🤖 Leanios_core Assistant")

if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

user_query = st.chat_input(placeholder="Ask me anything!")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

    response = requests.get("http://localhost:8000/query", params={"query": user_query})
    response_data = response.json().get("response")

    st.session_state.messages.append({"role": "assistant", "content": response_data})
    st.chat_message("assistant").write(response_data)
