import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

st.title("🇵🇰 PHC Digital Assistant")

# Bypass the heavy vector databases and just read the text directly
try:
    with open("phc_bootcamp_data.txt", "r", encoding="utf-8") as file:
        phc_data = file.read()
except FileNotFoundError:
    phc_data = ""
    st.warning("Please upload 'phc_bootcamp_data.txt' to your GitHub repository.")

# Ultra-light Llama 3 connection
llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful digital assistant for the Pakistan Hindu Council. Use ONLY this official data to answer questions:\n\n{context}\n\nIf the answer is not in the data, say you don't know."),
    ("human", "{input}")
])

# Simple, lightweight chain
chain = prompt | llm

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("Ask a question:"):
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        try:
            # Feed the text directly into the prompt
            response = chain.invoke({"context": phc_data, "input": user_input})
            st.markdown(response.content)
            st.session_state.messages.append({"role": "assistant", "content": response.content})
        except Exception as e:
            st.error(f"Error: {str(e)}")
