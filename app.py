import streamlit as st
import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

@st.cache_resource
def load_phc_bot():
    # If the text file isn't uploaded yet, don't crash, just return None
    if not os.path.exists("phc_bootcamp_data.txt"):
        return None
        
    loader = TextLoader("phc_bootcamp_data.txt", encoding="utf-8")
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=60)
    splits = text_splitter.split_documents(docs)
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    llm = ChatGroq(model_name="llama3-8b-8192", temperature=0)
    
    system_prompt = (
        "You are a helpful digital assistant for the Pakistan Hindu Council. "
        "Use only the following pieces of retrieved context to answer the user's question. "
        "If the answer is not in the context, politely say that you don't know. "
        "\n\nContext: {context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, question_answer_chain)

rag_chain = load_phc_bot()

st.title("🇵🇰 PHC Digital Assistant")
st.markdown("Running on Groq (Llama 3) for the Pakistan Hindu Council.")

if rag_chain is None:
    st.warning("Please upload 'phc_bootcamp_data.txt' to your GitHub repository to enable the bot.")
else:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_input := st.chat_input("Ask a PHC question:"):
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("assistant"):
            try:
                response = rag_chain.invoke({"input": user_input})
                answer = response["answer"]
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"Error: {str(e)}")
