import streamlit as st
import os
import time
from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever 
from langchain_community.vectorstores import FAISS, Chroma 
from langchain_community.document_loaders import PyPDFDirectoryLoader, PyPDFLoader 
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from dotenv import load_dotenv

load_dotenv()
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")
embeddings = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2")

st.title("RAG & Chat History Chatbot with LP")
st.write("Upload PDF and chat with their content")

api_key = st.text_input("Groq API Key:", type = "password")

if api_key:
    llm = ChatGroq(groq_api_key = api_key, model_name = "Gemma2-9b-It")

    session_id = st.text_input("Session ID", value = "default_session")

    if 'store' not in st.session_state:
        st.session_state.store = {}

    uploaded_files = st.file_uploader("Choose A Pdf file", type  = "pdf", accept_multiple_files = True)

    if uploaded_files:
        documents = []
        for uploaded_file in uploaded_files:
            temp_pdf = f"./temp.pdf"
            with open(temp_pdf, "wb") as file:
                file.write(uploaded_file.getvalue())
                file_name = uploaded_file.name

            loader = PyPDFLoader (temp_pdf)
            docs = loader.load()
            documents.extend(docs)

        # Split
        text_splitter = RecursiveCharacterTextSplitter(chunk_size = 5000, chunk_overlap = 200)
        splits = text_splitter.split_documents(documents)
        vectorstore = Chroma.from_documents(documents = splits, embedding = embeddings)
        retriever = vectorstore.as_retriever()
    
    contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, " \
    "formulate a standlone question which can be understood " \
    "without the chat history. Do NOT answer the question, " \
    "just reformulate it if needed and otherwise return it as is"
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("user", "{input}"),
        ]
    )

    system_prompt = (
    "You are an assistant for question-answering tasks"
    "Use the following pieces of retrieved context to answer"
    "the question. If you don't known the answer, say that you don't know"
    "{context}"
    )

    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("user", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


    store = {}
    def get_session_history(session_id:str)->BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory() # Tạo mới history đối với session chưa có trong store 
        return store[session_id]

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key = "input",
        history_messages_key = "chat_history",
        output_messages_key = "answer",
    )

    user_input = st.text_input("Please ask:")
    if user_input:
        session_history = get_session_history(session_id)
        response = conversational_rag_chain.invoke(
            {"input": user_input},
            config = {
                "configurable": {"session_id": session_id}
            },
        )

        st.write(st.session_state.store)
        st.success(f"Assistant:, {response['answer']}")
        st.write("Chat History:", session_history.messages)
else:
    st.warning("Please enter your Groq Key")



