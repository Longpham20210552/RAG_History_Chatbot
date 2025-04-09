import streamlit as st
import os
import time
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import OllamaEmbeddings, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.vectorstores import FAISS, Chroma 
from langchain_community.document_loaders import PyPDFDirectoryLoader
from dotenv import load_dotenv

env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path=env_path)
load_dotenv(dotenv_path=".env")
os.environ["OPENAI_API_KEY"] = "your_api_key"
print(os.getenv("OPENAI_API_KEY"))
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_KEY")
print("Loaded GROQ key:", os.getenv("GROQ_KEY"))
groq_api_key = os.getenv("GROQ_KEY")

llm = ChatGroq(groq_api_key = groq_api_key, model_name = "Llama3-8b-8192")
prompt = ChatPromptTemplate.from_template(
    """
    Answer the question based on the provided context only
    Please provide the most accurate response based on the question
    <context>
    {context}
    <context>
    Question: {input}
    """
)

def create_vector_embedding():
    if "vector" not in st.session_state:
        st.session_state.embeddings = OpenAIEmbeddings()
        st.session_state.loader = PyPDFDirectoryLoader("research_papers")
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

st.title("RAG Document Q&A LP - Groq  and LLama3")
user_prompt = st.text_input("Enter your query from the research paper")


if st.button("Document Embedding"):
    create_vector_embedding()
    st.write("Vector Database 's ready")

if user_prompt:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retriever_chain = create_retrieval_chain(retriever, document_chain)

    start = time.process_time()
    response = retriever_chain.invoke({'input': user_prompt})
    
    print(f"Response time:{time.process_time()-start}")

    st.write(response['answer'])

    # Expander
    with st.expander("Document similarity Search"):
        for i, doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write('----------------------')



    
