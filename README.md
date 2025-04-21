# RAG History Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that allows users to **upload PDF documents** and interact with their content through natural language questions. Combines the power of **LLMs**, **vector databases**, and a **Streamlit-FastAPI** architecture.

---

## Features

- **PDF Upload & Parsing**: Upload one or multiple PDFs to the backend.
- **Smart Chunking**: Automatically splits documents into semantically meaningful chunks.
- **Embedding Generation**: Uses HuggingFace embeddings for chunk representation.
- **Vector Search**: Supports **Chroma** or **FAISS** for semantic retrieval.
- **RAG Pipeline**: Combines retrieved knowledge chunks with LLM (via **Groq API**) for contextual response generation.
- **Streamlit Frontend**: Clean and interactive chat interface.
- **FastAPI Backend**: Modular, production-ready API layer.

---

## Tech Stack

| Layer         | Tools Used                                 |
|---------------|---------------------------------------------|
| **Frontend**  | Streamlit                                   |
| **Backend**   | FastAPI                                     |
| **RAG Engine**| LangChain + Groq API (LLM)                  |
| **Embedding** | HuggingFace Transformers                    |
| **Vector DB** | Chroma or FAISS                             |
| **Others**    | PyMuPDF, sentence-transformers, Uvicorn     |

---
