import streamlit as st
import requests

st.title("RAG Chatbot LP")

api_key = st.text_input("Groq API Key:", type="password")
session_id = st.text_input("Session ID:", value="default_session")

uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
if uploaded_files and st.button("Upload PDFs"):
    files = [("files", (f.name, f.read(), "application/pdf")) for f in uploaded_files]
    res = requests.post("http://localhost:8000/upload", files=files)
    st.success(res.json())

if api_key and session_id:
    user_input = st.text_input("Please ask:")
    if user_input:
        res = requests.post(
            "http://localhost:8000/chat",
            data={
                "api_key": api_key,
                "session_id": session_id,
                "user_input": user_input,
            },
        )
        if res.status_code == 200:
            data = res.json()
            st.success(data["answer"])
            st.write("Chat History:", data["history"])
        else:
            st.error(res.json())