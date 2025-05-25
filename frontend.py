import streamlit as st
import requests

st.title("RAG Chatbot Demo")

user_question = st.text_input("Ask me anything:")

if st.button("Send"):
    if user_question.strip() == "":
        st.warning("Please enter a question.")
    else:
        response = requests.post("http://localhost:8000/chat/", json={"question": user_question})
        if response.status_code == 200:
            data = response.json()
            st.success(data['answer'])
        else:
            st.error("Failed to get response.") 