import streamlit as st
from google import genai


st.title("Мой первый AI-чат")

if "chat" not in st.session_state:
    st.session_state.client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
    st.session_state.chat = st.session_state.client.chats.create(model="gemini-3.5-flash")
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["text"])

if user_input := st.chat_input("Напиши сообщение..."):
    st.session_state.messages.append({"role": "user", "text": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    response = st.session_state.chat.send_message(user_input)
    st.session_state.messages.append({"role": "assistant", "text": response.text})
    with st.chat_message("assistant"):
        st.markdown(response.text)