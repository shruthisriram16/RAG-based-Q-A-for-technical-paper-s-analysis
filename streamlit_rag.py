import streamlit as st
from rag_query_v1 import run_rag
st.set_page_config(
    page_title="PDF Reasoning",
    layout="centered"
    )

st.markdown("<h1 style='text-align: center; color: Black;'>Ask Your Documents</h1>", 
            unsafe_allow_html=True)
st.markdown("<h4 style ='text-align: center;color:Black;'>What can I help with?<h4>",
            unsafe_allow_html=True)


#creating the user input
# 1. init
if "messages" not in st.session_state:
    st.session_state.messages = []

# 2. input FIRST
user_input = st.chat_input("Enter your question")

if user_input:
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })
    result=run_rag(user_input)

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"]
    })

# 3. render AFTER appending
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])





