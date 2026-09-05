"""
app.py - Minimal Streamlit UI for the Incerro RAG chatbot.
"""
import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Incerro AI Chatbot",
    page_icon="🤖",
)

st.title("🤖 Incerro AI Assistant")
st.caption("Ask me anything about Incerro — products, services, or company info.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Load collection once (cached)
@st.cache_resource
def load_collection():
    from chat import get_collection, get_embedding_fn
    embedding_fn = get_embedding_fn()
    return get_collection(embedding_fn)

collection = load_collection()

# Render existing chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            st.caption("**Sources:** " + " | ".join(f"[{s}]({s})" for s in msg["sources"]))

# Handle new user input
if prompt := st.chat_input("Ask about Incerro..."):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            from chat import chat
            answer, sources = chat(prompt, collection)

        st.markdown(answer)
        if sources:
            st.caption("**Sources:** " + " | ".join(f"[{s}]({s})" for s in sources))

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })
