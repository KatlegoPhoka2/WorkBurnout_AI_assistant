import streamlit as st
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src import ask

# Page config
st.set_page_config(
    page_title="Workplace Burnout Assistant",
    page_icon="🧠",
    layout="centered"
)

# Header
st.title("🧠 Workplace Burnout Assistant")
st.markdown(
    """
    Welcome! I'm here to help you understand and manage workplace burnout.
    Ask me anything about burnout symptoms, causes, prevention, or recovery.
    """
)

st.divider()

# Disclaimer
st.info(
    "⚠️ This assistant provides general wellness information only. "
    "For serious mental health concerns, please consult a qualified professional."
)

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_chain" not in st.session_state:
    with st.spinner("Loading AI assistant..."):
        from src.llm_chain import build_rag_chain
        st.session_state.rag_chain = build_rag_chain()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if query := st.chat_input("Ask about workplace burnout..."):

    # Display user message
    with st.chat_message("user"):
        st.markdown(query)

    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": query})

    # Get response from RAG chain
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer = st.session_state.rag_chain.invoke(query)
                st.markdown(answer)

                # Add assistant response to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })

            except Exception as e:
                error_msg = f"Something went wrong: {str(e)}"
                st.error(error_msg)

# Sidebar
with st.sidebar:
    st.header("About")
    st.markdown(
        """
        This assistant uses a **RAG (Retrieval-Augmented Generation)** 
        pipeline to provide grounded, accurate responses about workplace burnout.
        
        **Tech Stack:**
        - 🔗 LangChain
        - 🗄️ FAISS Vector Database
        - 🤗 HuggingFace Embeddings
        - 🤖 GPT-3.5 via OpenRouter
        - 🖥️ Streamlit
        """
    )

    st.divider()

    st.header("Sample Questions")
    sample_questions = [
        "What are the signs of burnout?",
        "How can I prevent burnout at work?",
        "What causes workplace burnout?",
        "How do I recover from burnout?",
        "How does burnout affect job performance?",
    ]

    for question in sample_questions:
        if st.button(question, use_container_width=True):
            st.session_state.messages.append({
                "role": "user",
                "content": question
            })
            with st.spinner("Thinking..."):
                answer = st.session_state.rag_chain.invoke(question)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })
            st.rerun()

    st.divider()

    # Clear chat button
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()