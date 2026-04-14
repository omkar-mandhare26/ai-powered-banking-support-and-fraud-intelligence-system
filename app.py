from main_pipeline import run_pipeline
import streamlit as st

MAX_HISTORY = 20

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "debug_mode" not in st.session_state:
    st.session_state.debug_mode = False

st.set_page_config(layout="wide")

st.title("AI-Powered Banking Support & Fraud Intelligence System using NLP + RAG")
st.write("---")

with st.sidebar:
    st.header("Settings")

    st.session_state.debug_mode = st.toggle(
        "Show Debug Info",
        value=st.session_state.debug_mode
    )

    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

for msg in st.session_state.chat_history[-MAX_HISTORY:]:
    if isinstance(msg, dict) and "role" in msg and "content" in msg:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

query = st.chat_input("Type your message...")

if query:
    query = query.strip()

    if not query:
        st.warning("Please enter a valid message.")
        st.stop()

    user_msg = {"role": "user", "content": query}
    st.session_state.chat_history.append(user_msg)

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = run_pipeline(query=query, chat_history=st.session_state.chat_history)
                response_text = result.get("response", "")

                if not isinstance(response_text, str):
                    response_text = str(response_text)
            except Exception as e:
                response_text = "Something went wrong. Please try again."
                result = {"intent": None}
        st.markdown(response_text)

    assistant_msg = {
        "role": "assistant",
        "content": response_text
    }
    st.session_state.chat_history.append(assistant_msg)

    if st.session_state.debug_mode:
        with st.expander("Debug Info"):
            st.write("Intent:", result.get("intent"))
            st.write("Intent Confidence:", result.get("intent_confidence"))
            st.write("RAG Confidence:", result.get("rag_confidence"))