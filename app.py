"""Streamlit chat UI for the Capstone RAG project.

Run locally:   streamlit run app.py
Deploy:        push to GitHub -> share.streamlit.io -> set OPENAI_API_KEY secret
"""

import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# Make the module folder importable (rag.py, pdf_ingestion.py live in notebooks/src/)
sys.path.append(str(Path(__file__).resolve().parent / "notebooks" / "src"))

from rag import build_conversational_rag, to_lc_messages  # noqa: E402

# ---------------------------------------------------------------------------
# API key: from Streamlit secrets (cloud) or .env / environment (local)
# ---------------------------------------------------------------------------
load_dotenv()
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

st.set_page_config(
    page_title="RBS Group Capstone Assistant", page_icon="📄", layout="centered"
)
st.title("📄 RBS Group Capstone Assistant")
st.caption("Ask questions about the Capstone Final Report.")

if not os.getenv("OPENAI_API_KEY"):
    st.error(
        "No OPENAI_API_KEY found. Set it in `.env` locally, or in the app's "
        "Secrets on Streamlit Cloud (Settings → Secrets)."
    )
    st.stop()


# ---------------------------------------------------------------------------
# Build the RAG chain once and cache it (survives reruns, not rebuilt per query)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Indexing the document… (first run only)")
def get_chain():
    return build_conversational_rag(k=3, model="gpt-4o-mini")


try:
    chain = get_chain()
except Exception as exc:  # surface build errors in the UI instead of a blank page
    st.error(f"Failed to build the RAG pipeline: {exc}")
    st.stop()

# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    if st.button("🗑️ Clear conversation"):
        st.session_state.messages = []
        st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if question := st.chat_input("Ask about the report…"):
    # History is everything said *before* this turn; the chain rewrites the
    # follow-up into a standalone query using it, then answers with it in view.
    chat_history = to_lc_messages(st.session_state.messages)

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                answer = chain.invoke(
                    {"question": question, "chat_history": chat_history}
                )
            except Exception as exc:
                answer = f"⚠️ Error: {exc}"
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
