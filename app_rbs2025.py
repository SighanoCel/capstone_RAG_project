"""Streamlit chat app — Capstone Project Assistant RBS 2025.

Conversational RAG with memory over the Capstone Final Report, faithful to
notebooks/Capstone_project_LCEL_Memory.ipynb: persistent Chroma store, k=4
retriever, and RunnableWithMessageHistory keyed by a per-session id. Reuses the
same ingestion module (pdf_ingestion.py) and chain builder (rag.build_memory_rag).

Run locally:   streamlit run app_rbs2025.py
Deploy:        push to GitHub -> share.streamlit.io (main file: app_rbs2025.py)
               -> set OPENAI_API_KEY in Secrets
"""

import os
import sys
import uuid
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# Make the module folder importable (rag.py, pdf_ingestion.py live in notebooks/src/)
sys.path.append(str(Path(__file__).resolve().parent / "notebooks" / "src"))

from rag import build_memory_rag  # noqa: E402

# ---------------------------------------------------------------------------
# API key: from Streamlit secrets (cloud) or .env / environment (local)
# ---------------------------------------------------------------------------
load_dotenv()
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

st.set_page_config(
    page_title="Capstone Project Assistant RBS 2025", page_icon="📄", layout="centered"
)
st.title("📄 Capstone Project Assistant RBS 2025")
st.caption("Ask Clara about the Capstone Final Report.")

if not os.getenv("OPENAI_API_KEY"):
    st.error(
        "No OPENAI_API_KEY found. Set it in `.env` locally, or in the app's "
        "Secrets on Streamlit Cloud (Settings → Secrets)."
    )
    st.stop()


# ---------------------------------------------------------------------------
# Build the memory RAG once and cache it (survives reruns, shared across users;
# per-session isolation comes from the session_id passed at invoke time).
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Indexing the document… (first run only)")
def get_chain():
    return build_memory_rag(k=4, model="gpt-4o-mini")


try:
    chain = get_chain()
except Exception as exc:  # surface build errors in the UI instead of a blank page
    st.error(f"Failed to build the RAG pipeline: {exc}")
    st.stop()

# ---------------------------------------------------------------------------
# Chat (memory managed by RunnableWithMessageHistory, keyed per browser session)
# ---------------------------------------------------------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    if st.button("🗑️ Clear conversation"):
        st.session_state.messages = []
        # New session id -> fresh, empty chat history in the chain's store.
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if question := st.chat_input("Ask about the report…"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                answer = chain.invoke(
                    {"question": question},
                    config={
                        "configurable": {"session_id": st.session_state.session_id}
                    },
                )
            except Exception as exc:
                answer = f"⚠️ Error: {exc}"
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
