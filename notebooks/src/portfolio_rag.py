"""Conversational RAG over Celestin Sighano's portfolio — assistant "Celestin".

Extracted from notebooks/portfolio_LCEL_Memory.ipynb so it can be served by
api_celestin.py, the chat API behind the "Ask to Celestin" widget.

Context = the live portfolio page (Webingestion, 500/100) + the notebook
summaries text file (Textingestion, 1000/250), embedded with
text-embedding-3-small into Chroma, k=3 retriever, gpt-4o-mini, memory via
RunnableWithMessageHistory keyed by session id.
"""

import os
from operator import itemgetter
from pathlib import Path

# WebBaseLoader warns if no user agent is set
os.environ.setdefault("USER_AGENT", "celestin-portfolio-assistant")

from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

try:
    from text_ingestion import Textingestion
    from web_ingestion import Webingestion
except ModuleNotFoundError:  # imported as notebooks.src.portfolio_rag
    from notebooks.src.text_ingestion import Textingestion
    from notebooks.src.web_ingestion import Webingestion

PORTFOLIO_URL = "https://sighanocel.github.io/My-data-science-project/"

# data/pdf_files/text_files lives at the project root
# (notebooks/src/portfolio_rag.py -> notebooks/src -> notebooks -> project root).
TEXT_FILE = (
    Path(__file__).resolve().parents[2]
    / "data" / "pdf_files" / "text_files" / "Notebook summary for RAG.txt"
)

PROMPT_CONV = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an assistant who provides information about the Celestin Sighano portfolio web site. "
            "Your name is Celestin. "
            "Answer questions based on the provided context. "
            "You must answer in the language used by the query. "
            "If the human greets you, you can also greet him and tell him what your name is. "
            "But you have to avoid any other conversation with him different from the provided context.",
        ),
        MessagesPlaceholder("history"),
        (
            "user",
            "These are the significant excerpts from the portfolio web site: \n\n{context}\n\n"
            "My Question:{question}",
        ),
    ]
)


def return_only_text(documents):
    return "\n\n...\n\n".join(doc.page_content for doc in documents)


def load_chunks(url: str = PORTFOLIO_URL, text_path: Path = TEXT_FILE):
    """Web chunks of the portfolio page + chunks of the notebook summaries."""
    web_chunks = Webingestion(chunk_size=500, chunk_overlap=100).process(url)
    text_chunks = Textingestion(chunk_size=1000, chunk_overlap=250).process(str(text_path))
    for c in text_chunks:
        c.metadata["source"] = text_path.name
    return web_chunks + text_chunks


def build_retriever(k: int = 3):
    """In-memory Chroma (rebuilt at startup — hosted filesystems are ephemeral)."""
    store = Chroma.from_documents(
        documents=load_chunks(),
        embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
        collection_name="portfolio_RAG",
    )
    return store.as_retriever(search_kwargs={"k": k})


def build_celestin_rag(k: int = 3, model: str = "gpt-4o-mini"):
    """Retriever + LCEL chain wrapped with per-session conversational memory."""
    retriever = build_retriever(k=k)

    rag_chain = (
        {
            "question": itemgetter("question"),
            "context": itemgetter("question") | retriever | RunnableLambda(return_only_text),
            "history": itemgetter("history"),
        }
        | PROMPT_CONV
        | ChatOpenAI(model=model)
        | StrOutputParser()
    )

    store_conv = {}

    def get_session_history(session_id: str):
        if session_id not in store_conv:
            store_conv[session_id] = ChatMessageHistory()
        return store_conv[session_id]

    return RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="history",
    )
