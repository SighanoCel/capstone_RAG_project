"""Reusable RAG pipeline for the Capstone project.

Extracted from notebooks/Capstone_project_RAG_LCEL.ipynb so it can be served
by app.py (Streamlit) or any other entry point. The OpenAI API key is read
from the environment (loaded from .env locally, or from Streamlit secrets in
the cloud).
"""

from operator import itemgetter
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Works both when notebooks/src is on sys.path (Streamlit app) and when the
# project root is (notebook doing `from notebooks.src...`).
try:
    from pdf_ingestion import PdfIngestion
except ModuleNotFoundError:  # pragma: no cover
    from notebooks.src.pdf_ingestion import PdfIngestion

# data/pdf_files lives at the project root, three levels up from this file
# (notebooks/src/rag.py -> notebooks/src -> notebooks -> project root).
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "pdf_files"

PROMPT = ChatPromptTemplate.from_template(
    """
You are a technical assistant for our data analytics team.
Answer the question below focusing on the context below.
If there is no answer in the context, just say: "there is no answer"


QUESTION:
{question}


CONTEXT:
{context}


ANSWER:
Be precise and very concise.
"""
)


def _format_docs(docs: list[Document]) -> str:
    """Turn retrieved documents into a single context string with sources."""
    return "\n\n".join(
        f"[page {d.metadata.get('page', '?')}] {d.page_content}" for d in docs
    )


def load_chunks(pdf_path: str | None = None):
    """Ingest one PDF (or every PDF in data/pdf_files) into chunks."""
    ingestor = PdfIngestion(chunk_size=1200, chunk_overlap=180)

    if pdf_path is not None:
        pdfs = [Path(pdf_path)]
    else:
        pdfs = sorted(DATA_DIR.glob("*.pdf"))

    if not pdfs:
        raise FileNotFoundError(f"No PDF found in {DATA_DIR}")

    chunks = []
    for pdf in pdfs:
        chunks.extend(ingestor.process(str(pdf)))
    return chunks


def build_retriever(pdf_path: str | None = None, k: int = 3):
    """Build an in-memory Chroma vector store and return a retriever.

    In-memory (no persist_directory) is intentional: the corpus is small
    (~76 chunks) so embedding at startup is fast and cheap, and it avoids
    the ephemeral-filesystem issues you hit on hosted platforms.
    """
    chunks = load_chunks(pdf_path)
    embed = OpenAIEmbeddings(model="text-embedding-3-small")
    store = Chroma.from_documents(
        documents=chunks,
        embedding=embed,
        collection_name="capstone_focused_docs",
    )
    return store.as_retriever(search_kwargs={"k": k})


def build_chain(retriever, model: str = "gpt-4o-mini"):
    """Assemble the LCEL RAG chain: retrieve -> prompt -> llm -> str."""
    llm = ChatOpenAI(model=model)
    parser = StrOutputParser()

    return (
        {
            "context": lambda x: _format_docs(retriever.invoke(x["question"])),
            "question": lambda x: x["question"],
        }
        | PROMPT
        | llm
        | parser
    )


def build_rag(pdf_path: str | None = None, k: int = 3, model: str = "gpt-4o-mini"):
    """Convenience: build retriever + chain in one call."""
    retriever = build_retriever(pdf_path=pdf_path, k=k)
    return build_chain(retriever, model=model)


# ---------------------------------------------------------------------------
# Conversational RAG (with chat history / memory)
# ---------------------------------------------------------------------------
# The chain itself stays stateless: history is passed in on every invoke under
# the "history" key. The caller (app.py) owns the conversation state, so the
# same cached chain can serve every turn and every user.

# Rewrites a follow-up question into a standalone one so the retriever gets a
# self-contained query (e.g. "and its revenue?" -> "What is RBS Group's revenue?").
CONTEXTUALIZE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Given the chat history and the latest user question, which might "
            "reference earlier turns, rephrase it into a standalone question "
            "understandable without the chat history. Do NOT answer it — only "
            "reformulate if needed, otherwise return it unchanged.",
        ),
        MessagesPlaceholder("history"),
        ("human", "{question}"),
    ]
)

# Answer prompt — Clara, the RBS Capstone Report assistant. Greets back with
# her name, answers in the human's language, and stays on the report.
CONVERSATIONAL_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert assistant who analyzes the RBS Capstone Project Report."
            "Your name is Clara."
            "Answer questions based on the provided context in the language used by the human"
            "If the human greats you, You can also great him and tell him what is your name"
            "But you have to avoid any other conversation with him different from the provided concept",
        ),
        MessagesPlaceholder("history"),
        (
            "user",
            "These are the significant excerpts from the report: \n\n{context}\n\n"
            "My Question:{question}",
        ),
    ]
)


def to_lc_messages(messages: list[dict]) -> list[BaseMessage]:
    """Convert Streamlit-style [{"role", "content"}] history to LC messages."""
    converted: list[BaseMessage] = []
    for m in messages:
        if m["role"] == "user":
            converted.append(HumanMessage(content=m["content"]))
        else:
            converted.append(AIMessage(content=m["content"]))
    return converted


def build_conversational_chain(retriever, model: str = "gpt-4o-mini"):
    """LCEL RAG chain that takes {"question", "history"} and remembers context.

    history is a list of langchain_core messages (see ``to_lc_messages``)
    covering the turns *before* the current question.
    """
    llm = ChatOpenAI(model=model)
    parser = StrOutputParser()

    contextualize = CONTEXTUALIZE_PROMPT | llm | parser

    def standalone_question(x: dict) -> str:
        # Only spend an LLM call rewriting when there is history to resolve.
        if x.get("history"):
            return contextualize.invoke(x)
        return x["question"]

    retrieve_context = (
        RunnableLambda(standalone_question) | retriever | _format_docs
    )

    return (
        RunnablePassthrough.assign(context=retrieve_context)
        | CONVERSATIONAL_PROMPT
        | llm
        | parser
    )


def build_conversational_rag(
    pdf_path: str | None = None, k: int = 3, model: str = "gpt-4o-mini"
):
    """Convenience: build retriever + conversational chain in one call."""
    retriever = build_retriever(pdf_path=pdf_path, k=k)
    return build_conversational_chain(retriever, model=model)


# ---------------------------------------------------------------------------
# Memory RAG — faithful to Capstone_project_LCEL_Memory.ipynb
# ---------------------------------------------------------------------------
# Differences from build_conversational_rag above (kept on purpose to match the
# notebook served by app_rbs2025.py):
#   * persistent Chroma store (collection "capstone_project_RAG"), retriever k=4
#   * no query-rewrite step — retrieves on the raw question; memory lives in the
#     prompt's history placeholder
#   * memory managed by RunnableWithMessageHistory + a per-session store


def _return_only_text(documents: list[Document]) -> str:
    """Notebook's context formatter: join chunk texts with a separator."""
    return "\n\n...\n\n".join(doc.page_content for doc in documents)


def build_memory_retriever(
    pdf_path: str | None = None,
    k: int = 4,
    persist_directory: str = "Chroma_capstone_store",
    collection_name: str = "capstone_project_RAG",
):
    """Persistent Chroma retriever, as in the notebook.

    Idempotent on purpose: only embeds the corpus when the collection is empty,
    so a warm restart (existing store on disk) does not duplicate vectors — the
    bug that left the notebook's store with 3x the chunks (228 = 76 x 3).
    """
    chunks = load_chunks(pdf_path)
    for c in chunks:
        c.metadata["source"] = "Capstone_FinalReport.pdf"

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )
    if store._collection.count() == 0:
        store.add_documents(chunks)
    return store.as_retriever(search_kwargs={"k": k})


def build_memory_chain(retriever, model: str = "gpt-4o-mini"):
    """The notebook's LCEL chain: retrieve on the raw question, answer with history."""
    llm = ChatOpenAI(model=model)
    parser = StrOutputParser()
    format_runnable = RunnableLambda(_return_only_text)

    return (
        {
            "question": itemgetter("question"),
            "context": itemgetter("question") | retriever | format_runnable,
            "history": itemgetter("history"),
        }
        | CONVERSATIONAL_PROMPT  # the Clara prompt_conv (same wording as the notebook)
        | llm
        | parser
    )


def build_memory_rag(
    pdf_path: str | None = None, k: int = 4, model: str = "gpt-4o-mini"
):
    """Notebook's conversational-memory RAG wrapped in RunnableWithMessageHistory.

    Invoke with ``{"question": ...}`` and a session id, e.g.::

        rag.invoke({"question": q}, config={"configurable": {"session_id": sid}})

    Each session id gets its own in-memory chat history, held in the closure
    below (lives as long as the returned object — cache it once per process).
    """
    retriever = build_memory_retriever(pdf_path=pdf_path, k=k)
    chain = build_memory_chain(retriever, model=model)

    store: dict[str, BaseChatMessageHistory] = {}

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]

    return RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="history",
    )
