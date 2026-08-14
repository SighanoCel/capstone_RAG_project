"""Reusable RAG pipeline for the Capstone project.

Extracted from notebooks/Capstone_project_RAG_LCEL.ipynb so it can be served
by app.py (Streamlit) or any other entry point. The OpenAI API key is read
from the environment (loaded from .env locally, or from Streamlit secrets in
the cloud).
"""

from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
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
# the "chat_history" key. The caller (app.py) owns the conversation state, so
# the same cached chain can serve every turn and every user.

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
        MessagesPlaceholder("chat_history"),
        ("human", "{question}"),
    ]
)

# Answer prompt: same instructions as PROMPT above, but message-based so it can
# carry the prior turns alongside the retrieved context.
CONVERSATIONAL_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a technical assistant for our data analytics team.\n"
            "Answer the question focusing on the context below.\n"
            'If there is no answer in the context, just say: "there is no answer"\n'
            "Be precise and very concise.\n\n"
            "CONTEXT:\n{context}",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}"),
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
    """LCEL RAG chain that takes {"question", "chat_history"} and remembers context.

    chat_history is a list of langchain_core messages (see ``to_lc_messages``)
    covering the turns *before* the current question.
    """
    llm = ChatOpenAI(model=model)
    parser = StrOutputParser()

    contextualize = CONTEXTUALIZE_PROMPT | llm | parser

    def standalone_question(x: dict) -> str:
        # Only spend an LLM call rewriting when there is history to resolve.
        if x.get("chat_history"):
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
