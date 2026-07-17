"""Reusable RAG pipeline for the Capstone project.

Extracted from notebooks/Capstone_project_RAG_LCEL.ipynb so it can be served
by app.py (Streamlit) or any other entry point. The OpenAI API key is read
from the environment (loaded from .env locally, or from Streamlit secrets in
the cloud).
"""

from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from pdf_ingestion import PdfIngestion

# data/pdf_files lives two levels up from this file (src/rag.py -> project root)
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "pdf_files"

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
