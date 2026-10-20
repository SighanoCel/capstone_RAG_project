"""Conversational RAG over Celestin Sighano's portfolio — assistant "Celestin".

Extracted from notebooks/portfolio_LCEL_Memory.ipynb so it can be served by
api_celestin.py, the chat API behind the "Ask to Celestin" widget.

Context = the live portfolio page (Webingestion, 500/100) + the notebook
summaries text file (Textingestion, 1000/250) + the English CV PDF (split by
section, long sections 500/100),
embedded with
text-embedding-3-small into Chroma, k retriever (API uses 5), gpt-4o-mini, memory via
RunnableWithMessageHistory keyed by session id.
"""

import os
import re
from operator import itemgetter
from pathlib import Path

# WebBaseLoader warns if no user agent is set
os.environ.setdefault("USER_AGENT", "celestin-portfolio-assistant")

from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

try:
    from text_ingestion import Textingestion, clean_text
    from web_ingestion import Webingestion
except ModuleNotFoundError:  # imported as notebooks.src.portfolio_rag
    from notebooks.src.text_ingestion import Textingestion, clean_text
    from notebooks.src.web_ingestion import Webingestion

PORTFOLIO_URL = "https://sighanocel.github.io/My-data-science-project/"

# data/pdf_files lives at the project root
# (notebooks/src/portfolio_rag.py -> notebooks/src -> notebooks -> project root).
PDF_DIR = Path(__file__).resolve().parents[2] / "data" / "pdf_files"
TEXT_FILE = PDF_DIR / "text_files" / "Notebook summary for RAG.txt"
CV_FILE = PDF_DIR / "data_Science_CV_english.pdf"

# Top-level section headings of the CV (the PDF hyphenates some across lines).
CV_SECTIONS = re.compile(
    r"\b(PROJECTS|WORK EXPERIENCE|EDUCATION AND TRAIN\W*\s*ING|LANGUAGE SKILLS|DRIVING LICENCE"
    r"|PROFESSIONAL SKILLS|COMMUNICATION AND INTERPERSONAL SKILLS|ORGANIZATIONAL SKILLS"
    r"|PERSONAL DATA PRO\W*\s*CESSING)\b"
)

# Plain-language synonyms added to each section label so generic questions
# ("professional experience", "where did he study") reach the right section.
CV_SECTION_ALIASES = {
    "ABOUT ME": "profile, summary, who he is",
    "PROJECTS": "data science and AI projects",
    "WORK EXPERIENCE": "professional experience, jobs, internships, employment history, companies",
    "EDUCATION AND TRAINING": "education, degrees, diplomas, studies, schools, universities",
    "LANGUAGE SKILLS": "languages spoken",
    "PROFESSIONAL SKILLS": "technical skills, tools, programming",
}

PROMPT_CONV = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an assistant who provides information about the Celestin Sighano portfolio web site and CV. "
            "Your name is Celestin. "
            "Answer questions based on the provided context. "
            "You must answer in the language used by the latest question, even if earlier messages used another language. "
            "If the human greets you, you can also greet him and tell him what your name is. "
            "But you have to avoid any other conversation with him different from the provided context.",
        ),
        MessagesPlaceholder("history"),
        (
            "user",
            "These are the significant excerpts from the portfolio web site and CV: \n\n{context}\n\n"
            "My Question:{question}",
        ),
    ]
)


def return_only_text(documents):
    return "\n\n...\n\n".join(doc.page_content for doc in documents)


def load_cv_chunks(cv_path: Path = CV_FILE):
    """Chunks of the CV PDF, each tagged as coming from the CV.

    Plain PyMuPDF text (not PdfIngestion): the CV layout uses tables, which
    PdfIngestion drops, and its block joining glues words across lines.
    """
    pages = PyMuPDFLoader(str(cv_path)).load()
    text = clean_text(" ".join(p.page_content for p in pages))

    # Split by CV section so every chunk carries its section title — helps the
    # retriever match e.g. "work experience" to the right part of the CV.
    parts = CV_SECTIONS.split(text)
    sections = [("ABOUT ME", parts[0])] + [
        (re.sub(r"\W*\s+(?=ING|CESSING)", "", parts[i]), parts[i + 1])  # "TRAIN‐ ING" -> "TRAINING"
        for i in range(1, len(parts) - 1, 2)
    ]

    # Short sections stay whole (splitting education mixed up schools and dates);
    # long ones get small chunks so they compete with the short skill sections.
    splitter = Textingestion(chunk_size=500, chunk_overlap=100).splitter
    chunks = []
    for title, body in sections:
        body = body.strip()
        if not body:
            continue
        pieces = [body] if len(body) <= 1000 else splitter.split_text(body)
        for piece in pieces:
            c = Document(page_content=piece, metadata={"source": cv_path.name})
            alias = f" ({CV_SECTION_ALIASES[title]})" if title in CV_SECTION_ALIASES else ""
            c.page_content = f"From Celestin Sighano's CV, section {title}{alias}: {c.page_content}"
            chunks.append(c)
    return chunks


def load_chunks(url: str = PORTFOLIO_URL, text_path: Path = TEXT_FILE, cv_path: Path = CV_FILE):
    """Web chunks of the portfolio page + notebook summaries + CV."""
    web_chunks = Webingestion(chunk_size=500, chunk_overlap=100).process(url)
    text_chunks = Textingestion(chunk_size=1000, chunk_overlap=250).process(str(text_path))
    for c in text_chunks:
        c.metadata["source"] = text_path.name
    return web_chunks + text_chunks + load_cv_chunks(cv_path)


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
