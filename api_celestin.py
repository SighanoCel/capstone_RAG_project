"""Chat API for "Ask to Celestin" — the assistant embedded in the portfolio.

The portfolio (GitHub Pages) is static, so the OpenAI key can't live in the
browser: the chat widget (My-data-science-project/js/celestin.js) calls this
API, which runs the RAG pipeline of notebooks/portfolio_LCEL_Memory.ipynb
(notebooks/src/portfolio_rag.py) with per-session conversational memory.

Run locally:   uvicorn api_celestin:app --reload --port 8000
Deploy:        Render (render.yaml) -> set OPENAI_API_KEY in the dashboard
"""

import os
import sys
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

# Make the module folder importable (portfolio_rag.py lives in notebooks/src/)
sys.path.append(str(Path(__file__).resolve().parent / "notebooks" / "src"))

from portfolio_rag import build_celestin_rag  # noqa: E402

load_dotenv(Path(__file__).resolve().parent / ".env")

# Only the portfolio (and local testing) may call the API from a browser.
ALLOWED_ORIGINS = [
    "https://sighanocel.github.io",
    "http://localhost:5500",
    "http://127.0.0.1:5500",
    "http://localhost:8080",
    "http://127.0.0.1:8080",
]

# Basic abuse protection — the endpoint spends your OpenAI credits.
MAX_QUESTION_CHARS = 1000
RATE_LIMIT = 20  # questions per IP...
RATE_WINDOW = 60 * 10  # ...per 10 minutes
_hits: dict[str, deque] = defaultdict(deque)

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Build the vector store once at startup (~150 chunks, a few seconds).
    state["chain"] = await run_in_threadpool(build_celestin_rag, k=3, model="gpt-4o-mini")
    yield


app = FastAPI(title="Ask to Celestin", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    session_id: str = Field(min_length=8, max_length=64)


class ChatResponse(BaseModel):
    answer: str


def _check_rate_limit(ip: str):
    now = time.time()
    hits = _hits[ip]
    while hits and now - hits[0] > RATE_WINDOW:
        hits.popleft()
    if len(hits) >= RATE_LIMIT:
        raise HTTPException(429, "Too many questions — please try again in a few minutes.")
    hits.append(now)


@app.get("/health")
def health():
    return {"status": "ok", "ready": "chain" in state}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request):
    ip = request.headers.get("x-forwarded-for", request.client.host).split(",")[0].strip()
    _check_rate_limit(ip)
    answer = await run_in_threadpool(
        state["chain"].invoke,
        {"question": req.question},
        config={"configurable": {"session_id": req.session_id}},
    )
    return ChatResponse(answer=answer)
