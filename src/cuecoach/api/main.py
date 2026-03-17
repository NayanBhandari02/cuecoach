from typing import List

from fastapi import FastAPI
from pydantic import BaseModel

from cuecoach.rag.ask import answer
from cuecoach.rag.langchain_ask import answer_langchain

from fastapi.middleware.cors import CORSMiddleware



app = FastAPI(title="CueCoach QA API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatMessage(BaseModel):
    role: str
    content: str


class AskRequest(BaseModel):
    question: str
    mode: str = "explain"
    top_k: int = 5
    min_score: float = 0.42
    max_context_chars: int = 12000
    chat_history: List[ChatMessage] = []


class AskResponse(BaseModel):
    answer: str
    mode: str


@app.get("/")
def root() -> dict:
    return {
        "message": "CueCoach QA API is running",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask_question(payload: AskRequest) -> AskResponse:
    mode = (payload.mode or "explain").strip().lower()

    if mode not in {"strict", "explain"}:
        mode = "explain"

    history = [{"role": m.role, "content": m.content} for m in payload.chat_history]

    if mode == "strict":
        ans = answer(
            payload.question,
            top_k=payload.top_k,
            min_score=payload.min_score,
            max_context_chars=payload.max_context_chars,
            chat_history=history,
        )
    else:
        ans = answer_langchain(
            payload.question,
            top_k=payload.top_k,
            min_score=payload.min_score,
            max_context_chars=payload.max_context_chars,
            chat_history=history,
        )

    return AskResponse(answer=ans, mode=mode)