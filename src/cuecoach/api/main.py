from fastapi import FastAPI
from pydantic import BaseModel

from cuecoach.rag.ask import answer
from cuecoach.rag.langchain_ask import answer_langchain

app = FastAPI(title="CueCoach QA API")


class AskRequest(BaseModel):
    question: str
    mode: str = "explain"
    top_k: int = 8
    min_score: float = 0.0
    max_context_chars: int = 12000


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

    if mode == "strict":
        ans = answer(
            payload.question,
            top_k=payload.top_k,
            min_score=payload.min_score,
            max_context_chars=payload.max_context_chars,
        )
    else:
        ans = answer_langchain(
            payload.question,
            top_k=payload.top_k,
            min_score=payload.min_score,
            max_context_chars=payload.max_context_chars,
        )

    return AskResponse(answer=ans, mode=mode)