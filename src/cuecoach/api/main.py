from fastapi import FastAPI
from pydantic import BaseModel

from cuecoach.rag.langchain_ask import answer_langchain

app = FastAPI(title="CueCoach QA API")


class AskRequest(BaseModel):
    question: str
    top_k: int = 8
    min_score: float = 0.0
    max_context_chars: int = 12000


class AskResponse(BaseModel):
    answer: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask_question(payload: AskRequest) -> AskResponse:
    ans = answer_langchain(
        payload.question,
        top_k=payload.top_k,
        min_score=payload.min_score,
        max_context_chars=payload.max_context_chars,
    )
    return AskResponse(answer=ans)

@app.get("/")
def root() -> dict:
    return {
        "message": "CueCoach QA API is running",
        "docs": "/docs",
        "health": "/health",
    }