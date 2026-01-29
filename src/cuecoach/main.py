from fastapi import FastAPI
from fastapi.responses import ORJSONResponse

app = FastAPI(
    title="CueCoach RAG API",
    version="0.1.0",
    default_response_class=ORJSONResponse,
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
