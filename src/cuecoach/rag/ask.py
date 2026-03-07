from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone


# -----------------------------
# Defaults
# -----------------------------
DEFAULT_NAMESPACE = "default"
DEFAULT_EMBED_MODEL = "text-embedding-3-small"
DEFAULT_CHAT_MODEL = "gpt-4.1-mini"
DEFAULT_TOP_K = 6


# -----------------------------
# Env
# -----------------------------
def _load_env() -> None:
    # load from repo root .env
    load_dotenv(dotenv_path=Path(".env"), override=False)


def require_env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise SystemExit(f"Missing required env var: {name}")
    return v


def optional_env(name: str, default: str) -> str:
    return os.getenv(name, default)


# -----------------------------
# Retrieval
# -----------------------------
def embed_query(client: OpenAI, model: str, text: str) -> List[float]:
    resp = client.embeddings.create(model=model, input=text)
    return resp.data[0].embedding


def pinecone_query(
    index: Any,
    namespace: str,
    vector: List[float],
    top_k: int,
) -> List[Dict[str, Any]]:
    res = index.query(
        namespace=namespace,
        vector=vector,
        top_k=top_k,
        include_metadata=True,
    )
    matches = getattr(res, "matches", None) or []
    out: List[Dict[str, Any]] = []
    for m in matches:
        out.append(
            {
                "id": getattr(m, "id", ""),
                "score": float(getattr(m, "score", 0.0)),
                "metadata": getattr(m, "metadata", {}) or {},
            }
        )
    return out


def build_context(matches: List[Dict[str, Any]], *, text_key: str = "text") -> str:
    """
    We store chunk text inside Pinecone metadata['text'].
    Build a compact, structured context the model can quote from.
    """
    parts: List[str] = []
    for i, m in enumerate(matches, start=1):
        md = m.get("metadata", {}) or {}
        chunk_text = (md.get(text_key) or "").strip()
        if not chunk_text:
            continue

        title = (md.get("title") or "").strip()
        doc_id = (md.get("doc_id") or "").strip()
        section = (md.get("section") or "").strip()
        topic = (md.get("topic") or "").strip()
        skill = (md.get("skill_level") or "").strip()

        header_bits = [b for b in [title, doc_id, section, topic, skill] if b]
        header = " | ".join(header_bits) if header_bits else "unknown-source"

        parts.append(f"[SOURCE {i}: {header}]\n{chunk_text}")

    return "\n\n".join(parts).strip()


def truncate_context(context: str, max_chars: int) -> str:
    """
    Keep the context under a limit (rough safety against huge prompts).
    Truncate from the end because early sources are usually best matches.
    """
    context = context.strip()
    if max_chars <= 0:
        return context
    if len(context) <= max_chars:
        return context
    return context[:max_chars].rstrip() + "\n"


# -----------------------------
# Answering (core)
# -----------------------------
def answer_question(
    client: OpenAI,
    model: str,
    question: str,
    context: str,
) -> str:
    """
    Tight grounding:
    - Use ONLY the context
    - Quote exact phrases where possible (this helps smoke_eval keyword checks)
    - No citations, no top-k dumps, just the answer
    """
    system = (
        "You are a strict rules assistant.\n"
        "You must answer using ONLY the provided CONTEXT.\n"
        "If the CONTEXT does not contain the answer, respond exactly:\n"
        "I don't know based on the provided material.\n"
        "Style rules:\n"
        "- Prefer copying exact rule wording from the context instead of paraphrasing.\n"
        "- Keep answers short and direct.\n"
        "- Do not mention 'context' or 'sources' in the answer.\n"
        "- Do not add extra explanations beyond what the context states.\n"
    )

    user = f"""QUESTION:
{question}

CONTEXT:
{context if context else "(empty)"}
"""

    resp = client.chat.completions.create(
        model=model,
        temperature=0.0,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return (resp.choices[0].message.content or "").strip()


# -----------------------------
# Answering (wrapper for eval/scripts)
# -----------------------------
def answer(
    question: str,
    *,
    top_k: int = DEFAULT_TOP_K,
    min_score: float = 0.1,
    max_context_chars: int = 20000,
    namespace: Optional[str] = None,
    embed_model: Optional[str] = None,
    chat_model: Optional[str] = None,
) -> str:
    """
    Convenience wrapper that:
    - Loads env
    - Builds clients
    - Retrieves context
    - Answers the question
    """
    _load_env()

    # Required env vars (keep in script)
    openai_key = require_env("OPENAI_API_KEY")
    pinecone_key = require_env("PINECONE_API_KEY")
    pinecone_index = require_env("PINECONE_INDEX")

    # Config (env default, but overridable by function args)
    ns = namespace or optional_env("PINECONE_NAMESPACE", DEFAULT_NAMESPACE)
    emb_model = embed_model or optional_env("OPENAI_EMBED_MODEL", DEFAULT_EMBED_MODEL)
    ch_model = chat_model or optional_env("OPENAI_CHAT_MODEL", DEFAULT_CHAT_MODEL)

    oai = OpenAI(api_key=openai_key)
    pc = Pinecone(api_key=pinecone_key)
    idx = pc.Index(pinecone_index)

    qvec = embed_query(oai, emb_model, question)
    matches = pinecone_query(idx, ns, qvec, top_k)

    if min_score > 0:
        matches = [m for m in matches if m.get("score", 0.0) >= min_score]

    context = build_context(matches)
    context = truncate_context(context, max_context_chars)

    return answer_question(oai, ch_model, question, context)


# -----------------------------
# CLI
# -----------------------------
def main() -> None:
    _load_env()

    parser = argparse.ArgumentParser(description="Ask questions against Pinecone RAG index.")
    parser.add_argument("--question", required=True, help="User question.")
    parser.add_argument("--debug", action="store_true", help="Print retrieval debug info.")
    parser.add_argument("--top-k", type=int, default=6, help="How many matches to retrieve.")
    parser.add_argument("--min-score", type=float, default=0.0, help="Drop matches below this score.")
    parser.add_argument("--max-context-chars", type=int, default=12000, help="Max context chars fed to LLM")
    args = parser.parse_args()

    # Required env vars (keep in script)
    openai_key = require_env("OPENAI_API_KEY")
    pinecone_key = require_env("PINECONE_API_KEY")
    pinecone_index = require_env("PINECONE_INDEX")
    namespace = optional_env("PINECONE_NAMESPACE", "default")

    # Models (configurable)
    embed_model = optional_env("OPENAI_EMBED_MODEL", "text-embedding-3-small")
    chat_model = optional_env("OPENAI_CHAT_MODEL", "gpt-4.1-mini")

    oai = OpenAI(api_key=openai_key)
    pc = Pinecone(api_key=pinecone_key)
    idx = pc.Index(pinecone_index)

    qvec = embed_query(oai, embed_model, args.question)
    matches = pinecone_query(idx, namespace, qvec, args.top_k)

    # Filter low-score matches (kept simple, optional)
    if args.min_score > 0:
        matches = [m for m in matches if m.get("score", 0.0) >= args.min_score]

    context = build_context(matches)
    context = truncate_context(context, args.max_context_chars)

    if args.debug:
        for m in matches:
            md = m.get("metadata", {}) or {}
            preview = (md.get("text") or "").strip().replace("\n", " ")
            if len(preview) > 160:
                preview = preview[:160] + "..."
            print(f"score: {m.get('score')}")
            print(f"id: {m.get('id')}")
            print(f"title: {md.get('title')}")
            print(f"topic: {md.get('topic')}")
            print(f"text_preview: {preview}\n")

    ans = answer_question(oai, chat_model, args.question, context)
    print(ans)


if __name__ == "__main__":
    main()
