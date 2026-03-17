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
DEFAULT_TOP_K = 5


# -----------------------------
# Env
# -----------------------------
def _load_env() -> None:
    load_dotenv(dotenv_path=Path(".env"), override=False)


def require_env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise SystemExit(f"Missing required env var: {name}")
    return v


def optional_env(name: str, default: str) -> str:
    return os.getenv(name, default)

def contextualize_question_with_history(
    client: OpenAI,
    model: str,
    question: str,
    chat_history: Optional[List[Dict[str, str]]] = None,
) -> str:
    """
    Rewrite a follow-up user question into a standalone question using same-chat history.
    If no history is provided, return the original question unchanged.
    """
    if not chat_history:
        return question

    history_lines: List[str] = []
    for msg in chat_history[-8:]:
        role = str(msg.get("role", "")).strip().lower()
        content = str(msg.get("content", "")).strip()
        if not content or role not in {"user", "assistant"}:
            continue
        history_lines.append(f"{role}: {content}")

    if not history_lines:
        return question

    system = (
        "You rewrite the latest user question into a standalone question using the chat history.\n"
        "Rules:\n"
        "- Preserve the user's meaning.\n"
        "- Resolve references like 'it', 'that', 'this', 'those', 'him', 'her'.\n"
        "- Keep the rewritten question concise.\n"
        "- Do not answer the question.\n"
        "- Return only the rewritten standalone question.\n"
    )

    user = (
        "Chat history:\n"
        + "\n".join(history_lines)
        + f"\n\nLatest user question:\n{question}"
    )

    resp = client.chat.completions.create(
        model=model,
        temperature=0.0,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )

    rewritten = (resp.choices[0].message.content or "").strip()
    return rewritten or question

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
    Keep the context under a limit.
    Truncate from the end because early sources are usually best matches.
    """
    context = context.strip()
    if max_chars <= 0:
        return context
    if len(context) <= max_chars:
        return context
    return context[:max_chars].rstrip() + "\n"


def rewrite_query_for_retrieval(question: str) -> str:
    """
    Lightweight query rewriting for retrieval.
    Keeps user-facing question unchanged, but improves vector search wording.
    """
    q = question.strip().lower()

    rewrites = {
        "what makes a shot legally completed?": "legal shot completion cue ball contacts object ball then ball pocketed or driven to cushion rule",
        "when is the cue ball considered in play?": "cue ball in play rule definition",
        "what are the options after an illegal break in pyramid?": "illegal break pyramid options rerack accept assign rebreak",
        "what is the rule about double hit?": "double hit cue ball more than once foul rule",
    }

    return rewrites.get(q, question)


def select_matches_by_confidence(matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Confidence-aware trimming:
    - < 0.42: treat as too weak
    - 0.42 to < 0.52: keep top 4
    - 0.52 to < 0.62: keep top 3
    - >= 0.62: keep top 2
    """
    best_score = matches[0]["score"] if matches else 0.0

    if best_score < 0.42:
        return []
    if best_score >= 0.62:
        return matches[:2]
    if best_score >= 0.52:
        return matches[:3]
    return matches[:4]

def rewrite_query_with_llm(
    client: OpenAI,
    model: str,
    question: str,
) -> str:
    """
    Rewrite a user question into a better retrieval query.
    Focus on fixing typos, ambiguity, and converting natural wording
    into concise rulebook/search phrasing.
    """
    system = (
        "You rewrite user questions into short retrieval queries for a billiards rules knowledge base.\n"
        "Rules:\n"
        "- Fix spelling mistakes and typos.\n"
        "- Keep the meaning the same.\n"
        "- Make the query concise and search-friendly.\n"
        "- Prefer rulebook terms over conversational wording.\n"
        "- Return only the rewritten query.\n"
        "- Do not answer the question.\n"
    )

    user = f"User question:\n{question}"

    resp = client.chat.completions.create(
        model=model,
        temperature=0.0,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )

    rewritten = (resp.choices[0].message.content or "").strip()
    return rewritten or question

def retrieve_with_fallback(
    *,
    oai: OpenAI,
    idx: Any,
    namespace: str,
    emb_model: str,
    chat_model: str,
    question: str,
    top_k: int,
    min_score: float,
) -> Dict[str, Any]:
    """
    Retrieval flow:
    1. static/manual rewrite
    2. retrieve
    3. if weak, LLM rewrite and retrieve again
    4. keep the better result
    """
    first_query = rewrite_query_for_retrieval(question)
    first_vec = embed_query(oai, emb_model, first_query)
    first_matches = pinecone_query(idx, namespace, first_vec, top_k)

    if min_score > 0:
        first_matches = [m for m in first_matches if m.get("score", 0.0) >= min_score]

    first_best = first_matches[0]["score"] if first_matches else 0.0

    # If first retrieval is already decent, keep it
    if first_best >= 0.52:
        return {
            "retrieval_query": first_query,
            "matches": first_matches,
            "best_score": first_best,
            "used_fallback": False,
        }

    second_query = rewrite_query_with_llm(oai, chat_model, question)
    second_vec = embed_query(oai, emb_model, second_query)
    second_matches = pinecone_query(idx, namespace, second_vec, top_k)

    if min_score > 0:
        second_matches = [m for m in second_matches if m.get("score", 0.0) >= min_score]

    second_best = second_matches[0]["score"] if second_matches else 0.0

    if second_best > first_best:
        return {
            "retrieval_query": second_query,
            "matches": second_matches,
            "best_score": second_best,
            "used_fallback": True,
        }

    return {
        "retrieval_query": first_query,
        "matches": first_matches,
        "best_score": first_best,
        "used_fallback": False,
    }

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
    - Prefer exact wording where possible
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
# Debug retrieval
# -----------------------------
def retrieve_debug(
    question: str,
    *,
    top_k: int = DEFAULT_TOP_K,
    min_score: float = 0.42,
    namespace: Optional[str] = None,
    embed_model: Optional[str] = None,
) -> Dict[str, Any]:
    _load_env()

    openai_key = require_env("OPENAI_API_KEY")
    pinecone_key = require_env("PINECONE_API_KEY")
    pinecone_index = require_env("PINECONE_INDEX")

    ns = namespace or optional_env("PINECONE_NAMESPACE", DEFAULT_NAMESPACE)
    emb_model = embed_model or optional_env("OPENAI_EMBED_MODEL", DEFAULT_EMBED_MODEL)
    ch_model = optional_env("OPENAI_CHAT_MODEL", DEFAULT_CHAT_MODEL)

    oai = OpenAI(api_key=openai_key)
    pc = Pinecone(api_key=pinecone_key)
    idx = pc.Index(pinecone_index)

    return retrieve_with_fallback(
        oai=oai,
        idx=idx,
        namespace=ns,
        emb_model=emb_model,
        chat_model=ch_model,
        question=question,
        top_k=top_k,
        min_score=min_score,
    )
# -----------------------------
# Answering (wrapper for eval/scripts)
# -----------------------------
def answer(
    question: str,
    *,
    top_k: int = DEFAULT_TOP_K,
    min_score: float = 0.42,
    max_context_chars: int = 20000,
    namespace: Optional[str] = None,
    embed_model: Optional[str] = None,
    chat_model: Optional[str] = None,
    chat_history: Optional[List[Dict[str, str]]] = None,
) -> str:
    """
    Convenience wrapper that:
    - Loads env
    - Builds clients
    - Retrieves context
    - Applies confidence gating
    - Answers only when retrieval is strong enough
    """
    _load_env()

    openai_key = require_env("OPENAI_API_KEY")
    pinecone_key = require_env("PINECONE_API_KEY")
    pinecone_index = require_env("PINECONE_INDEX")

    ns = namespace or optional_env("PINECONE_NAMESPACE", DEFAULT_NAMESPACE)
    emb_model = embed_model or optional_env("OPENAI_EMBED_MODEL", DEFAULT_EMBED_MODEL)
    ch_model = chat_model or optional_env("OPENAI_CHAT_MODEL", DEFAULT_CHAT_MODEL)

    oai = OpenAI(api_key=openai_key)
    pc = Pinecone(api_key=pinecone_key)
    idx = pc.Index(pinecone_index)
    standalone_question = contextualize_question_with_history(
        oai,
        ch_model,
        question,
        chat_history,
    )
    retrieval_result = retrieve_with_fallback(
        oai=oai,
        idx=idx,
        namespace=ns,
        emb_model=emb_model,
        chat_model=ch_model,
        question=standalone_question,
        top_k=top_k,
        min_score=min_score,
    )

    matches = retrieval_result["matches"]

    matches = select_matches_by_confidence(matches)
    context = build_context(matches)
    context = truncate_context(context, max_context_chars)

    if not matches or not context.strip():
        return "I don't know based on the provided material."

    return answer_question(oai, ch_model, standalone_question, context)


# -----------------------------
# CLI
# -----------------------------
def main() -> None:
    _load_env()

    parser = argparse.ArgumentParser(description="Ask questions against Pinecone RAG index.")
    parser.add_argument("--question", required=True, help="User question.")
    parser.add_argument("--debug", action="store_true", help="Print retrieval debug info.")
    parser.add_argument("--top-k", type=int, default=5, help="How many matches to retrieve.")
    parser.add_argument("--min-score", type=float, default=0.42, help="Drop matches below this score.")
    parser.add_argument("--max-context-chars", type=int, default=12000, help="Max context chars fed to LLM")
    args = parser.parse_args()

    openai_key = require_env("OPENAI_API_KEY")
    pinecone_key = require_env("PINECONE_API_KEY")
    pinecone_index = require_env("PINECONE_INDEX")
    namespace = optional_env("PINECONE_NAMESPACE", DEFAULT_NAMESPACE)

    embed_model = optional_env("OPENAI_EMBED_MODEL", DEFAULT_EMBED_MODEL)
    chat_model = optional_env("OPENAI_CHAT_MODEL", DEFAULT_CHAT_MODEL)

    oai = OpenAI(api_key=openai_key)
    pc = Pinecone(api_key=pinecone_key)
    idx = pc.Index(pinecone_index)

    retrieval_query = rewrite_query_for_retrieval(args.question)
    qvec = embed_query(oai, embed_model, retrieval_query)
    matches = pinecone_query(idx, namespace, qvec, args.top_k)

    if args.min_score > 0:
        matches = [m for m in matches if m.get("score", 0.0) >= args.min_score]

    raw_matches = matches
    best_score = raw_matches[0]["score"] if raw_matches else 0.0

    matches = select_matches_by_confidence(raw_matches)
    context = build_context(matches)
    context = truncate_context(context, args.max_context_chars)

    if args.debug:
        print(f"best_score: {best_score:.4f}")
        print(f"raw_matches: {len(raw_matches)}")
        print(f"selected_matches: {len(matches)}\n")

        for m in matches:
            md = m.get("metadata", {}) or {}
            preview = (md.get("text") or "").strip().replace("\n", " ")
            if len(preview) > 160:
                preview = preview[:160] + "..."
            print(f"score: {m.get('score'):.4f}")
            print(f"id: {m.get('id')}")
            print(f"title: {md.get('title')}")
            print(f"topic: {md.get('topic')}")
            print(f"text_preview: {preview}\n")

    if not matches or not context.strip():
        print("I don't know based on the provided material.")
        return

    ans = answer_question(oai, chat_model, args.question, context)
    print(ans)


if __name__ == "__main__":
    main()