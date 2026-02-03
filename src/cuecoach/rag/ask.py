from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

DEFAULT_NAMESPACE = "default"
DEFAULT_EMBED_MODEL = "text-embedding-3-small"
DEFAULT_CHAT_MODEL = "gpt-4o-mini"  # cheap + good enough for grounded answers


def _load_env() -> None:
    load_dotenv(dotenv_path=Path(".env"), override=False)


def require_env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise SystemExit(f"Missing required env var: {name}")
    return v


def optional_env(name: str, default: str) -> str:
    return os.getenv(name, default)


def embed_query(oai: OpenAI, model: str, text: str) -> List[float]:
    resp = oai.embeddings.create(model=model, input=[text])
    return resp.data[0].embedding


def build_context(matches: List[Any], *, max_chars: int) -> Tuple[str, List[Dict[str, str]]]:
    """
    Pull chunk text from Pinecone metadata and assemble a compact context.
    Returns:
      - context string
      - citations list (id/title/doc_id) for debug/printing
    """
    parts: List[str] = []
    cites: List[Dict[str, str]] = []
    used = 0

    for m in matches:
        md = getattr(m, "metadata", None) or {}
        chunk_text = (md.get("text") or "").strip()
        if not chunk_text:
            continue

        chunk_id = str(getattr(m, "id", ""))
        title = str(md.get("title", ""))
        doc_id = str(md.get("doc_id", ""))

        header = f"[{chunk_id} | {title} | {doc_id}]"
        block = f"{header}\n{chunk_text}\n"
        if used + len(block) > max_chars:
            remaining = max_chars - used
            if remaining <= 0:
                break
            block = block[:remaining]
        parts.append(block)
        used += len(block)

        cites.append({"id": chunk_id, "title": title, "doc_id": doc_id})

        if used >= max_chars:
            break

    return "\n".join(parts).strip(), cites


def answer_question(
    oai: OpenAI,
    chat_model: str,
    question: str,
    context: str,
) -> str:
    if not context.strip():
        return "I don't know based on the provided material."

    system = (
        "You answer questions using ONLY the provided context. "
        "If the context does not contain the answer, say: "
        "'I don't know based on the provided material.' "
        "Be concise and specific."
    )

    user = f"""Context:
{context}

Question: {question}
"""

    resp = oai.chat.completions.create(
        model=chat_model,
        temperature=0.2,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return resp.choices[0].message.content.strip()


def main() -> None:
    _load_env()

    parser = argparse.ArgumentParser(description="Ask a question using Pinecone RAG (no UI, no frameworks).")
    parser.add_argument("--question", required=True, help="Question to ask.")
    parser.add_argument("--namespace", default=optional_env("PINECONE_NAMESPACE", DEFAULT_NAMESPACE))
    parser.add_argument("--top-k", type=int, default=6, help="How many chunks to retrieve.")
    parser.add_argument("--context-chars", type=int, default=9000, help="Max chars of context fed to the LLM.")
    parser.add_argument("--debug", action="store_true", help="Print retrieved matches and preview context.")
    args = parser.parse_args()

    openai_key = require_env("OPENAI_API_KEY")
    pinecone_key = require_env("PINECONE_API_KEY")
    index_name = require_env("PINECONE_INDEX")

    embed_model = optional_env("OPENAI_EMBED_MODEL", DEFAULT_EMBED_MODEL)
    chat_model = optional_env("OPENAI_CHAT_MODEL", DEFAULT_CHAT_MODEL)

    oai = OpenAI(api_key=openai_key)
    pc = Pinecone(api_key=pinecone_key)
    index = pc.Index(index_name)

    qvec = embed_query(oai, embed_model, args.question)

    res = index.query(
        namespace=args.namespace,
        vector=qvec,
        top_k=args.top_k,
        include_metadata=True,
    )

    matches = getattr(res, "matches", None) or []

    context, cites = build_context(matches, max_chars=args.context_chars)
    if args.debug:
        print("\nRetrieved matches:")
        for i, m in enumerate(matches, start=1):
            md = getattr(m, "metadata", None) or {}
            print(
                f"{i}. score={getattr(m,'score',None)} id={getattr(m,'id','')} "
                f"title={md.get('title','')} doc_id={md.get('doc_id','')} "
                f"text_len={len((md.get('text') or '').strip())}"
            )
        print("\n--- Context preview (first 1000 chars) ---")
        print(context[:1000] if context else "[EMPTY]")
        print("\n----------------------------------------\n")

    ans = answer_question(oai, chat_model, args.question, context)
    print(ans)


if __name__ == "__main__":
    main()
