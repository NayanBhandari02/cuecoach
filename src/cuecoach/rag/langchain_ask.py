from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from cuecoach.rag.ask import (
    DEFAULT_CHAT_MODEL,
    DEFAULT_EMBED_MODEL,
    DEFAULT_NAMESPACE,
    DEFAULT_TOP_K,
    _load_env,
    build_context,
    embed_query,
    optional_env,
    pinecone_query,
    require_env,
    select_matches_by_confidence,
    truncate_context,
)
from openai import OpenAI
from pinecone import Pinecone


def _matches_to_documents(matches: List[Dict[str, Any]]) -> List[Document]:
    docs: List[Document] = []
    for m in matches:
        md = m.get("metadata", {}) or {}
        text = (md.get("text") or "").strip()
        if not text:
            continue

        docs.append(
            Document(
                page_content=text,
                metadata={
                    "id": m.get("id"),
                    "score": m.get("score"),
                    "title": md.get("title"),
                    "doc_id": md.get("doc_id"),
                    "section": md.get("section"),
                    "topic": md.get("topic"),
                    "skill_level": md.get("skill_level"),
                },
            )
        )
    return docs


def answer_langchain(
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
    Hybrid approach:
    - keep your current Pinecone retrieval + confidence gating
    - use LangChain prompt + ChatOpenAI for a more natural explanation
    """
    _load_env()

    openai_key = require_env("OPENAI_API_KEY")
    pinecone_key = require_env("PINECONE_API_KEY")
    pinecone_index = require_env("PINECONE_INDEX")

    ns = namespace or optional_env("PINECONE_NAMESPACE", DEFAULT_NAMESPACE)
    emb_model = embed_model or optional_env("OPENAI_EMBED_MODEL", DEFAULT_EMBED_MODEL)
    ch_model = chat_model or optional_env("OPENAI_CHAT_MODEL", DEFAULT_CHAT_MODEL)

    # Keep your existing retrieval logic
    oai = OpenAI(api_key=openai_key)
    pc = Pinecone(api_key=pinecone_key)
    idx = pc.Index(pinecone_index)

    qvec = embed_query(oai, emb_model, question)
    matches = pinecone_query(idx, ns, qvec, top_k)

    if min_score > 0:
        matches = [m for m in matches if m.get("score", 0.0) >= min_score]

    matches = select_matches_by_confidence(matches)
    if not matches:
        return "I don't know based on the provided material."

    context = build_context(matches)
    context = truncate_context(context, max_context_chars)
    if not context.strip():
        return "I don't know based on the provided material."

    # LangChain LLM layer
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are a helpful billiards coach and rules explainer.\n"
                    "Use the provided material as your primary reference.\n"
                    "Explain the answer clearly and naturally in your own words.\n"
                    "Do not invent facts that are not supported by the provided material.\n"
                    "If the material does not contain enough information, respond exactly:\n"
                    "I don't know based on the provided material."
                ),
            ),
            (
                "human",
                (
                    "Question:\n{question}\n\n"
                    "Provided material:\n{context}"
                ),
            ),
        ]
    )

    llm = ChatOpenAI(
        model=ch_model,
        temperature=0.2,
    )

    chain = prompt | llm
    resp = chain.invoke({"question": question, "context": context})

    return str(resp.content).strip()