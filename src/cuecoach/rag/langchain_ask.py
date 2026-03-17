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
    contextualize_question_with_history,
    needs_contextualization,
    embed_query,
    optional_env,
    retrieve_with_fallback,
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
    min_score: float = 0.42,
    max_context_chars: int = 20000,
    namespace: Optional[str] = None,
    embed_model: Optional[str] = None,
    chat_model: Optional[str] = None,
    chat_history: Optional[list[dict[str, str]]] = None,
) -> str:
    """
    Hybrid approach:
    - keep current Pinecone retrieval + confidence gating
    - use LangChain prompt + ChatOpenAI for a more natural explanation
    - use same-chat history to rewrite follow-up questions into standalone ones
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

    if needs_contextualization(question, chat_history):
        standalone_question = contextualize_question_with_history(
            oai,
            ch_model,
            question,
            chat_history,
        )
    else:
        standalone_question = question

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
    if not matches:
        return "I don't know based on the provided material."

    context = build_context(matches)
    context = truncate_context(context, max_context_chars)
    if not context.strip():
        return "I don't know based on the provided material."

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
                    "I don't know based on the provided material.\n\n"
                    "When the answer is available, structure it like this:\n"
                    "1. Give a short direct answer first.\n"
                    "2. Then explain what the rule means in practice.\n"
                    "3. If helpful, give one short example.\n\n"
                    "Keep the answer concise, useful, and easy to understand.\n"
                    "Do not mention the provided material, sources, or context."
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
        temperature=0.1,
    )

    chain = prompt | llm
    resp = chain.invoke({"question": standalone_question, "context": context})

    return str(resp.content).strip()