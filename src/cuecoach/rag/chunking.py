from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import List, Optional

from cuecoach.rag.schemas import Chunk, SkillLevel, Topic


@dataclass(frozen=True)
class DocMeta:
    doc_id: str
    source: str
    title: str
    url: Optional[str] = None
    section: Optional[str] = None
    topic: Topic = "misc"
    skill_level: SkillLevel = "beginner"


def _stable_chunk_id(doc_id: str, section: Optional[str], idx: int, text: str) -> str:
    # Stable across machines and runs. Good for citations + dedupe.
    h = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
    sec = section or "root"
    return f"{doc_id}::{sec}::{idx:04d}::{h}"


def chunk_text(
    text: str,
    meta: DocMeta,
    *,
    max_chars: int = 1400,
    overlap_chars: int = 200,
) -> List[Chunk]:
    """
    Chunk text into overlapping segments.

    Why chars not tokens (for now):
    - Offline chunking should be fast and dependency-light.
    - We'll move to token-aware chunking later once OpenAI is in the loop.
    """
    cleaned = "\n".join([line.rstrip() for line in text.splitlines()]).strip()
    if not cleaned:
        return []

    # Split by blank lines (paragraph-ish). Works well for instructional articles and rules PDFs.
    parts = [p.strip() for p in cleaned.split("\n\n") if p.strip()]

    chunks: List[str] = []
    buf = ""

    for p in parts:
        candidate = (buf + "\n\n" + p).strip() if buf else p
        if len(candidate) <= max_chars:
            buf = candidate
            continue

        # Flush current buffer if it has content
        if buf:
            chunks.append(buf)

            # Start new buffer with overlap tail
            tail = buf[-overlap_chars:] if overlap_chars > 0 else ""
            buf = (tail + "\n\n" + p).strip() if tail else p
        else:
            # Single paragraph too large: hard split
            start = 0
            while start < len(p):
                end = min(start + max_chars, len(p))
                chunks.append(p[start:end].strip())
                start = max(0, end - overlap_chars)

            buf = ""

    if buf:
        chunks.append(buf)

    out: List[Chunk] = []
    for i, ctext in enumerate(chunks, start=1):
        out.append(
            Chunk(
                chunk_id=_stable_chunk_id(meta.doc_id, meta.section, i, ctext),
                doc_id=meta.doc_id,
                source=meta.source,
                title=meta.title,
                section=meta.section,
                url=meta.url,
                topic=meta.topic,
                skill_level=meta.skill_level,
                text=ctext,
            )
        )
    return out
