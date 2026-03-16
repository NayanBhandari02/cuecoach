from __future__ import annotations

import hashlib
import re
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
    h = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
    sec = section or "root"
    return f"{doc_id}::{sec}::{idx:04d}::{h}"


def _clean_text(text: str) -> str:
    """
    Light cleanup that preserves rule numbering and structure.
    """
    if not text:
        return ""

    lines: List[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()

        if not line:
            lines.append("")
            continue

        lowered = line.lower()
        if lowered in {
            "return to contents",
            "contents",
        }:
            continue

        line = re.sub(r"\s+", " ", line)
        lines.append(line)

    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def _looks_like_section_header(text: str) -> bool:
    """
    Heuristic for section-like lines.
    Examples:
    - 3.7 Double Hit / Frozen Balls
    - 11.3 Legal Break Requirements
    - Cue Ball In Hand
    """
    t = text.strip()
    if not t:
        return False

    if len(t) > 120:
        return False

    if "\n" in t:
        return False

    if re.match(r"^\d+(\.\d+)*[.)]?\s+\S+", t):
        return True

    words = t.split()
    if 1 <= len(words) <= 10 and t == t.title():
        return True

    return False


def _split_paragraphs(text: str) -> List[str]:
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def chunk_text(
    text: str,
    meta: DocMeta,
    *,
    max_chars: int = 2000,
    overlap_chars: int = 300,
    min_chunk_chars: int = 80,
) -> List[Chunk]:
    """
    Chunk text into overlapping segments with light structure awareness.
    """
    cleaned = _clean_text(text)
    if not cleaned:
        return []

    if overlap_chars >= max_chars:
        raise ValueError("overlap_chars must be < max_chars")

    parts = _split_paragraphs(cleaned)
    if not parts:
        return []

    chunks: List[str] = []
    buf = ""
    current_header: Optional[str] = None

    for p in parts:
        para = p.strip()
        if not para:
            continue

        if _looks_like_section_header(para):
            current_header = para
            continue

        para_with_header = f"{current_header}\n\n{para}" if current_header else para
        candidate = (buf + "\n\n" + para_with_header).strip() if buf else para_with_header

        if len(candidate) <= max_chars:
            buf = candidate
            continue

        if buf:
            if len(buf) >= min_chunk_chars:
                chunks.append(buf)

            tail = buf[-overlap_chars:] if overlap_chars > 0 else ""
            if tail:
                first_ws = None
                for j, ch in enumerate(tail):
                    if ch.isspace():
                        first_ws = j
                        break
                if first_ws is not None:
                    tail = tail[first_ws:].lstrip()

            buf = (tail + "\n\n" + para_with_header).strip() if tail else para_with_header
        else:
            start = 0
            n = len(para_with_header)

            while start < n:
                end = min(start + max_chars, n)
                piece = para_with_header[start:end].strip()

                if len(piece) >= min_chunk_chars:
                    chunks.append(piece)

                if end >= n:
                    break

                next_start = end - overlap_chars
                if next_start <= start:
                    next_start = end
                start = next_start

            buf = ""

    if buf and len(buf) >= min_chunk_chars:
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