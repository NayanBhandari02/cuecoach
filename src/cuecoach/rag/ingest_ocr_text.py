from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from cuecoach.rag.chunking import DocMeta, chunk_text
from cuecoach.rag.sanitize_chunks import repair_mojibake, strip_front_matter
from cuecoach.rag.schemas import Topic

OCR_TEXT_DIR = Path("data/extracted/ocr_text")
OUT_CHUNKS_DIR = Path("data/chunks/ocr")


def slugify(name: str) -> str:
    """
    Why:
    - Stable doc_id from filename (portable across machines).
    - Keeps vector-store ids clean later.
    """
    base = name.lower()
    base = re.sub(r"[^a-z0-9]+", "_", base)
    base = re.sub(r"_+", "_", base).strip("_")
    return base


def infer_topic(filename: str) -> Topic:
    """
    Why:
    - Topic helps retrieval filters later (rules vs drills vs aiming).
    - Simple heuristic now; can get smarter later.
    """
    f = filename.lower()
    if "rule" in f:
        return "rules"
    if "break" in f:
        return "break"
    if "aim" in f:
        return "aiming"
    if "spin" in f or "english" in f:
        return "spin"
    if "safety" in f:
        return "safety"
    if "pattern" in f or "runout" in f:
        return "pattern_play"
    return "misc"


def remove_page_markers(text: str) -> str:
    """
    Why:
    - OCR output often includes '--- page 12 ---' separators.
    - These add noise to embeddings and retrieval.
    """
    return re.sub(r"(?im)^\s*---\s*page\s*\d+\s*---\s*$\n?", "", text)


def clean_ocr_text(raw: str) -> str:
    """
    Why this order:
    1) repair_mojibake: fix 'â€œ' 'Â©' etc early so later matching works.
    2) strip_front_matter: removes copyright/ISBN/contents pages that
       you said you don't want in the RAG knowledge base.
    3) remove_page_markers: optional cleanup for better embeddings.
    """
    text = repair_mojibake(raw)
    text = strip_front_matter(text)
    text = remove_page_markers(text)

    # Normalize whitespace a bit without being aggressive.
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest OCR text files -> chunk JSONL."
    )
    parser.add_argument(
        "--in-dir",
        default=str(OCR_TEXT_DIR),
        help="Directory with OCR .txt files.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(OUT_CHUNKS_DIR),
        help="Output directory for chunk JSONL.",
    )
    parser.add_argument(
        "--source",
        default="OCR",
        help="Source label stored in chunks.",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=1400,
        help="Max chars per chunk.",
    )
    parser.add_argument(
        "--overlap-chars",
        type=int,
        default=200,
        help="Overlap chars between chunks.",
    )
    parser.add_argument(
        "--skill",
        default="beginner",
        help="Skill level metadata.",
    )
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    txt_files = sorted(in_dir.glob("*.txt"))
    if not txt_files:
        raise SystemExit(f"No .txt files found in: {in_dir}")

    total_chunks = 0

    for txt_path in txt_files:
        raw = txt_path.read_text(encoding="utf-8", errors="replace")
        cleaned = clean_ocr_text(raw)

        if not cleaned:
            print(f"Skip empty after cleaning: {txt_path.name}")
            continue

        doc_id = slugify(txt_path.stem)
        topic = infer_topic(txt_path.name)

        meta = DocMeta(
            doc_id=doc_id,
            source=args.source,
            title=txt_path.stem.replace("_", " ").strip(),
            url=None,
            section=None,
            topic=topic,
            skill_level=args.skill,
        )

        chunks = chunk_text(
            cleaned,
            meta,
            max_chars=args.max_chars,
            overlap_chars=args.overlap_chars,
        )

        out_path = out_dir / f"{doc_id}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps(c.model_dump(), ensure_ascii=False) + "\n")

        total_chunks += len(chunks)
        print(
            f"{txt_path.name} -> {out_path.name} "
            f"({len(chunks)} chunks, {len(raw)} -> {len(cleaned)} chars)"
        )

    print(f"Done. Total chunks: {total_chunks}")


if __name__ == "__main__":
    main()
