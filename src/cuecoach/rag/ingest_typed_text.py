from __future__ import annotations

import re
from pathlib import Path

from cuecoach.rag.chunking import DocMeta, chunk_text
from cuecoach.rag.extract_pdf_text import extract_pdf_text


def _slugify(name: str) -> str:
    """Simple, stable doc_id from filename."""
    base = Path(name).stem.lower()
    base = re.sub(r"[^a-z0-9]+", "_", base).strip("_")
    return base or "doc"


def ingest_typed_pdfs(
    raw_dir: Path,
    out_dir: Path,
    *,
    max_chars: int = 1400,
    overlap_chars: int = 200,
) -> None:
    raw_dir = raw_dir.resolve()
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(raw_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found in: {raw_dir}")

    print(f"Found {len(pdfs)} typed PDFs in: {raw_dir}")
    print(f"Writing chunks to: {out_dir}\n")

    for pdf_path in pdfs:
        doc_id = _slugify(pdf_path.name)
        title = pdf_path.stem

        # Text extraction (typed PDFs only)
        text = extract_pdf_text(pdf_path)

        meta = DocMeta(
            doc_id=doc_id,
            source="typed_pdf",
            title=title,
            topic="misc",
            skill_level="beginner",
        )

        chunks = chunk_text(text, meta, max_chars=max_chars, overlap_chars=overlap_chars)

        out_path = out_dir / f"{doc_id}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for c in chunks:
                f.write(c.model_dump_json())
                f.write("\n")

        chars = len(text)
        print(f"- {pdf_path.name}")
        print(f"  doc_id: {doc_id}")
        print(f"  extracted_chars: {chars}")
        print(f"  chunks: {len(chunks)}")
        print(f"  out: {out_path.relative_to(Path.cwd())}\n")


if __name__ == "__main__":
    ingest_typed_pdfs(
        raw_dir=Path("data/raw/typed"),
        out_dir=Path("data/chunks/typed"),
        max_chars=1400,
        overlap_chars=200,
    )
