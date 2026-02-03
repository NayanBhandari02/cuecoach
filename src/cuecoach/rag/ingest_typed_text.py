from __future__ import annotations

from pathlib import Path

from cuecoach.rag.chunking import DocMeta, chunk_text


def main() -> None:
    # Input extracted text (from your extraction step)
    txt_path = Path("data/extracted/typed_text/2025.07.19-Pyramid-Rules.txt")
    if not txt_path.exists():
        raise SystemExit(f"Missing file: {txt_path}")

    text = txt_path.read_text(encoding="utf-8")

    # Metadata: this becomes filters later in Pinecone
    meta = DocMeta(
        doc_id="pyramid_rules_2025_07_19",
        source="IPC / Pyramid",
        title="Pyramid General Rules (2025-07-19)",
        topic="rules",
        skill_level="beginner",
    )

    # Chunk it
    chunks = chunk_text(text, meta, max_chars=1400, overlap_chars=200)

    # Output location
    out_dir = Path("data/processed/chunks")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{meta.doc_id}.jsonl"

    # Save as JSONL (one chunk per line)
    with out_path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(c.model_dump_json())
            f.write("\n")

    print(f"Wrote {len(chunks)} chunks to {out_path}\n")

    # Preview first 3 chunks
    for c in chunks[:3]:
        print("=" * 90)
        print(c.chunk_id)
        print(c.text[:500])


if __name__ == "__main__":
    main()
