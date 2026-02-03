from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader


def extract_pdf_text(pdf_path: Path, *, max_pages: int | None = None) -> str:
    reader = PdfReader(str(pdf_path))
    pages = reader.pages if max_pages is None else reader.pages[:max_pages]

    out: list[str] = []
    for page in pages:
        text = page.extract_text() or ""
        text = " ".join(text.split())  # normalize whitespace
        if text:
            out.append(text)

    return "\n\n".join(out).strip()


def main() -> None:
    pdf_dir = Path("data/raw/typed")
    out_dir = Path("data/extracted/typed_text")
    out_dir.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        raise SystemExit("No PDFs found in data/raw/typed")

    # Pilot: first PDF only
    pdf_path = pdfs[0]

    text = extract_pdf_text(pdf_path)
    out_path = out_dir / f"{pdf_path.stem}.txt"
    out_path.write_text(text, encoding="utf-8")

    print(f"Input:  {pdf_path}")
    print(f"Output: {out_path}")
    print(f"Chars:  {len(text)}")
    print("\nPreview:\n")
    print(text[:1500])


if __name__ == "__main__":
    main()
