from __future__ import annotations

from pathlib import Path
from pypdf import PdfReader


def probe_pdf(pdf_path: Path, pages_to_check: int = 2, preview_chars: int = 300) -> tuple[int, str]:
    reader = PdfReader(str(pdf_path))
    n = min(len(reader.pages), pages_to_check)

    combined = []
    for i in range(n):
        txt = reader.pages[i].extract_text() or ""
        txt = " ".join(txt.split())  # normalize whitespace
        if txt:
            combined.append(txt)

    sample = " ".join(combined).strip()
    return len(sample), sample[:preview_chars]


def main() -> None:
    roots = [Path("data/raw/typed"), Path("data/raw/scanned")]
    pdfs: list[Path] = []
    for r in roots:
        if r.exists():
            pdfs.extend(sorted(r.glob("*.pdf")))

    if not pdfs:
        print("No PDFs found in data/raw/typed or data/raw/scanned")
        return

    print("Tiny PDF probe (first 2 pages, preview 300 chars):\n")
    for p in pdfs:
        chars, preview = probe_pdf(p)
        label = "TEXT_OK" if chars >= 300 else "LIKELY_OCR"
        print(f"{label:10}  {chars:6} chars  {p.as_posix()}")
        print(f"Preview: {preview!r}\n")


if __name__ == "__main__":
    main()
