from __future__ import annotations

import argparse
import os
from pathlib import Path

import pypdfium2 as pdfium
import pytesseract

RAW_SCANNED_DIR = Path("data/raw/scanned")
OUT_TEXT_DIR = Path("data/extracted/ocr_text")


def configure_tesseract(tesseract_exe: str | None) -> None:
    """
    Why:
    - On Windows, Tesseract might not be on PATH even if installed.
    - This makes the script deterministic across machines.
    """
    exe = tesseract_exe or os.environ.get("TESSERACT_EXE")
    if exe:
        pytesseract.pytesseract.tesseract_cmd = exe


def render_page_to_pil(pdf: pdfium.PdfDocument, page_index: int, dpi: int):
    """
    Why:
    - OCR needs an image.
    - PDFium renders PDFs reliably on Windows without Poppler.
    """
    page = pdf[page_index]
    scale = dpi / 72  # PDF coordinate space baseline
    return page.render(scale=scale).to_pil()


def ocr_image(pil_image, *, lang: str) -> str:
    """
    Why:
    - Central place to tune OCR settings later (psm/oem/whitelist).
    """
    # Basic config. We'll tune later if needed.
    return pytesseract.image_to_string(pil_image, lang=lang).strip()


def ocr_pdf_to_text(pdf_path: Path, *, dpi: int, max_pages: int | None, lang: str) -> str:
    pdf = pdfium.PdfDocument(str(pdf_path))
    total_pages = len(pdf)
    n_pages = total_pages if max_pages is None else min(total_pages, max_pages)

    pages_text: list[str] = []

    for i in range(n_pages):
        img = render_page_to_pil(pdf, i, dpi)
        text = ocr_image(img, lang=lang)

        # Keep page separators for traceability and later citation.
        if text:
            pages_text.append(f"\n\n--- page {i+1} ---\n\n{text}")
        else:
            pages_text.append(f"\n\n--- page {i+1} ---\n\n(ocr_empty)")

    return "\n".join(pages_text).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch OCR scanned PDFs using Tesseract + PDFium.")
    parser.add_argument("--in-dir", default=str(RAW_SCANNED_DIR), help="Input directory containing scanned PDFs.")
    parser.add_argument("--out-dir", default=str(OUT_TEXT_DIR), help="Output directory for extracted OCR text.")
    parser.add_argument("--dpi", type=int, default=200, help="Render DPI for OCR (higher = slower, better text).")
    parser.add_argument(
        "--max-pages",
        type=int,
        default=3,
        help="Max pages to OCR per PDF (use 0 or negative for all pages).",
    )
    parser.add_argument("--lang", default="eng", help="Tesseract language code (default: eng).")
    parser.add_argument(
        "--tesseract-exe",
        default=None,
        help="Optional full path to tesseract.exe (overrides TESSERACT_EXE env var).",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing .txt outputs.")
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    configure_tesseract(args.tesseract_exe)

    max_pages = None if args.max_pages <= 0 else args.max_pages

    pdfs = sorted(in_dir.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDFs found in: {in_dir}")

    for pdf_path in pdfs:
        out_path = out_dir / f"{pdf_path.stem}.txt"

        if out_path.exists() and not args.force:
            print(f"Skip (exists): {out_path}")
            continue

        try:
            text = ocr_pdf_to_text(pdf_path, dpi=args.dpi, max_pages=max_pages, lang=args.lang)
            out_path.write_text(text, encoding="utf-8")

            preview = text[:300].replace("\n", " ") if text else "(empty)"
            print(f"Input:   {pdf_path}")
            print(f"Output:  {out_path}")
            print(f"Pages:   {('all' if max_pages is None else max_pages)}")
            print(f"Chars:   {len(text)}")
            print(f"Preview: {preview}")
            print()

        except Exception as e:
            print(f"FAILED: {pdf_path}  ({type(e).__name__}: {e})")

    print("Done.")


if __name__ == "__main__":
    main()
