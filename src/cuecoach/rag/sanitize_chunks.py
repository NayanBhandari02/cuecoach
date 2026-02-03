from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


def stable_chunk_id(doc_id: str, section: str | None, idx: int, text: str) -> str:
    h = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
    sec = section or "root"
    return f"{doc_id}::{sec}::{idx:04d}::{h}"

def repair_mojibake(s: str) -> str:
    """
    Fix common OCR/encoding mojibake, not just the â€œ case.

    Strategy:
    - If it *looks* like mojibake, try CP1252->UTF-8 repair.
    - Then apply a small mapping for leftovers that often survive.
    """
    if not s:
        return s

    # Heuristics: these characters are strong mojibake signals
    mojibake_signals = ("â", "Â", "Ã", "�")
    if not any(sig in s for sig in mojibake_signals):
        return s

    original = s

    # 1) Attempt the classic repair route
    try:
        fixed = s.encode("cp1252", errors="ignore").decode("utf-8", errors="ignore")
        if fixed:
            s = fixed
    except Exception:
        s = original

    # 2) Patch common remnants (kept small on purpose)
    replacements = {
        "â€œ": "“",
        "â€\u009d": "”",
        "â€": "”",
        "â€˜": "‘",
        "â€™": "’",
        "â€”": "—",
        "â€“": "–",
        "Â©": "©",
        "Â®": "®",
        "Â·": "·",
        "Â": "",  # stray non-breaking-space marker
    }
    for bad, good in replacements.items():
        s = s.replace(bad, good)

    # Also normalize the replacement character if it appears a lot
    # (Don't delete single occurrences; only collapse repeated junk)
    s = re.sub(r"�{2,}", " ", s)

    return s

def strip_front_matter(text: str) -> str:
    """
    Remove boilerplate pages like copyright/ISBN/LoC/contents.
    Conservative: only strips if markers appear early.
    """
    if not text:
        return text

    head = text[:12000].lower()  # look early, but not just 1-2 pages

    markers = [
        "all rights reserved",
        "library of congress",
        "cataloging-in-publication",
        "isbn",
        "printed in the united states",
        "contents",
    ]
    if not any(m in head for m in markers):
        return text

    # Cut at first meaningful “real content” section
    cut_regexes = [
        r"(?im)^\s*introduction\s*$",
        r"(?im)^\s*foreword\b.*$",
        r"(?im)^\s*chapter\s+\d+\b.*$",
        r"(?im)^\s*\d+\.\s+[A-Z].*$",  # numbered headings like "1. MOTIVATION..."
    ]
    for rx in cut_regexes:
        m = re.search(rx, text)
        if m and m.start() < 20000:  # only cut if it happens relatively early
            return text[m.start():]

    return text

def clean_text(text: str) -> str:
    """
    Why:
    - Remove OCR watermarks + page markers that pollute embeddings.
    - Reduce noisy repeats so retrieval returns useful content.
    """
    patterns = [
        re.compile(r"(?im)^\s*---\s*page\s+\d+\s*---\s*$"),  # page markers
        re.compile(r"(?i)\boceanofpdf\.com\b"),
        re.compile(r"(?i)\bqceanofpde\.com\b"),
        re.compile(r"(?i)\boceanofpde\.com\b"),
    ]

    out = text
    for pat in patterns:
        out = pat.sub("", out)

    # Collapse excessive whitespace
    out = re.sub(r"[ \t]+\n", "\n", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def sanitize_record(
    rec: dict[str, Any],
    *,
    new_doc_id: str | None,
    new_source: str | None,
    title_prefix_re: re.Pattern[str] | None,
    idx: int,
) -> dict[str, Any]:
    rec["text"] = clean_text(rec.get("text", ""))

    if new_doc_id:
        rec["doc_id"] = new_doc_id
    if new_source:
        rec["source"] = new_source

    if title_prefix_re and isinstance(rec.get("title"), str):
        rec["title"] = title_prefix_re.sub("", rec["title"]).strip(" _-")

    rec["chunk_id"] = stable_chunk_id(rec["doc_id"], rec.get("section"), idx, rec["text"])
    return rec


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sanitize chunk JSONL (remove watermarks/page markers, normalize metadata)."
    )
    parser.add_argument(
        "--in-dir",
        default="data/chunks/ocr",
        help="Input directory with OCR chunk jsonl files.",
    )
    parser.add_argument(
        "--out-dir",
        default="data/chunks/ocr_clean",
        help="Output directory for cleaned chunk jsonl files.",
    )
    parser.add_argument(
        "--doc-id",
        default=None,
        help="Override doc_id for ALL chunks (optional).",
    )
    parser.add_argument(
        "--source",
        default=None,
        help="Override source for ALL chunks (optional).",
    )
    parser.add_argument(
        "--strip-title-prefix",
        default=r"(?i)^_?oceanofpdf\.com[_\- ]*",
        help="Regex to strip from start of title. Empty disables.",
    )
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    title_prefix_re = re.compile(args.strip_title_prefix) if args.strip_title_prefix else None

    files = sorted(in_dir.glob("*.jsonl"))
    if not files:
        raise SystemExit(f"No jsonl files found in {in_dir}")

    for fp in files:
        out_fp = out_dir / fp.name

        idx = 1
        kept = 0

        with fp.open("r", encoding="utf-8") as fin, out_fp.open("w", encoding="utf-8") as fout:
            for line in fin:
                line = line.strip()
                if not line:
                    continue

                rec = json.loads(line)
                rec = sanitize_record(
                    rec,
                    new_doc_id=args.doc_id,
                    new_source=args.source,
                    title_prefix_re=title_prefix_re,
                    idx=idx,
                )

                # Drop empty chunks after cleanup (rare but happens)
                if not rec.get("text"):
                    continue

                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                kept += 1
                idx += 1

        print(f"{fp.name} -> {out_fp.name} (kept {kept} chunks)")

    print("Done.")


if __name__ == "__main__":
    main()
