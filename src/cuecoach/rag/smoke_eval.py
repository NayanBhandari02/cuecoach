from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

from cuecoach.rag.ask import answer, retrieve_debug


DEFAULT_SMOKE_PATH = Path("data/eval/smoke_questions.jsonl")
REFUSAL_TEXT = "i don't know based on the provided material."


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def check_expect(answer_text: str, expects: List[str]) -> Tuple[bool, List[str]]:
    answer_l = answer_text.lower()
    missing = [e for e in expects if e.lower() not in answer_l]
    return (len(missing) == 0), missing


def check_refusal(answer_text: str) -> bool:
    return REFUSAL_TEXT in answer_text.lower()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke test RAG answers against keyword expectations or expected refusals."
    )
    parser.add_argument("--path", default=str(DEFAULT_SMOKE_PATH), help="Path to smoke_questions.jsonl")
    parser.add_argument("--top-k", type=int, default=8, help="Top K retrieval")
    parser.add_argument("--min-score", type=float, default=0.0, help="Optional minimum similarity score filter")
    parser.add_argument("--max-context-chars", type=int, default=12000, help="Max context chars fed to LLM")
    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        raise SystemExit(f"Smoke file not found: {path}")

    rows = load_jsonl(path)
    if not rows:
        raise SystemExit(f"No rows in: {path}")

    fails = 0

    for i, row in enumerate(rows, start=1):
        q = str(row["q"])
        mode = str(row.get("mode", "answer")).strip().lower()

        expects = row.get("expect") or []
        if not isinstance(expects, list):
            expects = [str(expects)]

        matches = retrieve_debug(
            q,
            top_k=args.top_k,
            min_score=args.min_score,
        )

        best_score = matches[0]["score"] if matches else 0.0
        print(f"\nQ{i} DEBUG: best_score={best_score:.4f}, matches={len(matches)}")

        for j, m in enumerate(matches[:3], start=1):
            metadata = m.get("metadata", {}) or {}
            preview = (metadata.get("text") or "")[:200].replace("\n", " ")
            print(f"  Match {j}: score={m.get('score', 0.0):.4f}")
            print(f"    Preview: {preview}")

        ans = answer(
            q,
            top_k=args.top_k,
            min_score=args.min_score,
            max_context_chars=args.max_context_chars,
        )

        if mode == "refuse":
            ok = check_refusal(ans)
            missing: List[str] = []
        else:
            ok, missing = check_expect(ans, expects)

        status = "PASS" if ok else "FAIL"
        print(f"\n[{status}] {i}. {q}\n{ans}\n")

        if not ok:
            fails += 1
            if mode == "refuse":
                print("  Expected refusal but got an answer.")
            else:
                print(f"  Missing keywords: {missing}")

    if fails:
        print(f"\nSmoke eval FAILED: {fails}/{len(rows)} failed.")
        sys.exit(1)

    print(f"\nSmoke eval PASSED: {len(rows)}/{len(rows)} passed.")
    sys.exit(0)


if __name__ == "__main__":
    main()