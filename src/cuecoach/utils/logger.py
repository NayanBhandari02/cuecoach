from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

LOG_FILE = Path("data/logs/query_logs.jsonl")


def log_query(event: Dict[str, Any]) -> None:
    """
    Append a single query event to the log file.
    """

    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **event,
    }

    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")