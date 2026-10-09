from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ITERATION_LOG = ROOT / "artifacts" / "iterations" / "agent_iterations.jsonl"


def persist_iteration(record: dict) -> dict:
    """Append an immutable structured record of one agent decision."""
    ITERATION_LOG.parent.mkdir(parents=True, exist_ok=True)
    event = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), **record}
    with ITERATION_LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")
    return event


def read_iterations() -> list[dict]:
    if not ITERATION_LOG.exists():
        return []
    with ITERATION_LOG.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
