from __future__ import annotations

import json
import os
from datetime import datetime, timezone

from .runtime import artifact_root, atomic_json, validate_identifier


def persist_iteration(record: dict) -> dict:
    """Append a structured decision without mixing distinct run directories."""
    path = artifact_root() / "iterations" / "agent_iterations.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    event = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), **record}
    # Immutable decision is authoritative; JSONL remains a readable report mirror.
    if record.get("iteration_id"):
        target = path.parent / "decisions" / f"{validate_identifier(record['iteration_id'])}.json"
        atomic_json(target, event, exclusive=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return event


def read_iterations() -> list[dict]:
    path = artifact_root() / "iterations" / "agent_iterations.jsonl"
    records = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                raise RuntimeError("Incomplete JSONL audit mirror; immutable decisions require manual reconciliation") from None
    by_id = {r["iteration_id"]: r for r in records if r.get("iteration_id")}
    legacy = [r for r in records if not r.get("iteration_id")]
    for file in (path.parent / "decisions").glob("*.json"):
        record = json.loads(file.read_text(encoding="utf-8"))
        by_id[record["iteration_id"]] = record
    return sorted(legacy + list(by_id.values()), key=lambda r: r["iteration_index"])


def persist_model_event(iteration_id: str, iteration_index: int, trace: dict) -> None:
    """Durable response evidence before validation or experiment execution."""
    target = artifact_root() / "iterations" / "model_responses" / f"{validate_identifier(iteration_id)}-{trace['attempt']}.json"
    atomic_json(target, {"iteration_id": iteration_id, "iteration_index": iteration_index,
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(), "trace": trace}, exclusive=True)


def orphan_response_events() -> list[dict]:
    completed = {r.get("iteration_id") for r in read_iterations()}
    directory = artifact_root() / "iterations" / "model_responses"
    events = [json.loads(p.read_text(encoding="utf-8")) for p in directory.glob("*.json")]
    return [e for e in events if e["iteration_id"] not in completed]
