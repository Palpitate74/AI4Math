"""Explicit local configuration and isolated, non-overwriting artifact runs."""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def artifact_root() -> Path:
    override = os.getenv("BLE_ARTIFACT_DIR")
    return Path(override).expanduser().resolve() if override else ROOT / "artifacts"


def load_local_config() -> None:
    # Only this project's explicitly named file is loaded; never search home/config.
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env.local", override=False)


def configure_run(run_id: str) -> Path:
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}", run_id):
        raise ValueError("run-id must be a safe alphanumeric identifier")
    path = ROOT / "artifacts" / "midterm" / run_id
    os.environ["BLE_ARTIFACT_DIR"] = str(path)
    return path


def validate_identifier(identifier: str) -> str:
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,159}", identifier):
        raise ValueError("unsafe experiment identifier")
    return identifier


def data_digest(scenarios: list[dict]) -> str:
    text = json.dumps(scenarios, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode()).hexdigest()


def atomic_json(path: Path, value: dict | list, *, exclusive: bool = False) -> None:
    """Publish only complete JSON; exclusive mode never replaces existing evidence."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".write-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        if exclusive:
            os.link(temporary, path)
        else:
            os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def generator_identity() -> dict:
    import numpy
    files = ["simulator.py", "calibration.py", "models.py"]
    return {"version": "observable-static-v2", "numpy": numpy.__version__,
            "source_sha256": {name: hashlib.sha256((ROOT / "src" / "ble_agent" / name).read_bytes()).hexdigest()
                              for name in files}}


def execution_identity() -> dict:
    import scipy
    files = ["solvers.py", "reliability.py", "metrics.py", "experiment.py",
             "agent.py", "prompts.py", "graph.py", "logging.py", "runtime.py"]
    hashes = {name: hashlib.sha256((ROOT / "src" / "ble_agent" / name).read_bytes()).hexdigest() for name in files}
    hashes["scripts/run_midterm.py"] = hashlib.sha256((ROOT / "scripts" / "run_midterm.py").read_bytes()).hexdigest()
    return {"scipy": scipy.__version__, "source_sha256": hashes}


def freeze_scenarios(splits: list[str]) -> dict:
    """Pre-run snapshots only. Existing experiments require a new run ID."""
    directory = artifact_root() / "protocol"
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["generator"] != generator_identity():
            raise RuntimeError("Generator changed; use a new run-id")
        if manifest.get("execution") != execution_identity():
            raise RuntimeError("Solver/Agent execution code changed; use a new run-id")
        for split in splits:
            raw = json.loads((directory / f"{split}_scenarios.json").read_text(encoding="utf-8"))
            if data_digest(raw) != manifest["scenario_digests"][split]:
                raise RuntimeError("Frozen scenario snapshot changed")
        return manifest
    if any((artifact_root() / "experiments").glob("*.json")):
        raise RuntimeError("Historical run lacks pre-run manifest; preserve it and use a new run-id")
    from .models import ScenarioConfig
    digests = {}
    for split in splits:
        if split not in {"midterm_validation", "midterm_holdout"}:
            raise ValueError("unsupported snapshot split")
        target = directory / f"{split}_scenarios.json"
        if not target.exists():
            raw = json.loads((ROOT / "configs" / f"{split}_scenarios.json").read_text(encoding="utf-8"))
            atomic_json(target, [ScenarioConfig(**item).to_dict() for item in raw], exclusive=True)
        expanded = json.loads(target.read_text(encoding="utf-8"))
        digests[split] = data_digest(expanded)
    manifest = {"provenance": "pre_run_snapshot", "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "scenario_digests": digests, "generator": generator_identity(), "execution": execution_identity()}
    atomic_json(manifest_path, manifest, exclusive=True)
    return manifest
