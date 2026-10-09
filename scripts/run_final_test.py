from __future__ import annotations

import json
from pathlib import Path

from ble_agent.experiment import run_experiment
from ble_agent.models import MethodConfig

ROOT = Path(__file__).resolve().parents[1]
STATE_PATH = ROOT / "artifacts" / "agent_state.json"

if __name__ == "__main__":
    if not STATE_PATH.exists():
        raise RuntimeError("Run scripts/run_baseline.py and scripts/run_agent.py before the final test.")
    state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    selected = MethodConfig(**state["current_method"])
    baseline = run_experiment(MethodConfig(name="nls"), "test", label="test-baseline-nls")
    final = run_experiment(selected, "test", label=f"test-final-{selected.name}")
    print("Baseline:", baseline.metrics)
    print("Final:", final.metrics)
