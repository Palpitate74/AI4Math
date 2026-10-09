from __future__ import annotations

import argparse
import json
from pathlib import Path

from ble_agent.graph import run_iteration
from ble_agent.models import MethodConfig

ROOT = Path(__file__).resolve().parents[1]
STATE_PATH = ROOT / "artifacts" / "agent_state.json"


def read_state() -> dict:
    with STATE_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


def save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with STATE_PATH.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one bounded LangGraph experiment-agent iteration.")
    parser.add_argument("--policy", choices=["scripted", "api"], default="scripted")
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    if args.reset or not STATE_PATH.exists():
        state = {
            "current_method": MethodConfig(name="nls").to_dict(),
            "current_experiment_id": "validation-baseline-nls",
            "completed_iterations": 0,
            "policy": args.policy,
        }
    else:
        state = read_state()
        if state["policy"] != args.policy:
            raise RuntimeError("Use --reset before changing the agent policy.")

    for index in range(state["completed_iterations"] + 1, args.iterations + 1):
        output = run_iteration(
            policy=args.policy,
            iteration_index=index,
            current_method=MethodConfig(**state["current_method"]),
            current_experiment_id=state["current_experiment_id"],
        )
        state["completed_iterations"] = index
        if output["decision"]["accepted"]:
            state["current_method"] = output["current_method"]
            state["current_experiment_id"] = output["current_experiment_id"]
        save_state(state)
        print(json.dumps(output["record"], ensure_ascii=False, indent=2))

    print("Final validation method:", state["current_method"])
    print("Final validation experiment:", state["current_experiment_id"])


if __name__ == "__main__":
    main()
