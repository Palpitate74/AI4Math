"""Isolated midterm workflow: snapshot, bounded Agent, freeze, missing-only holdout."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

from ble_agent.experiment import experiment_path, load_scenarios, read_experiment, run_experiment
from ble_agent.graph import run_iteration
from ble_agent.logging import orphan_response_events, read_iterations
from ble_agent.metrics import paired_bootstrap
from ble_agent.models import MethodConfig
from ble_agent.runtime import atomic_json, configure_run, data_digest, freeze_scenarios, generator_identity

COMPARISONS = {
    "nls": MethodConfig(name="nls"), "weighted_nls": MethodConfig(name="weighted_nls"),
    "huber": MethodConfig(name="huber"), "cauchy": MethodConfig(name="cauchy"),
    "adaptive": MethodConfig(name="adaptive"),
    "adaptive_temporal": MethodConfig(name="adaptive", use_temporal_weight=True),
}


def persist(path, summary):
    atomic_json(path, summary)


def recover_summary(summary: dict, records: list[dict]) -> dict:
    """Replay the entire decision chain, rather than trusting a stale summary."""
    if len(records) < summary.get("completed_iterations", 0):
        raise RuntimeError("Decision audit tail missing; cannot roll back recorded progress")
    if not records:
        if summary.get("completed_iterations", 0):
            raise RuntimeError("Summary references missing decisions")
        return summary
    current = summary["baseline_validation"]
    method = COMPARISONS["nls"].to_dict()
    evaluated = errors = 0
    for index, record in enumerate(records, 1):
        if record["iteration_index"] != index or record["input_experiment_id"] != current:
            raise RuntimeError("Decision chain is duplicated, missing or inconsistent")
        if record["policy"] != summary["policy"]:
            raise RuntimeError("Decision policy mismatch")
        if record["status"] == "error":
            errors += 1
        elif summary["policy"] == "api":
            evaluated += 1
        if record["decision"]["accepted"]:
            if record["status"] != "evaluated" or not record.get("candidate_experiment_id"):
                raise RuntimeError("Invalid accepted decision")
            current, method = record["candidate_experiment_id"], record["candidate_method"]
    if summary.get("frozen_at_utc") and (summary["final_validation"] != current or summary["final_method"] != method):
        raise RuntimeError("Frozen selection disagrees with decision chain")
    summary.update(completed_iterations=len(records), api_evaluated_iterations=evaluated,
                   error_iterations=errors, final_validation=current, final_method=method)
    return summary


def expected_identity(split):
    scenarios = load_scenarios(split)
    keys = [f"{s.name}:{s.seed}:{i}" for s in scenarios for i in range(s.trials)]
    return {"split": split, "scenario_sha256": data_digest([s.to_dict() for s in scenarios]),
            "generator": generator_identity(), "pair_order_sha256": data_digest(keys)}


def ensure_experiment(method, split, label):
    """Recover published artifacts after a crash, without rerunning/overwriting."""
    if not experiment_path(label).exists():
        run_experiment(method, split, label=label)
    record = read_experiment(label)
    if record["method"] != method.to_dict() or record.get("data_identity") != expected_identity(split):
        raise RuntimeError("Existing experiment does not match frozen method/data")
    scenarios = [s.to_dict() for s in load_scenarios(split)]
    keys = [f"{s['name']}:{s['seed']}:{i}" for s in scenarios for i in range(s['trials'])]
    if record["scenarios"] != scenarios or record.get("trial_keys") != keys or len(record["trial_errors_m"]) != len(keys):
        raise RuntimeError("Existing experiment scenario/pair payload mismatch")
    return record


def finalize_run(summary, path):
    if summary["policy"] == "api" and summary["api_evaluated_iterations"] < 3:
        raise RuntimeError("Need >=3 evaluated model-driven rounds for API midterm finalization")
    if not summary.get("frozen_at_utc"):
        summary["frozen_at_utc"] = datetime.now(timezone.utc).isoformat()
        summary["status"] = "finalizing"
        summary["holdout"] = {}
        persist(path, summary)
    methods = {**COMPARISONS, "selected": MethodConfig(**summary["final_method"])}
    for name, method in methods.items():
        record = ensure_experiment(method, "midterm_holdout", f"holdout-{name}")
        if name in summary["holdout"] and summary["holdout"][name] != record["experiment_id"]:
            raise RuntimeError("Frozen holdout identifier changed")
        summary["holdout"][name] = record["experiment_id"]
        persist(path, summary)
    baseline = read_experiment(summary["holdout"]["nls"])
    summary.setdefault("paired_bootstrap", {})
    for name, identifier in summary["holdout"].items():
        candidate = read_experiment(identifier)
        if (candidate["data_identity"] != baseline["data_identity"] or
                candidate["scenarios"] != baseline["scenarios"] or candidate["trial_keys"] != baseline["trial_keys"]):
            raise RuntimeError("Bootstrap full data identity/pairs do not align")
        if name not in summary["paired_bootstrap"]:
            summary["paired_bootstrap"][name] = paired_bootstrap(
                baseline["trial_errors_m"], candidate["trial_errors_m"], seed=91001, repeats=1000)
            persist(path, summary)
    summary["status"] = "completed"
    persist(path, summary)


def workflow(args, root):
    path = root / "summary.json"
    if path.exists():
        summary = json.loads(path.read_text(encoding="utf-8"))
        if summary["policy"] != args.policy:
            raise RuntimeError("Use a different run-id for a different policy")
        if summary.get("status") == "completed":
            print("Completed frozen run preserved; no model or experiment was rerun.")
            return
        if summary.get("frozen_at_utc") and not args.finalize:
            raise RuntimeError("Frozen run allows --finalize recovery only, never further tuning")
        manifest = freeze_scenarios(["midterm_validation", "midterm_holdout"])
        if summary["protocol"].get("snapshot") != manifest:
            raise RuntimeError("Summary and pre-run protocol manifest mismatch")
    else:
        manifest = freeze_scenarios(["midterm_validation", "midterm_holdout"])
        validation, holdout = load_scenarios("midterm_validation"), load_scenarios("midterm_holdout")
        if {s.seed for s in validation} & {s.seed for s in holdout}:
            raise RuntimeError("Validation/holdout seeds must be disjoint")
        if any(s.metadata_mode != "observable" for s in validation + holdout):
            raise RuntimeError("Midterm data must use observable metadata")
        summary = {"run_id": args.run_id, "policy": args.policy, "status": "in_progress",
                   "completed_iterations": 0, "api_evaluated_iterations": 0, "error_iterations": 0,
                   "comparisons": {}, "final_method": COMPARISONS["nls"].to_dict(),
                   "protocol": {"scope": "static target; helper drift is corruption, not tracking",
                       "snapshot": manifest, "validation_trials": sum(s.trials for s in validation),
                       "holdout_trials": sum(s.trials for s in holdout),
                       "validation_seeds": [s.seed for s in validation], "holdout_seeds": [s.seed for s in holdout],
                       "metadata": "independent observable precision/age proxies; no oracle",
                       "selection": "Agent candidates only; ablation table descriptive, not hidden selection source",
                       "acceptance": "CEP90 improves; median <=110%; failures do not increase; clean CEP90 <= incumbent+max(10%,0.25m); numerical success decline <=5pp",
                       "old_test": "historical only; new snapshots saved before experiments",
                       "bootstrap": "1000 ordinary paired resamples over fixed scenario mixture, not scenario generalization"}}
        persist(path, summary)
    for name, method in COMPARISONS.items():
        if name not in summary["comparisons"]:
            record = ensure_experiment(method, "midterm_validation", f"ablation-validation-{name}")
            summary["comparisons"][name] = record["experiment_id"]
            if name == "nls":
                summary["baseline_validation"] = summary["final_validation"] = record["experiment_id"]
            persist(path, summary)
            print(json.dumps({"ablation": name, "metrics": record["metrics"]}, ensure_ascii=False))
    recover_summary(summary, read_iterations())
    persist(path, summary)
    if orphan_response_events():
        raise RuntimeError("Interrupted model response evidence preserved; reconcile manually/use new run-id; no silent repeat")
    if not summary.get("frozen_at_utc"):
        records = read_iterations()
        if records and records[-1]["status"] == "error" and args.iterations > summary["completed_iterations"]:
            raise RuntimeError("Run blocked after audited error; use new run-id, or finalize existing eligible selection")
        for index in range(summary["completed_iterations"] + 1, args.iterations + 1):
            output = run_iteration(policy=args.policy, iteration_index=index,
                current_method=MethodConfig(**summary["final_method"]),
                current_experiment_id=summary["final_validation"], validation_split="midterm_validation")
            recover_summary(summary, read_iterations())
            persist(path, summary)
            record = output["record"]
            print(json.dumps({"iteration": index, "status": record["status"], "proposal": record["proposal"],
                              "decision": {k: v for k, v in record["decision"].items() if k != "candidate_slices"}}, ensure_ascii=False))
            if record["status"] == "error":
                summary["status"] = "blocked_api" if args.policy == "api" else "blocked_policy"
                persist(path, summary)
                raise SystemExit("Stopped after audited error; no scripted fallback or fabricated success")
    if args.finalize:
        finalize_run(summary, path)
    else:
        summary["status"] = "validation_completed"
        persist(path, summary)
    print(json.dumps({"run_id": args.run_id, "status": summary["status"],
                      "final_method": summary["final_method"], "api_evaluated_iterations": summary["api_evaluated_iterations"]}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--policy", choices=["api", "diagnostic", "scripted"], default="api")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.iterations <= 12:
        parser.error("iterations must be between 1 and 12")
    root = configure_run(args.run_id)
    root.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (root / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("This run is already active") from None
        workflow(args, root)


if __name__ == "__main__":
    main()
