from __future__ import annotations

import importlib.metadata
import json
import platform
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np

from .metrics import slice_trials, summarize_trials
from .models import ExperimentResult, MethodConfig, TrialResult
from .runtime import (ROOT, artifact_root, atomic_json, data_digest,
                      freeze_scenarios, generator_identity, validate_identifier)
from .simulator import simulate_observations
from .solvers import localize


def load_scenarios(split: str) -> list:
    if split not in {"train", "validation", "test", "midterm_validation", "midterm_holdout"}:
        raise ValueError("unsupported data split")
    snapshot = artifact_root() / "protocol" / f"{split}_scenarios.json"
    if (artifact_root() / "protocol" / "manifest.json").exists():
        freeze_scenarios([split])
    source = snapshot if snapshot.exists() else ROOT / "configs" / f"{split}_scenarios.json"
    with source.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    from .models import ScenarioConfig
    return [ScenarioConfig(**item) for item in raw]


def _mobile_instability_score(solver_score: np.ndarray) -> np.ndarray:
    # Solver emits normalized instability: large means unstable, not confidence.
    return np.maximum(np.asarray(solver_score, dtype=float), 0.0)


def run_experiment(method: MethodConfig, split: str, label: str | None = None) -> ExperimentResult:
    scenarios = load_scenarios(split)
    trial_results: list[TrialResult] = []
    rows: list[tuple[dict, TrialResult]] = []
    trial_keys = []
    for scenario in scenarios:
        for index in range(scenario.trials):
            observations = simulate_observations(scenario, index)
            started = time.perf_counter()
            solver = localize(observations.solver_inputs(), method)
            result = TrialResult(
                error_m=float(np.linalg.norm(solver.estimate - observations.true_position)),
                converged=solver.converged,
                runtime_ms=(time.perf_counter() - started) * 1000,
                mobile_true=observations.mobile_mask,
                mobile_scores=_mobile_instability_score(solver.mobile_scores),
            )
            trial_results.append(result)
            rows.append((scenario.to_dict(), result))
            trial_keys.append(f"{scenario.name}:{scenario.seed}:{index}")
    result = ExperimentResult(
        experiment_id=label or f"{split}-{method.name}-{uuid4().hex[:8]}",
        split=split, method=method, scenarios=scenarios,
        metrics=summarize_trials(trial_results), slices=slice_trials(rows), trials=trial_results,
        environment={"timestamp_utc": datetime.now(timezone.utc).isoformat(),
                     "python": sys.version.split()[0], "platform": platform.platform(),
                     "dependencies": {name: importlib.metadata.version(name)
                                      for name in ["numpy", "scipy", "langgraph", "pydantic"]}},
    )
    serializable = result.to_dict()
    serializable["trial_errors_m"] = [x.error_m for x in trial_results]
    serializable["trial_keys"] = trial_keys
    serializable["data_identity"] = {"split": split,
        "scenario_sha256": data_digest([s.to_dict() for s in scenarios]),
        "generator": generator_identity(), "pair_order_sha256": data_digest(trial_keys)}
    serializable["metric_scope"] = "static target localization; mobile metrics describe helper instability only"
    atomic_json(experiment_path(result.experiment_id), serializable, exclusive=True)
    return result


def experiment_path(experiment_id: str) -> Path:
    return artifact_root() / "experiments" / f"{validate_identifier(experiment_id)}.json"


def save_experiment(result: ExperimentResult) -> Path:
    path = experiment_path(result.experiment_id)
    record = result.to_dict()
    keys = [f"{s.name}:{s.seed}:{i}" for s in result.scenarios for i in range(s.trials)]
    if len(keys) != len(result.trials):
        raise ValueError("Trial count does not match scenario protocol")
    record["trial_errors_m"] = [trial.error_m for trial in result.trials]
    record["trial_keys"] = keys
    record["data_identity"] = {"split": result.split,
        "scenario_sha256": data_digest(record["scenarios"]), "generator": generator_identity(),
        "pair_order_sha256": data_digest(keys)}
    atomic_json(path, record, exclusive=True)
    return path


def read_experiment(experiment_id: str) -> dict:
    with experiment_path(experiment_id).open(encoding="utf-8") as handle:
        return json.load(handle)


def validation_summary(experiment_id: str) -> dict:
    record = read_experiment(experiment_id)
    if record["split"] not in {"validation", "midterm_validation"}:
        raise ValueError("agent may only inspect validation experiments during tuning")
    if any(s.get("metadata_mode", "observable") == "oracle" for s in record["scenarios"]):
        raise ValueError("oracle metadata is forbidden during agent tuning")
    return {"experiment_id": record["experiment_id"], "method": record["method"],
            "data_identity": record.get("data_identity"),
            "scenarios": record["scenarios"], "trial_keys": record.get("trial_keys"),
            "metrics": record["metrics"],
            "worst_slices": sorted(record["slices"], key=lambda r: r["cep90_m"], reverse=True)[:3],
            "clean_slices": [r for r in record["slices"] if r.get("clean")],
            "slices": record["slices"]}


def complete_data_identity(record: dict) -> bool:
    identity = record.get("data_identity")
    scenarios, keys = record.get("scenarios"), record.get("trial_keys")
    if not isinstance(identity, dict) or set(identity) != {"split", "scenario_sha256", "generator", "pair_order_sha256"}:
        return False
    if identity["split"] not in {"validation", "midterm_validation", "midterm_holdout", "train", "test"}:
        return False
    if not isinstance(scenarios, list) or not scenarios or not isinstance(keys, list) or not keys:
        return False
    if not all(isinstance(k, str) for k in keys) or len(set(keys)) != len(keys):
        return False
    generator = identity["generator"]
    if not isinstance(generator, dict) or set(generator) != {"version", "numpy", "source_sha256"}:
        return False
    if not all(isinstance(generator[k], str) and generator[k] for k in ["version", "numpy"]):
        return False
    hashes = generator["source_sha256"]
    if not isinstance(hashes, dict) or set(hashes) != {"simulator.py", "calibration.py", "models.py"}:
        return False
    if not all(isinstance(v, str) and len(v) == 64 and all(c in "0123456789abcdef" for c in v) for v in hashes.values()):
        return False
    return identity["scenario_sha256"] == data_digest(scenarios) and identity["pair_order_sha256"] == data_digest(keys)


def accepted_against(current: dict, candidate: dict,
                     clean_degradation_limit: float = 0.10) -> tuple[bool, str]:
    if not complete_data_identity(current) or not complete_data_identity(candidate):
        return False, "Rejected: complete data identity/pair evidence missing or invalid."
    if current.get("data_identity") != candidate.get("data_identity"):
        return False, "Rejected: complete data identity missing or mismatched."
    if current.get("scenarios") != candidate.get("scenarios") or current.get("trial_keys") != candidate.get("trial_keys"):
        return False, "Rejected: scenario configurations or trial pair order mismatch."
    a, b = current["metrics"], candidate["metrics"]
    if b["cep90_m"] >= a["cep90_m"]:
        return False, "Rejected: validation CEP90 did not improve."
    if b["median_error_m"] > a["median_error_m"] * (1 + clean_degradation_limit):
        return False, "Rejected: overall median error degraded beyond 10%."
    if b.get("failure_rate_over_30m", 0) > a.get("failure_rate_over_30m", 0) + 1e-12:
        return False, "Rejected: >30m failure rate increased."
    if b.get("convergence_rate", 1) < a.get("convergence_rate", 1) - .05:
        return False, "Rejected: numerical success rate declined by >5 percentage points."
    ca = {s["slice"]: s for s in current.get("clean_slices", [])}
    cb = {s["slice"]: s for s in candidate.get("clean_slices", [])}
    if set(ca) != set(cb):
        return False, "Rejected: clean-slice identities do not match."
    for key, value in ca.items():
        if cb[key]["cep90_m"] > value["cep90_m"] + max(value["cep90_m"] * clean_degradation_limit, .25):
            return False, f"Rejected: clean-slice CEP90 degraded: {key}."
    return True, "Accepted: CEP90 improved; median, failures, clean slices and numerical success passed."


def clone_method(method: MethodConfig, **change) -> MethodConfig:
    return replace(method, **change)
