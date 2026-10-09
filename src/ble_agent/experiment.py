from __future__ import annotations

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
from .simulator import simulate_observations
from .solvers import localize

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"


def load_scenarios(split: str) -> list:
    config_path = ROOT / "configs" / f"{split}_scenarios.json"
    with config_path.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    from .models import ScenarioConfig

    return [ScenarioConfig(**item) for item in raw]


def _mobile_instability_score(solver_score: np.ndarray) -> np.ndarray:
    # stability weights are high for stable reports; invert to obtain a mobile score.
    return 1.0 - np.clip(solver_score, 0.0, 1.0)


def run_experiment(method: MethodConfig, split: str, label: str | None = None) -> ExperimentResult:
    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation, or test")
    scenarios = load_scenarios(split)
    trial_results: list[TrialResult] = []
    rows: list[tuple[dict, TrialResult]] = []

    for scenario in scenarios:
        for trial_index in range(scenario.trials):
            observations = simulate_observations(scenario, trial_index)
            started = time.perf_counter()
            solver = localize(observations, method)
            runtime_ms = (time.perf_counter() - started) * 1000.0
            result = TrialResult(
                error_m=float(np.linalg.norm(solver.estimate - observations.true_position)),
                converged=solver.converged,
                runtime_ms=runtime_ms,
                mobile_true=observations.mobile_mask,
                mobile_scores=_mobile_instability_score(solver.mobile_scores),
            )
            trial_results.append(result)
            rows.append((scenario.to_dict(), result))

    experiment_id = label or f"{split}-{method.name}-{uuid4().hex[:8]}"
    result = ExperimentResult(
        experiment_id=experiment_id,
        split=split,
        method=method,
        scenarios=scenarios,
        metrics=summarize_trials(trial_results),
        slices=slice_trials(rows),
        trials=trial_results,
        environment={
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
    )
    save_experiment(result)
    return result


def experiment_path(experiment_id: str) -> Path:
    return ARTIFACTS / "experiments" / f"{experiment_id}.json"


def save_experiment(result: ExperimentResult) -> Path:
    path = experiment_path(result.experiment_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = result.to_dict()
    serializable["trial_errors_m"] = [trial.error_m for trial in result.trials]
    with path.open("w", encoding="utf-8") as handle:
        json.dump(serializable, handle, ensure_ascii=False, indent=2)
    return path


def read_experiment(experiment_id: str) -> dict:
    with experiment_path(experiment_id).open(encoding="utf-8") as handle:
        return json.load(handle)


def validation_summary(experiment_id: str) -> dict:
    record = read_experiment(experiment_id)
    if record["split"] != "validation":
        raise ValueError("agent may only inspect validation experiments during tuning")
    return {
        "experiment_id": record["experiment_id"],
        "method": record["method"],
        "metrics": record["metrics"],
        "worst_slices": sorted(record["slices"], key=lambda row: row["cep90_m"], reverse=True)[:3],
    }


def accepted_against(current: dict, candidate: dict, clean_degradation_limit: float = 0.10) -> tuple[bool, str]:
    current_metrics = current["metrics"]
    candidate_metrics = candidate["metrics"]
    if candidate_metrics["cep90_m"] >= current_metrics["cep90_m"]:
        return False, "Rejected: validation CEP90 did not improve."
    if candidate_metrics["median_error_m"] > current_metrics["median_error_m"] * (1.0 + clean_degradation_limit):
        return False, "Rejected: median error degraded beyond the pre-registered tolerance."
    return True, "Accepted: validation CEP90 improved without excessive median-error degradation."


def clone_method(method: MethodConfig, **change) -> MethodConfig:
    return replace(method, **change)
