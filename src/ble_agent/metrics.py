from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

from .models import TrialResult


def _finite_vector(values, name: str, *, nonempty: bool = True) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or (nonempty and array.size == 0):
        raise ValueError(f"{name} must be a {'nonempty ' if nonempty else ''}1D array")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def percentile(values: list[float], q: float) -> float:
    array = _finite_vector(values, "percentile values")
    if not np.isfinite(q) or not 0 <= q <= 100:
        raise ValueError("percentile q must be in [0, 100]")
    return float(np.percentile(array, q))


def binary_metrics(y_true: np.ndarray, score: np.ndarray, threshold: float = 0.80) -> dict:
    labels = _finite_vector(y_true, "binary labels", nonempty=False)
    if np.any((labels != 0) & (labels != 1)):
        raise ValueError("binary labels must be boolean or 0/1")
    actual = labels.astype(bool)
    scores = _finite_vector(score, "binary scores", nonempty=False)
    if actual.ndim != 1 or actual.shape != scores.shape:
        raise ValueError("binary labels and scores must have the same 1D shape")
    if not np.isfinite(threshold):
        raise ValueError("binary threshold must be finite")
    predicted = scores >= threshold
    tp = int(np.logical_and(actual, predicted).sum())
    fp = int(np.logical_and(~actual, predicted).sum())
    fn = int(np.logical_and(actual, ~predicted).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def summarize_trials(trials: list[TrialResult]) -> dict:
    """Summarize finite trials; reject empty/invalid input rather than emit NaN.

    mobile_detection is a legacy field for agreement of range-instability scores
    with simulated helper labels. It is not a claim of target-motion detection.
    """
    if not trials:
        raise ValueError("summarize_trials requires at least one trial")
    errors = _finite_vector([trial.error_m for trial in trials], "trial errors")
    runtimes = _finite_vector([trial.runtime_ms for trial in trials], "trial runtimes")
    convergence = _finite_vector([trial.converged for trial in trials], "trial convergence flags")
    if np.any((convergence != 0) & (convergence != 1)):
        raise ValueError("trial convergence flags must be boolean or 0/1")
    if np.any(errors < 0) or np.any(runtimes < 0):
        raise ValueError("trial errors and runtimes must be nonnegative")
    for trial in trials:
        binary_metrics(trial.mobile_true, trial.mobile_scores)
    mobile_true = np.concatenate([trial.mobile_true for trial in trials])
    mobile_scores = np.concatenate([trial.mobile_scores for trial in trials])
    return {
        "n_trials": len(trials),
        "mean_error_m": float(np.mean(errors)),
        "median_error_m": percentile(errors, 50),
        "cep50_m": percentile(errors, 50),
        "cep90_m": percentile(errors, 90),
        "failure_rate_over_30m": float(np.mean(errors > 30.0)),
        "convergence_rate": float(np.mean(convergence)),
        "mean_runtime_ms": float(np.mean(runtimes)),
        "mobile_detection": binary_metrics(mobile_true, mobile_scores),
    }


def slice_trials(rows: list[tuple[dict, TrialResult]]) -> list[dict]:
    """Group by full scenario identity, retaining old slice/metric fields.

    Clean means no outliers, mobile helpers, or calibration bias; ordinary range
    or position noise is permitted. Seed, metadata mode, and every supplied
    descriptor are part of the grouping key, without rounded-float collisions.
    """
    grouped: dict[str, list[TrialResult]] = defaultdict(list)
    descriptors: dict[str, dict] = {}
    for descriptor, result in rows:
        conditions = {
            **descriptor,
            "scenario_name": descriptor.get("scenario_name", descriptor.get("name", "unknown")),
            "layout": descriptor["layout"],
            "seed": descriptor.get("seed"),
            "metadata_mode": descriptor.get("metadata_mode", "observable"),
            "position_noise_std_m": descriptor.get("position_noise_std_m", 1.2),
            "clean": bool(
                descriptor.get("outlier_rate", 0.0) == 0.0
                and descriptor.get("mobile_rate", 0.0) == 0.0
                and descriptor.get("calibration_bias", 0.0) == 0.0
            ),
        }
        legacy = f"beacons={descriptor['beacon_count']}|outliers={descriptor['outlier_rate']:.2f}|layout={descriptor['layout']}|mobile={descriptor['mobile_rate']:.2f}"
        key = legacy + "|conditions=" + json.dumps(conditions, sort_keys=True, separators=(",", ":"), allow_nan=False)
        grouped[key].append(result)
        descriptors[key] = conditions

    return [
        {"slice": key, **descriptors[key], **summarize_trials(values)}
        for key, values in sorted(grouped.items())
    ]


def paired_bootstrap(
    baseline_errors,
    candidate_errors,
    *,
    seed: int = 2026,
    repeats: int = 1000,
) -> dict:
    """Ordinary paired percentile bootstrap on a fixed scenario mixture.

    Arrays must be aligned by the same scenario/trial (and thus target position).
    Resample the SAME index vector for both methods. This is not a stratified or
    hierarchical bootstrap, does not model new scenario mixtures, and is not
    industrial evidence. Deltas are candidate minus baseline: negative is better.
    ci95 is the 2.5/97.5 percentile interval, not a hypothesis-test guarantee.
    """
    baseline = _finite_vector(baseline_errors, "baseline errors")
    candidate = _finite_vector(candidate_errors, "candidate errors")
    if baseline.shape != candidate.shape:
        raise ValueError("paired errors must have equal lengths and aligned trial order")
    if np.any(baseline < 0) or np.any(candidate < 0):
        raise ValueError("paired errors must be nonnegative")
    if isinstance(repeats, bool) or not isinstance(repeats, (int, np.integer)) or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")

    def statistics(values: np.ndarray) -> np.ndarray:
        return np.array([
            np.percentile(values, 90),
            np.mean(values > 30.0),
            np.median(values),
        ])

    point = statistics(candidate) - statistics(baseline)
    rng = np.random.default_rng(seed)
    bootstrap_deltas = np.empty((repeats, 3))
    for repeat in range(repeats):
        indices = rng.integers(0, baseline.size, size=baseline.size)
        bootstrap_deltas[repeat] = statistics(candidate[indices]) - statistics(baseline[indices])
    intervals = np.percentile(bootstrap_deltas, [2.5, 97.5], axis=0)
    names = ("cep90_m", "failure_rate_over_30m", "median_error_m")
    return {
        "comparison": "candidate_minus_baseline",
        "n_pairs": int(baseline.size),
        "seed": int(seed),
        "repeats": int(repeats),
        "bootstrap_scope": "ordinary paired bootstrap; fixed scenario mixture only; not industrial evidence",
        **{
            name: {"point_estimate": float(point[index]), "ci95": intervals[:, index].tolist()}
            for index, name in enumerate(names)
        },
    }
