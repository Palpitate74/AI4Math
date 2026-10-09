from __future__ import annotations

from collections import defaultdict

import numpy as np

from .models import TrialResult


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), q))


def binary_metrics(y_true: np.ndarray, score: np.ndarray, threshold: float = 0.80) -> dict:
    actual = np.asarray(y_true, dtype=bool)
    predicted = np.asarray(score, dtype=float) >= threshold
    tp = int(np.logical_and(actual, predicted).sum())
    fp = int(np.logical_and(~actual, predicted).sum())
    fn = int(np.logical_and(actual, ~predicted).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def summarize_trials(trials: list[TrialResult]) -> dict:
    errors = [trial.error_m for trial in trials]
    mobile_true = np.concatenate([trial.mobile_true for trial in trials])
    mobile_scores = np.concatenate([trial.mobile_scores for trial in trials])
    return {
        "n_trials": len(trials),
        "mean_error_m": float(np.mean(errors)),
        "median_error_m": percentile(errors, 50),
        "cep50_m": percentile(errors, 50),
        "cep90_m": percentile(errors, 90),
        "failure_rate_over_30m": float(np.mean(np.asarray(errors) > 30.0)),
        "convergence_rate": float(np.mean([trial.converged for trial in trials])),
        "mean_runtime_ms": float(np.mean([trial.runtime_ms for trial in trials])),
        "mobile_detection": binary_metrics(mobile_true, mobile_scores),
    }


def slice_trials(rows: list[tuple[dict, TrialResult]]) -> list[dict]:
    grouped: dict[str, list[TrialResult]] = defaultdict(list)
    for descriptor, result in rows:
        key = f"beacons={descriptor['beacon_count']}|outliers={descriptor['outlier_rate']:.2f}|layout={descriptor['layout']}|mobile={descriptor['mobile_rate']:.2f}"
        grouped[key].append(result)

    output = []
    for key, values in sorted(grouped.items()):
        summary = summarize_trials(values)
        output.append({"slice": key, **summary})
    return output
