from __future__ import annotations

import numpy as np


def temporal_stability_weight(temporal_ranges_m: np.ndarray, loss_scale_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Return a reliability weight and instability score for each helper device."""
    variability = np.std(np.asarray(temporal_ranges_m, dtype=float), axis=0)
    normalized = variability / max(loss_scale_m, 1e-6)
    weights = 1.0 / (1.0 + normalized**2)
    return np.clip(weights, 0.03, 1.0), normalized


def residual_reliability_weight(residuals: np.ndarray, loss_scale_m: float) -> np.ndarray:
    scaled = np.abs(np.asarray(residuals, dtype=float)) / max(loss_scale_m, 1e-6)
    return np.clip(1.0 / (1.0 + scaled**2), 0.02, 1.0)


def combine_reliability(
    metadata_quality: np.ndarray,
    residuals: np.ndarray,
    temporal_ranges_m: np.ndarray,
    loss_scale_m: float,
    use_temporal_weight: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Combine exogenous metadata, residual evidence, and temporal stability."""
    residual_weight = residual_reliability_weight(residuals, loss_scale_m)
    temporal_weight, mobile_scores = temporal_stability_weight(temporal_ranges_m, loss_scale_m)
    weights = np.asarray(metadata_quality, dtype=float) * residual_weight
    if use_temporal_weight:
        weights *= temporal_weight
    return np.clip(weights, 0.01, 1.0), mobile_scores
