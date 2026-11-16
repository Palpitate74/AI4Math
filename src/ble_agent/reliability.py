from __future__ import annotations

import numpy as np


def _scale(loss_scale_m: float) -> float:
    if not np.isfinite(loss_scale_m) or loss_scale_m <= 0:
        raise ValueError("loss_scale_m must be finite and positive")
    return float(loss_scale_m)


def temporal_stability_weight(temporal_ranges_m: np.ndarray, loss_scale_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Return reliability and normalized instability (std / scale).

    Larger second values mean less stable observations. This detects repeated
    range variability, not target movement; scores are not probabilities.
    """
    ranges = np.asarray(temporal_ranges_m, dtype=float)
    if ranges.ndim != 2 or not all(ranges.shape) or not np.isfinite(ranges).all():
        raise ValueError("temporal_ranges_m must be a nonempty finite 2D array")
    normalized = np.std(ranges, axis=0) / _scale(loss_scale_m)
    weights = 1.0 / (1.0 + normalized**2)
    return np.clip(weights, 0.03, 1.0), normalized


def residual_reliability_weight(residuals: np.ndarray, loss_scale_m: float) -> np.ndarray:
    values = np.asarray(residuals, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("residuals must be a nonempty finite 1D array")
    scaled = np.abs(values) / _scale(loss_scale_m)
    return np.clip(1.0 / (1.0 + scaled**2), 0.02, 1.0)


def combine_reliability(
    metadata_quality: np.ndarray,
    residuals: np.ndarray,
    temporal_ranges_m: np.ndarray,
    loss_scale_m: float,
    use_temporal_weight: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Combine metadata and residuals; temporal evidence is explicitly opt-in.

    The legacy mobile_scores output is normalized instability, or all zeros
    when temporal weighting is disabled. It is not a target-motion classifier.
    """
    residual_weight = residual_reliability_weight(residuals, loss_scale_m)
    metadata = np.asarray(metadata_quality, dtype=float)
    if metadata.shape != residual_weight.shape or not np.isfinite(metadata).all():
        raise ValueError("metadata_quality must be finite and match residuals")
    if np.any((metadata < 0) | (metadata > 1)):
        raise ValueError("metadata_quality must be in [0, 1]")
    weights = metadata * residual_weight
    instability = np.zeros_like(residual_weight)
    if use_temporal_weight:
        temporal_weight, instability = temporal_stability_weight(temporal_ranges_m, loss_scale_m)
        if temporal_weight.shape != weights.shape:
            raise ValueError("temporal ranges must have one column per helper")
        weights *= temporal_weight
    return np.clip(weights, 0.01, 1.0), instability
