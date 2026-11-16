from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from .models import MethodConfig, ObservationSet, SolverObservations, SolverResult
from .reliability import combine_reliability, temporal_stability_weight


VALID_METHODS = {"nls", "weighted_nls", "huber", "cauchy", "adaptive"}


def _residuals(position: np.ndarray, points: np.ndarray, ranges: np.ndarray) -> np.ndarray:
    return np.linalg.norm(points - position, axis=1) - ranges


def _confidence_radius(residuals: np.ndarray, points: np.ndarray, weights: np.ndarray) -> float:
    effective_weights = np.maximum(weights, 1e-6)
    centered = points - np.average(points, axis=0, weights=effective_weights)
    geometry = np.linalg.svd(centered * np.sqrt(effective_weights)[:, None], compute_uv=False)
    geometry_penalty = 1.0 if len(geometry) < 2 else max(geometry[0] / max(geometry[1], 1e-4), 1.0)
    return float(np.sqrt(np.average(residuals**2, weights=effective_weights)) * geometry_penalty)


def _solve_weighted(
    points: np.ndarray,
    ranges: np.ndarray,
    weights: np.ndarray,
    loss: str,
    scale: float,
    max_iterations: int,
    initial: np.ndarray,
):
    sqrt_weights = np.sqrt(np.clip(weights, 1e-6, None))

    def objective(position: np.ndarray) -> np.ndarray:
        return sqrt_weights * _residuals(position, points, ranges)

    return least_squares(
        objective,
        x0=initial,
        loss=loss,
        f_scale=scale,
        max_nfev=max_iterations * 25,
    )


def _validate_inputs(observations: SolverObservations, method: MethodConfig) -> tuple[np.ndarray, ...]:
    if method.name not in VALID_METHODS:
        raise ValueError(f"unknown localization method: {method.name}")
    if not np.isfinite(method.loss_scale_m) or method.loss_scale_m <= 0:
        raise ValueError("loss_scale_m must be finite and positive")
    if not np.isfinite(method.reliability_threshold) or not 0 <= method.reliability_threshold <= 1:
        raise ValueError("reliability_threshold must be in [0, 1]")
    if (
        isinstance(method.max_irls_iterations, bool)
        or not isinstance(method.max_irls_iterations, (int, np.integer))
        or method.max_irls_iterations < 1
    ):
        raise ValueError("max_irls_iterations must be a positive integer")
    points = np.asarray(observations.reported_positions, dtype=float)
    ranges = np.asarray(observations.estimated_ranges_m, dtype=float)
    metadata = np.asarray(observations.metadata_quality, dtype=float)
    temporal = np.asarray(observations.temporal_ranges_m, dtype=float)
    rssi = np.asarray(observations.rssi_dbm, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
        raise ValueError("reported_positions must have shape (n, 2), n >= 3")
    n = len(points)
    if ranges.shape != (n,) or metadata.shape != (n,) or rssi.shape != (n,):
        raise ValueError("ranges, metadata and RSSI must have one value per helper")
    if temporal.ndim != 2 or temporal.shape[0] < 1 or temporal.shape[1] != n:
        raise ValueError("temporal_ranges_m must have shape (windows, n), windows >= 1")
    if not all(np.isfinite(array).all() for array in (points, ranges, metadata, temporal, rssi)):
        raise ValueError("solver inputs must be finite")
    if np.any(ranges <= 0) or np.any(temporal <= 0):
        raise ValueError("ranges must be positive metres")
    if np.any((metadata < 0) | (metadata > 1)):
        raise ValueError("metadata_quality must be in [0, 1]")
    return points, ranges, metadata, temporal


def _localize_inputs(observations: SolverObservations, method: MethodConfig) -> SolverResult:
    """Solver body cannot access simulation truth or scenario configuration."""
    points, ranges, metadata, temporal = _validate_inputs(observations, method)
    # Identical geometry-only initialization for every method; unweighted
    # baselines are not secretly advantaged by reliability metadata.
    initial = np.mean(points, axis=0)
    weights = np.ones(len(ranges))
    instability = np.zeros(len(ranges))

    if method.name == "adaptive":
        weights = np.clip(metadata.copy(), 1e-6, 1.0)
        position = initial
        for iteration in range(1, method.max_irls_iterations + 1):
            fit = _solve_weighted(
                points, ranges, weights, "huber", method.loss_scale_m,
                method.max_irls_iterations, position,
            )
            position = fit.x
            next_weights, instability = combine_reliability(
                metadata, _residuals(position, points, ranges), temporal,
                method.loss_scale_m, method.use_temporal_weight,
            )
            next_weights[next_weights < method.reliability_threshold] = 0.01
            stable = np.max(np.abs(next_weights - weights)) < 1e-3
            weights = next_weights
            if stable:
                break
        # The returned coordinates must solve the objective for the returned
        # weights. This final fit performs NO further reliability update.
        final = _solve_weighted(
            points, ranges, weights, "huber", method.loss_scale_m,
            method.max_irls_iterations, position,
        )
        iterations = iteration  # Number of IRLS weight updates, excludes final fit.
    else:
        loss = {"nls": "linear", "weighted_nls": "linear", "huber": "huber", "cauchy": "cauchy"}[method.name]
        if method.name == "weighted_nls":
            weights = metadata.copy()
        if method.use_temporal_weight:
            temporal_weight, instability = temporal_stability_weight(temporal, method.loss_scale_m)
            weights *= temporal_weight
        # Report the same effective weights as those used by the optimizer.
        weights = np.clip(weights, 1e-6, 1.0)
        final = _solve_weighted(
            points, ranges, weights, loss, method.loss_scale_m,
            method.max_irls_iterations, initial,
        )
        iterations = int(final.nfev)  # SciPy function evaluations, not IRLS updates.

    residuals = _residuals(final.x, points, ranges)
    return SolverResult(
        estimate=final.x,
        residuals=residuals,
        weights=weights,
        converged=bool(final.success),
        iterations=iterations,
        confidence_radius_m=_confidence_radius(residuals, points, weights),
        mobile_scores=instability,
    )


def localize(observations: SolverObservations | ObservationSet, method: MethodConfig) -> SolverResult:
    """Estimate from observables, converting legacy simulation containers first.

    For adaptive, iterations counts IRLS reliability updates and excludes the
    mandatory final fixed-weight solve. For other methods it is SciPy nfev.
    mobile_scores retains its old name but means normalized range instability;
    scores are zero when temporal weighting is off, not target-motion estimates.
    """
    if isinstance(observations, ObservationSet):
        observations = observations.solver_inputs()
    if not isinstance(observations, SolverObservations):
        raise TypeError("observations must be SolverObservations or ObservationSet")
    return _localize_inputs(observations, method)
