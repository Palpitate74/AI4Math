from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from .models import MethodConfig, ObservationSet, SolverResult
from .reliability import combine_reliability, temporal_stability_weight


def _residuals(position: np.ndarray, points: np.ndarray, ranges: np.ndarray) -> np.ndarray:
    return np.linalg.norm(points - position, axis=1) - ranges


def _confidence_radius(residuals: np.ndarray, points: np.ndarray, weights: np.ndarray) -> float:
    centered = points - np.average(points, axis=0, weights=np.maximum(weights, 1e-6))
    geometry = np.linalg.svd(centered * np.sqrt(weights)[:, None], compute_uv=False)
    geometry_penalty = 1.0 if len(geometry) < 2 else max(geometry[0] / max(geometry[1], 1e-4), 1.0)
    return float(np.sqrt(np.average(residuals**2, weights=np.maximum(weights, 1e-6))) * geometry_penalty)


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
        f_scale=max(scale, 0.1),
        max_nfev=max_iterations * 25,
    )


def localize(observations: ObservationSet, method: MethodConfig) -> SolverResult:
    """Estimate a device position with a baseline or robust nonlinear objective."""
    points = observations.reported_positions
    ranges = observations.estimated_ranges_m
    initial = np.average(points, axis=0, weights=np.maximum(observations.metadata_quality, 1e-3))
    n = len(ranges)
    weights = np.ones(n)
    mobile_scores = np.zeros(n)

    if method.name == "weighted_nls":
        weights = observations.metadata_quality.copy()
    elif method.name == "adaptive":
        # Alternating robust fit and reliability update: optimize the coordinate,
        # then update helper reliability from residual and temporal stability.
        weights = observations.metadata_quality.copy()
        position = initial
        final = None
        for iteration in range(1, method.max_irls_iterations + 1):
            final = _solve_weighted(
                points,
                ranges,
                weights,
                loss="huber",
                scale=method.loss_scale_m,
                max_iterations=method.max_irls_iterations,
                initial=position,
            )
            position = final.x
            residuals = _residuals(position, points, ranges)
            next_weights, mobile_scores = combine_reliability(
                observations.metadata_quality,
                residuals,
                observations.temporal_ranges_m,
                method.loss_scale_m,
                method.use_temporal_weight,
            )
            next_weights[next_weights < method.reliability_threshold] = 0.01
            if np.max(np.abs(next_weights - weights)) < 1e-3:
                weights = next_weights
                break
            weights = next_weights
        assert final is not None
        residuals = _residuals(position, points, ranges)
        return SolverResult(
            estimate=position,
            residuals=residuals,
            weights=weights,
            converged=bool(final.success),
            iterations=iteration,
            confidence_radius_m=_confidence_radius(residuals, points, weights),
            mobile_scores=mobile_scores,
        )
    else:
        if method.name == "nls":
            loss = "linear"
        elif method.name == "huber":
            loss = "huber"
        else:
            loss = "cauchy"

        if method.use_temporal_weight:
            temporal_weight, mobile_scores = temporal_stability_weight(
                observations.temporal_ranges_m, method.loss_scale_m
            )
            weights *= temporal_weight
        final = _solve_weighted(
            points,
            ranges,
            weights,
            loss=loss,
            scale=method.loss_scale_m,
            max_iterations=method.max_irls_iterations,
            initial=initial,
        )
        residuals = _residuals(final.x, points, ranges)
        return SolverResult(
            estimate=final.x,
            residuals=residuals,
            weights=weights,
            converged=bool(final.success),
            iterations=int(final.nfev),
            confidence_radius_m=_confidence_radius(residuals, points, weights),
            mobile_scores=mobile_scores,
        )
