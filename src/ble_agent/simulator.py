from __future__ import annotations

import numpy as np

from .calibration import range_to_rssi, rssi_to_range
from .models import ObservationSet, ScenarioConfig

MAP_SIDE_M = 100.0


def _beacon_positions(rng: np.random.Generator, config: ScenarioConfig) -> np.ndarray:
    count = config.beacon_count
    center = np.array([MAP_SIDE_M / 2.0, MAP_SIDE_M / 2.0])

    if config.layout == "surrounding":
        angles = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
        angles += rng.normal(0.0, 0.10, size=count)
        radii = rng.uniform(36.0, 58.0, size=count)
        points = center + np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
    elif config.layout == "clustered":
        points = rng.normal(loc=[25.0, 30.0], scale=[11.0, 12.0], size=(count, 2))
    else:  # collinear (validated by simulate_observations)
        x = np.linspace(8.0, 92.0, count) + rng.normal(0.0, 2.0, size=count)
        y = 50.0 + rng.normal(0.0, 2.0, size=count)
        points = np.column_stack((x, y))

    return np.clip(points, 1.0, MAP_SIDE_M - 1.0)


def _validate_config(config: ScenarioConfig, trial_index: int) -> None:
    if config.metadata_mode not in {"observable", "oracle"}:
        raise ValueError("metadata_mode must be observable or oracle")
    if config.layout not in {"surrounding", "clustered", "collinear"}:
        raise ValueError("unknown beacon layout")
    for name, value, minimum in (
        ("seed", config.seed, 0),
        ("trial_index", trial_index, 0),
        ("beacon_count", config.beacon_count, 3),
        ("temporal_windows", config.temporal_windows, 1),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    for name in ("noise_std_m", "position_noise_std_m", "outlier_bias_m", "mobile_drift_m"):
        value = getattr(config, name)
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    for name in ("mobile_rate", "outlier_rate"):
        value = getattr(config, name)
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{name} must be in [0, 1]")
    if not np.isfinite(config.calibration_bias):
        raise ValueError("calibration_bias must be finite (dBm)")


def simulate_observations(config: ScenarioConfig, trial_index: int) -> ObservationSet:
    """Deterministic synthetic BLE observations, not a target-motion model.

    The default metadata consists of synthetic accuracy/age proxies drawn from
    a separate random stream, independent of mobile/outlier labels and drift.
    It is not a measured positioning-quality model. Explicit oracle metadata
    leaks simulated corruption and is suitable only as an idealized upper bound.
    position_noise_std_m=0 is required for exact, noiseless recovery.
    calibration_bias is applied once in dBm, including to temporal observations.
    """
    _validate_config(config, trial_index)
    rng = np.random.default_rng(config.seed + trial_index * 10_007)
    metadata_rng = np.random.default_rng(np.random.SeedSequence([config.seed, trial_index, 71_029]))
    count = config.beacon_count
    position_accuracy_m = metadata_rng.uniform(0.5, 4.0, size=count)
    observation_age_s = metadata_rng.uniform(0.0, 60.0, size=count)
    metadata_quality = 1.0 / (1.0 + 0.15 * position_accuracy_m + observation_age_s / 120.0)

    true_position = rng.uniform(20.0, 80.0, size=2)
    actual_beacons = _beacon_positions(rng, config)
    mobile_mask = rng.random(count) < config.mobile_rate
    outlier_mask = rng.random(count) < config.outlier_rate
    drifts = np.zeros_like(actual_beacons)
    if mobile_mask.any():
        directions = rng.normal(size=(mobile_mask.sum(), 2))
        directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-12)
        magnitudes = rng.uniform(config.mobile_drift_m * 0.45, config.mobile_drift_m, mobile_mask.sum())
        drifts[mobile_mask] = directions * magnitudes[:, None]

    reported_positions = actual_beacons + drifts + rng.normal(
        0.0, config.position_noise_std_m, size=(count, 2)
    )
    true_ranges = np.linalg.norm(actual_beacons - true_position, axis=1)
    measured_ranges = true_ranges + rng.normal(0.0, config.noise_std_m, size=count)
    measured_ranges[outlier_mask] += rng.exponential(config.outlier_bias_m, outlier_mask.sum())
    measured_ranges = np.clip(measured_ranges, 0.5, None)

    # Repeated range variability can have many causes; it is not evidence of
    # target motion. Both current and temporal ranges use the same RSSI model.
    temporal_ranges = measured_ranges[None, :] + rng.normal(
        0.0, config.noise_std_m, size=(config.temporal_windows, count)
    )
    if mobile_mask.any():
        temporal_ranges[:, mobile_mask] += rng.normal(
            0.0,
            config.mobile_drift_m * 0.60,
            size=(config.temporal_windows, mobile_mask.sum()),
        )
    temporal_ranges = np.clip(temporal_ranges, 0.5, None)

    rssi_dbm = range_to_rssi(measured_ranges, config.calibration_bias)
    estimated_ranges = rssi_to_range(rssi_dbm)
    temporal_ranges = rssi_to_range(range_to_rssi(temporal_ranges, config.calibration_bias))
    if config.metadata_mode == "oracle":
        metadata_quality = metadata_quality / (
            1.0 + 0.06 * np.linalg.norm(drifts, axis=1) + outlier_mask.astype(float) * 1.8
        )

    return ObservationSet(
        true_position=true_position,
        reported_positions=reported_positions,
        rssi_dbm=rssi_dbm,
        estimated_ranges_m=estimated_ranges,
        metadata_quality=metadata_quality,
        temporal_ranges_m=temporal_ranges,
        mobile_mask=mobile_mask,
        outlier_mask=outlier_mask,
        scenario=config,
        position_accuracy_m=position_accuracy_m,
        observation_age_s=observation_age_s,
    )
