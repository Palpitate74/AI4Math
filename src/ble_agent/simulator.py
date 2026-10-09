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
    else:  # collinear
        x = np.linspace(8.0, 92.0, count) + rng.normal(0.0, 2.0, size=count)
        y = 50.0 + rng.normal(0.0, 2.0, size=count)
        points = np.column_stack((x, y))

    return np.clip(points, 1.0, MAP_SIDE_M - 1.0)


def simulate_observations(config: ScenarioConfig, trial_index: int) -> ObservationSet:
    """Generate a deterministic crowdsourced BLE localization observation set."""
    rng = np.random.default_rng(config.seed + trial_index * 10_007)
    true_position = rng.uniform(20.0, 80.0, size=2)
    actual_beacons = _beacon_positions(rng, config)
    count = config.beacon_count

    mobile_mask = rng.random(count) < config.mobile_rate
    outlier_mask = rng.random(count) < config.outlier_rate
    drifts = np.zeros_like(actual_beacons)
    if mobile_mask.any():
        directions = rng.normal(size=(mobile_mask.sum(), 2))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        magnitudes = rng.uniform(config.mobile_drift_m * 0.45, config.mobile_drift_m, mobile_mask.sum())
        drifts[mobile_mask] = directions * magnitudes[:, None]

    # The phone reports its possibly stale/inaccurate own location, while the BLE
    # range is emitted at its actual location.
    position_noise_std = 0.0 if (config.mobile_rate == 0.0 and config.outlier_rate == 0.0 and config.noise_std_m == 0.0) else 1.2
    reported_positions = actual_beacons + drifts + rng.normal(0.0, position_noise_std, size=(count, 2))
    true_ranges = np.linalg.norm(actual_beacons - true_position, axis=1)
    measured_ranges = true_ranges + rng.normal(0.0, config.noise_std_m, size=count)
    measured_ranges += config.calibration_bias
    measured_ranges[outlier_mask] += rng.exponential(config.outlier_bias_m, outlier_mask.sum())
    measured_ranges = np.clip(measured_ranges, 0.5, None)

    # Generate temporally repeated observations. Moving beacons have unstable range
    # observations, which makes their temporal-consistency score useful.
    temporal_noise = rng.normal(0.0, config.noise_std_m, size=(config.temporal_windows, count))
    temporal_ranges = measured_ranges[None, :] + temporal_noise
    if mobile_mask.any():
        temporal_ranges[:, mobile_mask] += rng.normal(
            0.0,
            config.mobile_drift_m * 0.60,
            size=(config.temporal_windows, mobile_mask.sum()),
        )
    temporal_ranges = np.clip(temporal_ranges, 0.5, None)

    rssi_dbm = range_to_rssi(measured_ranges, config.calibration_bias)
    estimated_ranges = rssi_to_range(rssi_dbm)
    metadata_quality = 1.0 / (
        1.0
        + 0.06 * np.linalg.norm(drifts, axis=1)
        + outlier_mask.astype(float) * 1.8
        + rng.uniform(0.0, 0.15, size=count)
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
    )
