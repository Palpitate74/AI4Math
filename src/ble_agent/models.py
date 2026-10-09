from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np

MethodName = Literal["nls", "weighted_nls", "huber", "cauchy", "adaptive"]
LayoutName = Literal["surrounding", "clustered", "collinear"]


@dataclass(frozen=True)
class ScenarioConfig:
    name: str
    seed: int
    trials: int = 40
    beacon_count: int = 15
    layout: LayoutName = "surrounding"
    noise_std_m: float = 2.5
    outlier_rate: float = 0.20
    outlier_bias_m: float = 25.0
    mobile_rate: float = 0.10
    mobile_drift_m: float = 12.0
    calibration_bias: float = 0.0
    temporal_windows: int = 3

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class MethodConfig:
    name: MethodName
    loss_scale_m: float = 8.0
    reliability_threshold: float = 0.20
    max_irls_iterations: int = 10
    use_temporal_weight: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ObservationSet:
    true_position: np.ndarray
    reported_positions: np.ndarray
    rssi_dbm: np.ndarray
    estimated_ranges_m: np.ndarray
    metadata_quality: np.ndarray
    temporal_ranges_m: np.ndarray
    mobile_mask: np.ndarray
    outlier_mask: np.ndarray
    scenario: ScenarioConfig


@dataclass
class SolverResult:
    estimate: np.ndarray
    residuals: np.ndarray
    weights: np.ndarray
    converged: bool
    iterations: int
    confidence_radius_m: float
    mobile_scores: np.ndarray


@dataclass
class TrialResult:
    error_m: float
    converged: bool
    runtime_ms: float
    mobile_true: np.ndarray
    mobile_scores: np.ndarray


@dataclass
class ExperimentResult:
    experiment_id: str
    split: str
    method: MethodConfig
    scenarios: list[ScenarioConfig]
    metrics: dict
    slices: list[dict]
    trials: list[TrialResult]
    environment: dict

    def to_dict(self) -> dict:
        return {
            "experiment_id": self.experiment_id,
            "split": self.split,
            "method": self.method.to_dict(),
            "scenarios": [scenario.to_dict() for scenario in self.scenarios],
            "metrics": self.metrics,
            "slices": self.slices,
            "environment": self.environment,
        }
