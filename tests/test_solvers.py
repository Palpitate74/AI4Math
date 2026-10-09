from dataclasses import replace

from ble_agent.models import MethodConfig, ScenarioConfig
from ble_agent.simulator import simulate_observations
from ble_agent.solvers import localize


def test_nls_recovers_near_exact_position_without_noise_or_corruption():
    config = ScenarioConfig(
        name="noiseless",
        seed=7,
        trials=1,
        beacon_count=20,
        noise_std_m=0.0,
        outlier_rate=0.0,
        mobile_rate=0.0,
        calibration_bias=0.0,
    )
    observation = simulate_observations(config, 0)
    result = localize(observation, MethodConfig(name="nls"))
    assert result.converged
    assert float(((result.estimate - observation.true_position) ** 2).sum() ** 0.5) < 1e-4


def test_adaptive_method_returns_bounded_weights():
    config = ScenarioConfig(name="corrupt", seed=8, trials=1, outlier_rate=0.3, mobile_rate=0.2)
    observation = simulate_observations(config, 1)
    result = localize(observation, MethodConfig(name="adaptive", use_temporal_weight=True))
    assert result.weights.min() >= 0.01
    assert result.weights.max() <= 1.0
    assert result.iterations <= 10
