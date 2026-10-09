import numpy as np

from ble_agent.models import ScenarioConfig
from ble_agent.simulator import simulate_observations


def test_simulator_is_seed_reproducible():
    config = ScenarioConfig(name="test", seed=42, trials=1)
    first = simulate_observations(config, 0)
    second = simulate_observations(config, 0)
    np.testing.assert_allclose(first.true_position, second.true_position)
    np.testing.assert_allclose(first.reported_positions, second.reported_positions)
    np.testing.assert_allclose(first.estimated_ranges_m, second.estimated_ranges_m)
    assert np.array_equal(first.mobile_mask, second.mobile_mask)
