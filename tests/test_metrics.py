import numpy as np

from ble_agent.metrics import binary_metrics, percentile


def test_cep_percentile():
    assert percentile([0.0, 10.0, 20.0, 30.0, 40.0], 50) == 20.0


def test_binary_metrics_perfect_predictions():
    result = binary_metrics(np.array([True, False, True]), np.array([0.9, 0.1, 0.95]))
    assert result == {"precision": 1.0, "recall": 1.0, "f1": 1.0}
