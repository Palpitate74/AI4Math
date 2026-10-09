from __future__ import annotations

from ble_agent.experiment import run_experiment
from ble_agent.models import MethodConfig

if __name__ == "__main__":
    result = run_experiment(MethodConfig(name="nls"), "validation", label="validation-baseline-nls")
    print(result.experiment_id)
    print(result.metrics)
