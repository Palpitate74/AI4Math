import pytest
from pydantic import ValidationError

from ble_agent.agent import apply_proposal, validate_proposal
from ble_agent.models import MethodConfig


def test_rejects_unallowlisted_agent_change():
    with pytest.raises(ValidationError):
        validate_proposal(
            {
                "hypothesis": "Try arbitrary code execution.",
                "change": {"field": "loss_scale_m", "value": 999.0},
                "rationale": "This should never be executed.",
            }
        )


def test_applies_exactly_one_safe_change():
    proposal = validate_proposal(
        {
            "hypothesis": "A robust loss should reduce heavy-tailed range residuals.",
            "change": {"field": "name", "value": "huber"},
            "rationale": "Validation CEP90 is dominated by the outlier tail.",
        }
    )
    method = apply_proposal(MethodConfig(name="nls"), proposal)
    assert method.name == "huber"
    assert method.loss_scale_m == 8.0
