from __future__ import annotations

import json

import pytest

from ble_agent.agent import apply_proposal, diagnostic_proposal, validate_proposal
from ble_agent.experiment import accepted_against, experiment_path, validation_summary
from ble_agent.graph import run_iteration
from ble_agent.models import MethodConfig
from ble_agent.runtime import data_digest


def proposal(field="name", value="huber"):
    return {"hypothesis": "Robust estimation should suppress the observed outlier tail.",
            "change": {"field": field, "value": value}, "rationale": "The observed tail is large relative to its median."}


@pytest.mark.parametrize("raw", [
    {**proposal(), "shell": "command"},
    {**proposal(), "change": {"field": "name", "value": "huber", "extra": 1}},
    proposal("use_temporal_weight", 1), proposal("loss_scale_m", True),
    proposal("max_irls_iterations", 10.0), proposal("name", ["huber"]),
])
def test_strict_schema(raw):
    with pytest.raises(ValueError):
        validate_proposal(raw)


@pytest.mark.parametrize("current,field,value", [
    ("huber", "name", "huber"), ("nls", "loss_scale_m", 4),
    ("cauchy", "max_irls_iterations", 14), ("huber", "reliability_threshold", .3),
])
def test_inactive_or_noop_rejected(current, field, value):
    with pytest.raises(ValueError):
        apply_proposal(MethodConfig(name=current), validate_proposal(proposal(field, value)))


def summary(cep=10., failure=.05, clean=2.):
    return {"data_identity": {"split": "midterm_validation",
                "scenario_sha256": data_digest([{"name": "mock-only"}]),
                "pair_order_sha256": data_digest(["mock-only:0"]),
                "generator": {"version": "mock-only", "numpy": "mock-only",
                    "source_sha256": {name: "0" * 64 for name in ["simulator.py", "calibration.py", "models.py"]}}},
            "scenarios": [{"name": "mock-only"}], "trial_keys": ["mock-only:0"],
            "metrics": {"cep90_m": cep, "median_error_m": 2.,
                        "failure_rate_over_30m": failure, "convergence_rate": 1.},
            "clean_slices": [{"slice": "clean", "cep90_m": clean}]}


def test_failure_and_clean_constraints():
    assert not accepted_against(summary(), summary(9., .06))[0]
    assert not accepted_against(summary(), summary(9., .05, 3.))[0]
    assert accepted_against(summary(), summary(9., .05, 2.1))[0]


def test_path_traversal():
    with pytest.raises(ValueError):
        experiment_path("../../.env.local")


def test_holdout_reader_denied(tmp_path, monkeypatch):
    monkeypatch.setenv("BLE_ARTIFACT_DIR", str(tmp_path))
    (tmp_path / "experiments").mkdir()
    (tmp_path / "experiments" / "holdout.json").write_text(json.dumps({"split": "midterm_holdout"}))
    with pytest.raises(ValueError):
        validation_summary("holdout")


def test_diagnostic_uses_observed_mobile_slice():
    current = MethodConfig(name="huber")
    data = {"metrics": {"cep90_m": 10, "median_error_m": 3},
            "worst_slices": [{"slice": "drift", "cep90_m": 20, "mobile_rate": .25}]}
    assert diagnostic_proposal(current, data).change.field == "use_temporal_weight"


def test_graph_live_flow_mock_transport(tmp_path, monkeypatch):
    import ble_agent.graph as graph
    monkeypatch.setenv("BLE_ARTIFACT_DIR", str(tmp_path))
    current, candidate = summary(10), summary(8)
    current.update(experiment_id="base", method=MethodConfig(name="nls").to_dict(), slices=[], worst_slices=[])
    candidate.update(experiment_id="candidate", slices=[])
    monkeypatch.setattr(graph, "validation_summary", lambda identifier: current if identifier == "base" else candidate)
    calls = []
    def fake_request(*args):
        calls.append(args)
        return proposal(), {"response_id": "mock-only", "model": "mock", "raw_response": json.dumps(proposal())}
    monkeypatch.setattr(graph, "request_model_proposal", fake_request)
    class Result:
        experiment_id = "candidate"
    monkeypatch.setattr(graph, "run_experiment", lambda *a, **k: Result())
    out = run_iteration(policy="api", iteration_index=1, current_method=MethodConfig(name="nls"),
                        current_experiment_id="base", validation_split="midterm_validation")
    assert out["decision"]["accepted"]
    assert out["current_method"]["name"] == "huber"
    assert out["record"]["model_traces"][0]["response_id"] == "mock-only"
    assert len(calls) == 1


def test_api_error_logged_without_fallback_or_secret(tmp_path, monkeypatch):
    import ble_agent.graph as graph
    monkeypatch.setenv("BLE_ARTIFACT_DIR", str(tmp_path))
    monkeypatch.setattr(graph, "validation_summary", lambda _: {"metrics": {}, "worst_slices": []})
    def fail(*args):
        raise RuntimeError("SECRET must not be logged")
    monkeypatch.setattr(graph, "request_model_proposal", fail)
    out = run_iteration(policy="api", iteration_index=1, current_method=MethodConfig(name="nls"),
                        current_experiment_id="base", validation_split="midterm_validation")
    assert out["record"]["status"] == "error"
    assert "SECRET" not in json.dumps(out)
    assert out["record"]["proposal"] is None
    assert not out["decision"]["accepted"]


def test_executor_split_guard():
    with pytest.raises(ValueError):
        run_iteration(policy="api", iteration_index=1, current_method=MethodConfig(name="nls"),
                      current_experiment_id="base", validation_split="midterm_holdout")
