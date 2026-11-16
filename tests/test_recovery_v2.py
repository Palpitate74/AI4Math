"""Controlled local faults only. Mock responses are not real Agent evidence."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ble_agent import experiment, graph, runtime
from ble_agent.agent import apply_proposal, validate_proposal
from ble_agent.logging import orphan_response_events, read_iterations
from ble_agent.models import MethodConfig
from test_agent_v2 import proposal, summary

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("midterm_workflow_test", REPO / "scripts" / "run_midterm.py")
workflow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workflow)


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    source = tmp_path / "project"
    (source / "src" / "ble_agent").mkdir(parents=True)
    for file in (REPO / "src" / "ble_agent").glob("*.py"):
        (source / "src" / "ble_agent" / file.name).write_bytes(file.read_bytes())
    (source / "scripts").mkdir()
    (source / "scripts" / "run_midterm.py").write_bytes((REPO / "scripts" / "run_midterm.py").read_bytes())
    (source / "configs").mkdir()
    for split, seed in [("midterm_validation", 71), ("midterm_holdout", 81)]:
        (source / "configs" / f"{split}_scenarios.json").write_text(json.dumps([
            {"name": "tiny-test", "seed": seed, "trials": 4, "beacon_count": 10,
             "outlier_rate": .1, "mobile_rate": .1}]))
    output = tmp_path / "run"
    monkeypatch.setattr(runtime, "ROOT", source)
    monkeypatch.setattr(experiment, "ROOT", source)
    monkeypatch.setenv("BLE_ARTIFACT_DIR", str(output))
    return source, output


def test_snapshot_uses_original_conditions_after_config_edit(isolated):
    source, output = isolated
    manifest = runtime.freeze_scenarios(["midterm_validation", "midterm_holdout"])
    config = source / "configs" / "midterm_validation_scenarios.json"
    raw = json.loads(config.read_text())
    raw[0]["outlier_bias_m"] = 0
    config.write_text(json.dumps(raw))
    assert experiment.load_scenarios("midterm_validation")[0].outlier_bias_m == 25
    assert manifest == runtime.freeze_scenarios(["midterm_validation", "midterm_holdout"])
    snapshot = output / "protocol" / "midterm_validation_scenarios.json"
    raw = json.loads(snapshot.read_text())
    raw[0]["outlier_bias_m"] = 0
    snapshot.write_text(json.dumps(raw))
    with pytest.raises(RuntimeError, match="snapshot changed"):
        experiment.load_scenarios("midterm_validation")


def test_generator_change_rejected(isolated):
    source, _ = isolated
    runtime.freeze_scenarios(["midterm_validation", "midterm_holdout"])
    with (source / "src" / "ble_agent" / "simulator.py").open("a") as handle:
        handle.write("\n# generator changed\n")
    with pytest.raises(RuntimeError, match="Generator changed"):
        runtime.freeze_scenarios(["midterm_validation"])


def test_solver_execution_change_requires_new_run(isolated):
    source, _ = isolated
    runtime.freeze_scenarios(["midterm_validation", "midterm_holdout"])
    with (source / "src" / "ble_agent" / "solvers.py").open("a") as handle:
        handle.write("\n# solver changed\n")
    with pytest.raises(RuntimeError, match="execution code changed"):
        runtime.freeze_scenarios(["midterm_validation"])


def test_no_retroactive_preregistration(isolated):
    _, output = isolated
    (output / "experiments").mkdir(parents=True)
    (output / "experiments" / "old.json").write_text("{}")
    with pytest.raises(RuntimeError, match="Historical run"):
        runtime.freeze_scenarios(["midterm_validation"])


def test_full_data_identity_rejects_false_gain():
    current, candidate = summary(20), summary(10)
    candidate["data_identity"] = {**candidate["data_identity"], "scenario_sha256": "different-outlier-bias"}
    assert not experiment.accepted_against(current, candidate)[0]
    candidate = summary(10)
    candidate["scenarios"] = [{"name": "changed"}]
    assert not experiment.accepted_against(current, candidate)[0]
    candidate = summary(10)
    candidate["trial_keys"] = ["wrong-order"]
    assert not experiment.accepted_against(current, candidate)[0]
    candidate = summary(10)
    candidate.pop("data_identity")
    assert not experiment.accepted_against(current, candidate)[0]


@pytest.mark.parametrize("method", ["nls", "weighted_nls"])
def test_temporal_scale_is_effective_allowlisted_control(method):
    change = validate_proposal(proposal("loss_scale_m", 4))
    assert apply_proposal(MethodConfig(name=method, use_temporal_weight=True), change).loss_scale_m == 4
    with pytest.raises(ValueError):
        apply_proposal(MethodConfig(name=method), change)


def mock_graph(monkeypatch):
    current, candidate = summary(10), summary(8)
    current.update(experiment_id="base", method=MethodConfig(name="nls").to_dict(), slices=[], worst_slices=[])
    candidate.update(experiment_id="candidate", slices=[])
    monkeypatch.setattr(graph, "validation_summary", lambda identifier: current if identifier == "base" else candidate)
    monkeypatch.setattr(graph, "request_model_proposal", lambda *args: (proposal(), {"response_id": "mock-only", "raw_response": json.dumps(proposal())}))
    monkeypatch.setattr(graph, "run_experiment", lambda *args, **kwargs: SimpleNamespace(experiment_id="candidate"))


def test_accepted_decision_recovers_stale_summary(isolated, monkeypatch):
    mock_graph(monkeypatch)
    out = graph.run_iteration(policy="api", iteration_index=1, current_method=MethodConfig(name="nls"),
                              current_experiment_id="base", validation_split="midterm_validation")
    assert out["decision"]["accepted"]
    stale = {"policy": "api", "baseline_validation": "base", "final_validation": "base",
             "final_method": MethodConfig(name="nls").to_dict(), "completed_iterations": 0}
    workflow.recover_summary(stale, read_iterations())
    assert stale["final_validation"] == "candidate"
    assert stale["final_method"]["name"] == "huber"
    assert stale["api_evaluated_iterations"] == 1
    assert not orphan_response_events()
    with pytest.raises(RuntimeError, match="authoritative decision"):
        graph.run_iteration(policy="api", iteration_index=1, current_method=MethodConfig(name="nls"), current_experiment_id="base")


@pytest.mark.parametrize("stage", ["execute", "evaluate"])
def test_execution_failure_retains_model_evidence(isolated, monkeypatch, stage):
    mock_graph(monkeypatch)
    def fail(*args, **kwargs):
        raise RuntimeError("SECRET do not log this exception")
    if stage == "execute":
        monkeypatch.setattr(graph, "run_experiment", fail)
    else:
        original = graph.validation_summary
        monkeypatch.setattr(graph, "validation_summary", lambda x: fail() if x == "candidate" else original(x))
    output = graph.run_iteration(policy="api", iteration_index=1, current_method=MethodConfig(name="nls"),
                                 current_experiment_id="base", validation_split="midterm_validation")
    assert output["record"]["status"] == "error"
    assert output["record"]["model_traces"][0]["response_id"] == "mock-only"
    assert "SECRET" not in json.dumps(output)
    _, root = isolated
    assert len(list((root / "iterations" / "model_responses").glob("*.json"))) == 1


def test_immutable_decision_survives_jsonl_mirror_gap(isolated, monkeypatch):
    mock_graph(monkeypatch)
    graph.run_iteration(policy="api", iteration_index=1, current_method=MethodConfig(name="nls"), current_experiment_id="base")
    _, output = isolated
    (output / "iterations" / "agent_iterations.jsonl").unlink()
    assert read_iterations()[0]["decision"]["accepted"]


def test_finalize_recovers_artifact_published_before_summary(isolated, monkeypatch):
    _, root = isolated
    runtime.freeze_scenarios(["midterm_validation", "midterm_holdout"])
    baseline = workflow.ensure_experiment(MethodConfig(name="nls"), "midterm_holdout", "holdout-nls")
    original_bytes = experiment.experiment_path("holdout-nls").read_bytes()
    summary_data = {"policy": "diagnostic", "api_evaluated_iterations": 0,
                    "final_method": MethodConfig(name="huber", use_temporal_weight=True).to_dict()}
    path = root / "summary.json"
    workflow.finalize_run(summary_data, path)
    assert summary_data["status"] == "completed"
    assert experiment.experiment_path("holdout-nls").read_bytes() == original_bytes
    frozen_at = summary_data["frozen_at_utc"]
    def forbid(*args, **kwargs):
        raise AssertionError("published experiments must not rerun")
    monkeypatch.setattr(workflow, "run_experiment", forbid)
    summary_data["status"] = "finalizing"
    summary_data["holdout"].pop("selected")
    summary_data["paired_bootstrap"].pop("selected")
    workflow.finalize_run(summary_data, path)
    assert summary_data["frozen_at_utc"] == frozen_at
    assert summary_data["status"] == "completed"
    assert baseline["data_identity"] == experiment.read_experiment("holdout-selected")["data_identity"]


def test_atomic_publish_never_overwrites(isolated):
    _, root = isolated
    target = root / "test.json"
    runtime.atomic_json(target, {"original": True}, exclusive=True)
    with pytest.raises(FileExistsError):
        runtime.atomic_json(target, {"original": False}, exclusive=True)
    assert json.loads(target.read_text()) == {"original": True}


def test_incomplete_equal_identities_cannot_be_accepted():
    current, candidate = summary(20), summary(10)
    for record in [current, candidate]:
        record["data_identity"] = {"split": "midterm_validation"}
        record.pop("trial_keys")
    assert not experiment.accepted_against(current, candidate)[0]


def test_missing_rejected_audit_tail_cannot_roll_back_frozen_progress():
    baseline = MethodConfig(name="nls").to_dict()
    state = {"policy": "api", "baseline_validation": "base", "final_validation": "base",
             "final_method": baseline, "completed_iterations": 3, "api_evaluated_iterations": 3,
             "frozen_at_utc": "test-only"}
    records = [{"iteration_index": i, "input_experiment_id": "base", "policy": "api",
                "status": "evaluated", "decision": {"accepted": False}} for i in [1, 2]]
    with pytest.raises(RuntimeError, match="audit tail missing"):
        workflow.recover_summary(state, records)
    state["api_evaluated_iterations"] = 2
    with pytest.raises(RuntimeError, match=">=3"):
        workflow.finalize_run(state, Path("unused-test-only.json"))
