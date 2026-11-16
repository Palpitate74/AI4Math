from dataclasses import fields, replace

import numpy as np
import pytest
from scipy.optimize import least_squares

from ble_agent.calibration import PATH_LOSS_EXPONENT, range_to_rssi
from ble_agent.metrics import paired_bootstrap, percentile, slice_trials, summarize_trials
from ble_agent.models import MethodConfig, ScenarioConfig, SolverObservations, TrialResult
from ble_agent.reliability import (
    combine_reliability,
    residual_reliability_weight,
    temporal_stability_weight,
)
from ble_agent.simulator import simulate_observations
from ble_agent.solvers import localize
import ble_agent.solvers as solvers


METHODS = ["nls", "weighted_nls", "huber", "cauchy", "adaptive"]


def _trial(error=5.0):
    return TrialResult(error, True, 1.0, np.array([False, True]), np.array([0.0, 1.0]))


@pytest.mark.parametrize("name", METHODS)
@pytest.mark.parametrize("temporal", [False, True])
@pytest.mark.parametrize("layout", ["surrounding", "clustered", "collinear"])
def test_every_solver_branch_is_finite(name, temporal, layout):
    obs = simulate_observations(ScenarioConfig(name="finite", seed=17, layout=layout), 2)
    result = localize(obs.solver_inputs(), MethodConfig(name=name, use_temporal_weight=temporal))
    for values in (result.estimate, result.residuals, result.weights, result.mobile_scores):
        assert np.isfinite(values).all()
    assert np.isfinite(result.confidence_radius_m)
    assert result.estimate.shape == (2,)
    assert result.weights.shape == (obs.scenario.beacon_count,)
    np.testing.assert_allclose(
        result.residuals,
        np.linalg.norm(obs.reported_positions - result.estimate, axis=1) - obs.estimated_ranges_m,
    )
    assert (result.weights > 0).all() and (result.weights <= 1).all()
    assert result.iterations >= 1
    if not temporal:
        np.testing.assert_array_equal(result.mobile_scores, np.zeros(obs.scenario.beacon_count))


@pytest.mark.parametrize("name", METHODS)
def test_solver_boundary_has_no_truth_and_truth_changes_do_not_change_output(name):
    obs = simulate_observations(ScenarioConfig(name="boundary", seed=21), 0)
    inputs = obs.solver_inputs()
    assert {field.name for field in fields(SolverObservations)} == {
        "reported_positions", "rssi_dbm", "estimated_ranges_m", "metadata_quality",
        "temporal_ranges_m", "position_accuracy_m", "observation_age_s",
    }
    for name_absent in ("true_position", "mobile_mask", "outlier_mask", "scenario"):
        assert not hasattr(inputs, name_absent)
    changed = replace(
        obs, true_position=np.array([1e9, -1e9]), mobile_mask=~obs.mobile_mask,
        outlier_mask=~obs.outlier_mask,
        scenario=replace(obs.scenario, seed=900, metadata_mode="oracle"),
    )
    for field in fields(SolverObservations):
        np.testing.assert_array_equal(getattr(inputs, field.name), getattr(changed.solver_inputs(), field.name))
    method = MethodConfig(name=name, use_temporal_weight=True)
    first = localize(obs, method)
    second = localize(changed, method)
    third = localize(inputs, method)
    for field in fields(type(first)):
        np.testing.assert_array_equal(getattr(first, field.name), getattr(second, field.name))
        np.testing.assert_array_equal(getattr(first, field.name), getattr(third, field.name))


def test_solver_inputs_copies_observable_arrays():
    obs = simulate_observations(ScenarioConfig(name="copy", seed=3), 0)
    inputs = obs.solver_inputs()
    for field in fields(SolverObservations):
        left, right = getattr(obs, field.name), getattr(inputs, field.name)
        assert not np.shares_memory(left, right)


@pytest.mark.parametrize("name", METHODS)
def test_all_initializations_are_geometry_mean(name, monkeypatch):
    obs = simulate_observations(ScenarioConfig(name="initial", seed=31), 0).solver_inputs()
    obs = replace(obs, metadata_quality=np.linspace(0.01, 1, len(obs.estimated_ranges_m)))
    starts = []
    original = solvers._solve_weighted

    def record(*args, **kwargs):
        starts.append(np.asarray(args[6] if len(args) > 6 else kwargs["initial"]).copy())
        return original(*args, **kwargs)

    monkeypatch.setattr(solvers, "_solve_weighted", record)
    localize(obs, MethodConfig(name=name))
    np.testing.assert_array_equal(starts[0], obs.reported_positions.mean(axis=0))


@pytest.mark.parametrize("name", ["nls", "huber", "cauchy"])
def test_unweighted_baselines_are_independent_of_metadata(name):
    inputs = simulate_observations(ScenarioConfig(name="metadata", seed=11), 0).solver_inputs()
    changed = replace(inputs, metadata_quality=1 - inputs.metadata_quality)
    np.testing.assert_array_equal(
        localize(inputs, MethodConfig(name=name)).estimate,
        localize(changed, MethodConfig(name=name)).estimate,
    )


def test_weighted_nls_solves_actual_metadata_objective():
    inputs = simulate_observations(ScenarioConfig(name="weighted", seed=10), 0).solver_inputs()
    inputs = replace(inputs, metadata_quality=np.linspace(0, 1, len(inputs.estimated_ranges_m)))
    result = localize(inputs, MethodConfig(name="weighted_nls"))
    effective = np.maximum(inputs.metadata_quality, 1e-6)
    reference = least_squares(
        lambda position: np.sqrt(effective) * (
            np.linalg.norm(inputs.reported_positions - position, axis=1) - inputs.estimated_ranges_m
        ),
        inputs.reported_positions.mean(axis=0), max_nfev=250,
    )
    np.testing.assert_allclose(result.estimate, reference.x, atol=1e-8)
    np.testing.assert_array_equal(result.weights, effective)
    assert result.converged == reference.success
    assert result.iterations == reference.nfev


@pytest.mark.parametrize("max_updates", [1, 3])
def test_adaptive_final_fit_matches_returned_weights(max_updates, monkeypatch):
    inputs = simulate_observations(ScenarioConfig(name="adaptive", seed=51), 0).solver_inputs()
    calls = []
    updates = []
    original_solve, original_combine = solvers._solve_weighted, solvers.combine_reliability

    def record_solve(*args, **kwargs):
        calls.append(np.asarray(args[2]).copy())
        return original_solve(*args, **kwargs)

    def record_update(*args, **kwargs):
        value = original_combine(*args, **kwargs)
        updates.append(value[0].copy())
        return value

    monkeypatch.setattr(solvers, "_solve_weighted", record_solve)
    monkeypatch.setattr(solvers, "combine_reliability", record_update)
    method = MethodConfig(name="adaptive", max_irls_iterations=max_updates, use_temporal_weight=True)
    result = localize(inputs, method)
    assert len(updates) == result.iterations <= max_updates
    assert len(calls) == result.iterations + 1
    np.testing.assert_array_equal(calls[-1], result.weights)
    if max_updates == 1:
        assert not np.allclose(calls[0], calls[-1])
    reference = original_solve(
        inputs.reported_positions, inputs.estimated_ranges_m, result.weights,
        "huber", method.loss_scale_m, 100, result.estimate,
    )
    np.testing.assert_allclose(result.estimate, reference.x, atol=0.03)


def test_adaptive_early_stop_still_does_final_solve(monkeypatch):
    obs = simulate_observations(ScenarioConfig(
        name="exact", seed=9, noise_std_m=0, position_noise_std_m=0,
        mobile_rate=0, outlier_rate=0,
    ), 0).solver_inputs()
    obs = replace(obs, metadata_quality=np.full(len(obs.estimated_ranges_m), 0.5))
    calls = []
    original = solvers._solve_weighted

    def record(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(solvers, "_solve_weighted", record)
    result = localize(obs, MethodConfig(name="adaptive"))
    assert result.iterations == 1
    assert len(calls) == 2


def test_default_metadata_is_independent_of_pollution_labels_and_drift():
    config = ScenarioConfig(name="proxy", seed=27)
    clean = simulate_observations(replace(config, mobile_rate=0, outlier_rate=0), 0)
    polluted = simulate_observations(replace(
        config, mobile_rate=1, outlier_rate=1, mobile_drift_m=200, outlier_bias_m=100,
    ), 0)
    assert config.metadata_mode == "observable"
    assert config.position_noise_std_m == 1.2
    assert not clean.mobile_mask.any() and polluted.mobile_mask.all()
    assert not clean.outlier_mask.any() and polluted.outlier_mask.all()
    for name in ("metadata_quality", "position_accuracy_m", "observation_age_s"):
        np.testing.assert_array_equal(getattr(clean, name), getattr(polluted, name))
    oracle = simulate_observations(replace(polluted.scenario, metadata_mode="oracle"), 0)
    assert (oracle.metadata_quality < polluted.metadata_quality).all()
    np.testing.assert_array_equal(oracle.estimated_ranges_m, polluted.estimated_ranges_m)
    np.testing.assert_array_equal(oracle.position_accuracy_m, polluted.position_accuracy_m)


@pytest.mark.parametrize("bias", [-8.0, 5.0])
def test_calibration_bias_is_applied_once_in_dbm_and_temporal_ranges_share_units(bias):
    config = ScenarioConfig(name="bias", seed=32)
    unbiased = simulate_observations(config, 0)
    biased = simulate_observations(replace(config, calibration_bias=bias), 0)
    factor = 10 ** (-bias / (10 * PATH_LOSS_EXPONENT))
    np.testing.assert_allclose(biased.rssi_dbm, unbiased.rssi_dbm + bias, atol=1e-12)
    np.testing.assert_allclose(biased.estimated_ranges_m, unbiased.estimated_ranges_m * factor)
    np.testing.assert_allclose(biased.temporal_ranges_m, unbiased.temporal_ranges_m * factor)
    np.testing.assert_array_equal(biased.reported_positions, unbiased.reported_positions)
    noiseless = simulate_observations(replace(
        config, noise_std_m=0, position_noise_std_m=0, mobile_rate=0, outlier_rate=0,
        calibration_bias=bias,
    ), 0)
    physical_ranges = np.linalg.norm(noiseless.reported_positions - noiseless.true_position, axis=1)
    np.testing.assert_allclose(noiseless.rssi_dbm, range_to_rssi(physical_ranges) + bias)
    np.testing.assert_allclose(noiseless.temporal_ranges_m, np.tile(noiseless.estimated_ranges_m, (3, 1)))


def test_instability_score_direction_and_disabled_temporal_zero():
    ranges = np.array([[10, 10, 10], [10, 12, 20], [10, 14, 30]], dtype=float)
    weights, instability = temporal_stability_weight(ranges, 5)
    np.testing.assert_allclose(instability, ranges.std(axis=0) / 5)
    assert instability[0] < instability[1] < instability[2]
    assert weights[0] > weights[1] > weights[2]
    plain, off_scores = combine_reliability(np.ones(3), np.zeros(3), ranges, 5, False)
    temporal, on_scores = combine_reliability(np.ones(3), np.zeros(3), ranges, 5, True)
    np.testing.assert_array_equal(plain, np.ones(3))
    np.testing.assert_array_equal(off_scores, np.zeros(3))
    np.testing.assert_array_equal(on_scores, instability)
    np.testing.assert_array_equal(temporal, weights)
    weights_one, score_one = temporal_stability_weight(ranges[:1], 5)
    np.testing.assert_array_equal(weights_one, np.ones(3))
    np.testing.assert_array_equal(score_one, np.zeros(3))
    residual_weights = residual_reliability_weight(np.array([0, 5, 50]), 5)
    assert residual_weights[0] > residual_weights[1] > residual_weights[2]


@pytest.mark.parametrize("change", [
    {"name": "unknown"}, {"max_irls_iterations": 0}, {"max_irls_iterations": 1.5},
    {"loss_scale_m": 0}, {"loss_scale_m": np.nan}, {"reliability_threshold": -1},
    {"reliability_threshold": np.inf},
])
def test_invalid_methods_raise_clear_errors(change):
    inputs = simulate_observations(ScenarioConfig(name="invalid", seed=2), 0).solver_inputs()
    with pytest.raises(ValueError):
        localize(inputs, replace(MethodConfig(name="nls"), **change))


@pytest.mark.parametrize("field,value", [
    ("reported_positions", np.ones((2, 2))), ("reported_positions", np.ones((15, 3))),
    ("estimated_ranges_m", np.ones(14)), ("estimated_ranges_m", np.full(15, np.nan)),
    ("estimated_ranges_m", np.zeros(15)), ("metadata_quality", np.full(15, -0.1)),
    ("temporal_ranges_m", np.ones((0, 15))), ("temporal_ranges_m", np.ones((3, 14))),
    ("temporal_ranges_m", np.full((3, 15), np.inf)), ("rssi_dbm", np.ones(14)),
])
def test_invalid_solver_observations_rejected(field, value):
    inputs = simulate_observations(ScenarioConfig(name="invalid", seed=2), 0).solver_inputs()
    with pytest.raises(ValueError):
        localize(replace(inputs, **{field: value}), MethodConfig(name="nls"))


def test_solver_rejects_wrong_container():
    with pytest.raises(TypeError):
        localize(object(), MethodConfig(name="nls"))


@pytest.mark.parametrize("change", [
    {"metadata_mode": "secret"}, {"layout": "unknown"}, {"beacon_count": 2},
    {"temporal_windows": 0}, {"position_noise_std_m": -1}, {"noise_std_m": np.nan},
    {"mobile_rate": 1.1}, {"outlier_rate": -0.1}, {"calibration_bias": np.inf},
])
def test_simulator_rejects_invalid_config(change):
    with pytest.raises(ValueError):
        simulate_observations(replace(ScenarioConfig(name="bad", seed=0), **change), 0)


def test_simulator_all_observables_are_seed_reproducible():
    config = ScenarioConfig(name="seed", seed=12)
    first, second = simulate_observations(config, 1), simulate_observations(config, 1)
    for field in fields(SolverObservations):
        np.testing.assert_array_equal(getattr(first, field.name), getattr(second, field.name))
    assert not np.array_equal(first.metadata_quality, simulate_observations(config, 2).metadata_quality)
    assert not np.array_equal(first.reported_positions, simulate_observations(replace(config, seed=13), 1).reported_positions)


def test_slices_keep_full_condition_identity_and_clean_definition():
    config = ScenarioConfig(name="clean", seed=1, mobile_rate=0, outlier_rate=0, noise_std_m=9)
    configs = [
        config, replace(config, name="other"), replace(config, seed=2),
        replace(config, metadata_mode="oracle"), replace(config, calibration_bias=0.5),
        replace(config, mobile_rate=0.1), replace(config, outlier_rate=0.1),
        replace(config, noise_std_m=4), replace(config, position_noise_std_m=0),
        replace(config, outlier_rate=0.1001), replace(config, layout="clustered"),
    ]
    rows = [(item.to_dict(), _trial()) for item in configs]
    slices = slice_trials(rows + [rows[0]])
    assert len(slices) == len(configs)
    assert sum(item["n_trials"] for item in slices) == len(configs) + 1
    for item in slices:
        assert item["scenario_name"] == item["name"]
        assert "seed" in item and "metadata_mode" in item and "layout" in item
        assert item["slice"].startswith("beacons=")
        assert item["clean"] == (
            item["outlier_rate"] == 0 and item["mobile_rate"] == 0 and item["calibration_bias"] == 0
        )
    assert next(item for item in slices if item["n_trials"] == 2)["clean"] is True


@pytest.mark.parametrize("trials", [
    [], [_trial(np.nan)], [_trial(np.inf)], [_trial(-1)],
    [replace(_trial(), runtime_ms=np.nan)],
    [replace(_trial(), mobile_scores=np.array([0, np.inf]))],
    [replace(_trial(), mobile_scores=np.array([0]))],
    [replace(_trial(), mobile_true=np.array([0, np.nan]))],
    [replace(_trial(), converged=np.nan)],
])
def test_summarize_rejects_empty_or_nonfinite_trials(trials):
    with pytest.raises(ValueError):
        summarize_trials(trials)


def test_summarize_valid_values_and_legacy_fields():
    summary = summarize_trials([_trial(0), _trial(40)])
    assert summary["n_trials"] == 2
    assert summary["cep90_m"] == 36
    assert summary["failure_rate_over_30m"] == 0.5
    assert summary["mobile_detection"]["f1"] == 1
    with pytest.raises(ValueError):
        percentile([], 50)
    with pytest.raises(ValueError):
        percentile([np.nan], 50)


def test_paired_bootstrap_same_errors_have_zero_differences():
    errors = np.array([0, 4, 20, 30, 50], dtype=float)
    result = paired_bootstrap(errors, errors, seed=8, repeats=80)
    assert result["comparison"] == "candidate_minus_baseline"
    assert result["n_pairs"] == len(errors)
    assert "fixed scenario mixture" in result["bootstrap_scope"]
    for name in ("cep90_m", "failure_rate_over_30m", "median_error_m"):
        assert result[name] == {"point_estimate": 0.0, "ci95": [0.0, 0.0]}


def test_paired_bootstrap_preserves_pairs_and_known_signs():
    baseline = np.array([15, 20, 31, 35, 100], dtype=float)
    candidate = baseline - 10
    result = paired_bootstrap(baseline, candidate, seed=91, repeats=120)
    for name in ("cep90_m", "median_error_m"):
        assert result[name]["point_estimate"] == pytest.approx(-10)
        np.testing.assert_allclose(result[name]["ci95"], [-10, -10], atol=1e-12)
    assert result["failure_rate_over_30m"]["point_estimate"] == pytest.approx(-0.4)


def test_paired_bootstrap_seed_and_ci_reproducible():
    data_rng = np.random.default_rng(15)
    baseline = data_rng.uniform(0, 80, size=64)
    candidate = baseline * data_rng.uniform(0.3, 1.5, size=64)
    first = paired_bootstrap(baseline, candidate, seed=42, repeats=100)
    assert first == paired_bootstrap(baseline, candidate, seed=42, repeats=100)
    second = paired_bootstrap(baseline, candidate, seed=43, repeats=100)
    assert any(first[name]["ci95"] != second[name]["ci95"] for name in (
        "cep90_m", "median_error_m", "failure_rate_over_30m",
    ))
    assert first["cep90_m"]["point_estimate"] == pytest.approx(
        np.percentile(candidate, 90) - np.percentile(baseline, 90)
    )


@pytest.mark.parametrize("ranges,scale", [
    (np.array([]), 5), (np.ones((0, 3)), 5), (np.ones((3, 0)), 5),
    (np.full((2, 3), np.nan), 5), (np.ones((2, 3)), 0),
])
def test_temporal_reliability_invalid_inputs(ranges, scale):
    with pytest.raises(ValueError):
        temporal_stability_weight(ranges, scale)


@pytest.mark.parametrize("metadata,residuals,temporal", [
    (np.ones(2), np.zeros(3), np.ones((2, 3))),
    (np.full(3, np.nan), np.zeros(3), np.ones((2, 3))),
    (np.full(3, -1), np.zeros(3), np.ones((2, 3))),
    (np.ones(3), np.full(3, np.inf), np.ones((2, 3))),
    (np.ones(3), np.zeros(3), np.ones((2, 2))),
])
def test_combined_reliability_invalid_inputs(metadata, residuals, temporal):
    with pytest.raises(ValueError):
        combine_reliability(metadata, residuals, temporal, 5, True)


@pytest.mark.parametrize("baseline,candidate,kwargs", [
    ([], [], {}), ([1], [1, 2], {}), ([np.nan], [1], {}), ([1], [np.inf], {}),
    ([-1], [1], {}), ([1], [1], {"repeats": 0}), ([1], [1], {"repeats": 1.5}),
    ([1], [1], {"seed": -1}), ([1], [1], {"seed": True}),
])
def test_paired_bootstrap_invalid_input(baseline, candidate, kwargs):
    with pytest.raises(ValueError):
        paired_bootstrap(baseline, candidate, **kwargs)
