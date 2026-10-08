"""Scientific identities and intervention boundaries for the 10H-Q preparation."""

import json
from pathlib import Path

import numpy as np
import pytest
from experiment_10hq_preflight import parent_preflight

from federated_lpv.feedback_diagnostics import (
    command_error_components,
    command_error_summary,
    corrected_controller_state,
    diagnostic_cases,
)

CFG = json.loads((Path(__file__).parents[1] / "config/experiment_10hq.json").read_text())


def test_locked_cases_cover_factorial_and_only_three_partial_interventions():
    cases = diagnostic_cases(CFG)
    assert len(cases) == 12
    assert len({case.name for case in cases}) == 12
    main_cases = [case for case in cases if case.correction in {"none", "all"}]
    assert {(case.controller, case.state_source) for case in main_cases} == {
        (controller, state_source)
        for controller in ("nominal", "learned", "oracle")
        for state_source in ("nominal_kf", "learned_kf", "true_state")
    }
    partial = [case for case in cases if case.correction not in {"none", "all"}]
    assert {(case.controller, case.state_source) for case in partial} == {("learned", "learned_kf")}
    assert {case.correction for case in partial} == {"beta", "forces", "beta_forces"}
    assert len(cases) * 5 * 10 * 2 * 2 * 2 == CFG["planned_closed_loop_rollouts"]


def test_force_correction_does_not_overwrite_observer_or_other_states():
    estimate = np.array([[0.1, 0.2, 0.3, 0.4, 0.5], [1, 2, 3, 4, 5]])
    truth = -estimate
    before = estimate.copy()
    supplied = corrected_controller_state(estimate, truth, "forces")
    np.testing.assert_array_equal(supplied[:, [2, 3]], truth[:, [2, 3]])
    np.testing.assert_array_equal(supplied[:, [0, 1, 4]], before[:, [0, 1, 4]])
    supplied[:] = 100
    np.testing.assert_array_equal(estimate, before)
    np.testing.assert_array_equal(truth, -before)


def test_scheduled_command_error_matches_full_lqi_command_difference():
    rng = np.random.default_rng(709)
    truth = rng.normal(size=(40, 5))
    estimate = truth + rng.normal(scale=0.2, size=truth.shape)
    gains = rng.normal(size=(40, 6))
    integral = rng.normal(size=40)
    reference_term = rng.normal(size=40)
    estimated_command = -(gains[:, :5] * estimate).sum(1) - gains[:, 5] * integral + reference_term
    true_command = -(gains[:, :5] * truth).sum(1) - gains[:, 5] * integral + reference_term
    contributions = command_error_components(estimate, truth, gains[:, :5])
    np.testing.assert_allclose(contributions.sum(1), estimated_command - true_command, atol=2e-15)


def test_large_axle_errors_can_cancel_completely_in_command():
    truth = np.zeros((3, 5))
    estimate = np.array([[0, 0, 10, -10, 0]] * 3, dtype=float)
    contribution = command_error_components(estimate, truth, [0, 0, 2, 2, 0])
    summary = command_error_summary(contribution)
    assert summary["total_rms_rad"] == 0
    assert summary["joint_force_rms_rad"] == 0
    assert summary["component_rms_rad"]["front_force_per_mass"] == 20
    moments = np.array(summary["second_moment_rad2"])
    assert moments[2, 3] == -400
    assert moments.sum() == 0


def test_second_moments_recover_command_rms_with_bias_and_warmup():
    contributions = np.array([[100, 100, 100, 100, 100], [1, 2, 3, -2, 0], [2, 1, -3, 1, 0]])
    summary = command_error_summary(contributions, start=1)
    moments = np.array(summary["second_moment_rad2"])
    assert summary["samples"] == 2
    assert summary["mean_command_error_rad"] == 2.5
    np.testing.assert_allclose(summary["total_rms_rad"] ** 2, moments.sum())
    np.testing.assert_allclose(summary["total_rms_rad"], np.sqrt((4**2 + 1**2) / 2))


def test_partial_correction_removes_only_selected_command_contributions():
    estimate = np.arange(1, 6, dtype=float)
    truth = np.zeros(5)
    gains = np.array([2, -3, 4, -5, 6], dtype=float)
    original = command_error_components(estimate, truth, gains)
    corrected = corrected_controller_state(estimate, truth, "beta_forces")
    remaining = command_error_components(corrected, truth, gains)
    np.testing.assert_array_equal(remaining[[0, 2, 3]], 0)
    np.testing.assert_array_equal(remaining[[1, 4]], original[[1, 4]])


def test_command_identity_is_pre_clipping_not_a_clipped_difference():
    estimate = np.ones(5)
    truth = np.zeros(5)
    contribution = command_error_components(estimate, truth, np.ones(5))
    assert contribution.sum() == -5
    assert np.clip(-5, -0.25, 0.25) - np.clip(0, -0.25, 0.25) == -0.25


@pytest.mark.parametrize("gain", [np.ones(6), np.ones((2, 5)), [1, 1, np.nan, 1, 1]])
def test_gain_shape_or_nonfinite_errors_are_not_silently_broadcast(gain):
    with pytest.raises(ValueError):
        command_error_components(np.ones(5), np.zeros(5), gain)


@pytest.mark.parametrize("start", [-1, 2, 0.5])
def test_invalid_scoring_window_is_rejected(start):
    with pytest.raises(ValueError):
        command_error_summary(np.ones((2, 5)), start)


def test_nonfinite_state_is_a_failure_not_a_finite_score():
    with pytest.raises(ValueError, match="failures"):
        command_error_components([0, 0, np.inf, 0, 0], np.zeros(5), np.ones(5))


def test_missing_parent_preflight_does_not_fabricate_execution(tmp_path):
    status = parent_preflight(tmp_path, CFG)
    assert status["status"] == "blocked_missing_parent"
    assert status["missing_parent_paths"] == CFG["required_parent_paths"]
    assert not status["parent_revision_resolves"]
    assert status["fleet_rollouts_executed"] == 0
    assert not status["frozen_models_refitted"]
    assert not status["confirmation_run"]
