"""Calibration causality, deployed aggregation and training-only CV selection."""

import json
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose
from test_experiment_10hj import Q, R, Z, data_fixture

from federated_lpv.causal_model_selection import (
    candidate_forecasts,
    deploy_rows,
    first_record_prefix,
    forecast_choices,
    prefix_scores,
    select_inner_candidate,
)
from federated_lpv.innovation_likelihood import MeasuredDataset
from federated_lpv.output_validation import causal_forecasts


def test_prefix_scores_ignore_future_outputs_and_other_records():
    data = data_fixture(noisy=True)
    parameters = [Z, Z + np.linspace(-0.03, 0.05, 9), Z + np.linspace(0.04, -0.05, 9)]
    clients, before = prefix_scores(data, [8, 8], parameters, 0.01, Q, R, 30, [1, 5, 10], 10)
    changed = data.measurements.copy()
    changed[0, 30:] += 100
    changed[1] += 500
    alternative = MeasuredDataset(data.speeds, data.commands, changed)
    _, after = prefix_scores(alternative, [8, 8], parameters, 0.01, Q, R, 30, [1, 5, 10], 10)
    assert_allclose(before, after, rtol=0, atol=0)
    assert_allclose(forecast_choices(before), forecast_choices(after))
    assert clients.tolist() == [8]


def test_calibration_filter_uses_earlier_outputs_but_forecast_has_no_corrections():
    data = data_fixture(noisy=True)
    rows_before = candidate_forecasts(data, [1, 2], [Z], 0.01, Q, R, [1, 5], 55)[1]
    changed = data.measurements.copy()
    changed[:, 55:59] += 100
    rows_after = candidate_forecasts(
        MeasuredDataset(data.speeds, data.commands, changed), [1, 2], [Z], 0.01, Q, R, [1, 5], 55
    )[1]
    assert_allclose(
        [r["sensor_normalized_mse"] for r in rows_before if r["horizon_samples"] == 5],
        [r["sensor_normalized_mse"] for r in rows_after if r["horizon_samples"] == 5],
        rtol=0,
        atol=0,
    )


def test_frozen_deployment_reproduces_independent_forecast_errors():
    data = data_fixture(noisy=True)
    clients, rows, _, _ = candidate_forecasts(data, [1, 2], [Z], 0.01, Q, R, [1, 5, 20], 10)
    deployed = deploy_rows(rows, clients, np.zeros(2, dtype=int))
    independent = causal_forecasts(data, Z, 0.01, Q, R, [1, 5, 20], 10)
    for expected in independent[:-1]:
        result = next(
            row
            for row in deployed
            if row["client_position"] == -1
            and row["horizon_samples"] == expected["horizon_samples"]
        )
        for key in (
            "sensor_normalized_mse",
            "yaw_rmse_deg_s",
            "acceleration_rmse_mps2",
            "steering_rmse_deg",
        ):
            assert_allclose(result[key], expected[key], rtol=1e-12)


def test_fallback_requires_strict_prefix_improvement_and_honors_margin():
    scores = np.array([[10, 9, 12], [10, 10, 10], [10, 7, 4]])
    assert forecast_choices(scores, margin=0).tolist() == [1, 0, 2]
    assert forecast_choices(scores, margin=0.1).tolist() == [0, 0, 2]
    assert forecast_choices(scores, fallback=False).tolist() == [1, 1, 2]


def test_validation_selection_uses_all_folds_eligibility_and_declared_ties():
    records = [
        {"fold": fold, "restart": restart, "margin": margin, "selection_score": -0.4}
        for fold in (0, 1)
        for restart in (0, 1)
        for margin in (0, 0.25)
    ]
    selected, candidates = select_inner_candidate(records, {0: True, 1: True}, 2)
    assert selected["restart"] == 0 and selected["margin"] == 0.25
    assert len(candidates) == 4
    selected, _ = select_inner_candidate(records, {0: False, 1: True}, 2)
    assert selected["restart"] == 1
    selected, candidates = select_inner_candidate(records, {0: False, 1: False}, 2)
    assert selected is None and candidates == []
    incomplete = [row for row in records if row["fold"] == 0]
    assert select_inner_candidate(incomplete, {0: True, 1: True}, 2)[0] is None


def test_configuration_and_prefix_windows_preserve_confirmation_seal():
    root = Path(__file__).parents[1] / "config"
    own = json.loads((root / "experiment_10hl.json").read_text())
    cfg = json.loads((root / own["base_config"]).read_text()) | own
    assert set(cfg["development_seeds"]).isdisjoint(cfg["reserved_confirmation_seeds"])
    assert set(cfg["reserved_confirmation_seeds"]) == set(range(591, 601)) | set(range(611, 621))
    assert cfg["process_noise_scales"] == [0.01, 1]
    assert cfg["restart_count"] == 6 and cfg["inner_client_fold_count"] == 2
    assert cfg["membership_prefix_samples"] == cfg["forecast_burn_in_samples"] == 100
    assert cfg["calibration_burn_in_samples"] + max(cfg["calibration_horizons_samples"]) <= 100
    clients, data = first_record_prefix(data_fixture(), [2, 1], 30)
    assert clients.tolist() == [1, 2]
    assert data.commands.shape == (2, 30)
