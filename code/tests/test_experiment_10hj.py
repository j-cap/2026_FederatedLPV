"""Forecast causality, independent scoring, record boundaries and sealed seeds."""

import itertools
import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from federated_lpv.innovation_likelihood import (
    C,
    InnovationLikelihood,
    MeasuredDataset,
    structured_matrices,
)
from federated_lpv.output_validation import causal_forecasts, filter_records, residual_correlations

Z = np.log([0.72, 0.9, 100.0, 120.0, 90.0, 135.0, 2.0, 1.7, 8.0])
Q = np.diag([1e-7, 1e-6, 2e-4, 2e-4, 1e-6])
R = np.diag([1e-5, 1e-3, 1e-6])


def data_fixture(noisy=False):
    rng = np.random.default_rng(7211)
    speeds = np.array([12.0, 23.0])
    commands = rng.normal(scale=0.015, size=(2, 60))
    measured = np.zeros((2, 60, 3))
    for record, speed in enumerate(speeds):
        a, b = structured_matrices(Z, speed, 0.01)
        state = np.zeros(5)
        for time, command in enumerate(commands[record]):
            state = a @ state + b * command
            measured[record, time] = C @ state
    if noisy:
        measured += rng.normal(size=measured.shape) * np.sqrt(np.diag(R))
    return MeasuredDataset(speeds, commands, measured)


def test_one_step_matches_independent_working_likelihood_sensor_metric():
    data = data_fixture(noisy=True)
    proposal = Z + np.linspace(-0.1, 0.15, 9)
    rows = causal_forecasts(data, proposal, 0.01, Q, R, [1], burn_in=0)
    independent = InnovationLikelihood(data, 0.01, Q, R).evaluate(proposal, diagnostics=True)
    assert_allclose(
        rows[0]["sensor_normalized_mse"], independent["sensor_normalized_mse"], rtol=1e-12
    )
    assert rows[0]["scored_output_samples"] == 120


def test_forecasts_ignore_intervening_outputs_and_use_correct_command_alignment():
    data = data_fixture(noisy=True)
    proposal = Z + np.linspace(-0.1, 0.15, 9)
    rows = causal_forecasts(data, proposal, 0.01, Q, R, [1, 5], burn_in=55)
    changed = data.measurements.copy()
    changed[:, 55:59] += np.array([2.0, 8.0, 0.5])
    alternative = MeasuredDataset(data.speeds, data.commands, changed)
    after = causal_forecasts(alternative, proposal, 0.01, Q, R, [1, 5], burn_in=55)
    assert_allclose(
        rows[1]["sensor_normalized_mse"], after[1]["sensor_normalized_mse"], rtol=0, atol=0
    )
    assert rows[0]["sensor_normalized_mse"] != after[0]["sensor_normalized_mse"]
    history, _, a, b = filter_records(data, proposal, 0.01, Q, R)
    error = []
    for record in range(2):
        state = history[record, 55].copy()
        for time in range(55, 60):
            state = a[record] @ state + b[record] * data.commands[record, time]
        error.append(data.measurements[record, 59] - C @ state)
    expected = sum(e @ np.linalg.solve(R, e) for e in error) / 6
    assert_allclose(rows[1]["sensor_normalized_mse"], expected, rtol=1e-12)
    assert all(row["origins_per_record"] == 1 for row in rows)


def test_planted_noiseless_model_has_zero_forecast_and_simulation_error():
    rows = causal_forecasts(data_fixture(), Z, 0.01, Q, R, [1, 5, 20], burn_in=10)
    assert all(row["sensor_normalized_mse"] < 1e-22 for row in rows)
    assert all(row["origins_per_record"] == 31 for row in rows[:-1])
    assert rows[-1]["mode"] == "zero_start_simulation"


def test_record_correlations_never_pair_different_records():
    rng = np.random.default_rng(99)
    errors = rng.normal(size=(2, 10, 3))
    commands = rng.normal(size=(2, 10))
    rows = residual_correlations(errors, commands, maximum_lag=2)
    centered = errors - errors.mean(axis=(0, 1), keepdims=True)
    following, previous = centered[:, 1:, 0], centered[:, :-1, 0]
    expected = np.sum(following * previous) / np.sqrt(np.sum(following**2) * np.sum(previous**2))
    row = next(
        r
        for r in rows
        if r["centering"] == "pooled"
        and r["kind"] == "autocorrelation"
        and r["lag"] == 1
        and r["output"] == "yaw"
    )
    assert_allclose(row["correlation"], expected)
    shifted = errors + np.array([[[10.0]], [[-20.0]]])
    shifted_rows = residual_correlations(shifted, commands, maximum_lag=2)
    assert_allclose(
        [r["correlation"] for r in rows if r["centering"] == "within_record"],
        [r["correlation"] for r in shifted_rows if r["centering"] == "within_record"],
        atol=1e-14,
    )


def test_invalid_forecast_windows_are_rejected():
    for horizons, burn in [([0], 0), ([1.5], 0), ([20], 50), ([1], -1)]:
        with pytest.raises(ValueError):
            causal_forecasts(data_fixture(), Z, 0.01, Q, R, horizons, burn)


def test_protocol_is_nested_fixed_and_confirmation_is_sealed():
    cfg = json.loads((Path(__file__).parents[1] / "config/experiment_10hj.json").read_text())
    assert set(cfg["development_seeds"]).isdisjoint(cfg["reserved_confirmation_seeds"])
    assert set(cfg["reserved_confirmation_seeds"]) == set(range(591, 601)) | set(range(611, 621))
    assert cfg["process_noise_scales"] == [0.01, 1.0]
    boxes = cfg["bound_boxes"]
    assert [(b["lower"], b["upper"]) for b in boxes] == [(0.35, 3.0), (0.2, 5.0), (0.1, 10.0)]
    assert cfg["forecast_horizons_samples"] == [1, 5, 20, 50]
    for first, second in itertools.pairwise(boxes):
        assert second["lower"] < first["lower"] and second["upper"] > first["upper"]
