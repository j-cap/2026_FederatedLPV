"""Client pooling, mixture derivatives, causal membership and forecast checks."""

import json
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose
from test_experiment_10hj import Q, R, Z, data_fixture

from federated_lpv.client_mixture import (
    ClientLikelihood,
    ClientMixture,
    calibrate_memberships,
    mixture_forecasts,
    project_split,
)
from federated_lpv.innovation_likelihood import InnovationLikelihood, MeasuredDataset
from federated_lpv.output_validation import causal_forecasts
from federated_lpv.physical_coupled_fit import CoupledLikelihood, PhysicalCoordinates


def fixture():
    data = data_fixture(noisy=True)
    coordinates = PhysicalCoordinates(Z, Z + np.log(0.1), Z + np.log(10))
    return data, coordinates


def test_client_totals_and_gradients_match_existing_likelihood():
    data, coordinates = fixture()
    eta = np.linspace(-0.05, 0.07, 8)
    client = ClientLikelihood(data, [7, 7], 0.01, Q, R, coordinates)
    costs, derivatives = client.evaluate(eta, gradient=True)
    previous = CoupledLikelihood(InnovationLikelihood(data, 0.01, Q, R), coordinates)
    value, gradient = previous.value_gradient(eta)
    assert_allclose(2 * costs.sum() / client.sample_count, value, rtol=1e-12)
    assert_allclose(2 * derivatives.sum(axis=0) / client.sample_count, gradient, rtol=1e-10)
    separate = ClientLikelihood(data, [7, 8], 0.01, Q, R, coordinates)
    assert_allclose(separate.evaluate(eta).sum(), costs[0], rtol=1e-12)
    assert costs.shape == (1,)


def test_mixture_total_derivative_and_permutation_invariance():
    data, coordinates = fixture()
    evaluator = ClientMixture(ClientLikelihood(data, [1, 2], 0.01, Q, R, coordinates))
    values = np.r_[np.linspace(-0.07, 0.05, 8), np.linspace(0.03, -0.09, 8), 0.43]
    value, analytic = evaluator.value_gradient(values)
    step = 1e-5
    numerical = np.array(
        [
            (
                evaluator.value(values + np.eye(17)[j] * step)
                - evaluator.value(values - np.eye(17)[j] * step)
            )
            / (2 * step)
            for j in range(17)
        ]
    )
    assert_allclose(analytic, numerical, rtol=2e-5, atol=2e-7)
    swapped = np.r_[values[8:16], values[:8], 1 - values[16]]
    assert_allclose(evaluator.value(swapped), value, rtol=1e-12)
    eta = np.linspace(-0.02, 0.03, 8)
    collapsed = np.r_[eta, eta, 0.43]
    expected = 2 * evaluator.evaluator.evaluate(eta).sum() / evaluator.evaluator.sample_count
    assert_allclose(evaluator.value(collapsed), expected, rtol=1e-12)


def test_calibration_uses_first_record_prefix_only():
    data, coordinates = fixture()
    parameters = [Z, coordinates.effective(np.linspace(-0.04, 0.05, 8))]
    prefix = MeasuredDataset(data.speeds, data.commands[:, :20], data.measurements[:, :20])
    _, before = calibrate_memberships(prefix, [8, 8], parameters, [0.4, 0.6], 0.01, Q, R)
    changed = data.measurements.copy()
    changed[0, 20:] += 100
    changed[1] += 200
    alternative = MeasuredDataset(data.speeds, data.commands[:, :20], changed[:, :20])
    _, after = calibrate_memberships(alternative, [8, 8], parameters, [0.4, 0.6], 0.01, Q, R)
    assert_allclose(before, after, rtol=0, atol=0)


def test_single_component_forecasts_reproduce_independent_implementation():
    data, _ = fixture()
    rows, _, _ = mixture_forecasts(data, [1, 2], [Z], np.ones((2, 1)), 0.01, Q, R, [1, 5, 20], 10)
    independent = causal_forecasts(data, Z, 0.01, Q, R, [1, 5, 20], 10)
    for expected in independent[:-1]:
        row = next(
            r
            for r in rows
            if r["client_position"] == -1 and r["horizon_samples"] == expected["horizon_samples"]
        )
        assert_allclose(row["sensor_normalized_mse"], expected["sensor_normalized_mse"], rtol=1e-12)


def test_forecast_excludes_intervening_measurements_with_fixed_membership():
    data, coordinates = fixture()
    parameters = [Z, coordinates.effective(np.linspace(-0.04, 0.05, 8))]
    weights = np.array([[0.7, 0.3], [0.2, 0.8]])
    rows, _, _ = mixture_forecasts(data, [1, 2], parameters, weights, 0.01, Q, R, [1, 5], 55)
    changed = data.measurements.copy()
    changed[:, 55:59] += 100
    after, _, _ = mixture_forecasts(
        MeasuredDataset(data.speeds, data.commands, changed),
        [1, 2],
        parameters,
        weights,
        0.01,
        Q,
        R,
        [1, 5],
        55,
    )
    assert_allclose(
        [r["sensor_normalized_mse"] for r in rows if r["horizon_samples"] == 5],
        [r["sensor_normalized_mse"] for r in after if r["horizon_samples"] == 5],
        rtol=0,
        atol=0,
    )


def test_split_feasibility_coupling_and_sealed_configuration():
    _, coordinates = fixture()
    values = project_split(np.zeros(8), np.arange(1, 9), 10, coordinates)
    for eta in values[:16].reshape(2, 8):
        z = coordinates.effective(eta)
        assert np.all(z >= coordinates.lower - 1e-10)
        assert np.all(z <= coordinates.upper + 1e-10)
        assert_allclose(z[0] - z[1] + z[2] - z[3] - z[4] + z[5], 0, atol=1e-12)
    cfg = json.loads((Path(__file__).parents[1] / "config/experiment_10hk.json").read_text())
    assert set(cfg["development_seeds"]).isdisjoint(cfg["reserved_confirmation_seeds"])
    assert set(cfg["reserved_confirmation_seeds"]) == set(range(591, 601)) | set(range(611, 621))
    assert cfg["component_counts"] == [1, 2]
    assert cfg["membership_prefix_samples"] == cfg["forecast_burn_in_samples"] == 50
    assert cfg["process_noise_scales"] == [0.01, 1.0]
