"""Audit total multihorizon sensitivities against independent forecasts/FD."""

import numpy as np
from numpy.testing import assert_allclose
from test_experiment_10hj import Q, R, Z, data_fixture

from federated_lpv.multihorizon_fit import MultihorizonLoss
from federated_lpv.output_validation import causal_forecasts
from federated_lpv.physical_coupled_fit import PhysicalCoordinates


def test_loss_equals_independent_causal_forecasts_and_total_gradient():
    coords = PhysicalCoordinates(Z, Z - 2, Z + 2)
    eta = np.linspace(-0.03, 0.02, 8)
    data = data_fixture(noisy=True)
    loss = MultihorizonLoss(data, coords, 0.01, Q, R, [1, 5, 20], 10)
    value, gradient = loss.value_gradient(eta)
    rows = causal_forecasts(data, coords.effective(eta), 0.01, Q, R, [1, 5, 20], 10)
    assert_allclose(value, np.mean([r["sensor_normalized_mse"] for r in rows[:-1]]), rtol=1e-12)
    numerical = []
    for j in range(8):
        direction = np.eye(8)[j] * 1e-5
        numerical.append(
            (loss.value_gradient(eta + direction)[0] - loss.value_gradient(eta - direction)[0])
            / 2e-5
        )
    assert_allclose(gradient, numerical, rtol=2e-5, atol=1e-6)


def test_zero_weight_records_do_not_affect_fit():
    coords = PhysicalCoordinates(Z, Z - 2, Z + 2)
    data = data_fixture(noisy=True)
    before = MultihorizonLoss(data, coords, 0.01, Q, R, [1, 5], 10, [1, 0]).value_gradient(
        np.zeros(8)
    )
    data.measurements[1] += 100
    after = MultihorizonLoss(data, coords, 0.01, Q, R, [1, 5], 10, [1, 0]).value_gradient(
        np.zeros(8)
    )
    assert_allclose(before[0], after[0], rtol=0, atol=0)
    assert_allclose(before[1], after[1], rtol=0, atol=0)
