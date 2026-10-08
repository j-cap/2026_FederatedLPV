"""Feedback timing, KF design, truth intervention and replay checks."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.linalg import solve_discrete_are
from test_experiment_10hj import Q, R, Z

from federated_lpv.innovation_likelihood import C, structured_matrices
from federated_lpv.output_feedback_control import (
    design_schedule,
    feedback_radius,
    prior_kalman_gain,
    replay_observer,
    rollout,
)

WEIGHTS = {
    "q_beta": 50.0,
    "q_yaw": 75.0,
    "q_force": 0.05,
    "q_actuator": 5.0,
    "q_integral": 1500.0,
    "r": 0.2,
}


def setup():
    matrices = [structured_matrices(Z, v, 0.01) for v in [10.0, 30.0]]
    schedule = design_schedule(matrices, [10.0, 30.0], Q, R, WEIGHTS, 0.01)
    scenario = {
        "time": np.arange(20) * 0.01,
        "speed": np.ones(20) * 10.0,
        "reference": np.ones(20) * 0.01,
    }
    noise = np.random.default_rng(19).multivariate_normal(np.zeros(3), R, size=20)
    return matrices, schedule, scenario, noise


def test_prior_gain_has_no_double_prediction():
    a, _ = structured_matrices(Z, 15.0, 0.01)
    p = solve_discrete_are(a.T, C.T, Q, R)
    expected = p @ C.T @ np.linalg.inv(C @ p @ C.T + R)
    assert_allclose(prior_kalman_gain(a, Q, R), expected, atol=1e-12)


def test_closed_loop_current_yaw_integrator_clipped_input_and_replay():
    matrices, schedule, scenario, noise = setup()
    record = rollout(matrices, [10.0, 30.0], schedule, schedule, scenario, noise, limit=1e-4)
    a, b = matrices[0]
    assert_allclose(record["truth"][1], b * record["command"][0])
    gain = schedule.observer[0]
    predicted = a @ record["estimate"][0] + b * record["command"][0]
    assert_allclose(
        record["estimate"][1], predicted + gain @ (record["measurement"][1] - C @ predicted)
    )
    increment = 0.01 * (record["measurement"][0, 0] - 0.01)
    raw = record["raw_command"][0]
    expected = (
        0 if abs(raw) > 1e-4 and raw * (-schedule.feedback[0, 5] * increment) > 0 else increment
    )
    assert_allclose(record["integral"][1], expected)
    replay = replay_observer(
        {k: record[k] for k in ["command", "measurement"]}, schedule, scenario["speed"]
    )
    assert_allclose(replay, record["estimate"], atol=1e-14)


def test_truth_corrected_input_does_not_overwrite_kf():
    matrices, schedule, scenario, noise = setup()
    record = rollout(
        matrices, [10.0, 30.0], schedule, schedule, scenario, noise, correction="forces"
    )
    assert_allclose(record["supplied"][:, 2:4], record["truth"][:, 2:4])
    assert np.max(abs(record["estimate"][:, 2:4] - record["truth"][:, 2:4])) > 1e-6
    assert_allclose(
        replay_observer(record, schedule, scenario["speed"]), record["estimate"], atol=1e-14
    )


@pytest.mark.parametrize(
    "correction,indices",
    [("none", []), ("beta", [0]), ("forces", [2, 3]), ("beta_forces", [0, 2, 3])],
)
def test_frozen_radius_agrees_with_independent_closed_loop_jacobian(correction, indices):
    matrices, schedule, _, _ = setup()
    a, b = matrices[0]
    l, gain = schedule.observer[0], schedule.feedback[0]

    def step(z):
        x, xhat, eta = z[:5], z[5:10], z[10]
        supplied = xhat.copy()
        supplied[indices] = x[indices]
        u = -gain @ np.r_[supplied, eta]
        xn = a @ x + b * u
        prediction = a @ xhat + b * u
        return np.r_[xn, prediction + l @ (C @ xn - C @ prediction), eta + 0.01 * x[1]]

    jacobian = np.column_stack([step(column) for column in np.eye(11)])
    radius = np.max(abs(np.linalg.eigvals(jacobian)))
    assert_allclose(
        feedback_radius(a, b, a, b, l, gain, 0.01, correction=correction), radius, atol=1e-12
    )
