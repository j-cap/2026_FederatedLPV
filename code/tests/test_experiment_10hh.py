"""Independent numerical checks for the measured-output estimator repair."""

import json
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose

from federated_lpv.innovation_likelihood import (
    C, InnovationLikelihood, MeasuredDataset, projected_gradient, physical_coupling_diagnostic,
    steady_filter, structured_matrices,
)

NOMINAL = np.log([.764, .947, 103., 129., 96., 149., 2., 1.667, 8.333])
Q = np.diag(np.array([1e-7, 1e-6, 2e-4, 2e-4, 1e-6]) * .01)
R = np.diag([np.deg2rad(.03)**2, .03**2, np.deg2rad(.02)**2])


def fixture():
    rng = np.random.default_rng(712)
    speeds = np.array([10., 20., 30.])
    commands = rng.normal(scale=.008, size=(3, 90))
    y = rng.normal(size=(3, 90, 3)) * np.sqrt(np.diag(R))
    for i, speed in enumerate(speeds):
        a, b = structured_matrices(NOMINAL, speed, .01)
        x = np.zeros(5)
        for t in range(90):
            x = a @ x + b * commands[i, t]
            y[i, t] += C @ x
    return MeasuredDataset(speeds, commands, y)


def test_discretization_frechet_matches_central_differences():
    z = NOMINAL + np.linspace(-.12, .15, 9)
    a, b, da, db = structured_matrices(z, 20., .01, True)
    for j in range(9):
        offset = np.eye(9)[j] * 1e-5
        ap, bp = structured_matrices(z + offset, 20., .01)
        am, bm = structured_matrices(z - offset, 20., .01)
        assert_allclose(da[j], (ap-am)/2e-5, atol=1e-8, rtol=1e-5)
        assert_allclose(db[j], (bp-bm)/2e-5, atol=1e-8, rtol=1e-5)


def test_prior_covariance_is_fixed_point_and_has_correct_gain():
    for speed in [10., 20., 30.]:
        f = steady_filter(NOMINAL, speed, .01, Q, R)
        posterior = f['p'] - f['k'] @ C @ f['p']
        assert_allclose(f['a'] @ posterior @ f['a'].T + Q, f['p'], atol=1e-13)
        assert_allclose(f['k'] @ f['s'], f['p'] @ C.T, atol=1e-13)


def test_full_likelihood_agrees_with_independent_filter_recursion():
    data = fixture()
    z = NOMINAL + .09
    evaluator = InnovationLikelihood(data, .01, Q, R)
    value = 0.
    for i, speed in enumerate(data.speeds):
        f = steady_filter(z, speed, .01, Q, R)
        x = np.zeros(5)
        posterior = f['p'] - f['k'] @ C @ f['p']
        for command, measured in zip(data.commands[i], data.measurements[i]):
            x = f['a'] @ x + f['b'] * command
            p = f['a'] @ posterior @ f['a'].T + Q
            s = C @ p @ C.T + R
            gain = np.linalg.solve(s, C @ p).T
            e = measured - C @ x
            value += e @ np.linalg.solve(s, e) + np.linalg.slogdet(s)[1] - np.linalg.slogdet(R)[1]
            x = x + gain @ e
            posterior = p - gain @ C @ p
    assert_allclose(evaluator.value(z), value/data.measurements.size, rtol=1e-10)


def test_total_likelihood_gradient_includes_covariance_and_recursion():
    evaluator = InnovationLikelihood(fixture(), .01, Q, R)
    for z in [NOMINAL, NOMINAL + np.linspace(-.1, .12, 9)]:
        value, analytic = evaluator.value_gradient(z)
        numerical = np.array([
            (evaluator.value(z + np.eye(9)[j]*1e-5)
             - evaluator.value(z - np.eye(9)[j]*1e-5))/2e-5
            for j in range(9)
        ])
        assert np.isfinite(value)
        assert_allclose(analytic, numerical, atol=1e-6, rtol=2e-5)


def test_boundary_gradient_uses_correct_kkt_sign():
    x = np.array([0., 0., 1., 1., .5])
    g = np.array([2., -2., -3., 3., 4.])
    assert_allclose(projected_gradient(x, g, np.zeros(5), np.ones(5)), [0., -2., 0., 3., 4.])


def test_confirmation_seeds_are_not_part_of_development():
    cfg = json.loads((Path(__file__).parents[1]/'config/experiment_10hh.json').read_text())
    assert set(cfg['development_seeds']).isdisjoint(cfg['reserved_confirmation_seeds'])
    assert cfg['reserved_confirmation_seeds'] == list(range(611, 621))


def test_physical_coupling_check_uses_an_exact_nominal_identity():
    # Known nominal fixture, no client truths are supplied to the diagnostic.
    v = np.array([.6*1.2, .6*1.5, 100., 1.2*100., 90., 1.5*90., 2., 1.7, 8.])
    result = physical_coupling_diagnostic(np.log(v))
    assert_allclose(result['log_coupling_residual'],0.,atol=1e-14)
    v[0] *= 1.2
    assert physical_coupling_diagnostic(np.log(v))['relative_coupling_discrepancy'] > .1
