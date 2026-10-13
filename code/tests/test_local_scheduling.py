import json
from pathlib import Path
import sys

import numpy as np
import pytest

from experiment_10a_basis_coverage_audit import raw_basis
from federated_lpv.local_scheduling import curvature_factor, fit_smooth_local, select_one_se

ROOT = Path(__file__).resolve().parents[2]
CFG = json.loads((ROOT / "code/config/experiment_10a.json").read_text())


def test_two_observations_recover_affine_map_across_envelope():
    train = np.array([28., 30.])
    grid = np.linspace(10., 30., 51)
    values = lambda v: np.column_stack([.2+.01*v, .8-.002*v])
    for alpha in [1e-6, 1., 1e6]:
        theta = fit_smooth_local(raw_basis(train, 7, CFG), values(train), curvature_factor(), alpha)
        np.testing.assert_allclose(raw_basis(grid, 7, CFG)@theta, values(grid), atol=3e-7, rtol=0)


def test_loss_normalization_preserves_fit_under_repeated_rows():
    phi = raw_basis(np.array([10., 12., 14.]), 7, CFG)
    y = np.array([[1., .3], [.9, .32], [.86, .28]])
    factor = curvature_factor()
    a = fit_smooth_local(phi, y, factor, .01)
    b = fit_smooth_local(np.repeat(phi, 5, axis=0), np.repeat(y, 5, axis=0), factor, .01)
    np.testing.assert_allclose(a, b, atol=1e-10, rtol=1e-10)


def test_curvature_is_derivative_of_function_in_fixed_qr_basis():
    grid = np.linspace(10., 30., 401)
    step = 1e-4  # z increment; speed increment = 10*step
    derivative = (raw_basis(grid+10*step, 7, CFG)-2*raw_basis(grid, 7, CFG)
                  +raw_basis(grid-10*step, 7, CFG))/step**2
    weights = np.ones(401)/400
    weights[[0, -1]] *= .5
    np.testing.assert_allclose(curvature_factor(), np.sqrt(weights)[:, None]*derivative,
                               atol=2e-6, rtol=2e-5)
    assert np.linalg.matrix_rank(curvature_factor(), tol=1e-10) == 5


def test_one_se_prefers_largest_eligible_strength_and_matches_fleets():
    selected, detail = select_one_se({1e-6: [.9, 1.1], 1.: [.99, 1.03], 100.: [1.3, 1.3]})
    assert selected == 1.
    assert detail["best_strength"] == 1e-6
    with pytest.raises(ValueError):
        select_one_se({1.: [1., 2.], 100.: [1.]})


def test_prior_centered_fit_preserves_null_directions():
    phi = raw_basis(np.array([28., 30.]), 7, CFG)
    rng = np.random.default_rng(5)
    prior = rng.normal(size=(7, 6))
    y = rng.normal(size=(2, 6))
    alpha, eps = .01, 1e-12
    theta = np.linalg.solve(phi.T@phi+(alpha+eps)*np.eye(7), phi.T@y+alpha*prior)
    null = np.linalg.svd(phi, full_matrices=True)[2][2:]
    np.testing.assert_allclose(null@(theta-prior), -eps/(alpha+eps)*(null@prior), atol=1e-13)


def test_frozen_hashes_reject_changed_inputs(tmp_path, monkeypatch):
    sys.path.insert(0, str(ROOT / "paper"))
    import local_baseline_validation as runner
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    path = tmp_path / "input.json"
    path.write_text('{"target": 1}\n')
    lock = {"input.json": runner.digest(path)}
    path.write_text('{"target": 2}\n')
    with pytest.raises(ValueError, match="Frozen input changed"):
        runner.verify_hashes(lock)
