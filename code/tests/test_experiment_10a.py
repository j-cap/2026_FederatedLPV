import json
from pathlib import Path

import numpy as np

from experiment_10a_basis_coverage_audit import discrete_jacobian, raw_basis
from federated_lpv import family_centers


CFG = json.loads((Path(__file__).parents[1] / "config/experiment_10a.json").read_text())


def test_basis_is_nested_and_has_requested_order():
    speeds = np.array([10, 14, 18, 22, 26, 30], dtype=float)
    for order in CFG["candidate_orders"]:
        basis = raw_basis(speeds, order, CFG)
        assert basis.shape == (len(speeds), order)
        assert np.linalg.matrix_rank(basis) == min(len(speeds), order)


def test_straight_line_jacobian_is_finite_and_equilibrium_is_zero():
    a, b, operating = discrete_jacobian(family_centers()["nominal"], 20.0, 0.0, CFG["sample_time"])
    assert np.isfinite(a).all() and np.isfinite(b).all()
    assert np.allclose(operating, 0.0, atol=1e-10)


def test_nonlinear_cornering_operating_point_is_feasible():
    _, _, operating = discrete_jacobian(family_centers()["nominal"], 30.0, 0.005, CFG["sample_time"])
    assert abs(operating[0]) < np.deg2rad(10)
    assert abs(operating[2]) < np.deg2rad(15)
