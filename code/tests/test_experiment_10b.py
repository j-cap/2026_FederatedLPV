import json
from pathlib import Path

import numpy as np

from experiment_10a_basis_coverage_audit import raw_basis
from experiment_10b_oracle_complementary_gate import predict, ridge_fit


CFG10A = json.loads((Path(__file__).parents[1] / "config/experiment_10a.json").read_text())


def test_ridge_fit_recovers_full_rank_basis_coefficients():
    speeds = np.asarray(CFG10A["speed_grid"], dtype=float)
    rng = np.random.default_rng(4)
    coefficients = rng.normal(size=(7, 6))
    targets = raw_basis(speeds, 7, CFG10A) @ coefficients
    fitted = ridge_fit(speeds, targets, 7, CFG10A)
    assert np.allclose(predict(fitted, speeds, CFG10A), targets, atol=1e-9)


def test_local_blocks_are_rank_deficient_for_frozen_order():
    blocks = [[10, 12, 14], [16, 18, 20], [22, 24, 26], [28, 30]]
    assert all(np.linalg.matrix_rank(raw_basis(np.asarray(block), 7, CFG10A)) < 7
               for block in blocks)
