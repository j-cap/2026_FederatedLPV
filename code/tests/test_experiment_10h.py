import json
from pathlib import Path

import numpy as np
from experiment_10h_higher_order_lpv_gate import (
    continuous_matrices,
    discrete_matrices,
    raw_basis,
    sample_fleet,
)

CFG = json.loads((Path(__file__).parents[1] / "config/experiment_10h.json").read_text())


def test_relaxation_fleet_has_positive_time_and_length_scales():
    clients = sample_fleet(1, CFG)
    assert len(clients) == 30
    assert all(client.actuator_time_constant > 0 for client in clients)
    assert all(client.front_relaxation_length > 0 for client in clients)
    assert all(client.rear_relaxation_length > 0 for client in clients)


def test_higher_order_model_shapes_and_actuator_structure():
    client = sample_fleet(2, CFG)[0]
    a, b = continuous_matrices(client, 20.0)
    assert a.shape == (5, 5)
    assert b.shape == (5, 1)
    np.testing.assert_allclose(a[4, 4], -1 / client.actuator_time_constant)
    np.testing.assert_allclose(b.ravel()[:4], 0)
    assert b[4, 0] > 0


def test_discrete_model_is_stable_and_basis_has_five_columns():
    client = sample_fleet(3, CFG)[0]
    a, b = discrete_matrices(client, 20.0, CFG["sample_time"])
    assert max(abs(np.linalg.eigvals(a))) < 1
    assert b.shape == (5, 1)
    assert raw_basis([10, 20, 30]).shape == (3, CFG["lpv_basis_order"])
