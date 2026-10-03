import json
from pathlib import Path

import numpy as np
from experiment_10g_vehicle_class_gate import (
    continuous_augmented_matrices,
    discrete_augmented_matrices,
    sample_extended_fleet,
)

CFG = json.loads((Path(__file__).parents[1] / "config/experiment_10g.json").read_text())


def test_extended_fleet_has_three_classes_and_positive_parameters():
    clients = sample_extended_fleet(1, CFG)
    assert len(clients) == 3 * CFG["clients_per_class"]
    assert {client.vehicle_class for client in clients} == {"compact", "suv", "sport"}
    assert all(client.actuator_time_constant > 0 for client in clients)


def test_actuator_linearization_has_expected_structure():
    client = sample_extended_fleet(2, CFG)[0]
    a, b = continuous_augmented_matrices(client, 20.0)
    assert a.shape == (3, 3)
    assert b.shape == (3, 1)
    np.testing.assert_allclose(a[2], [0, 0, -1 / client.actuator_time_constant])
    np.testing.assert_allclose(b.ravel(), [0, 0, 1 / client.actuator_time_constant])


def test_discrete_actuator_is_stable_and_commanded():
    client = sample_extended_fleet(3, CFG)[0]
    a, b = discrete_augmented_matrices(client, 20.0, CFG["sample_time"])
    assert max(abs(np.linalg.eigvals(a))) < 1
    assert b[2, 0] > 0
