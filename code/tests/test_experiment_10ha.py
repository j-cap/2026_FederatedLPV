import json
from pathlib import Path

import numpy as np
from experiment_10h_higher_order_lpv_gate import architecture_matrices, sample_fleet
from experiment_10ha_tire_force_observer import C, design_observer_schedule, observability_audit

CFG = json.loads((Path(__file__).parents[1] / "config/experiment_10ha.json").read_text())
CFG10H = json.loads((Path(__file__).parents[1] / "config/experiment_10h.json").read_text())


def test_measurement_matrix_observes_yaw_force_sum_and_steering():
    np.testing.assert_allclose(C @ np.array([1, 2, 3, 4, 5]), [2, 7, 5])


def test_exact_higher_order_models_are_observable():
    clients = sample_fleet(1, CFG10H)
    exact, _, _ = architecture_matrices(clients, CFG10H["speed_grid"], CFG10H["sample_time"])
    rank, condition = observability_audit(exact[clients[0].client_id])
    assert rank == 5
    assert np.isfinite(condition)


def test_observer_schedule_has_expected_shape():
    clients = sample_fleet(2, CFG10H)
    _, global_model, _ = architecture_matrices(clients, CFG10H["speed_grid"], CFG10H["sample_time"])
    gains = design_observer_schedule(global_model, 0.1, CFG)
    assert gains.shape == (len(CFG10H["speed_grid"]), 5, 3)
