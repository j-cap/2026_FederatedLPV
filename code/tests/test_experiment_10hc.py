import json
from pathlib import Path

import numpy as np
from experiment_10hc_learned_observer_groups import (
    fit_schedule,
    predict_schedule,
    unflatten_schedule,
)

CFG10H = json.loads((Path(__file__).parents[1] / "config/experiment_10h.json").read_text())


def test_basis_schedule_round_trip_on_training_grid():
    speeds = np.asarray(CFG10H["speed_grid"], dtype=float)
    rng = np.random.default_rng(10)
    coefficients = rng.normal(size=(5, 30))
    targets = predict_schedule(coefficients, speeds)
    fitted = fit_schedule(speeds, targets, 1e-12)
    np.testing.assert_allclose(predict_schedule(fitted, speeds), targets, atol=1e-8)


def test_unflatten_schedule_has_state_space_shapes():
    schedule = unflatten_schedule(np.zeros((3, 30)))
    assert len(schedule) == 3
    assert schedule[0][0].shape == (5, 5)
    assert schedule[0][1].shape == (5, 1)
