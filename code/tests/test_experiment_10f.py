import json
from pathlib import Path

import numpy as np
from experiment_10f_control_relevance import scenarios, simulate_tracking, smooth_three_point

CFG = json.loads((Path(__file__).parents[1] / "config/experiment_10f.json").read_text())


def test_smooth_profile_hits_endpoints_and_midpoint():
    time = np.linspace(0, 12, 1201)
    profile = smooth_three_point(time, (11, 29, 15))
    np.testing.assert_allclose(profile[[0, 600, 1200]], [11, 29, 15])


def test_scenarios_respect_speed_envelope():
    defined = scenarios(CFG)
    for scenario in defined.values():
        assert scenario["speed"].min() >= 10
        assert scenario["speed"].max() <= 30
        assert len(scenario["time"]) == len(scenario["speed"]) == len(scenario["curvature"])


def test_missing_schedule_is_infeasible():
    class Client:
        parameters = None

    assert simulate_tracking(Client(), None, scenarios(CFG)["moderate"], CFG) is None
