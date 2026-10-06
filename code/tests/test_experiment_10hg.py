import json
from pathlib import Path

import numpy as np
from experiment_10hg_global_prediction_error import prediction_objective


def test_prediction_objective_empty_segments_is_finite():
    cfg10h = json.loads((Path(__file__).parents[1] / "config/experiment_10h.json").read_text())
    cfg10ha = json.loads((Path(__file__).parents[1] / "config/experiment_10ha.json").read_text())
    value = prediction_objective(np.zeros(9), [], cfg10h, cfg10ha)
    assert value == 0.0


def test_10hg_split_is_label_blind_in_configuration():
    cfg = json.loads((Path(__file__).parents[1] / "config/experiment_10hg.json").read_text())
    heldout = [i for i in range(30) if i % cfg["heldout_client_modulus"] == cfg["heldout_client_remainder"]]
    assert len(heldout) == 10
