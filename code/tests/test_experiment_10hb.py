import json
from pathlib import Path

from experiment_10hb_observer_mechanism import condition_config, paired_rng

CFG = json.loads((Path(__file__).parents[1] / "config/experiment_10ha.json").read_text())


def test_noise_only_condition_removes_all_biases_without_mutating_source():
    local = condition_config(CFG, "noise_only")
    assert all(value == 0 for value in local["measurement_bias_std"].values())
    assert any(value > 0 for value in CFG["measurement_bias_std"].values())


def test_paired_rng_repeats_common_random_numbers():
    first = paired_rng(511, 2, 1).normal(size=10)
    second = paired_rng(511, 2, 1).normal(size=10)
    assert (first == second).all()
