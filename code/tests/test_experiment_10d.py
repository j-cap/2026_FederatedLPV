import json
from pathlib import Path

import numpy as np
from experiment_10d_learned_compatible_groups import balanced_categories, fit_mixture

CFG10A = json.loads((Path(__file__).parents[1] / "config/experiment_10a.json").read_text())


class Client:
    def __init__(self, client_id, family):
        self.client_id = client_id
        self.family = family


def test_balanced_categories_cover_every_block_within_family():
    clients = [Client(f"{family}_{j}", family) for family in ("a", "b", "c") for j in range(10)]
    categories = balanced_categories(clients, np.random.default_rng(2), [0, 1, 2, 3])
    for family in ("a", "b", "c"):
        assert {categories[client.client_id] for client in clients if client.family == family} == {
            0,
            1,
            2,
            3,
        }


def test_mixture_selects_separated_partial_regressions():
    rng = np.random.default_rng(3)
    speeds, targets = [], []
    blocks = [
        np.asarray([10, 12, 14], dtype=float),
        np.asarray([16, 18, 20], dtype=float),
        np.asarray([22, 24, 26], dtype=float),
        np.asarray([28, 30], dtype=float),
    ]
    for cluster in range(2):
        # Twenty clients per component give BIC enough evidence to pay for the
        # 42 additional coefficients of a second order-seven LPV backbone.
        for client in range(20):
            local_speed = blocks[client % len(blocks)]
            value = np.tile(np.asarray([cluster * 8, cluster * 4, 1, 2, 3, 4], dtype=float), (3, 1))
            value = np.tile(value[0], (len(local_speed), 1))
            speeds.append(local_speed)
            targets.append(value + rng.normal(scale=0.01, size=value.shape))
    result, _, _ = fit_mixture(speeds, targets, [1, 2], 5, 20, 4, 7, CFG10A)
    assert result["k"] == 2
