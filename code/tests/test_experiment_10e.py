import json
from pathlib import Path

import numpy as np
from experiment_10b_oracle_complementary_gate import ridge_fit
from experiment_10e_federated_latent_groups import (
    aligned_model_error,
    client_statistics,
    fit_federated_mixture,
    label_free_categories,
    solve_statistics,
)

CFG10A = json.loads((Path(__file__).parents[1] / "config/experiment_10a.json").read_text())


class Client:
    def __init__(self, client_id, family):
        self.client_id = client_id
        self.family = family


def test_label_free_categories_balance_without_family_pattern():
    clients = [
        Client(f"{family}_{index}", family) for family in ("a", "b", "c") for index in range(10)
    ]
    categories = label_free_categories(clients, np.random.default_rng(4), range(4))
    counts = np.bincount(list(categories.values()), minlength=4)
    assert counts.max() - counts.min() <= 1


def test_sufficient_statistics_equal_central_ridge():
    speeds = [np.asarray([10, 12, 14]), np.asarray([16, 18, 20]), np.asarray([22, 26, 30])]
    rng = np.random.default_rng(5)
    targets = [rng.normal(size=(len(values), 6)) for values in speeds]
    statistics = [client_statistics(x, y, CFG10A) for x, y in zip(speeds, targets)]
    federated = solve_statistics(statistics, np.arange(3), 1e-8)
    central = ridge_fit(np.concatenate(speeds), np.vstack(targets), 7, CFG10A, strength=1e-8)
    np.testing.assert_allclose(federated, central, atol=1e-11, rtol=1e-11)


def test_federated_mixture_is_central_equivalent():
    rng = np.random.default_rng(6)
    blocks = [
        np.asarray([10, 12, 14], dtype=float),
        np.asarray([16, 18, 20], dtype=float),
        np.asarray([22, 24, 26], dtype=float),
        np.asarray([28, 30], dtype=float),
    ]
    speeds, targets = [], []
    for cluster in range(2):
        for client in range(20):
            local_speed = blocks[client % 4]
            base = np.asarray([cluster * 8, cluster * 4, 1, 2, 3, 4], dtype=float)
            speeds.append(local_speed)
            targets.append(
                np.tile(base, (len(local_speed), 1))
                + rng.normal(scale=0.01, size=(len(local_speed), 6))
            )
    from experiment_10d_learned_compatible_groups import fit_mixture

    args = (speeds, targets, [1, 2], 5, 20, 4, 7, CFG10A)
    central, _, _ = fit_mixture(*args)
    federated, _, _, communication = fit_federated_mixture(*args)
    error, same = aligned_model_error(central, federated)
    assert same
    assert error < 1e-10
    assert communication["uplink_scalars"] > 0
