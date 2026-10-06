import json
from pathlib import Path

import numpy as np
from experiment_10hd_measured_output_observer_groups import (
    nominal_client,
    prior_regularized_matrix,
)

CFG10H = json.loads((Path(__file__).parents[1] / "config/experiment_10h.json").read_text())


def test_nominal_client_is_five_state_compatible():
    client = nominal_client(CFG10H)
    assert client.vehicle_class == "nominal"
    assert client.parameters.mass > 0
    assert client.front_relaxation_length > 0


def test_prior_regularization_returns_prior_without_excitation():
    prior = np.arange(30, dtype=float).reshape(5, 6)
    regressors = np.zeros((20, 6))
    targets = np.zeros((20, 5))
    estimate = prior_regularized_matrix(regressors, targets, prior, 0.1)
    assert estimate.shape == prior.shape
    assert np.isfinite(estimate).all()
