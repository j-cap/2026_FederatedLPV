import json
from pathlib import Path

import numpy as np
from experiment_10he_innovation_separability import (
    nominal_client,
    residualize_blocks,
    signature,
)

CFG = json.loads((Path(__file__).parents[1] / "config/experiment_10he.json").read_text())


def test_nominal_design_is_fixed_by_config():
    client = nominal_client(CFG)
    assert client.parameters.mass == CFG["nominal_design"]["mass"]
    assert client.vehicle_class == "fixed_nominal"


def test_signature_uses_only_innovations_and_commands():
    rng = np.random.default_rng(10)
    innovations = rng.normal(size=(500, 3))
    commands = rng.normal(size=500)
    result = signature(innovations, commands, np.eye(3))
    assert result.shape == (23,)
    assert np.isfinite(result).all()


def test_block_residualization_removes_block_means():
    features = np.arange(48, dtype=float).reshape(8, 6)
    blocks = np.repeat([0, 1], 4)
    result = residualize_blocks(features, blocks)
    np.testing.assert_allclose(result[blocks == 0].mean(axis=0), 0, atol=1e-12)
    np.testing.assert_allclose(result[blocks == 1].mean(axis=0), 0, atol=1e-12)
