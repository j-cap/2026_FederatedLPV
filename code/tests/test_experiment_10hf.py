import json
from pathlib import Path

import numpy as np
from experiment_10hf_output_identifiability import (
    PARAMETER_NAMES,
    information_diagnostics,
    nominal_effective_parameters,
    structured_discrete,
)

CFG = json.loads((Path(__file__).parents[1] / "config/experiment_10hf.json").read_text())


def test_effective_parameterization_has_nine_positive_coordinates():
    values = nominal_effective_parameters(CFG)
    assert len(values) == len(PARAMETER_NAMES) == 9
    assert (values > 0).all()


def test_structured_model_has_expected_shape():
    a, b = structured_discrete(np.log(nominal_effective_parameters(CFG)), 20.0, 0.01)
    assert a.shape == (5, 5)
    assert b.shape == (5, 1)
    assert np.isfinite(a).all() and np.isfinite(b).all()


def test_information_rank_detects_missing_direction():
    matrix = np.diag([1.0] * 8 + [0.0])
    rank, condition, _ = information_diagnostics(matrix, 1e-6)
    assert rank == 8
    assert condition == 1.0
