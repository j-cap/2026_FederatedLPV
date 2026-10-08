"""Validation-only choice must not refit or depend on outer results."""

import pandas as pd
from experiment_10hm_frozen_libraries import choose_library


def test_actual_library_selection_eligibility_and_ties():
    scores = pd.DataFrame(
        [
            {"restart": 0, "margin": 0.1, "selection_score": -0.4},
            {"restart": 1, "margin": 0.25, "selection_score": -0.4},
            {"restart": 2, "margin": 0, "selection_score": -10},
        ]
    )
    assert choose_library(scores, {0: True, 1: True, 2: False}) == {
        "restart": 1,
        "margin": 0.25,
        "score": -0.4,
    }
    assert choose_library(scores, {0: False, 1: False, 2: False}) is None
