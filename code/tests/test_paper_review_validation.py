"""Checks that the review supplement cannot silently change its evidence."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import paper_review_validation as review


def test_preflight_locks_original_seed_set_and_sources():
    cfg, parent, lock = review.preflight()
    assert cfg["seeds"] == parent["seeds"] == list(range(371, 381))
    assert "code/src/federated_lpv/vehicle.py" in lock


def test_preflight_rejects_changed_historical_source(monkeypatch):
    monkeypatch.setattr(review, "digest", lambda _: "changed")
    with pytest.raises(ValueError, match="Frozen source changed"):
        review.preflight()


def test_size_alone_does_not_establish_scheduling_information():
    import json
    basis = json.loads((review.ROOT / "code/config/experiment_10a.json").read_text())
    local = review.information_diagnostic([10, 12, 14] * 20, basis)
    assert local["rank"] == 3
    assert not local["full_rank"]
    assert local["design_condition"] is None
    covered = review.information_diagnostic(list(range(10, 31, 2)), basis)
    assert covered["rank"] == 7 and covered["full_rank"]
    assert np.isfinite(covered["design_condition"])


def test_parent_check_rejects_an_inconsistent_prediction():
    cfg, _, _ = review.preflight()
    wrong = {m: np.zeros((30, 11)) for m in ["Local", "Global", "FederatedLearned"]}
    with pytest.raises(ValueError, match="Parent mismatch"):
        review.compare_parent(371, wrong, 2, cfg)
