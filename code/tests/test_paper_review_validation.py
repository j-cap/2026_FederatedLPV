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


def test_source_digest_accepts_windows_line_endings_but_preserves_binary(tmp_path):
    lf = tmp_path / "lf.py"
    crlf = tmp_path / "crlf.py"
    lf.write_bytes(b"value = 1\nprint(value)\n")
    crlf.write_bytes(b"value = 1\r\nprint(value)\r\n")
    assert review.digest(lf) == review.digest(crlf)
    crlf.write_bytes(b"value = 2\r\nprint(value)\r\n")
    assert review.digest(lf) != review.digest(crlf)
    binary_lf = tmp_path / "lf.gz"
    binary_crlf = tmp_path / "crlf.gz"
    binary_lf.write_bytes(b"record\n")
    binary_crlf.write_bytes(b"record\r\n")
    assert review.digest(binary_lf) != review.digest(binary_crlf)


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
