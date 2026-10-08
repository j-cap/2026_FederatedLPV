"""Frozen parent provenance and full-state intervention isolation."""
import hashlib
import json
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose
from test_experiment_10hp import Q, R, WEIGHTS, Z, setup

from experiment_10hq_preflight import parent_preflight
from federated_lpv.feedback_diagnostics import diagnostic_cases
from federated_lpv.innovation_likelihood import structured_matrices
from federated_lpv.output_feedback_control import design_schedule, rollout


def test_frozen_parent_pins_and_factorial_scope():
    root = Path(__file__).parents[2]
    cfg = json.loads((root/"code/config/experiment_10hq.json").read_text())
    for field in ["required_parent_manifest", "required_parent_audit"]:
        assert hashlib.sha256((root/cfg[field]).read_bytes()).hexdigest() == cfg[field+"_sha256"]
    assert len(diagnostic_cases(cfg)) == 12
    assert 5*10*2*2*2*12 == cfg["planned_closed_loop_rollouts"]
    assert not cfg["fit_or_select_models"] and not cfg["confirmation_run"]


def test_preflight_rejects_changed_parent_pin(tmp_path):
    cfg = {"required_parent_revision": "missing", "required_parent_paths": [],
           "required_parent_artifact_patterns": [], "parent_mode": "reconstructed",
           "required_parent_manifest": "parent.json", "required_parent_manifest_sha256": "0"*64,
           "required_parent_audit": "audit.json", "required_parent_audit_sha256": "0"*64}
    (tmp_path/"parent.json").write_text(json.dumps({"source_sha256": {}, "output_sha256": {}, "rollouts": 1600, "confirmation_run": False}))
    (tmp_path/"audit.json").write_text(json.dumps({"integrity_checks_pass": True}))
    status = parent_preflight(tmp_path, cfg)
    assert status["status"] == "blocked_missing_parent"
    assert len(status["provenance_errors"]) == 2
    assert status["fleet_rollouts_executed"] == 0


def test_full_state_command_path_is_independent_of_kf_internals():
    matrices, controller, scenario, noise = setup()
    alternate = design_schedule([structured_matrices(Z+.02, v, .01) for v in [10., 30.]],
                                [10., 30.], Q, R, WEIGHTS, .01)
    before = rollout(matrices, [10., 30.], controller, controller, scenario, noise, correction="all")
    after = rollout(matrices, [10., 30.], alternate, controller, scenario, noise, correction="all")
    for key in ["truth", "command", "integral", "measurement"]:
        assert_allclose(before[key], after[key], rtol=0, atol=0)
    assert np.max(abs(before["estimate"]-after["estimate"])) > 1e-6
