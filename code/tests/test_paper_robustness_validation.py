"""Mechanism, pairing, and provenance tests for the focused local-run study."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/"code/src"), str(ROOT/"code/experiments")]
import paper_robustness_validation as study
from experiment_10b_oracle_complementary_gate import estimate_matrix

CFG = json.loads(study.CONFIG.read_text())
PARENT = json.loads((ROOT/"code/config/experiment_10e.json").read_text())
BASIS = json.loads((ROOT/"code/config/experiment_10a.json").read_text())
CASES = {c["name"]: c for c in CFG["cases"]}


def records(case_name, seed=1200, index=0):
    client = study.sample_fleet(seed)[index]
    rng, mrng = study.cell_rngs(seed, index, 0, CFG)
    return study.sample_records(client, 10., CASES[case_name], PARENT, rng, mrng,
                                study.center_bias(seed, index, CASES[case_name], CFG))


def test_coverage_preserves_budgets_and_has_the_predeclared_ranks():
    clients = study.sample_fleet(1200)
    baseline = study.observed_support(1200, clients, CASES["reference"], CFG, PARENT)
    for name, global_rank in [("group_coverage", 7), ("fleet_coverage", 6)]:
        observed = study.observed_support(1200, clients, CASES[name], CFG, PARENT)
        assert list(map(len, observed)) == list(map(len, baseline))
        assert study.information(np.concatenate(observed), BASIS)["rank"] == global_rank
        if name == "group_coverage":
            heavy = np.concatenate([s for c, s in zip(clients, observed) if c.family == "heavy"])
            assert study.information(heavy, BASIS)["rank"] == 3
            assert sum(c.family == "heavy" for c in clients) >= PARENT["minimum_cluster_size"]
            for c, original, changed in zip(clients, baseline, observed):
                if c.family != "heavy":
                    np.testing.assert_array_equal(original, changed)


def test_reference_calibration_matches_existing_estimator():
    client = study.sample_fleet(1200)[0]
    rng, _ = study.cell_rngs(1200, 0, 0, CFG)
    expected = estimate_matrix(client, 10., 36, PARENT, rng)
    data, _, assumed = records("reference")
    actual, diagnostic = study.estimate_records(data, assumed)
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)
    assert diagnostic["regressor_rank"] == 3


def test_target_noise_is_paired_and_does_not_change_regressors():
    base, _, _ = records("reference")
    high, _, _ = records("target_noise")
    np.testing.assert_array_equal(base[:, :3], high[:, :3])
    case = {**CASES["reference"], "target_noise_scale": 0.}
    client = study.sample_fleet(1200)[0]
    rng, mrng = study.cell_rngs(1200, 0, 0, CFG)
    noiseless, _, _ = study.sample_records(client, 10., case, PARENT, rng, mrng, np.zeros(3))
    np.testing.assert_allclose(high[:, 3:]-noiseless[:, 3:], 2*(base[:, 3:]-noiseless[:, 3:]), atol=1e-16)


def test_center_error_changes_preprocessing_not_physical_records():
    base, center, assumed_base = records("reference")
    biased, center2, assumed = records("center_bias")
    np.testing.assert_array_equal(base, biased)
    np.testing.assert_array_equal(center, center2)
    np.testing.assert_allclose(abs(assumed-assumed_base), CASES["center_bias"]["center_bias_magnitude"], atol=1e-17)
    z0, z1 = base[:, :3]-assumed_base, biased[:, :3]-assumed
    np.testing.assert_allclose(z1-z0, np.broadcast_to(-(assumed-assumed_base), z0.shape), atol=1e-17)


def test_current_noise_does_not_change_next_state_targets():
    base, _, _ = records("reference")
    noisy, _, _ = records("regressor_noise")
    np.testing.assert_array_equal(base[:, 3:], noisy[:, 3:])
    assert not np.array_equal(base[:, :3], noisy[:, :3])


def test_weak_excitation_halves_current_deviations_with_same_budget():
    base, center, _ = records("reference")
    weak, _, _ = records("weak_excitation")
    assert len(base) == len(weak) == 36
    np.testing.assert_allclose(weak[:, :3]-center, .5*(base[:, :3]-center), atol=1e-17)


def test_rank_fallback_uses_global_only_for_flagged_recipients():
    model = dict(Global=np.full((7, 6), 3.).tolist(), Federated=[np.ones((7, 6)).tolist()],
                 assignments=[0, 0], fallback_flags=[True, False])
    np.testing.assert_array_equal(study.client_model(model, "RankFallback", 0), model["Global"])
    np.testing.assert_array_equal(study.client_model(model, "RankFallback", 1), model["Federated"][0])


def test_record_digest_is_portable_and_detects_measurement_changes():
    data = np.arange(20.).reshape(4, 5)
    assert study.record_digest(data.astype(">f8")) == study.record_digest(data)
    changed = data.copy(); changed[0, 0] += 1e-6
    assert study.record_digest(changed) != study.record_digest(data)


def test_checkpoint_rejects_corruption_and_changed_provenance(tmp_path):
    result = dict(seed=1201, case="reference", status="execution_complete", provenance={"lock": "one"},
        frozen_inputs=[{} for _ in range(30)], reference=np.ones((30, 11, 6)).tolist(),
        errors={m: np.zeros((30, 11)).tolist() for m in study.METHODS})
    path = study.save_checkpoint(result, tmp_path)
    study.read_checkpoint(path, 1201, "reference", {"lock": "one"})
    with pytest.raises(ValueError, match="Stale/incompatible"):
        study.read_checkpoint(path, 1201, "reference", {"lock": "two"})
    envelope = json.loads(path.read_text()); envelope["payload"]["errors"]["Global"][0][0] = .5
    path.write_text(json.dumps(envelope))
    with pytest.raises(ValueError, match="corrupted"):
        study.read_checkpoint(path, 1201, "reference", {"lock": "one"})


def test_source_hash_accepts_crlf_and_rejects_semantic_edits(tmp_path):
    path = tmp_path/"source.py"; path.write_bytes(b"value = 1\n")
    original = study.digest(path); path.write_bytes(b"value = 1\r\n")
    assert study.digest(path) == original
    path.write_bytes(b"value = 2\r\n")
    assert study.digest(path) != original


def test_restoration_uses_frozen_centers_and_preserves_checkpoint(tmp_path, monkeypatch):
    client = study.sample_fleet(1200)[0]
    data, center, assumed = records("reference")
    item = dict(client=client.client_id, family=client.family, parameters=study.asdict(client.parameters),
        cells=[dict(speed=10., true_center=center.tolist(), assumed_center=assumed.tolist(), record_sha256=study.record_digest(data))])
    result = dict(seed=1200, case="reference", case_configuration=CASES["reference"], frozen_inputs=[item])
    monkeypatch.setattr(study, "equilibrium", lambda *args: pytest.fail("Restoration must use frozen centers"))
    monkeypatch.setattr(study, "estimate_records", lambda *args: pytest.fail("Restoration must not fit matrices"))
    assert study.restore_job(result, tmp_path, CFG, PARENT) == 1
    with np.load(tmp_path/"seed1200_reference.npz", allow_pickle=False) as restored:
        assert study.record_digest(restored[client.client_id+"_speed0"]) == study.record_digest(data)


def test_partial_results_cannot_be_summarized(tmp_path):
    with pytest.raises(ValueError, match="Complete 70-job matrix required"):
        study.summarize(tmp_path, CFG, PARENT, {})


def test_smoke_seed_is_separate_and_sources_are_pinned():
    cfg, _, _, strength, provenance = study.preflight()
    assert cfg["smoke_seed"] not in cfg["seeds"]
    assert strength == 1e6
    assert "code/experiments/experiment_10e_federated_latent_groups.py" in provenance["source_sha256"]
