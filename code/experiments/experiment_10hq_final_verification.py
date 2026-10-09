"""Check provenance, paired aggregation and full-state isolation for frozen Q."""

import json
import subprocess

import numpy as np
import pandas as pd
from experiment_10hk_joint_client_mixture import OUT, ROOT, sha

from federated_lpv.feedback_diagnostics import diagnostic_cases

BASE = "91bbdd6ab8ec1d2dc2461192483163b5f934c313"


def run():
    checked = 0
    for mode in ["10hp", "10hq"]:
        complete = json.loads((OUT / f"experiment_{mode}_complete.json").read_text())
        manifests = [complete] + [
            json.loads((OUT / f"{name}_complete.json").read_text())
            for name in complete["job_manifest_sha256"]
        ]
        for manifest in manifests:
            for path, expected in manifest["source_sha256"].items():
                assert sha(ROOT / path) == expected, path
                checked += 1
            for path, expected in manifest["output_sha256"].items():
                assert sha(OUT / path) == expected, path
                checked += 1
        for name, expected in complete["job_manifest_sha256"].items():
            assert sha(OUT / f"{name}_complete.json") == expected
            checked += 1

    metrics = pd.read_csv(OUT / "experiment_10hq_metrics.csv")
    common = pd.read_csv(OUT / "experiment_10hq_common_input.csv")
    cfg = json.loads((ROOT / "code/config/experiment_10hq.json").read_text())
    assert len(metrics) == 4800 and len(common) == 2400
    assert metrics.finite.all() and not metrics.design_failed.any()
    assert set(metrics.seed) == set(cfg["development_seeds"]) == {601, 602, 603, 604, 605}
    assert set(metrics.case) == {case.name for case in diagnostic_cases(cfg)}
    keys = ["seed", "process_noise_scale", "scenario", "case", "client_position"]
    assert metrics.groupby(keys).size().eq(2).all()
    assert not metrics.duplicated(keys + ["donor_fold"]).any()
    recipients = metrics.groupby(keys).tracking_rmse_deg_s.mean().reset_index()
    fleets = recipients.groupby(keys[:-1]).tracking_rmse_deg_s.mean().reset_index()
    saved = pd.read_csv(OUT / "experiment_10hq_fleets.csv")
    paired = fleets.merge(saved, on=keys[:-1], suffixes=("_new", "_saved"), validate="one_to_one")
    error = float(np.max(abs(paired.tracking_rmse_deg_s_new - paired.tracking_rmse_deg_s_saved)))
    assert error <= 1e-12
    nominal = fleets[fleets.case == "nominal_with_nominal_kf"].drop(columns="case")
    gain = fleets.merge(nominal, on=keys[:3], suffixes=("", "_baseline"), validate="many_to_one")
    gain["new_gain"] = 100 * (1 - gain.tracking_rmse_deg_s / gain.tracking_rmse_deg_s_baseline)
    gain = gain.merge(saved[keys[:-1] + ["tracking_gain_pct"]], on=keys[:-1], validate="one_to_one")
    gain_error = float(np.max(abs(gain.new_gain - gain.tracking_gain_pct)))
    assert gain_error <= 1e-10

    true = metrics[metrics.case.isin(["nominal_with_true_state", "oracle_with_true_state"])]
    equal = 0
    for _, rows in true.groupby(["seed", "donor_fold", "client_position", "scenario", "case"]):
        assert len(rows) == 2
        with np.load(ROOT / rows.iloc[0].record_path) as first, np.load(ROOT / rows.iloc[1].record_path) as second:
            for key in ["truth", "command", "raw_command", "integral", "measurement", "supplied", "noise"]:
                assert np.array_equal(first[key], second[key]), key
        equal += 1
    assert equal == 400
    audit_sources = {
        "experiment_10hq_audit.json": "code/experiments/experiment_10hp_result_audit.py",
        "experiment_10hq_projection_audit.json": "code/experiments/experiment_10hq_result_audit.py",
    }
    for name, source in audit_sources.items():
        audit = json.loads((OUT / name).read_text())
        assert audit["audit_source_sha256"] == sha(ROOT / source), name
        assert audit["result_manifest_sha256"] == sha(OUT / "experiment_10hq_complete.json")
        assert audit["audit_table_sha256"] == sha(OUT / name.replace(".json", ".csv").replace(
            "experiment_10hq_audit.csv", "experiment_10hq_replay_audit.csv"
        )), name
    conclusions = json.loads((OUT / "experiment_10hq_conclusions.json").read_text())
    assert conclusions["source_sha256"] == sha(ROOT / "code/experiments/experiment_10hq_reporting.py")
    assert conclusions["audit_sha256"] == sha(OUT / "experiment_10hq_audit.json")
    assert conclusions["projection_audit_sha256"] == sha(OUT / "experiment_10hq_projection_audit.json")
    regeneration = json.loads((OUT / "experiment_10hq_regeneration_audit.json").read_text())
    assert regeneration["all_checks_pass"] and regeneration["verified_or_restored"] == 320
    changed = subprocess.check_output(
        ["git", "diff", "--name-only", BASE, "--", "code"], cwd=ROOT, text=True
    ).splitlines()
    for path in changed:
        exists = subprocess.run(["git", "cat-file", "-e", f"{BASE}:{path}"], cwd=ROOT,
                                capture_output=True, check=False).returncode == 0
        if exists:
            old = subprocess.check_output(["git", "show", f"{BASE}:{path}"], cwd=ROOT)
            assert old == (ROOT / path).read_bytes(), f"Historical source changed: {path}"
    result = {
        "all_checks_pass": True, "provenance_references_checked": checked,
        "closed_loop_runs": 4800, "common_input_replays": 2400,
        "cross_panel_true_state_pairs_identical": equal,
        "maximum_fleet_aggregation_error": error, "maximum_gain_aggregation_error": gain_error,
        "audit_and_conclusion_source_hashes_match": True,
        "sampled_parent_and_q_hash_regenerations": regeneration["verified_or_restored"],
        "historical_source_unchanged_from": BASE,
        "source_sha256": sha(ROOT / "code/experiments/experiment_10hq_final_verification.py"),
    }
    (OUT / "experiment_10hq_final_verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    run()
