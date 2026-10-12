"""Audit committed review results without fitting or calibration/tracking runs.

Reference Jacobians are recomputed on the existing grid to check saved model
errors. Original checkpoints, summaries and source locks remain read-only.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code/src"))
sys.path.insert(0, str(ROOT / "code/experiments"))

import numpy as np
import pandas as pd
from federated_lpv import sample_fleet
from experiment_10a_basis_coverage_audit import collect_targets, raw_basis
from experiment_10b_oracle_complementary_gate import predict

OUT = ROOT / "results/tables"


def digest(path):
    data = path.read_bytes()
    if path.suffix in {".py", ".json", ".csv"}:
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def normalize_lock(lock):
    result = {}
    for path, sha in lock.items():
        key = path.replace("\\", "/")
        if key in result:
            raise ValueError(f"Duplicate provenance path: {key}")
        result[key] = sha
    return result


def check_frame(actual, saved, keys, columns):
    a = actual.set_index(keys).sort_index()
    b = saved.set_index(keys).sort_index()
    if not a.index.equals(b.index):
        raise ValueError(f"Summary keys differ: {keys}")
    np.testing.assert_allclose(a[columns], b[columns], rtol=1e-10, atol=1e-12)


def main():
    cfg = json.loads((ROOT / "code/config/paper_review_validation.json").read_text())
    parent = json.loads((ROOT / "code/config/experiment_10e.json").read_text())
    basis = json.loads((ROOT / "code/config/experiment_10a.json").read_text())
    manifest = json.loads((OUT / "paper_review_validation_manifest.json").read_text())
    lock = normalize_lock(manifest["source_sha256"])
    expected = {**cfg["source_sha256"],
                "code/config/paper_review_validation.json": digest(ROOT / "code/config/paper_review_validation.json"),
                "code/experiments/paper_review_validation.py": digest(ROOT / "code/experiments/paper_review_validation.py")}
    if lock != expected:
        raise ValueError("Manifest source lock is incomplete or changed")
    for path, sha in lock.items():
        if digest(ROOT / path) != sha:
            raise ValueError(f"Source hash mismatch: {path}")
    if manifest["status"] != "complete_parent_verified" or manifest["fleets"] != 10:
        raise ValueError("Incomplete run")
    grid = np.asarray(parent["speed_grid"], dtype=float)
    checks, group_rows, all_errors, local_rows = [], [], [], []
    maximum_coefficient_error = 0.0
    maximum_parent_record_error = 0.0
    maximum_parent_mean_error = 0.0
    for seed in cfg["seeds"]:
        name = f"paper_review_validation_seed{seed}.json"
        path = OUT / name
        if digest(path) != manifest["checkpoint_sha256"][name]:
            raise ValueError(f"Checkpoint hash mismatch: {name}")
        result = json.loads(path.read_text())
        if result["seed"] != seed or result["status"] != "parent_verified" or normalize_lock(result["source_sha256"]) != lock:
            raise ValueError(f"Checkpoint provenance mismatch: {seed}")
        clients = sample_fleet(seed)
        inputs = result["frozen_inputs"]
        if [r["client"] for r in inputs] != [c.client_id for c in clients]:
            raise ValueError("Client order mismatch")
        assignments = np.asarray(result["frozen_models"]["assignments"], dtype=int)
        if assignments.shape != (30,) or set(assignments) != set(range(result["selected_k"])):
            raise ValueError("Invalid assignments")
        errors = pd.DataFrame(result["errors"])
        if len(errors) != 30 * 11 * 4 or errors.duplicated(["client", "method", "speed"]).any():
            raise ValueError("Missing or duplicated client-speed error")
        expected_methods = {"Local", "Global", "FederatedLearned", "LocalRegularized"}
        if set(errors.method) != expected_methods or set(errors.client) != {c.client_id for c in clients}:
            raise ValueError("Invalid method/client set")
        if not np.isfinite(errors.prediction_error).all():
            raise ValueError("Nonfinite errors")
        for i, (client, item) in enumerate(zip(clients, inputs)):
            speeds = np.asarray(item["speeds"])
            if list(speeds) not in parent["coverage_blocks"]:
                raise ValueError("Unexpected coverage block")
            if np.asarray(item["targets"]).shape != (len(speeds), 6):
                raise ValueError("Invalid local matrix targets")
            design = raw_basis(speeds, 7, basis)
            local_rows.append(dict(seed=seed, client=client.client_id, local_speeds=len(speeds),
                                   local_rank=int(np.linalg.matrix_rank(design)),
                                   ridge_strength=item["local_regularization"]))
            rows = errors[errors.client == client.client_id]
            for row in rows.itertuples():
                if row.region != ("seen" if row.speed in speeds else "unseen"):
                    raise ValueError("Incorrect observed/unobserved region")
            # Recompute only evaluation references; no transitions or fits.
            _, reference = collect_targets(client, parent["curvature_per_m"], basis)
            frozen = result["frozen_models"]
            models = {"Global": frozen["global_model"],
                      "FederatedLearned": frozen["groups"][assignments[i]],
                      "LocalRegularized": frozen["local_regularized"][i]}
            for method, coefficients in models.items():
                predictions = predict(np.asarray(coefficients), grid, basis)
                actual = np.linalg.norm(predictions - reference, axis=1) / np.linalg.norm(reference, axis=1)
                saved = rows[rows.method == method].sort_values("speed").prediction_error.to_numpy()
                maximum_coefficient_error = max(maximum_coefficient_error, float(np.max(np.abs(actual - saved))))
                np.testing.assert_allclose(actual, saved, rtol=0, atol=1e-8)
        groups = result["groups"]
        if len(groups) != result["selected_k"]:
            raise ValueError("Missing group diagnostics")
        for g, saved in enumerate(groups):
            indices = np.flatnonzero(assignments == g)
            members = [clients[i].client_id for i in indices]
            if saved["members"] != members or saved["clients"] != len(indices):
                raise ValueError("Membership mismatch")
            speeds = np.concatenate([np.asarray(inputs[i]["speeds"]) for i in indices])
            design = raw_basis(speeds, 7, basis)
            singular = np.linalg.svd(design, compute_uv=False)
            rank = int(np.linalg.matrix_rank(design))
            if rank != saved["rank"] or sorted(set(speeds)) != saved["unique_speeds"]:
                raise ValueError("Group information mismatch")
            if rank == 7:
                np.testing.assert_allclose([singular[0] / singular[-1], singular[-1]],
                                           [saved["design_condition"], saved["minimum_singular_value"]], rtol=1e-10)
            group_rows.append(dict(seed=seed, group=g, clients=len(indices), rank=rank,
                                   unique_speeds=len(set(speeds)), condition=saved["design_condition"]))
        original = pd.read_csv(OUT / f"experiment_10e_seed{seed}_label_free_clients.csv.gz")
        for method in ["Local", "Global", "FederatedLearned"]:
            a = errors[errors.method == method].set_index(["client", "speed"]).prediction_error.sort_index()
            b = original[original.method == method].set_index(["client", "speed"]).prediction_error.sort_index()
            if not a.index.equals(b.index):
                raise ValueError("Parent record index mismatch")
            maximum_parent_record_error = max(maximum_parent_record_error, float(np.max(np.abs(a - b))))
            delta = float(a.mean() - b.mean())
            maximum_parent_mean_error = max(maximum_parent_mean_error, abs(delta))
            np.testing.assert_allclose(a, b, rtol=0, atol=cfg["parent_absolute_error_tolerance"])
            np.testing.assert_allclose(delta, result["parent_error_differences"][method], rtol=0, atol=1e-14)
        all_errors.extend(result["errors"])
        checks.append(dict(seed=seed, checkpoint_verified=True, selected_k=result["selected_k"],
                           model_error_verified=True, group_information_verified=True))

    raw = pd.DataFrame(all_errors)
    units = raw.groupby(["seed", "region", "method"]).prediction_error.mean().reset_index()
    full = raw.groupby(["seed", "method"]).prediction_error.mean().reset_index().assign(region="full")
    units = pd.concat([units, full], ignore_index=True)
    check_frame(units, pd.read_csv(OUT / "paper_review_validation_seed_summary.csv"),
                ["seed", "region", "method"], ["prediction_error"])
    summary = units.groupby(["region", "method"]).prediction_error.agg(["mean", "std"]).reset_index()
    check_frame(summary, pd.read_csv(OUT / "paper_review_validation_summary.csv"),
                ["region", "method"], ["mean", "std"])
    rng = np.random.default_rng(cfg["bootstrap_seed"])
    comparisons = []
    for region in ["full", "seen", "unseen"]:
        view = units[units.region == region].pivot(index="seed", columns="method", values="prediction_error")
        for method, baseline in [("LocalRegularized", "Local"), ("FederatedLearned", "LocalRegularized")]:
            gains = 100 * (1 - view[method] / view[baseline])
            low, high = np.quantile(rng.choice(gains, size=(20000, 10), replace=True).mean(axis=1), [.025, .975])
            comparisons.append(dict(region=region, method=method, baseline=baseline,
                                    improvement_pct=gains.mean(), ci_low=low, ci_high=high,
                                    wins=int((gains > 0).sum())))
    check_frame(pd.DataFrame(comparisons), pd.read_csv(OUT / "paper_review_validation_comparisons.csv"),
                ["region", "method", "baseline"], ["improvement_pct", "ci_low", "ci_high", "wins"])
    groups = pd.DataFrame(group_rows)
    saved_groups = pd.read_csv(OUT / "paper_review_validation_groups.csv").rename(columns={"design_condition": "condition"})
    check_frame(groups, saved_groups, ["seed", "group"], ["clients", "rank", "condition"])
    if len(groups) != manifest["groups"] or int((groups['rank'] == 7).sum()) != manifest["full_rank_groups"]:
        raise ValueError("Manifest group totals mismatch")
    np.testing.assert_allclose(maximum_parent_mean_error, manifest["maximum_parent_error_difference"], rtol=0, atol=1e-14)

    tails = raw.groupby(["seed", "client", "region", "method"]).prediction_error.mean().unstack("method")
    tail_rows = []
    for region, data in tails.groupby(level="region"):
        delta = data.FederatedLearned - data.Global
        tail_rows.append(dict(region=region, clients=len(data), grouped_worse=int((delta > 1e-12).sum()),
                              grouped_better=int((delta < -1e-12).sum()), numerical_ties=int((delta.abs() <= 1e-12).sum()),
                              maximum_deterioration_percentage_points=float(100 * delta.max()),
                              maximum_local_regularization_change_percentage_points=float(100 * (data.LocalRegularized-data.Local).abs().max())))
    pd.DataFrame(tail_rows).to_csv(OUT / "paper_review_validation_tail_audit.csv", index=False)
    pd.DataFrame(local_rows).to_csv(OUT / "paper_review_validation_local_rank_audit.csv", index=False)
    source_names = ["paper_review_validation_manifest.json", "paper_review_validation_summary.csv",
                    "paper_review_validation_comparisons.csv", "paper_review_validation_groups.csv",
                    "paper_review_validation_seed_summary.csv"] + list(manifest["checkpoint_sha256"])
    audit = dict(status="independently_audited", result_commit=subprocess.check_output(
        ["git", "log", "-1", "--format=%H", "--", "results/tables/paper_review_validation_manifest.json"], cwd=ROOT, text=True).strip(),
        fleets=10, clients=300, errors=len(raw), checked_sources=len(lock), checkpoints=10,
        reference_evaluations=300 * 11, model_error_entries_verified=300 * 11 * 3,
        maximum_saved_model_error_difference=maximum_coefficient_error,
        maximum_parent_record_error_difference=maximum_parent_record_error,
        maximum_parent_mean_error_difference=maximum_parent_mean_error,
        groups=len(groups), all_groups_full_rank=bool((groups['rank'] == 7).all()),
        all_groups_cover_eleven_speeds=bool((groups.unique_speeds == 11).all()),
        group_design_condition_range=[float(groups.condition.min()), float(groups.condition.max())],
        group_client_range=[int(groups.clients.min()), int(groups.clients.max())],
        local_rank_counts={str(k): int(v) for k, v in Counter(r["local_rank"] for r in local_rows).items()},
        regularization_strength_counts={str(k): int(v) for k, v in Counter(r["ridge_strength"] for r in local_rows).items()},
        client_tails=tail_rows, fleet_checks=checks,
        operation="read_only_results_audit_with_reference_jacobian_evaluation_no_fitting_or_rollouts",
        source_sha256={str((OUT / n).relative_to(ROOT)): digest(OUT / n) for n in source_names})
    (OUT / "paper_review_validation_independent_audit.json").write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in audit.items() if k not in {"source_sha256", "fleet_checks"}}, indent=2))


if __name__ == "__main__":
    main()
