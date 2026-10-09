"""Focused review supplement: restore 10E fits and evaluate LocalRegularized.

Historical inputs/results are read-only. A full run repeats only the original
one-step calibration and identification; it launches no tracking rollouts and
does not change the primary experiment. Exact Jacobians are evaluation-only.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from federated_lpv import sample_fleet
from experiment_10a_basis_coverage_audit import collect_targets, raw_basis
from experiment_10b_oracle_complementary_gate import (
    estimate_matrix, lift_prior, predict, ridge_fit, select_strength,
)
from experiment_10e_federated_latent_groups import label_free_categories, fit_federated_mixture

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "code/config/paper_review_validation.json"
OUT = ROOT / "results/tables"


def digest(path):
    data = path.read_bytes()
    # Git for Windows may check out CRLF. Preserve the frozen LF text hashes
    # without weakening checks for substantive edits or binary record changes.
    if path.suffix in {".py", ".json", ".csv"}:
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def preflight():
    cfg = json.loads(CONFIG.read_text())
    for name, expected in cfg["source_sha256"].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"Frozen source changed: {name}")
    parent = json.loads((ROOT / "code/config/experiment_10e.json").read_text())
    if cfg["seeds"] != parent["seeds"]:
        raise ValueError("Supplement must use exactly the original ten label-free fleets")
    lock = {**cfg["source_sha256"], str(CONFIG.relative_to(ROOT)): digest(CONFIG),
            str(Path(__file__).relative_to(ROOT)): digest(Path(__file__))}
    return cfg, parent, lock


def information_diagnostic(speeds, cfg10a):
    design = raw_basis(np.asarray(speeds, dtype=float), 7, cfg10a)
    rank = int(np.linalg.matrix_rank(design))
    singular = np.linalg.svd(design, compute_uv=False)
    return dict(rank=rank, full_rank=rank == 7, observations=len(speeds),
                unique_speeds=sorted(set(float(v) for v in speeds)),
                design_condition=float(singular[0] / singular[-1]) if rank == 7 else None,
                minimum_singular_value=float(singular[-1]) if rank == 7 else 0.0)


def compare_parent(seed, errors, selected_k, cfg):
    frozen = pd.read_csv(OUT / "experiment_10e_seed_summary.csv")
    view = frozen[(frozen.seed == seed) & (frozen.protocol == "label_free")].set_index("method")
    delta = {}
    for method in ["Local", "Global", "FederatedLearned"]:
        actual = float(np.mean(errors[method]))
        expected = float(view.loc[method, "prediction_error"])
        delta[method] = actual - expected
        if abs(delta[method]) > cfg["parent_absolute_error_tolerance"]:
            raise ValueError(f"Parent mismatch seed={seed}, method={method}: {actual} != {expected}")
    diagnostic = pd.read_csv(OUT / "experiment_10e_diagnostics.csv")
    row = diagnostic[(diagnostic.seed == seed) & (diagnostic.protocol == "label_free")].iloc[0]
    if int(row.selected_k) != selected_k:
        raise ValueError(f"Selected group count changed for seed {seed}")
    # When saved per-client records exist, enforce the stronger comparison too.
    path = OUT / f"experiment_10e_seed{seed}_label_free_clients.csv.gz"
    if path.exists():
        records = pd.read_csv(path)
        for method in ["Local", "Global", "FederatedLearned"]:
            actual = np.asarray(errors[method]).reshape(-1)
            # Parent record order is fixed client order, then ascending speed.
            expected = records[records.method == method].prediction_error.to_numpy()
            np.testing.assert_allclose(actual, expected, rtol=0,
                                       atol=cfg["parent_absolute_error_tolerance"])
    return delta


def run_seed(seed):
    cfg, parent, lock = preflight()
    basis = json.loads((ROOT / "code/config/experiment_10a.json").read_text())
    local_cfg = json.loads((ROOT / "code/config/experiment_10b.json").read_text())
    clients = sample_fleet(seed)
    rng = np.random.default_rng(seed + 500000)
    categories = label_free_categories(clients, rng, parent["coverage_blocks"])
    speeds, targets = [], []
    for client in clients:
        observed = np.asarray(parent["coverage_blocks"][categories[client.client_id]], dtype=float)
        target = np.asarray([estimate_matrix(client, v, parent["samples_per_local_speed"], parent, rng)
                             for v in observed])
        speeds.append(observed)
        targets.append(target)
    # No exact matrices or client family labels are passed to fitting/selection.
    with threadpool_limits(limits=1):
        selected, candidates, global_model, _ = fit_federated_mixture(
            speeds, targets, parent["candidate_clusters"], parent["mixture_restarts"],
            parent["mixture_iterations"], parent["minimum_cluster_size"], seed, basis,
            strength=parent["ridge_strength"])
    models, strengths = [], []
    for observed, target in zip(speeds, targets):
        local = ridge_fit(observed, target, 3, basis, strength=1e-8)
        prior = lift_prior(local, basis)
        strength = select_strength(
            observed, target, local_cfg, basis,
            lambda keep: lift_prior(ridge_fit(observed[keep], target[keep], 3, basis,
                                              strength=1e-6), basis))
        regularized = ridge_fit(observed, target, 7, basis, prior, strength)
        models.append((prior, regularized))
        strengths.append(float(strength))

    group_info = []
    for group in range(selected["k"]):
        indices = np.flatnonzero(selected["assignments"] == group)
        info = information_diagnostic(np.concatenate([speeds[i] for i in indices]), basis)
        group_info.append(dict(group=group, clients=len(indices),
                               members=[clients[i].client_id for i in indices], **info))
    grid = np.asarray(parent["speed_grid"], dtype=float)
    rows, errors = [], {m: [] for m in ["Local", "Global", "FederatedLearned", "LocalRegularized"]}
    for i, client in enumerate(clients):
        _, truth = collect_targets(client, parent["curvature_per_m"], basis)
        available = {"Local": models[i][0], "Global": global_model,
                     "FederatedLearned": selected["models"][selected["assignments"][i]],
                     "LocalRegularized": models[i][1]}
        for method, model in available.items():
            error = np.linalg.norm(predict(model, grid, basis) - truth, axis=1) / np.linalg.norm(truth, axis=1)
            errors[method].append(error.tolist())
            for v, value in zip(grid, error):
                rows.append(dict(seed=seed, client=client.client_id, method=method, speed=float(v),
                                 region="seen" if v in speeds[i] else "unseen", prediction_error=float(value)))
    delta = compare_parent(seed, errors, selected["k"], cfg)
    return dict(seed=seed, status="parent_verified", source_sha256=lock,
                selected_k=int(selected["k"]), parent_error_differences=delta,
                groups=group_info, candidates=candidates.to_dict("records"), errors=rows,
                frozen_inputs=[dict(client=c.client_id, speeds=s.tolist(), targets=t.tolist(),
                                    local_regularization=strengths[i])
                               for i, (c, s, t) in enumerate(zip(clients, speeds, targets))],
                frozen_models=dict(global_model=global_model.tolist(),
                                   groups=[m.tolist() for m in selected["models"]],
                                   assignments=selected["assignments"].tolist(),
                                   local_regularized=[m[1].tolist() for m in models]))


def save_result(result, directory):
    path = directory / f"paper_review_validation_seed{result['seed']}.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
    print(json.dumps({"seed": result["seed"], "status": result["status"],
                      "selected_k": result["selected_k"], "output": str(path)}), flush=True)


def summarize(directory):
    cfg, _, lock = preflight()
    results = []
    for seed in cfg["seeds"]:
        result = json.loads((directory / f"paper_review_validation_seed{seed}.json").read_text())
        if result["source_sha256"] != lock or result["seed"] != seed or result["status"] != "parent_verified":
            raise ValueError(f"Invalid or stale checkpoint for seed {seed}")
        results.append(result)
    raw = pd.DataFrame([row for result in results for row in result["errors"]])
    regions = raw.groupby(["seed", "region", "method"]).prediction_error.mean().reset_index()
    full = raw.groupby(["seed", "method"]).prediction_error.mean().reset_index().assign(region="full")
    units = pd.concat([regions, full], ignore_index=True)
    summary = units.groupby(["region", "method"]).prediction_error.agg(["mean", "std"]).reset_index()
    comparisons = []
    rng = np.random.default_rng(cfg["bootstrap_seed"])
    for region in ["full", "seen", "unseen"]:
        values = units[units.region == region].pivot(index="seed", columns="method", values="prediction_error")
        for method, baseline in [("LocalRegularized", "Local"), ("FederatedLearned", "LocalRegularized")]:
            gains = 100 * (1 - values[method] / values[baseline])
            draws = rng.choice(gains, size=(20000, len(gains)), replace=True).mean(axis=1)
            low, high = np.quantile(draws, [.025, .975])
            comparisons.append(dict(region=region, method=method, baseline=baseline,
                                    improvement_pct=float(gains.mean()), ci_low=float(low),
                                    ci_high=float(high), wins=int((gains > 0).sum())))
    groups = pd.DataFrame([dict(seed=r["seed"], **g) for r in results for g in r["groups"]])
    units.to_csv(directory / "paper_review_validation_seed_summary.csv", index=False)
    summary.to_csv(directory / "paper_review_validation_summary.csv", index=False)
    pd.DataFrame(comparisons).to_csv(directory / "paper_review_validation_comparisons.csv", index=False)
    groups.drop(columns="members").to_csv(directory / "paper_review_validation_groups.csv", index=False)
    manifest = dict(status="complete_parent_verified", fleets=len(results), source_sha256=lock,
                    groups=len(groups), full_rank_groups=int(groups.full_rank.sum()),
                    maximum_parent_error_difference=max(abs(v) for r in results for v in r["parent_error_differences"].values()),
                    checkpoint_sha256={f"paper_review_validation_seed{s}.json": digest(directory / f"paper_review_validation_seed{s}.json") for s in cfg["seeds"]})
    (directory / "paper_review_validation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "source_sha256" and k != "checkpoint_sha256"}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--preflight", action="store_true")
    action.add_argument("--smoke", action="store_true", help="One full original fleet, temporary output only")
    action.add_argument("--run", action="store_true", help="Ten original fleets, resumable durable checkpoints")
    action.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()
    cfg, _, lock = preflight()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.preflight:
        print(json.dumps(dict(status="preflight_passed", fleets=len(cfg["seeds"]), locked_sources=len(lock))))
    elif args.smoke:
        with tempfile.TemporaryDirectory(prefix="lpv_review_smoke_") as temp:
            save_result(run_seed(cfg["seeds"][0]), Path(temp))
        print(json.dumps(dict(status="smoke_passed", production_results_written=False)))
    elif args.summarize_only:
        summarize(OUT)
    else:
        tasks = []
        for seed in cfg["seeds"]:
            path = OUT / f"paper_review_validation_seed{seed}.json"
            if path.exists():
                result = json.loads(path.read_text())
                if result["source_sha256"] != lock or result["status"] != "parent_verified" or result["seed"] != seed:
                    raise ValueError(f"Stale checkpoint: {path}. Move it aside explicitly before rerunning.")
            else:
                tasks.append(seed)
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(run_seed, seed) for seed in tasks]
            for future in as_completed(futures):
                save_result(future.result(), OUT)
        summarize(OUT)


if __name__ == "__main__":
    main()
