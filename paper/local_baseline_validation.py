"""Frozen development selection and matched recipient-only smoothness baseline.

--run saves execution outputs only. --audit verifies saved predictions against
fresh Jacobians and writes manuscript numbers; it never fits local models.
Original evidence is read-only. Text hashes normalize Windows line endings.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "code/src"), str(ROOT / "code/experiments")]
import numpy as np
import pandas as pd
from federated_lpv import sample_fleet
from federated_lpv.local_scheduling import curvature_factor, fit_smooth_local, select_one_se
from experiment_10a_basis_coverage_audit import collect_targets, raw_basis
from experiment_10b_oracle_complementary_gate import estimate_matrix, ridge_fit, predict, lift_prior

OUT = ROOT / "results/tables"
PREFIX = "paper_local_smoothness"
CONFIG = ROOT / "code/config/paper_local_smoothness.json"


def digest(path):
    data = path.read_bytes()
    if path.suffix in {".py", ".json", ".csv"}:
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def load(path):
    return json.loads(path.read_text())


def save(path, obj):
    # Completed JSONs only; incomplete temporary files are never resumed.
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def verify_hashes(lock):
    for name, sha in lock.items():
        if digest(ROOT / name.replace("\\", "/")) != sha:
            raise ValueError(f"Frozen input changed: {name}")


def preflight():
    cfg = load(CONFIG)
    if set(cfg["development_seeds"]) & set(cfg["primary_seeds"]):
        raise ValueError("Development and primary seeds overlap")
    review = load(OUT / "paper_review_validation_manifest.json")
    verify_hashes(review["source_sha256"])
    verify_hashes({f"results/tables/{n}": h for n, h in review["checkpoint_sha256"].items()})
    if review["status"] != "complete_parent_verified":
        raise ValueError("Complete frozen parent required")
    basis = load(ROOT / "code/config/experiment_10a.json")
    if basis["orthogonalization_grid"] != {"minimum_mps": 10.0, "maximum_mps": 30.0, "points": 401}:
        raise ValueError("Curvature factor requires the fixed 401-point 10--30 basis")
    names = ["code/config/paper_local_smoothness.json", "paper/local_baseline_validation.py",
             "code/config/experiment_10a.json", "code/config/experiment_10b.json",
             "code/experiments/experiment_10a_basis_coverage_audit.py",
             "code/experiments/experiment_10b_oracle_complementary_gate.py",
             "results/tables/experiment_10b_seed_summary.csv",
             "results/tables/paper_review_validation_manifest.json"]
    names += [p.relative_to(ROOT).as_posix() for p in (ROOT / "code/src/federated_lpv").glob("*.py")]
    names += [f"results/tables/paper_review_validation_seed{s}.json" for s in cfg["primary_seeds"]]
    return {n: digest(ROOT / n) for n in sorted(names)}


def relative(prediction, reference):
    return np.linalg.norm(prediction-reference, axis=1) / np.linalg.norm(reference, axis=1)


def development_seed(args):
    seed, lock = args
    path = OUT / f"{PREFIX}_development{seed}.json"
    if path.exists():
        result = load(path)
        if result["source_sha256"] != lock:
            raise ValueError("Stale development checkpoint")
        print(f"Development {seed}: resumed", flush=True)
        return result
    cfg, gen, basis = load(CONFIG), load(ROOT / "code/config/experiment_10b.json"), load(ROOT / "code/config/experiment_10a.json")
    clients = sample_fleet(seed)
    rng = np.random.default_rng(seed+100000)
    order = rng.permutation(len(clients))
    items, fs, ft = [], [], []
    for j, client in enumerate(clients):
        speeds = gen["coverage_blocks"][int(order[j] % 4)]
        targets = np.array([estimate_matrix(client, v, gen["samples_per_local_speed"], gen, rng) for v in speeds])
        _, reference = collect_targets(client, gen["curvature_per_m"], basis)
        items.append(dict(client=client.client_id, speeds=speeds, targets=targets.tolist(), reference=reference.tolist()))
        fs.extend(speeds); ft.extend(targets)
    global_map = ridge_fit(fs, ft, 7, basis, strength=1e-8)
    historical = []
    grid = np.array(gen["speed_grid"])
    for item in items:
        local = ridge_fit(item["speeds"], item["targets"], 3, basis, strength=1e-8)
        for method, model in [("Local", local), ("Global", global_map)]:
            err = relative(predict(model, grid, basis), np.array(item["reference"]))
            for v, e in zip(grid, err):
                historical.append(dict(seed=seed, method=method, region="seen" if v in item["speeds"] else "unseen", prediction_error=e))
    units = pd.DataFrame(historical).groupby(["seed", "region", "method"]).prediction_error.mean()
    frozen = pd.read_csv(OUT / "experiment_10b_seed_summary.csv").set_index(["seed", "region", "method"]).prediction_error
    difference = float(np.max(np.abs(units - frozen.loc[units.index])))
    if difference > cfg["development_reconstruction_tolerance"]:
        raise ValueError(f"Development calibration differs from frozen Local/Global errors: {difference}")
    factor = curvature_factor(cfg["quadrature_points"])
    scores = []
    for alpha in cfg["strength_candidates"]:
        unseen = []
        for item in items:
            model = fit_smooth_local(raw_basis(item["speeds"], 7, basis), np.array(item["targets"]), factor, alpha, cfg["numerical_ridge"])
            mask = ~np.isin(grid, item["speeds"])
            unseen.extend(relative(predict(model, grid, basis), np.array(item["reference"]))[mask])
        scores.append(dict(seed=seed, strength=alpha, unobserved_error=float(np.mean(unseen))))
    result = dict(seed=seed, status="development_verified", source_sha256=lock,
                  maximum_historical_mean_difference=difference, frozen_inputs=items, candidate_scores=scores)
    save(path, result)
    print(f"Development {seed}: reconstructed and verified", flush=True)
    return result


def freeze_selection(development, lock):
    cfg = load(CONFIG)
    if sorted(r["seed"] for r in development) != cfg["development_seeds"]:
        raise ValueError("Selection requires exactly the predeclared development fleets")
    scores = pd.DataFrame([s for r in development for s in r["candidate_scores"]])
    selected, details = select_one_se({a: scores[scores.strength == a].sort_values("seed").unobserved_error.to_numpy()
                                      for a in cfg["strength_candidates"]})
    result = dict(status="development_frozen_before_primary", selected_strength=selected,
                  selection=details, development_seeds=cfg["development_seeds"], source_sha256=lock,
                  development_sha256={f"results/tables/{PREFIX}_development{s}.json": digest(OUT / f"{PREFIX}_development{s}.json") for s in cfg["development_seeds"]})
    path = OUT / f"{PREFIX}_selection.json"
    if path.exists() and load(path) != result:
        raise ValueError("Existing frozen selection differs; never overwrite it")
    save(path, result)
    scores.to_csv(OUT / f"{PREFIX}_development_scores.csv", index=False)
    print(f"Development selection frozen: strength={selected:g}", flush=True)
    return result


def primary_seed(args):
    seed, selection = args
    cfg, basis = load(CONFIG), load(ROOT / "code/config/experiment_10a.json")
    path = OUT / f"{PREFIX}_primary{seed}.json"
    selection_sha = digest(OUT / f"{PREFIX}_selection.json")
    if path.exists():
        result = load(path)
        if result["selection_sha256"] != selection_sha or result["source_sha256"] != selection["source_sha256"]:
            raise ValueError("Stale primary checkpoint")
        print(f"Primary {seed}: resumed", flush=True)
        return result
    if seed not in cfg["primary_seeds"] or seed in selection["development_seeds"]:
        raise ValueError("Unexpected primary fleet")
    verify_hashes(selection["source_sha256"])
    verify_hashes(selection["development_sha256"])
    parent = load(OUT / f"paper_review_validation_seed{seed}.json")
    gen = load(ROOT / "code/config/experiment_10b.json")
    grid = np.array(gen["speed_grid"])
    factor = curvature_factor(cfg["quadrature_points"])
    models, rows = [], []
    for client, item in zip(sample_fleet(seed), parent["frozen_inputs"]):
        if client.client_id != item["client"]:
            raise ValueError("Frozen primary client order mismatch")
        model = fit_smooth_local(raw_basis(item["speeds"], 7, basis), np.array(item["targets"]), factor,
                                 selection["selected_strength"], cfg["numerical_ridge"])
        # Evaluation references are computed only after fitting.
        _, reference = collect_targets(client, gen["curvature_per_m"], basis)
        models.append(dict(client=client.client_id, coefficients=model.tolist()))
        for v, e in zip(grid, relative(predict(model, grid, basis), reference)):
            rows.append(dict(seed=seed, client=client.client_id, method="SmoothLocal", speed=float(v),
                             region="seen" if v in item["speeds"] else "unseen", prediction_error=float(e)))
    result = dict(seed=seed, status="matched_primary_complete", source_sha256=selection["source_sha256"],
                  selection_sha256=selection_sha, models=models, errors=rows)
    save(path, result)
    print(f"Primary {seed}: complete", flush=True)
    return result


def fleet_units(raw):
    regions = raw.groupby(["seed", "method", "region"]).prediction_error.mean().reset_index()
    full = raw.groupby(["seed", "method"]).prediction_error.mean().reset_index().assign(region="full")
    return pd.concat([regions, full], ignore_index=True).sort_values(["seed", "method", "region"])


def execution(workers):
    lock, cfg = preflight(), load(CONFIG)
    completed = OUT / f"{PREFIX}_manifest.json"
    if completed.exists():
        previous = load(completed)
        if previous["source_sha256"] != lock:
            raise ValueError("Completed execution has a different source lock")
        verify_hashes(previous["output_sha256"])
        print("Complete frozen execution verified; no refitting.", flush=True)
        return
    # Ordered phases: no primary fitting/evaluation before selection is saved.
    with ProcessPoolExecutor(max_workers=workers) as pool:
        development = list(pool.map(development_seed, [(s, lock) for s in cfg["development_seeds"]]))
    selection = freeze_selection(development, lock)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        primary = list(pool.map(primary_seed, [(s, selection) for s in cfg["primary_seeds"]]))
    raw = pd.DataFrame([e for r in primary for e in r["errors"]])
    parents = pd.DataFrame([e for s in cfg["primary_seeds"] for e in load(OUT / f"paper_review_validation_seed{s}.json")["errors"]])
    raw = pd.concat([raw, parents], ignore_index=True)
    raw.to_csv(OUT / f"{PREFIX}_errors.csv", index=False)
    fleet_units(raw).to_csv(OUT / f"{PREFIX}_seed_summary.csv", index=False)
    names = [f"{PREFIX}_development{s}.json" for s in cfg["development_seeds"]]
    names += [f"{PREFIX}_primary{s}.json" for s in cfg["primary_seeds"]]
    names += [f"{PREFIX}_{x}" for x in ["selection.json", "development_scores.csv", "errors.csv", "seed_summary.csv"]]
    save(OUT / f"{PREFIX}_manifest.json", dict(status="execution_complete", source_sha256=lock,
         output_sha256={f"results/tables/{n}": digest(OUT / n) for n in names},
         development_fleets=10, primary_fleets=10, primary_clients=300,
         operation="reconstructed_development_calibration_and_matched_primary_local_fits_no_new_fleets_or_control_rollouts"))
    print("Execution checkpoint complete. Commit/push before --audit.", flush=True)


def audit():
    cfg, basis, gen = load(CONFIG), load(ROOT / "code/config/experiment_10a.json"), load(ROOT / "code/config/experiment_10b.json")
    lock = preflight()
    manifest = load(OUT / f"{PREFIX}_manifest.json")
    if manifest["source_sha256"] != lock or manifest["status"] != "execution_complete":
        raise ValueError("Complete current execution checkpoint required")
    verify_hashes(manifest["output_sha256"])
    selection = load(OUT / f"{PREFIX}_selection.json")
    scores = pd.read_csv(OUT / f"{PREFIX}_development_scores.csv")
    chosen, _ = select_one_se({a: scores[scores.strength == a].sort_values("seed").unobserved_error.to_numpy() for a in cfg["strength_candidates"]})
    if chosen != selection["selected_strength"]:
        raise ValueError("Selection audit mismatch")
    grid, factor = np.array(gen["speed_grid"]), curvature_factor(cfg["quadrature_points"])
    reconstructed, prior_rows = [], []
    maximum_reference_difference, maximum_stationarity = 0., 0.
    for seed in cfg["primary_seeds"]:
        result = load(OUT / f"{PREFIX}_primary{seed}.json")
        parent = load(OUT / f"paper_review_validation_seed{seed}.json")
        for index, (client, item, saved) in enumerate(zip(sample_fleet(seed), parent["frozen_inputs"], result["models"])):
            theta = np.array(saved["coefficients"])
            phi, y = raw_basis(item["speeds"], 7, basis), np.array(item["targets"])
            # First-order condition in relative units; no refitting.
            terms = [phi.T@(phi@theta-y)/len(phi), chosen*factor.T@(factor@theta), cfg["numerical_ridge"]*theta]
            stationary = np.linalg.norm(sum(terms)) / max(1., sum(np.linalg.norm(t) for t in terms))
            maximum_stationarity = max(maximum_stationarity, float(stationary))
            if stationary > 1e-5:
                raise ValueError(f"Local optimum audit failed: {stationary}")
            _, reference = collect_targets(client, gen["curvature_per_m"], basis)
            errors = relative(predict(theta, grid, basis), reference)
            saved_frame = pd.DataFrame(result["errors"])
            saved_errors = saved_frame[saved_frame.client == client.client_id].sort_values("speed").prediction_error.to_numpy()
            maximum_reference_difference = max(maximum_reference_difference, float(np.max(np.abs(errors-saved_errors))))
            np.testing.assert_allclose(errors, saved_errors, rtol=0, atol=1e-10)
            # Analytical audit of the old prior-centered comparator.
            prior = lift_prior(ridge_fit(item["speeds"], y, 3, basis, strength=1e-8), basis)
            centered = np.array(parent["frozen_models"]["local_regularized"][index])
            delta = centered-prior
            alpha, eps = item["local_regularization"], 1e-12
            residual = y-phi@prior
            normal_residual = (phi.T@phi+(alpha+eps)*np.eye(7))@delta - phi.T@residual + eps*prior
            null = np.linalg.svd(phi, full_matrices=True)[2][len(phi):]
            null_error = np.max(np.abs(null@delta + eps/(alpha+eps)*(null@prior)))
            if np.linalg.norm(normal_residual) > 1e-8 or null_error > 1e-8:
                raise ValueError("Prior-centered algebra audit failed")
            prior_rows.append(dict(seed=seed, client=client.client_id,
                 observed_prior_relative_residual=float(np.linalg.norm(residual)/np.linalg.norm(y)),
                 relative_coefficient_change=float(np.linalg.norm(delta)/max(1., np.linalg.norm(prior))),
                 maximum_nullspace_identity_error=float(null_error)))
        reconstructed.extend(result["errors"])
        reconstructed.extend(parent["errors"])
    units = fleet_units(pd.DataFrame(reconstructed))
    frozen = pd.read_csv(OUT / f"{PREFIX}_seed_summary.csv")
    keys = ["seed", "method", "region"]
    np.testing.assert_allclose(units.set_index(keys).sort_index().prediction_error,
                               frozen.set_index(keys).sort_index().prediction_error, rtol=0, atol=1e-12)
    summary = units.groupby(["method", "region"]).prediction_error.agg(["mean", "std"]).reset_index()
    summary.to_csv(OUT / f"{PREFIX}_summary.csv", index=False)
    rng, comparisons = np.random.default_rng(cfg["bootstrap_seed"]), []
    for region in ["full", "seen", "unseen"]:
        pivot = units[units.region == region].pivot(index="seed", columns="method", values="prediction_error")
        gains = 100*(1-pivot.FederatedLearned/pivot.SmoothLocal)
        low, high = np.quantile(rng.choice(gains, (cfg["bootstrap_resamples"], len(gains)), replace=True).mean(axis=1), [.025, .975])
        comparisons.append(dict(region=region, method="FederatedLearned", baseline="SmoothLocal",
           improvement_pct=float(gains.mean()), ci_low=float(low), ci_high=float(high), wins=int((gains>0).sum())))
    pd.DataFrame(comparisons).to_csv(OUT / f"{PREFIX}_comparisons.csv", index=False)
    prior_frame = pd.DataFrame(prior_rows)
    prior_frame.to_csv(OUT / f"{PREFIX}_prior_audit.csv", index=False)
    numbers = {}
    for region, suffix in [("full", "Full"), ("seen", "Seen"), ("unseen", "Unseen")]:
        numbers[f"SmoothLocal{suffix}Error"] = f"{100*float(summary[(summary.method=='SmoothLocal') & (summary.region==region)]['mean'].iloc[0]):.3f}"
    for row in comparisons:
        if row["region"] in ["full", "unseen"]:
            suffix = row["region"].title()
            for key, ending in [("improvement_pct", "Gain"), ("ci_low", "Low"), ("ci_high", "High")]:
                numbers[f"SmoothLocal{suffix}{ending}"] = f"{row[key]:.2f}"
    numbers["SmoothLocalStrength"] = f"{chosen:g}"
    numbers["PriorResidualMedian"] = f"{100*prior_frame.observed_prior_relative_residual.median():.4f}"
    numbers["PriorResidualMax"] = f"{100*prior_frame.observed_prior_relative_residual.max():.4f}"
    (ROOT / "paper/local_baseline_numbers.tex").write_text("% Generated by local_baseline_validation.py --audit.\n" +
        "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in sorted(numbers.items())))
    save(OUT / f"{PREFIX}_audit.json", dict(status="saved_models_audited_without_refitting", selected_strength=chosen,
         maximum_reference_error_difference=maximum_reference_difference, maximum_relative_stationarity=maximum_stationarity,
         primary_clients=300, numerical_claims=numbers, source_sha256=lock, execution_sha256=manifest["output_sha256"],
         manifest_sha256=digest(OUT / f"{PREFIX}_manifest.json")))
    print(json.dumps(dict(selected_strength=chosen, numbers=numbers, comparisons=comparisons,
                         maximum_relative_stationarity=maximum_stationarity), indent=2))


def main():
    parser = argparse.ArgumentParser(__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--preflight", action="store_true")
    actions.add_argument("--run", action="store_true")
    actions.add_argument("--audit", action="store_true")
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    if args.preflight:
        print(f"Preflight passed: {len(preflight())} frozen sources; development321--330, primary371--380")
    elif args.run:
        execution(args.workers)
    else:
        audit()


if __name__ == "__main__":
    main()
