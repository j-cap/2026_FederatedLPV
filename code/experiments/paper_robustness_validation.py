"""Focused rank-deficiency and calibration-sensitivity validation.

Run from the repository root; no PYTHONPATH setup is needed. --smoke uses a
separate seed and temporary outputs. --run checkpoints every fleet/condition.
Push execution outputs before --audit. Historical evidence is read-only.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from multiprocessing import get_context
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "code/src"), str(ROOT / "code/experiments")]
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from federated_lpv import VehicleClient, VehicleParameters, sample_fleet
from federated_lpv.local_scheduling import curvature_factor, fit_smooth_local
from experiment_10a_basis_coverage_audit import collect_targets, equilibrium, raw_basis
from experiment_10b_oracle_complementary_gate import predict, ridge_fit, rk4_step
from experiment_10e_federated_latent_groups import fit_federated_mixture, label_free_categories

CONFIG = ROOT / "code/config/paper_robustness_validation.json"
OUT = ROOT / "results/tables"
DATA = ROOT / "results/data/paper_robustness_validation"
PREFIX = "paper_robustness_validation"
METHODS = ("RestrictedLocal", "SmoothLocal", "Global", "Federated", "RankFallback", "OracleFamily")


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(path):
    raw = path.read_bytes()
    if path.suffix in {".py", ".json", ".csv"}:
        raw = raw.replace(b"\r\n", b"\n")
    return hashlib.sha256(raw).hexdigest()


def record_digest(records):
    a = np.asarray(records, dtype="<f8", order="C")
    return hashlib.sha256(canonical(list(a.shape)) + a.tobytes()).hexdigest()


def check_hashes(lock):
    for name, sha in lock.items():
        if digest(ROOT / name) != sha:
            raise ValueError(f"Frozen source changed: {name}")


def information(speeds, basis):
    phi = raw_basis(np.asarray(speeds, dtype=float), 7, basis)
    singular = np.linalg.svd(phi, compute_uv=False)
    rank = int(np.linalg.matrix_rank(phi))
    return dict(rank=rank, full_rank=rank == 7, observations=len(phi),
                unique_speeds=sorted(set(map(float, speeds))), singular_values=singular.tolist(),
                design_condition=float(singular[0]/singular[-1]) if rank == 7 else None,
                nonzero_condition=float(singular[0]/singular[rank-1]) if rank else None)


def observed_support(seed, clients, case, cfg, parent):
    tags = label_free_categories(clients, np.random.default_rng(seed+cfg["coverage_rng_offset"]),
                                parent["coverage_blocks"])
    result = []
    for client in clients:
        original = parent["coverage_blocks"][tags[client.client_id]]
        if case["coverage"] == "one_family" and client.family == cfg["restricted_family"]:
            block = cfg["restricted_family_speeds"][:len(original)]
        elif case["coverage"] == "fleet":
            block = cfg["fleet_restriction_blocks"][tags[client.client_id]]
        else:
            block = original
        if len(block) != len(original):
            raise ValueError("Coverage manipulation changed recipient budget")
        result.append(np.asarray(block, dtype=float))
    return result


def preflight():
    cfg = load(CONFIG)
    if cfg["seeds"] != list(range(1201, 1211)) or cfg["smoke_seed"] != 1200:
        raise ValueError("The frozen ten-fleet seed set or separate smoke seed changed")
    if tuple(cfg["methods"]) != METHODS or len(cfg["cases"]) != 7:
        raise ValueError("Expected six methods and seven predeclared conditions")
    if not cfg["source_sha256"]:
        raise ValueError("Historical source lock is missing")
    check_hashes(cfg["source_sha256"])
    versions = {n: importlib.metadata.version(n) for n in cfg["package_versions"]}
    if versions != cfg["package_versions"] or f"{sys.version_info.major}.{sys.version_info.minor}" != cfg["python_minor"]:
        raise ValueError("Use the supplied federated_lpv conda environment; package/Python pins differ")
    parent = load(ROOT / "code/config/experiment_10e.json")
    basis = load(ROOT / "code/config/experiment_10a.json")
    smooth = load(OUT / "paper_local_smoothness_selection.json")
    if smooth["status"] != "development_frozen_before_primary" or smooth["selected_strength"] != 1e6:
        raise ValueError("The completed development-only Smooth Local selection is required")
    names = [c["name"] for c in cfg["cases"]]
    if len(set(names)) != 7 or names[0] != "reference":
        raise ValueError("Duplicate or missing reference condition")
    for seed in [cfg["smoke_seed"], *cfg["seeds"]]:
        clients = sample_fleet(seed)
        for case in cfg["cases"]:
            speeds = observed_support(seed, clients, case, cfg, parent)
            global_info = information(np.concatenate(speeds), basis)
            expected = 6 if case["coverage"] == "fleet" else 7
            if global_info["rank"] != expected:
                raise ValueError(f"Unexpected planned fleet coverage for {seed}/{case['name']}")
            if case["coverage"] == "one_family":
                group = np.concatenate([s for c, s in zip(clients, speeds) if c.family == cfg["restricted_family"]])
                if information(group, basis)["rank"] != 3:
                    raise ValueError("Planned restricted-family design is not rank three")
                if global_info["unique_speeds"] != list(map(float, parent["speed_grid"])):
                    raise ValueError("One-family stress must preserve the full fleet speed union")
    lock = {**cfg["source_sha256"], CONFIG.relative_to(ROOT).as_posix(): digest(CONFIG),
            Path(__file__).relative_to(ROOT).as_posix(): digest(Path(__file__))}
    return cfg, parent, basis, smooth["selected_strength"], dict(source_sha256=lock,
        environment={"python_minor": cfg["python_minor"], "packages": versions})


def cell_rngs(seed, client_index, speed_index, cfg):
    return (np.random.default_rng(np.random.SeedSequence([cfg["calibration_rng_offset"], seed, client_index, speed_index])),
            np.random.default_rng(np.random.SeedSequence([cfg["measurement_rng_offset"], seed, client_index, speed_index])))


def center_bias(seed, client_index, case, cfg):
    rng = np.random.default_rng(np.random.SeedSequence([cfg["center_rng_offset"], seed, client_index]))
    return rng.choice([-1., 1.], size=3) * np.asarray(case["center_bias_magnitude"])


def sample_records(client, speed, case, parent, rng, measurement_rng, bias, center=None):
    """Generate measured arrays only; no hidden quantities are passed to OLS."""
    center = equilibrium(client.parameters, speed, parent["curvature_per_m"]) if center is None else np.asarray(center)
    n = parent["samples_per_local_speed"]
    dx = rng.normal(size=(n, 2))*np.asarray(parent["state_excitation_std"])*case["excitation_scale"]
    du = rng.normal(size=n)*parent["steering_excitation_std"]*case["excitation_scale"]
    noise = rng.normal(size=(n, 2))*np.asarray(parent["next_state_noise_std"])*case["target_noise_scale"]
    x = center[:2]+dx
    u = center[2]+du
    next_x = np.asarray([rk4_step(client.parameters, xx, uu, speed, parent["sample_time"])
                         for xx, uu in zip(x, u)])
    observed_x = x + measurement_rng.normal(size=(n, 2))*np.asarray(case["current_state_noise_std"])
    observed_u = u + measurement_rng.normal(size=n)*case["current_input_noise_std"]
    records = np.column_stack([observed_x, observed_u, next_x+noise])
    return records, center, center+np.asarray(bias)


def estimate_records(records, assumed_center):
    """Local zero-intercept matrix regression from supplied measured arrays."""
    design = records[:, :3]-np.asarray(assumed_center)
    target = records[:, 3:]-np.asarray(assumed_center)[:2]
    coefficient, _, rank, singular = np.linalg.lstsq(design, target, rcond=None)
    if rank != 3:
        raise RuntimeError("Local transition regressor is rank deficient")
    a, b = coefficient[:2].T, coefficient[2:].T
    return np.r_[a.ravel(), b.ravel()], dict(regressor_rank=int(rank),
        regressor_condition=float(singular[0]/singular[-1]), regressor_singular_values=singular.tolist(),
        residual_rms=float(np.sqrt(np.mean((target-design@coefficient)**2))))


def fit_observed(speeds, targets, seed, parent, basis, smooth_strength):
    """Learned methods receive no physical parameters, references, or labels."""
    selected, candidates, global_map, _ = fit_federated_mixture(speeds, targets,
        parent["candidate_clusters"], parent["mixture_restarts"], parent["mixture_iterations"],
        parent["minimum_cluster_size"], seed, basis, strength=parent["ridge_strength"])
    group_info = []
    for group in range(selected["k"]):
        indices = np.flatnonzero(selected["assignments"] == group)
        group_info.append(dict(group=group, clients=len(indices),
            **information(np.concatenate([speeds[i] for i in indices]), basis)))
    factor = curvature_factor()
    local, smooth = [], []
    for observed, y in zip(speeds, targets):
        local.append(ridge_fit(observed, y, 3, basis, strength=1e-8).tolist())
        smooth.append(fit_smooth_local(raw_basis(observed, 7, basis), y, factor,
                                       smooth_strength, epsilon=1e-16).tolist())
    assignments = selected["assignments"].tolist()
    return dict(selected_k=int(selected["k"]), selected_bic=float(selected["bic"]),
        selected_sse=float(selected["sse"]), candidates=candidates.to_dict("records"),
        global_information=information(np.concatenate(speeds), basis), groups=group_info,
        models=dict(RestrictedLocal=local, SmoothLocal=smooth, Global=global_map.tolist(),
                    Federated=[m.tolist() for m in selected["models"]], assignments=assignments,
                    fallback_flags=[not group_info[g]["full_rank"] for g in assignments]))


def add_oracle(fitted, clients, speeds, targets, parent, basis):
    """Separate known-family diagnostic after label-free model selection."""
    names = sorted({c.family for c in clients})
    oracle, info = [], []
    for group, family in enumerate(names):
        indices = [i for i, c in enumerate(clients) if c.family == family]
        observed = np.concatenate([speeds[i] for i in indices])
        y = np.vstack([targets[i] for i in indices])
        oracle.append(ridge_fit(observed, y, 7, basis, strength=parent["ridge_strength"]).tolist())
        info.append(dict(group=group, family=family, clients=len(indices), **information(observed, basis)))
    fitted["models"]["OracleFamily"] = oracle
    fitted["models"]["oracle_assignments"] = [names.index(c.family) for c in clients]
    fitted["oracle_groups"] = info


def client_model(models, method, index):
    if method in {"RestrictedLocal", "SmoothLocal"}:
        return np.asarray(models[method][index])
    if method == "Global" or (method == "RankFallback" and models["fallback_flags"][index]):
        return np.asarray(models["Global"])
    if method == "OracleFamily":
        return np.asarray(models[method][models["oracle_assignments"][index]])
    return np.asarray(models["Federated"][models["assignments"][index]])


def model_errors(models, reference, grid, basis):
    result = {}
    for method in METHODS:
        result[method] = []
        for i, truth in enumerate(reference):
            estimate = predict(client_model(models, method, i), grid, basis)
            error = np.linalg.norm(estimate-truth, axis=1)/np.linalg.norm(truth, axis=1)
            if not np.isfinite(error).all():
                raise RuntimeError(f"Nonfinite predictions for {method}")
            result[method].append(error.tolist())
    return result


def run_job(seed, case_name):
    cfg, parent, basis, smooth, provenance = preflight()
    case = next(c for c in cfg["cases"] if c["name"] == case_name)
    clients = sample_fleet(seed)
    speeds = observed_support(seed, clients, case, cfg, parent)
    inputs, targets = [], []
    with threadpool_limits(limits=1):
        for i, (client, observed) in enumerate(zip(clients, speeds)):
            ys, cells = [], []
            bias = center_bias(seed, i, case, cfg)
            for j, speed in enumerate(observed):
                rng, mrng = cell_rngs(seed, i, j, cfg)
                records, center, assumed = sample_records(client, speed, case, parent, rng, mrng, bias)
                y, diag = estimate_records(records, assumed)
                ys.append(y)
                cells.append(dict(speed=float(speed), true_center=center.tolist(), assumed_center=assumed.tolist(),
                    record_sha256=record_digest(records), records=len(records), **diag))
            targets.append(np.asarray(ys))
            inputs.append(dict(client=client.client_id, family=client.family, parameters=asdict(client.parameters),
                               speeds=observed.tolist(), targets=np.asarray(ys).tolist(), cells=cells))
        fitted = fit_observed(speeds, targets, seed, parent, basis, smooth)
        add_oracle(fitted, clients, speeds, targets, parent, basis)
        # Exact references enter only after all finite-data models are fitted.
        reference = np.asarray([collect_targets(c, parent["curvature_per_m"], basis)[1] for c in clients])
        errors = model_errors(fitted["models"], reference, np.asarray(parent["speed_grid"]), basis)
    return dict(status="execution_complete", seed=seed, case=case_name, provenance=provenance,
        case_configuration=case, frozen_inputs=inputs, reference=reference.tolist(), errors=errors, **fitted)


def checkpoint_path(directory, seed, case):
    return directory / f"{PREFIX}_seed{seed}_{case}.json"


def save_checkpoint(result, directory):
    directory.mkdir(parents=True, exist_ok=True)
    path = checkpoint_path(directory, result["seed"], result["case"])
    envelope = dict(payload_sha256=hashlib.sha256(canonical(result)).hexdigest(), payload=result)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(envelope, indent=2, allow_nan=False)+"\n", encoding="utf-8", newline="\n")
    temporary.replace(path)
    return path


def read_checkpoint(path, seed, case, provenance):
    envelope = load(path)
    result = envelope["payload"]
    if hashlib.sha256(canonical(result)).hexdigest() != envelope["payload_sha256"]:
        raise ValueError(f"Checkpoint payload corrupted: {path.name}")
    if result["provenance"] != provenance or result["seed"] != seed or result["case"] != case or result["status"] != "execution_complete":
        raise ValueError(f"Stale/incompatible checkpoint: {path.name}")
    if len(result["frozen_inputs"]) != 30 or np.asarray(result["reference"]).shape != (30, 11, 6):
        raise ValueError("Incomplete frozen fleet inputs/references")
    if set(result["errors"]) != set(METHODS) or any(np.asarray(e).shape != (30, 11) or not np.isfinite(e).all() for e in result["errors"].values()):
        raise ValueError("Incomplete or nonfinite error arrays")
    return result


def jobs(cfg):
    return [(s, c["name"]) for s in cfg["seeds"] for c in cfg["cases"]]


def scan(directory, cfg, provenance):
    completed, missing = [], []
    for seed, case in jobs(cfg):
        path = checkpoint_path(directory, seed, case)
        if path.exists():
            completed.append(read_checkpoint(path, seed, case, provenance))
        else:
            missing.append((seed, case))
    return completed, missing


def progress(result):
    return dict(seed=result["seed"], case=result["case"], selected_k=result["selected_k"],
        deficient_selected_groups=sum(not g["full_rank"] for g in result["groups"]),
        global_rank=result["global_information"]["rank"], status="checkpoint_saved")


def summarize(directory, cfg, parent, provenance, announce=True):
    results, missing = scan(directory, cfg, provenance)
    if missing:
        raise ValueError(f"Complete {len(jobs(cfg))}-job matrix required; missing {len(missing)}")
    grid = np.asarray(parent["speed_grid"])
    rows, group_rows, diagnostics = [], [], []
    for r in results:
        observed = np.asarray([np.isin(grid, item["speeds"]) for item in r["frozen_inputs"]])
        fleet_unseen = np.broadcast_to(~np.isin(grid, r["global_information"]["unique_speeds"]), observed.shape)
        for method, values in r["errors"].items():
            values = np.asarray(values)
            for region, mask in [("full", np.ones_like(observed)), ("seen", observed), ("unseen", ~observed), ("fleet_unseen", fleet_unseen)]:
                if mask.any():
                    rows.append(dict(seed=r["seed"], case=r["case"], method=method, region=region,
                                     prediction_error=float(values[mask].mean()), eligible_pairs=int(mask.sum())))
        for kind, groups in [("learned", r["groups"]), ("oracle", r["oracle_groups"]), ("global", [r["global_information"]])]:
            group_rows.extend(dict(seed=r["seed"], case=r["case"], partition=kind, **g) for g in groups)
        diagnostics.append(dict(seed=r["seed"], case=r["case"], selected_k=r["selected_k"],
            deficient_selected_groups=sum(not g["full_rank"] for g in r["groups"]),
            selected_groups=len(r["groups"]), global_rank=r["global_information"]["rank"],
            fallback_clients=sum(r["models"]["fallback_flags"]),
            fallback_global_deficient=not r["global_information"]["full_rank"]))
    units = pd.DataFrame(rows)
    units.to_csv(directory / f"{PREFIX}_seed_summary.csv", index=False)
    units.groupby(["case", "method", "region"]).prediction_error.agg(["mean", "std"]).reset_index().to_csv(directory / f"{PREFIX}_summary.csv", index=False)
    pd.DataFrame(group_rows).to_csv(directory / f"{PREFIX}_groups.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(directory / f"{PREFIX}_diagnostics.csv", index=False)
    comparisons = []
    rng = np.random.default_rng(cfg["bootstrap_seed"])
    for (case, region), view in units.groupby(["case", "region"], sort=True):
        if region not in {"full", "unseen"}:
            continue
        pivot = view.pivot(index="seed", columns="method", values="prediction_error").sort_index()
        for method, baseline in [("Federated", "Global"), ("Federated", "SmoothLocal"), ("RankFallback", "Federated")]:
            gains = 100*(1-pivot[method]/pivot[baseline])
            draws = rng.choice(gains.to_numpy(), (cfg["bootstrap_resamples"], len(gains)), replace=True).mean(axis=1)
            low, high = np.quantile(draws, [.025, .975])
            comparisons.append(dict(case=case, region=region, method=method, baseline=baseline,
                paired_gain_pct=float(gains.mean()), ci_low=float(low), ci_high=float(high), wins=int((gains>0).sum())))
    pd.DataFrame(comparisons).to_csv(directory / f"{PREFIX}_comparisons.csv", index=False)
    # Paired changes to the reference condition have the same local-pair count.
    delta = units[units.region.isin(["full", "unseen"])].pivot(index=["seed", "method", "region"], columns="case", values="prediction_error")
    for case in sorted(set(delta.columns)-{"reference"}):
        delta[case] = 100*(delta[case]-delta.reference)
    changes = delta.drop(columns="reference").reset_index()
    changes.to_csv(directory / f"{PREFIX}_sensitivity_deltas.csv", index=False)
    changes = changes.melt(id_vars=["seed", "method", "region"], var_name="case", value_name="change_pp")
    sensitivity = []
    for (case, method, region), view in changes.groupby(["case", "method", "region"], sort=True):
        values = view.sort_values("seed").change_pp.to_numpy()
        draws = rng.choice(values, (cfg["bootstrap_resamples"], len(values)), replace=True).mean(axis=1)
        low, high = np.quantile(draws, [.025, .975])
        sensitivity.append(dict(case=case, method=method, region=region, mean_change_pp=float(values.mean()),
            ci_low_pp=float(low), ci_high_pp=float(high), deteriorating_fleets=int((values>0).sum())))
    pd.DataFrame(sensitivity).to_csv(directory / f"{PREFIX}_sensitivity_comparisons.csv", index=False)
    outputs = [checkpoint_path(directory, s, c) for s, c in jobs(cfg)]
    outputs += [directory/f"{PREFIX}_{name}.csv" for name in ["seed_summary", "summary", "groups", "diagnostics", "comparisons", "sensitivity_deltas", "sensitivity_comparisons"]]
    manifest = dict(status="complete_execution", fleets=len(cfg["seeds"]), jobs=len(results), provenance=provenance,
        checkpoint_sha256={p.name: digest(p) for p in outputs},
        scope="controlled_matrix_error_and_rank_stress_no_controller_rollouts_no_adaptive_tuning")
    (directory/f"{PREFIX}_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8", newline="\n")
    if announce:
        print(json.dumps(dict(status="complete_execution", fleets=len(cfg["seeds"]), jobs=len(results),
                              next_step="commit_and_push_execution_outputs_before_audit")), flush=True)


def restore_job(result, data_directory, cfg, parent):
    """Recreate ignored measured records, using frozen centers; no OLS/fitting."""
    archive = {}
    for i, item in enumerate(result["frozen_inputs"]):
        client = VehicleClient(item["client"], item["family"], VehicleParameters(**item["parameters"]))
        bias = center_bias(result["seed"], i, result["case_configuration"], cfg)
        for j, cell in enumerate(item["cells"]):
            rng, mrng = cell_rngs(result["seed"], i, j, cfg)
            records, _, assumed = sample_records(client, cell["speed"], result["case_configuration"],
                parent, rng, mrng, bias, center=cell["true_center"])
            if record_digest(records) != cell["record_sha256"] or not np.array_equal(assumed, cell["assumed_center"]):
                raise ValueError("Regenerated record/centering hash mismatch")
            archive[f"{item['client']}_speed{j}"] = records
    data_directory.mkdir(parents=True, exist_ok=True)
    path = data_directory / f"seed{result['seed']}_{result['case']}.npz"
    np.savez_compressed(path, **archive)
    # Archive timestamps may differ; each underlying float64 array is checked.
    with np.load(path, allow_pickle=False) as restored:
        for name, records in archive.items():
            if record_digest(restored[name]) != record_digest(records):
                raise ValueError("Restored archive array differs")
    return len(archive)


def audit_result(result, reference, parent, basis, cfg):
    np.testing.assert_allclose(result["reference"], reference, rtol=0, atol=cfg["reference_audit_tolerance"])
    actual = model_errors(result["models"], reference, np.asarray(parent["speed_grid"]), basis)
    for method in METHODS:
        np.testing.assert_allclose(actual[method], result["errors"][method], rtol=0, atol=cfg["reference_audit_tolerance"])
    inputs = result["frozen_inputs"]
    models = result["models"]
    maximum = 0.
    all_speeds = np.concatenate([item["speeds"] for item in inputs])
    all_y = np.vstack([item["targets"] for item in inputs])
    checks = [(raw_basis(all_speeds, 7, basis), all_y, np.asarray(models["Global"]),
               parent["ridge_strength"]+1e-12, None)]
    factor = curvature_factor()
    for i, item in enumerate(inputs):
        checks.append((raw_basis(item["speeds"], 3, basis), np.asarray(item["targets"]),
                       np.asarray(models["RestrictedLocal"][i]), 1e-8+1e-12, None))
        checks.append((raw_basis(item["speeds"], 7, basis), np.asarray(item["targets"]),
                       np.asarray(models["SmoothLocal"][i]), 1e-16, factor))
    for phi, y, theta, ridge, curvature in checks:
        gradient = phi.T@(phi@theta-y)
        if curvature is not None:
            gradient = gradient/len(phi)+1e6*curvature.T@(curvature@theta)
        residual = gradient+ridge*theta
        relative = float(np.linalg.norm(residual)/max(1., np.linalg.norm(phi.T@y)))
        maximum = max(maximum, relative)
        if relative > cfg["stationarity_tolerance"]:
            raise ValueError("Saved global/local coefficients fail stationarity")
    for kind, assignment_key in [("Federated", "assignments"), ("OracleFamily", "oracle_assignments")]:
        saved_info = result["groups"] if kind == "Federated" else result["oracle_groups"]
        for g, theta in enumerate(models[kind]):
            indices = [i for i, a in enumerate(models[assignment_key]) if a == g]
            speeds = np.concatenate([inputs[i]["speeds"] for i in indices])
            phi = raw_basis(speeds, 7, basis)
            y = np.vstack([inputs[i]["targets"] for i in indices])
            theta = np.asarray(theta)
            residual = phi.T@(phi@theta-y)+(parent["ridge_strength"]+1e-12)*theta
            relative = float(np.linalg.norm(residual)/max(1., np.linalg.norm(phi.T@y)))
            maximum = max(maximum, relative)
            if relative > cfg["stationarity_tolerance"]:
                raise ValueError("Saved grouped coefficients fail stationarity")
            if information(speeds, basis)["rank"] != saved_info[g]["rank"]:
                raise ValueError("Saved group scheduling rank differs")
    expected_flags = [not result["groups"][g]["full_rank"] for g in models["assignments"]]
    if expected_flags != models["fallback_flags"]:
        raise ValueError("Rank-only fallback rule changed")
    if information(all_speeds, basis)["rank"] != result["global_information"]["rank"]:
        raise ValueError("Saved Global scheduling rank differs")
    return maximum


def audit(directory, cfg, parent, basis, provenance):
    results, missing = scan(directory, cfg, provenance)
    if missing:
        raise ValueError("Audit requires the complete execution checkpoint")
    manifest_path = directory/f"{PREFIX}_manifest.json"
    manifest = load(manifest_path)
    if manifest["provenance"] != provenance or manifest["status"] != "complete_execution":
        raise ValueError("Execution manifest differs")
    for name, sha in manifest["checkpoint_sha256"].items():
        if digest(directory/name) != sha:
            raise ValueError(f"Frozen execution output changed: {name}")
    maximum = 0.
    with threadpool_limits(limits=1):
        for seed in cfg["seeds"]:
            reference = np.asarray([collect_targets(c, parent["curvature_per_m"], basis)[1] for c in sample_fleet(seed)])
            for result in [r for r in results if r["seed"] == seed]:
                maximum = max(maximum, audit_result(result, reference, parent, basis, cfg))
    record = dict(status="audited_saved_models_no_refitting", jobs=len(results),
        maximum_grouped_stationarity_residual=maximum, provenance=provenance,
        execution_manifest_sha256=digest(manifest_path))
    (directory/f"{PREFIX}_audit.json").write_text(json.dumps(record, indent=2)+"\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: v for k, v in record.items() if k != "provenance"}))


def smoke(cfg, parent, basis, provenance, workers):
    with tempfile.TemporaryDirectory(prefix="lpv_robustness_smoke_") as temporary:
        directory = Path(temporary)
        # Spawn is also the Windows worker start method; exercise it here.
        with ProcessPoolExecutor(max_workers=min(workers, 2), mp_context=get_context("spawn")) as pool:
            futures = [pool.submit(run_job, cfg["smoke_seed"], case["name"]) for case in cfg["cases"]]
            for future in as_completed(futures):
                result = future.result()
                path = save_checkpoint(result, directory)
                saved = read_checkpoint(path, result["seed"], result["case"], provenance)
                audit_result(saved, np.asarray(saved["reference"]), parent, basis, cfg)
                restore_job(saved, directory/"records", cfg, parent)
                print(json.dumps(progress(saved)), flush=True)
        # Exercise real summary and independent-reference audit paths only in
        # the temporary smoke directory. This one-fleet table is not evidence.
        smoke_cfg = {**cfg, "seeds": [cfg["smoke_seed"]], "bootstrap_resamples": 100}
        summarize(directory, smoke_cfg, parent, provenance, announce=False)
        audit(directory, smoke_cfg, parent, basis, provenance)
    print(json.dumps(dict(status="smoke_passed", seed=cfg["smoke_seed"], cases=7,
                          production_results_written=False)), flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    for name in ["preflight", "smoke", "run", "status", "summarize-only", "audit", "restore-data"]:
        action.add_argument("--"+name, action="store_true")
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--seed", type=int, help="Optional production job/restore filter")
    parser.add_argument("--case", help="Optional production condition/restore filter")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    cfg, parent, basis, _, provenance = preflight()
    if args.seed is not None and args.seed not in cfg["seeds"]:
        parser.error("seed must be one of the ten frozen production seeds")
    if args.case is not None and args.case not in {c["name"] for c in cfg["cases"]}:
        parser.error("unknown condition")
    if (args.seed is not None or args.case is not None) and not (args.run or args.restore_data):
        parser.error("filters apply only to --run or --restore-data")
    if args.preflight:
        print(json.dumps(dict(status="preflight_passed", fleets=10, jobs=70,
                              cases=[c["name"] for c in cfg["cases"]], threads_per_worker=1)))
        return
    if args.smoke:
        smoke(cfg, parent, basis, provenance, args.workers)
        return
    completed, missing = scan(OUT, cfg, provenance)
    if args.status:
        print(json.dumps(dict(completed=len(completed), total=70, missing=len(missing),
             per_case={c["name"]: sum(r["case"] == c["name"] for r in completed) for c in cfg["cases"]})))
    elif args.summarize_only:
        summarize(OUT, cfg, parent, provenance)
    elif args.audit:
        audit(OUT, cfg, parent, basis, provenance)
    elif args.restore_data:
        selected = [r for r in completed if (args.seed is None or r["seed"] == args.seed) and (args.case is None or r["case"] == args.case)]
        if not selected or (args.seed is None and args.case is None and missing):
            raise ValueError("Requested frozen checkpoints are incomplete")
        for r in selected:
            before = digest(checkpoint_path(OUT, r["seed"], r["case"]))
            with threadpool_limits(limits=1):
                count = restore_job(r, DATA, cfg, parent)
            if before != digest(checkpoint_path(OUT, r["seed"], r["case"])):
                raise ValueError("Restoration modified a frozen checkpoint")
            print(json.dumps(dict(seed=r["seed"], case=r["case"], record_cells_verified=count)))
    else:
        tasks = [(s, c) for s, c in missing if (args.seed is None or s == args.seed) and (args.case is None or c == args.case)]
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn")) as pool:
            futures = [pool.submit(run_job, s, c) for s, c in tasks]
            for future in as_completed(futures):
                result = future.result()
                save_checkpoint(result, OUT)
                print(json.dumps(progress(result)), flush=True)
        _, remaining = scan(OUT, cfg, provenance)
        if remaining:
            print(json.dumps(dict(status="partial_execution", remaining_jobs=len(remaining))))
        else:
            summarize(OUT, cfg, parent, provenance)


if __name__ == "__main__":
    main()
