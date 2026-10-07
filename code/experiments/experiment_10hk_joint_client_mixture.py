"""10H-K: label-blind K=1/K=2 plants and frozen causal client memberships."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from experiment_10hf_output_identifiability import measurement_covariance
from experiment_10hh_single_model_repair import dataset_hash, measured_datasets
from experiment_10hi_physically_coupled_fit import make_coordinates

from federated_lpv.client_mixture import (
    ClientLikelihood,
    ClientMixture,
    calibrate_memberships,
    mixture_audit,
    mixture_forecasts,
    optimize_mixture,
    project_split,
)
from federated_lpv.innovation_likelihood import (
    PARAMETER_NAMES,
    InnovationLikelihood,
    MeasuredDataset,
    information_scale,
    physical_coupling_diagnostic,
)
from federated_lpv.output_validation import residual_correlations
from federated_lpv.physical_coupled_fit import LOG_MAP, PHYSICAL_NAMES, CoupledLikelihood

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/tables"
PREFIX = "experiment_10hk"
CONFIG = ROOT / "code/config/experiment_10hk.json"
TABLES = (
    "runs",
    "restarts",
    "phases",
    "initials",
    "parameters",
    "physical_parameters",
    "memberships",
    "forecasts",
    "residuals",
    "correlations",
    "gradient_checks",
    "splits",
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes(seed, scale):
    paths = [
        CONFIG,
        Path(__file__),
        ROOT / "docs/experiment_10hk_protocol.md",
        ROOT / "code/tests/test_experiment_10hk.py",
        *[
            ROOT / f"code/src/federated_lpv/{name}.py"
            for name in (
                "client_mixture",
                "physical_coupled_fit",
                "innovation_likelihood",
                "output_validation",
            )
        ],
        *[
            ROOT / f"code/experiments/{name}.py"
            for name in (
                "experiment_10hh_single_model_repair",
                "experiment_10hi_physically_coupled_fit",
                "experiment_10hf_output_identifiability",
                "experiment_10ha_tire_force_observer",
                "experiment_10h_higher_order_lpv_gate",
            )
        ],
        *[ROOT / f"code/config/experiment_{name}.json" for name in ("10h", "10ha", "10hf")],
        *[
            OUT / f"experiment_10hj_seed{seed}_Q{scale:g}_{name}"
            for name in ("complete.json", "runs.csv", "physical_parameters.csv", "restarts.csv")
        ],
    ]
    return {str(p.relative_to(ROOT)): sha(p) for p in paths}


def load_configuration():
    cfg = json.loads(CONFIG.read_text())
    if set(cfg["development_seeds"]) & set(cfg["reserved_confirmation_seeds"]):
        raise ValueError("Development and confirmation overlap")
    inherited = {
        name: json.loads((ROOT / f"code/config/experiment_{name}.json").read_text())
        for name in ("10h", "10ha", "10hf")
    }
    return cfg, inherited


def job_prefix(seed, scale):
    return f"{PREFIX}_seed{seed}_Q{scale:g}"


def archived_baseline(seed, scale, hashes):
    prefix = f"experiment_10hj_seed{seed}_Q{scale:g}"
    manifest = json.loads((OUT / f"{prefix}_complete.json").read_text())
    if hashes != manifest["measured_array_sha256"]:
        raise RuntimeError("Measured arrays differ from 10H-J")
    for name in ("runs", "physical_parameters", "restarts"):
        path = OUT / f"{prefix}_{name}.csv"
        if sha(path) != manifest["output_sha256"][path.name]:
            raise RuntimeError(f"Archived baseline hash mismatch: {path}")
    frame = pd.read_csv(OUT / f"{prefix}_physical_parameters.csv")
    frame = frame[frame.box == "widest"].set_index("parameter").loc[list(PHYSICAL_NAMES)]
    eta = np.log(frame.fitted.to_numpy() / frame.nominal.to_numpy())
    runs = pd.read_csv(OUT / f"{prefix}_runs.csv")
    return eta, runs[runs.box == "widest"].iloc[0].to_dict()


def gradient_audit(evaluator, initial, cfg):
    step = cfg["gradient_check_log_step"]
    analytic = evaluator.value_gradient(initial)[1]
    numerical = np.array(
        [
            (
                evaluator.value(initial + np.eye(17)[j] * step)
                - evaluator.value(initial - np.eye(17)[j] * step)
            )
            / (2 * step)
            for j in range(17)
        ]
    )
    relative = np.linalg.norm(analytic - numerical) / max(np.linalg.norm(numerical), 1e-8)
    return pd.DataFrame(
        {
            "index": np.arange(17),
            "analytic": analytic,
            "central_difference": numerical,
            "norm_relative_error": relative,
        }
    )


def partition_ari(left, right):
    """Adjusted Rand agreement between two learned partitions, without labels."""
    table = np.zeros((2, 2), dtype=int)
    np.add.at(table, (left, right), 1)

    def choose(a):
        return np.sum(a * (a - 1) / 2)

    total = len(left) * (len(left) - 1) / 2
    a, b = choose(table.sum(axis=0)), choose(table.sum(axis=1))
    expected = a * b / total
    denominator = (a + b) / 2 - expected
    return float((choose(table) - expected) / denominator) if denominator else 1.0


def scalar_audit(audit):
    return {key: value for key, value in audit.items() if key != "face_basis"}


def residual_stats(whitened, commands, maximum_lag):
    flat = whitened.reshape(-1, 3)
    eigenvalues = np.linalg.eigvalsh(np.cov(flat, rowvar=False, bias=True))
    correlations = residual_correlations(whitened, commands, maximum_lag)
    stats = {
        "normalized_innovation_squared_mean": float(np.mean(flat**2)),
        "whitened_covariance_minimum_eigenvalue": float(eigenvalues[0]),
        "whitened_covariance_maximum_eigenvalue": float(eigenvalues[-1]),
        "maximum_absolute_whitened_mean": float(np.max(np.abs(flat.mean(axis=0)))),
    }
    for centering in ("pooled", "within_record"):
        for kind in ("autocorrelation", "past_input"):
            stats[f"maximum_absolute_{centering}_{kind}"] = max(
                abs(row["correlation"])
                for row in correlations
                if row["centering"] == centering and row["kind"] == kind
            )
    return stats, correlations


def selected_curvature(evaluator, fitted, audit, cfg):
    step = cfg["hessian_log_step"]
    hessian = np.column_stack(
        [
            (
                evaluator.value_gradient(fitted + np.eye(17)[j] * step)[1]
                - evaluator.value_gradient(fitted - np.eye(17)[j] * step)[1]
            )
            / (2 * step)
            for j in range(17)
        ]
    )
    asymmetry = np.linalg.norm(hessian - hessian.T) / max(np.linalg.norm(hessian), 1e-12)
    hessian = (hessian + hessian.T) / 2
    face = audit["face_basis"]
    return {
        "minimum_full_hessian_eigenvalue": float(np.linalg.eigvalsh(hessian)[0]),
        "minimum_face_hessian_eigenvalue": float(np.linalg.eigvalsh(face.T @ hessian @ face)[0]),
        "hessian_asymmetry": float(asymmetry),
        "face_dimension": face.shape[1],
    }


def evaluate_seed(seed, scale):
    cfg, inherited = load_configuration()
    if seed not in cfg["development_seeds"] or scale not in cfg["process_noise_scales"]:
        raise ValueError("Only declared opened development jobs may execute")
    started = time.perf_counter()
    sources = source_hashes(seed, scale)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    data, split = measured_datasets(seed, cfg, inherited)
    hashes = {name: dataset_hash(records) for name, records in data.items()}
    global_eta, archived = archived_baseline(seed, scale, hashes)
    coordinates = make_coordinates(cfg, inherited["10hf"])
    dt = inherited["10h"]["sample_time"]
    q = np.diag(np.asarray(inherited["10ha"]["base_process_noise_diagonal"]) * scale)
    r = measurement_covariance(inherited["10ha"])
    train_positions = split.loc[split.split == "train", "client_position"].to_numpy()
    test_positions = split.loc[split.split == "heldout", "client_position"].to_numpy()
    client = ClientLikelihood(data["train"], train_positions, dt, q, r, coordinates)
    evaluator = ClientMixture(client)
    baseline_value = 2 * client.evaluate(global_eta).sum() / client.sample_count
    if abs(baseline_value - archived["train_objective"]) > 1e-10:
        raise RuntimeError("Client-level likelihood does not reproduce the K=1 baseline")
    global_evaluator = CoupledLikelihood(InnovationLikelihood(data["train"], dt, q, r), coordinates)
    scale_vector = information_scale(
        global_evaluator.evaluate(np.zeros(8), information=True)["information"],
        cfg["information_scale_minimum"],
        cfg["information_scale_maximum"],
    )
    frames = {name: [] for name in TABLES}
    rng = np.random.default_rng(seed + 2_700_000)
    initials = [
        project_split(global_eta, rng.normal(size=8), cfg["split_log_amplitude"], coordinates)
        for _ in range(cfg["restart_count"])
    ]
    frames["gradient_checks"].append(gradient_audit(evaluator, initials[0], cfg))
    estimates, restarts = [], []
    for restart, initial in enumerate(initials):
        fitted, result, phases, calls, invalid, audit = optimize_mixture(
            initial, evaluator, coordinates, cfg, scale_vector
        )
        final = evaluator.evaluate(fitted)
        estimates.append(fitted)
        restarts.append(
            {
                "restart": restart,
                "objective": final["objective"],
                "success": bool(result.success),
                "iterations": int(result.nit),
                "evaluations": calls,
                "invalid_evaluations": invalid,
                "message": str(result.message),
                **scalar_audit(audit),
                **{f"value_{j}": float(value) for j, value in enumerate(fitted)},
            }
        )
        frames["initials"].append(
            pd.DataFrame({"restart": restart, "index": np.arange(17), "value": initial})
        )
        frames["phases"].append(pd.DataFrame(phases).assign(restart=restart))
        print(
            f"10H-K seed={seed} Q={scale:g} restart={restart} L={final['objective']:.7f} "
            f"KKT={audit['kkt_residual']:.2g} pi={fitted[16]:.3f} success={result.success}",
            flush=True,
        )
    best = int(np.argmin([row["objective"] for row in restarts]))
    fitted = estimates[best]
    use_reference = evaluator.value(fitted) > baseline_value + 1e-10
    if use_reference:
        fitted = np.r_[global_eta, global_eta, 0.5]
    final = evaluator.evaluate(fitted, gradient=True)
    selected_audit = mixture_audit(fitted, final["gradient"], coordinates, cfg)
    weights = final["memberships"]
    aligned = []
    for index, estimate in enumerate(estimates):
        plants = estimate[:16].reshape(2, 8)
        target = fitted[:16].reshape(2, 8)
        swap = np.sum((plants[::-1] - target) ** 2) < np.sum((plants - target) ** 2)
        aligned.append(plants[::-1] if swap else plants)
        restart_weights = evaluator.evaluate(estimate)["memberships"]
        restarts[index]["alignment_swapped"] = bool(swap)
        restarts[index]["partition_ari_to_selected"] = partition_ari(
            np.argmax(restart_weights, axis=1), np.argmax(weights, axis=1)
        )
        restarts[index]["membership_mae_to_selected"] = float(
            np.mean(np.abs((restart_weights[:, ::-1] if swap else restart_weights) - weights))
        )
        frames["memberships"].append(
            pd.DataFrame(
                {
                    "split": "train",
                    "restart": index,
                    "client_position": client.clients,
                    "weight_0": restart_weights[:, 0],
                    "weight_1": restart_weights[:, 1],
                    "selected_component": np.argmax(restart_weights, axis=1),
                }
            )
        )
    aligned = np.exp(aligned)
    cv = aligned.std(axis=0) / aligned.mean(axis=0)
    occupancy = np.bincount(np.argmax(weights, axis=1), minlength=2)
    run = {
        "component_count": 2,
        "train_objective": final["objective"],
        "global_train_objective": baseline_value,
        "train_gain_pct": 100 * (baseline_value - final["objective"]) / abs(baseline_value),
        "best_restart": best,
        "collapsed_reference_selected": use_reference,
        "all_restart_success": all(row["success"] for row in restarts),
        "maximum_restart_kkt_residual": max(row["kkt_residual"] for row in restarts),
        "maximum_restart_constraint_violation": max(
            row["constraint_violation"] for row in restarts
        ),
        "invalid_evaluations": sum(row["invalid_evaluations"] for row in restarts),
        "maximum_aligned_parameter_cv": float(cv.max()),
        "restart_objective_spread_pct": 100
        * np.ptp([row["objective"] for row in restarts])
        / abs(final["objective"]),
        "minimum_partition_ari_to_selected": min(
            row["partition_ari_to_selected"] for row in restarts
        ),
        "mixing_weight_0": float(fitted[16]),
        "minimum_hard_training_occupancy": int(occupancy.min()),
        "minimum_effective_training_occupancy": float(weights.sum(axis=0).min()),
        "plant_log_distance": float(np.linalg.norm(fitted[:8] - fitted[8:16])),
        "maximum_gradient_relative_error": float(
            frames["gradient_checks"][0].norm_relative_error.max()
        ),
        **scalar_audit(selected_audit),
        **selected_curvature(evaluator, fitted, selected_audit, cfg),
    }
    frames["restarts"].append(pd.DataFrame(restarts))
    frames["runs"].append(
        pd.DataFrame(
            [
                run,
                {
                    "component_count": 1,
                    "train_objective": baseline_value,
                    "kkt_residual": archived["kkt_residual"],
                    "constraint_violation": archived["constraint_violation"],
                    "active_coefficient_bounds": archived["active_coefficient_bounds"],
                    "maximum_restart_kkt_residual": archived["maximum_restart_kkt_residual"],
                    "maximum_restart_constraint_violation": archived[
                        "maximum_restart_constraint_violation"
                    ],
                },
            ]
        )
    )
    models = {
        "K1": ([coordinates.effective(global_eta)], np.ones(1)),
        "K2": ([coordinates.effective(eta) for eta in fitted[:16].reshape(2, 8)], final["mixing"]),
        "nominal": ([coordinates.nominal], np.ones(1)),
    }
    for model, (parameters, mixing) in models.items():
        prefix = cfg["membership_prefix_samples"]
        calibration = MeasuredDataset(
            data["heldout"].speeds,
            data["heldout"].commands[:, :prefix],
            data["heldout"].measurements[:, :prefix],
        )
        test_clients, frozen = calibrate_memberships(
            calibration, test_positions, parameters, mixing, dt, q, r
        )
        if model == "K2":
            frames["memberships"].append(
                pd.DataFrame(
                    {
                        "split": "heldout_prefix",
                        "restart": -1,
                        "client_position": test_clients,
                        "weight_0": frozen[:, 0],
                        "weight_1": frozen[:, 1],
                        "selected_component": np.argmax(frozen, axis=1),
                    }
                )
            )
        forecasts, whitened, radius = mixture_forecasts(
            data["heldout"],
            test_positions,
            parameters,
            frozen,
            dt,
            q,
            r,
            cfg["forecast_horizons_samples"],
            cfg["forecast_burn_in_samples"],
        )
        stats, correlations = residual_stats(
            whitened, data["heldout"].commands[:, prefix:], cfg["residual_maximum_lag"]
        )
        frames["forecasts"].append(pd.DataFrame(forecasts).assign(model=model))
        frames["residuals"].append(
            pd.DataFrame([stats]).assign(model=model, maximum_plant_spectral_radius=radius)
        )
        frames["correlations"].append(pd.DataFrame(correlations).assign(model=model))
        for group, z in enumerate(parameters):
            frames["parameters"].append(
                pd.DataFrame(
                    {
                        "model": model,
                        "component": group,
                        "parameter": PARAMETER_NAMES,
                        "fitted": np.exp(z),
                        "nominal": np.exp(coordinates.nominal),
                        "coupling_log_residual": physical_coupling_diagnostic(z)[
                            "log_coupling_residual"
                        ],
                    }
                )
            )
            eta = np.linalg.lstsq(LOG_MAP, z - coordinates.nominal, rcond=None)[0]
            frames["physical_parameters"].append(
                pd.DataFrame(
                    {
                        "model": model,
                        "component": group,
                        "parameter": PHYSICAL_NAMES,
                        "fitted": np.exp(coordinates.nominal_physical_logs + eta),
                        "nominal": np.exp(coordinates.nominal_physical_logs),
                    }
                )
            )
    frames["splits"].append(split)
    outputs = {}
    for name, pieces in frames.items():
        path = OUT / f"{job_prefix(seed, scale)}_{name}.csv"
        pd.concat(pieces, ignore_index=True).assign(seed=seed, process_noise_scale=scale).to_csv(
            path, index=False
        )
        outputs[path.name] = sha(path)
    manifest = {
        "seed": seed,
        "process_noise_scale": scale,
        "revision": revision,
        "source_sha256": sources,
        "measured_array_sha256": hashes,
        "output_sha256": outputs,
        "elapsed_seconds": time.perf_counter() - started,
        "confirmation_run": False,
    }
    (OUT / f"{job_prefix(seed, scale)}_complete.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    return seed, scale


def summarize():
    cfg, _ = load_configuration()
    manifests = []
    frames = {name: [] for name in TABLES}
    for seed in cfg["development_seeds"]:
        for scale in cfg["process_noise_scales"]:
            manifest = json.loads((OUT / f"{job_prefix(seed, scale)}_complete.json").read_text())
            if manifest["source_sha256"] != source_hashes(seed, scale):
                raise RuntimeError("Execution source provenance changed")
            for name in TABLES:
                path = OUT / f"{job_prefix(seed, scale)}_{name}.csv"
                if sha(path) != manifest["output_sha256"][path.name]:
                    raise RuntimeError(f"Execution output changed: {path}")
                frames[name].append(pd.read_csv(path))
            manifests.append(manifest)
    frames = {name: pd.concat(pieces, ignore_index=True) for name, pieces in frames.items()}
    for name, frame in frames.items():
        frame.to_csv(OUT / f"{PREFIX}_{name}.csv", index=False)
    forecasts = frames["forecasts"]
    aggregate = forecasts[forecasts.client_position == -1]
    forecast_summary = (
        aggregate.groupby(["process_noise_scale", "model", "rule", "mode", "horizon_samples"])[
            "sensor_normalized_mse"
        ]
        .agg(["mean", "min", "max"])
        .reset_index()
    )
    forecast_summary.to_csv(OUT / f"{PREFIX}_forecast_summary.csv", index=False)
    paired = []
    for (scale, mode, horizon), local in aggregate.groupby(
        ["process_noise_scale", "mode", "horizon_samples"]
    ):
        reference = local[(local.model == "K1") & (local.rule == "map")].set_index("seed")
        for rule in ("map", "ensemble"):
            fitted = local[(local.model == "K2") & (local.rule == rule)].set_index("seed")
            for seed in cfg["development_seeds"]:
                first, second = (
                    reference.loc[seed, "sensor_normalized_mse"],
                    fitted.loc[seed, "sensor_normalized_mse"],
                )
                paired.append(
                    {
                        "seed": seed,
                        "process_noise_scale": scale,
                        "mode": mode,
                        "horizon_samples": horizon,
                        "rule": rule,
                        "K1_error": first,
                        "K2_error": second,
                        "gain_pct": 100 * (first - second) / first,
                    }
                )
    paired = pd.DataFrame(paired)
    paired.to_csv(OUT / f"{PREFIX}_paired_forecasts.csv", index=False)
    paired.groupby(["process_noise_scale", "mode", "horizon_samples", "rule"])["gain_pct"].agg(
        ["mean", "min", "max"]
    ).reset_index().to_csv(OUT / f"{PREFIX}_paired_summary.csv", index=False)
    runs, restarts = frames["runs"], frames["restarts"]
    selected = runs[runs.component_count == 2]
    summary = selected.groupby("process_noise_scale")[
        [
            "train_gain_pct",
            "maximum_aligned_parameter_cv",
            "minimum_partition_ari_to_selected",
            "minimum_hard_training_occupancy",
            "plant_log_distance",
        ]
    ].mean()
    summary.to_csv(OUT / f"{PREFIX}_summary.csv")
    checks = {
        "all_restart_success": bool(restarts.success.all()),
        "stationary_restarts": int((restarts.kkt_residual <= cfg["maximum_kkt_residual"]).sum()),
        "restart_count": len(restarts),
        "maximum_kkt_residual": float(restarts.kkt_residual.max()),
        "maximum_constraint_violation": float(restarts.constraint_violation.max()),
        "maximum_gradient_relative_error": float(
            frames["gradient_checks"].norm_relative_error.max()
        ),
        "maximum_coupling_log_residual": float(
            frames["parameters"].coupling_log_residual.abs().max()
        ),
        "maximum_nested_objective_increase": float(
            (selected.train_objective - selected.global_train_objective).max()
        ),
        "minimum_selected_face_eigenvalue": float(selected.minimum_face_hessian_eigenvalue.min()),
        "sources_and_output_hashes_verified": True,
        "confirmation_run": False,
        "reserved_confirmation_seeds": cfg["reserved_confirmation_seeds"],
        "limitations": "Opened development; uncalibrated Gaussian working weights, heuristic search bounds, fixed-Q panels and ignored quiet-bias uncertainty. No labels/truth/state/control/FL communication claims.",
    }
    (OUT / f"{PREFIX}_conclusions.json").write_text(json.dumps(checks, indent=2) + "\n")
    (OUT / f"{PREFIX}_execution_manifest.json").write_text(json.dumps(manifests, indent=2) + "\n")
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for row, scale in enumerate(cfg["process_noise_scales"]):
        local = aggregate[
            (aggregate.process_noise_scale == scale) & (aggregate["mode"] == "forecast")
        ]
        for model, rule, label in [
            ("nominal", "map", "Nominal"),
            ("K1", "map", "K=1"),
            ("K2", "map", "K=2 prefix MAP"),
            ("K2", "ensemble", "K=2 ensemble"),
        ]:
            curve = (
                local[(local.model == model) & (local.rule == rule)]
                .groupby("horizon_samples")
                .sensor_normalized_mse.mean()
            )
            axes[row, 0].plot(curve.index * 0.01, curve, "o-", label=label)
        axes[row, 0].set(
            title=f"Q scale {scale:g}: future outputs",
            xlabel="Horizon (s)",
            ylabel="Fixed-R normalized MSE",
            yscale="log",
        )
        axes[row, 0].legend(fontsize=8)
        local = paired[(paired.process_noise_scale == scale) & (paired.rule == "map")]
        for seed in cfg["development_seeds"]:
            curve = local[(local.seed == seed) & (local["mode"] == "forecast")].sort_values(
                "horizon_samples"
            )
            axes[row, 1].plot(curve.horizon_samples * 0.01, curve.gain_pct, "o-", label=str(seed))
        axes[row, 1].axhline(0, color="black", linewidth=0.7)
        axes[row, 1].set(
            title="Paired K=2 improvement over K=1", xlabel="Horizon (s)", ylabel="Improvement (%)"
        )
        axes[row, 1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT / "results/figures/experiment_10hk_joint_client_mixture.pdf")
    plt.close(fig)
    print(forecast_summary.to_string(index=False), flush=True)
    print(json.dumps(checks, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--q-scale", type=float)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    cfg, _ = load_configuration()
    if args.summarize_only:
        summarize()
        return
    seeds = cfg["development_seeds"] if args.seed is None else [args.seed]
    scales = cfg["process_noise_scales"] if args.q_scale is None else [args.q_scale]
    if any(seed not in cfg["development_seeds"] for seed in seeds) or any(
        scale not in cfg["process_noise_scales"] for scale in scales
    ):
        parser.error("Only predeclared development jobs may execute")
    jobs = [
        (seed, scale)
        for seed in seeds
        for scale in scales
        if not (args.resume and (OUT / f"{job_prefix(seed, scale)}_complete.json").exists())
    ]
    failures = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(evaluate_seed, *job): job for job in jobs}
        for future in as_completed(futures):
            error = future.exception()
            if error is None:
                print(f"Completed 10H-K {future.result()}", flush=True)
            else:
                failures.append(futures[future])
                print(f"Failed 10H-K {futures[future]}: {error!r}", flush=True)
    if failures:
        raise RuntimeError(f"Incomplete jobs {failures}")
    if args.seed is None and args.q_scale is None:
        summarize()


if __name__ == "__main__":
    main()
