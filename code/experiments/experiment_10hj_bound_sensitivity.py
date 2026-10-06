"""10H-J: predeclared nested-bound and causal output-validation diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import itertools
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
from experiment_10hi_physically_coupled_fit import (
    curvature_audit,
    gradient_audit,
    make_coordinates,
    make_initials,
    nuisance_profiles,
    scalar_audit,
)

from federated_lpv.innovation_likelihood import (
    PARAMETER_NAMES,
    InnovationLikelihood,
    physical_coupling_diagnostic,
)
from federated_lpv.output_validation import causal_forecasts, residual_diagnostics
from federated_lpv.physical_coupled_fit import (
    PHYSICAL_NAMES,
    CoupledLikelihood,
    constraint_audit,
    optimize_coupled,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "code/config/experiment_10hj.json"
OUT = ROOT / "results/tables"
PREFIX = "experiment_10hj"
TABLES = (
    "runs",
    "restarts",
    "phases",
    "parameters",
    "physical_parameters",
    "profiles",
    "rear_gain_profiles",
    "curvature",
    "forecasts",
    "residuals",
    "correlations",
    "gradient_checks",
)


def source_hashes():
    paths = [
        CONFIG,
        Path(__file__),
        ROOT / "docs/experiment_10hj_protocol.md",
        ROOT / "code/src/federated_lpv/output_validation.py",
        ROOT / "code/src/federated_lpv/physical_coupled_fit.py",
        ROOT / "code/src/federated_lpv/innovation_likelihood.py",
        ROOT / "code/experiments/experiment_10hi_physically_coupled_fit.py",
        ROOT / "code/experiments/experiment_10hh_single_model_repair.py",
        ROOT / "code/experiments/experiment_10hf_output_identifiability.py",
        ROOT / "code/experiments/experiment_10h_higher_order_lpv_gate.py",
        ROOT / "code/experiments/experiment_10ha_tire_force_observer.py",
        *[ROOT / f"code/config/experiment_{name}.json" for name in ("10h", "10ha", "10hf")],
        ROOT / "code/tests/test_experiment_10hj.py",
        OUT / "experiment_10hi_runs.csv",
    ]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def load_configuration():
    cfg = json.loads(CONFIG.read_text())
    if set(cfg["development_seeds"]) & set(cfg["reserved_confirmation_seeds"]):
        raise ValueError("Development and confirmation overlap")
    boxes = cfg["bound_boxes"]
    for previous, following in itertools.pairwise(boxes):
        if following["lower"] > previous["lower"] or following["upper"] < previous["upper"]:
            raise ValueError("Diagnostic boxes must be nested")
    inherited = {
        name: json.loads((ROOT / f"code/config/experiment_{name}.json").read_text())
        for name in ("10h", "10ha", "10hf")
    }
    return cfg, inherited


def job_prefix(seed, scale):
    return f"{PREFIX}_seed{seed}_Q{scale:g}"


def fit_model(seed, evaluator, coordinates, cfg):
    estimates, rows, phases = [], [], []
    for restart, initial in enumerate(make_initials(seed, cfg, coordinates)):
        fitted, result, trace, calls, invalid, audit = optimize_coupled(
            initial, evaluator, coordinates, cfg, "staged"
        )
        value = evaluator.value(fitted)
        estimates.append(fitted)
        rows.append(
            dict(
                restart=restart,
                objective=value,
                success=bool(result.success),
                iterations=int(result.nit),
                evaluations=calls,
                invalid_evaluations=invalid,
                message=str(result.message),
                **scalar_audit(audit),
            )
        )
        phases.extend(dict(restart=restart, **p) for p in trace)
        print(
            f"10H-J seed={seed} restart={restart} L={value:.7f} "
            f"KKT={audit['kkt_residual']:.2g} success={result.success}",
            flush=True,
        )
    values = np.array([row["objective"] for row in rows])
    best = int(np.argmin(values))  # Training working likelihood only.
    relative_parameters = np.exp(estimates)
    cv = relative_parameters.std(axis=0) / relative_parameters.mean(axis=0)
    stats = {
        "best_restart": best,
        "all_restart_success": all(row["success"] for row in rows),
        "maximum_restart_kkt_residual": max(row["kkt_residual"] for row in rows),
        "maximum_restart_constraint_violation": max(row["constraint_violation"] for row in rows),
        "invalid_evaluations": sum(row["invalid_evaluations"] for row in rows),
        "maximum_restart_parameter_cv": float(cv.max()),
        "restart_objective_spread_pct": float(100 * np.ptp(values) / abs(values.min())),
    }
    return estimates[best], stats, pd.DataFrame(rows), pd.DataFrame(phases)


def absolute_rear_profiles(evaluator, fitted, coordinates, cfg):
    baseline = evaluator.value(fitted)
    rows = []
    for multiplier in cfg["rear_gain_profile_nominal_multipliers"]:
        target = np.log(multiplier)
        start = coordinates.feasible_profile_start(fitted, 4, target)
        if start is None:
            rows.append({"nominal_multiplier": multiplier, "feasible": False})
            continue
        estimate, result, _, _, invalid, audit = optimize_coupled(
            start, evaluator, coordinates, cfg, "joint", {4: target}
        )
        value = evaluator.value(estimate)
        rows.append(
            dict(
                nominal_multiplier=multiplier,
                feasible=True,
                success=bool(result.success),
                objective=value,
                increase_pct=100 * (value - baseline) / abs(baseline),
                actual_rear_log_coordinate=float(estimate[4]),
                invalid_evaluations=invalid,
                **scalar_audit(audit),
            )
        )
    return pd.DataFrame(rows)


def evaluate_seed(seed, scale):
    cfg, inherited = load_configuration()
    if seed not in cfg["development_seeds"] or scale not in cfg["process_noise_scales"]:
        raise ValueError("Only declared opened development jobs may execute")
    started = time.perf_counter()
    provenance = source_hashes()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    data, split = measured_datasets(seed, cfg, inherited)
    archived = pd.read_csv(OUT / "experiment_10hi_runs.csv")
    archived = archived[(archived.seed == seed) & (archived.method == "staged")]
    hashes = {name: dataset_hash(value) for name, value in data.items()}
    if not all((archived[f"{name}_data_sha256"] == value).all() for name, value in hashes.items()):
        raise RuntimeError("Measured records differ from 10H-I")
    dt = inherited["10h"]["sample_time"]
    q = np.diag(np.asarray(inherited["10ha"]["base_process_noise_diagonal"]) * scale)
    r = measurement_covariance(inherited["10ha"])
    frames = {name: [] for name in TABLES}
    previous_objective = None
    nominal_forecasts = nominal_residuals = nominal_correlations = None
    for box in cfg["bound_boxes"]:
        coordinates = make_coordinates(
            {
                "parameter_lower_multiplier": box["lower"],
                "parameter_upper_multiplier": box["upper"],
            },
            inherited["10hf"],
        )
        train = CoupledLikelihood(InnovationLikelihood(data["train"], dt, q, r), coordinates)
        test = CoupledLikelihood(InnovationLikelihood(data["heldout"], dt, q, r), coordinates)
        meta = {"seed": seed, "process_noise_scale": scale, "box": box["name"]}
        checks = gradient_audit(train, cfg)
        fitted, stats, restarts, phases = fit_model(seed, train, coordinates, cfg)
        final = train.evaluate(fitted, gradient=True, diagnostics=True)
        heldout = test.evaluate(fitted, diagnostics=True)
        audit = constraint_audit(
            fitted, final["gradient"], coordinates, cfg["active_bound_tolerance"]
        )
        curvature, eigenvalues = curvature_audit(train, fitted, coordinates, cfg)
        z = coordinates.effective(fitted)
        row = dict(
            **meta,
            lower_multiplier=box["lower"],
            upper_multiplier=box["upper"],
            train_objective=final["objective"],
            heldout_objective=heldout["objective"],
            heldout_sensor_mse=heldout["sensor_normalized_mse"],
            maximum_gradient_relative_error=float(checks.norm_relative_error.max()),
            coupling_log_residual=physical_coupling_diagnostic(z)["log_coupling_residual"],
            nested_training_objective_increase=0.0
            if previous_objective is None
            else final["objective"] - previous_objective,
            train_data_sha256=hashes["train"],
            heldout_data_sha256=hashes["heldout"],
            **stats,
            **scalar_audit(audit),
            **curvature,
        )
        if box["name"] == cfg["profile_box"]:
            profile_stats, profiles = nuisance_profiles(train, fitted, coordinates, cfg)
            row.update(profile_stats)
            frames["profiles"].append(profiles.assign(**meta))
            frames["rear_gain_profiles"].append(
                absolute_rear_profiles(train, fitted, coordinates, cfg).assign(**meta)
            )
        previous_objective = final["objective"]
        frames["runs"].append(pd.DataFrame([row]))
        frames["restarts"].append(restarts.assign(**meta))
        frames["phases"].append(phases.assign(**meta))
        frames["gradient_checks"].append(checks.assign(**meta))
        frames["curvature"].append(eigenvalues.assign(**meta))
        multipliers = iter(audit["multipliers"])
        bound_multipliers = {}
        for side in ("lower", "upper"):
            for index in np.flatnonzero(audit[f"at_{side}"]):
                bound_multipliers[index, side] = float(next(multipliers))
        frames["parameters"].append(
            pd.DataFrame(
                [
                    dict(
                        **meta,
                        parameter=name,
                        nominal=np.exp(coordinates.nominal[j]),
                        fitted=np.exp(z[j]),
                        at_lower=bool(audit["at_lower"][j]),
                        at_upper=bool(audit["at_upper"][j]),
                        lower_kkt_multiplier=bound_multipliers.get((j, "lower"), 0.0),
                        upper_kkt_multiplier=bound_multipliers.get((j, "upper"), 0.0),
                    )
                    for j, name in enumerate(PARAMETER_NAMES)
                ]
            )
        )
        w = np.exp(coordinates.nominal_physical_logs + fitted)
        physical = [
            dict(
                **meta,
                parameter=name,
                nominal=np.exp(coordinates.nominal_physical_logs[j]),
                fitted=w[j],
            )
            for j, name in enumerate(PHYSICAL_NAMES)
        ]
        derived = {
            "wheelbase": (w[1] + w[2], float(np.exp(coordinates.nominal_physical_logs[1:3]).sum())),
            "front_relaxation_length": (1 / w[5], 1 / np.exp(coordinates.nominal_physical_logs[5])),
            "rear_relaxation_length": (1 / w[6], 1 / np.exp(coordinates.nominal_physical_logs[6])),
            "front_normalized_stiffness": (
                w[3] / w[5],
                np.exp(coordinates.nominal[2] - coordinates.nominal[6]),
            ),
            "rear_normalized_stiffness": (
                w[4] / w[6],
                np.exp(coordinates.nominal[4] - coordinates.nominal[7]),
            ),
            "actuator_time_constant": (1 / w[7], 1 / np.exp(coordinates.nominal_physical_logs[7])),
        }
        physical.extend(
            dict(**meta, parameter=name, fitted=value, nominal=nominal)
            for name, (value, nominal) in derived.items()
        )
        frames["physical_parameters"].append(pd.DataFrame(physical))
        if nominal_forecasts is None:
            nominal_forecasts = causal_forecasts(
                data["heldout"],
                coordinates.nominal,
                dt,
                q,
                r,
                cfg["forecast_horizons_samples"],
                cfg["forecast_burn_in_samples"],
            )
            nominal_residuals, nominal_correlations = residual_diagnostics(
                data["heldout"], coordinates.nominal, dt, q, r, cfg["residual_maximum_lag"]
            )
        forecasts = causal_forecasts(
            data["heldout"],
            z,
            dt,
            q,
            r,
            cfg["forecast_horizons_samples"],
            cfg["forecast_burn_in_samples"],
        )
        residuals, correlations = residual_diagnostics(
            data["heldout"], z, dt, q, r, cfg["residual_maximum_lag"]
        )
        for model, forecast, residual, correlation in (
            ("nominal", nominal_forecasts, nominal_residuals, nominal_correlations),
            ("fitted", forecasts, residuals, correlations),
        ):
            frames["forecasts"].append(pd.DataFrame(forecast).assign(model=model, **meta))
            frames["residuals"].append(pd.DataFrame([residual]).assign(model=model, **meta))
            frames["correlations"].append(pd.DataFrame(correlation).assign(model=model, **meta))
        print(
            f"10H-J seed={seed} Q={scale:g} box={box['name']} "
            f"E_R={heldout['sensor_normalized_mse']:.6f} bounds={audit['active_coefficient_bounds']}",
            flush=True,
        )
    stem = job_prefix(seed, scale)
    for name, parts in frames.items():
        pd.concat(parts, ignore_index=True).to_csv(OUT / f"{stem}_{name}.csv", index=False)
    split.to_csv(OUT / f"{stem}_splits.csv", index=False)
    if provenance != source_hashes():
        raise RuntimeError("Execution sources changed during this job")
    manifest = {
        "seed": seed,
        "process_noise_scale": scale,
        "revision": revision,
        "source_sha256": provenance,
        "measured_array_sha256": hashes,
        "elapsed_seconds": time.perf_counter() - started,
        "confirmation_run": False,
        "output_sha256": {
            f"{stem}_{name}.csv": hashlib.sha256(
                (OUT / f"{stem}_{name}.csv").read_bytes()
            ).hexdigest()
            for name in (*TABLES, "splits")
        },
    }
    (OUT / f"{stem}_complete.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return seed, scale


def summarize():
    cfg, _ = load_configuration()
    manifests = []
    expected = source_hashes()
    frames = {}
    for seed in cfg["development_seeds"]:
        for scale in cfg["process_noise_scales"]:
            manifest = json.loads((OUT / f"{job_prefix(seed, scale)}_complete.json").read_text())
            if manifest["source_sha256"] != expected:
                raise RuntimeError("Completed job does not match execution sources")
            for name, digest in manifest["output_sha256"].items():
                if hashlib.sha256((OUT / name).read_bytes()).hexdigest() != digest:
                    raise RuntimeError(f"Completed output changed: {name}")
            manifests.append(manifest)
    for name in (*TABLES, "splits"):
        frames[name] = pd.concat(
            [
                pd.read_csv(OUT / f"{job_prefix(seed, scale)}_{name}.csv")
                for seed in cfg["development_seeds"]
                for scale in cfg["process_noise_scales"]
            ],
            ignore_index=True,
        )
        frames[name].to_csv(OUT / f"{PREFIX}_{name}.csv", index=False)
    runs = frames["runs"]
    summary = (
        runs.groupby(["process_noise_scale", "box"], sort=False)
        .agg(
            fleets=("seed", "count"),
            train_objective_mean=("train_objective", "mean"),
            heldout_sensor_mse_mean=("heldout_sensor_mse", "mean"),
            active_bound_fleets=("active_coefficient_bounds", lambda x: int((x > 0).sum())),
            maximum_restart_kkt=("maximum_restart_kkt_residual", "max"),
            maximum_restart_parameter_cv=("maximum_restart_parameter_cv", "max"),
            minimum_face_curvature=("minimum_face_hessian_eigenvalue", "min"),
        )
        .reset_index()
    )
    summary.to_csv(OUT / f"{PREFIX}_summary.csv", index=False)
    restart = frames["restarts"]
    profiles = frames["profiles"]
    absolute = frames["rear_gain_profiles"]
    checks = {
        "all_restart_solver_flags": bool(restart.success.all()),
        "all_restart_kkt": bool((restart.kkt_residual <= cfg["maximum_kkt_residual"]).all()),
        "all_restart_feasibility": bool(
            (restart.constraint_violation <= cfg["maximum_constraint_violation"]).all()
        ),
        "no_invalid_fitting_evaluations": bool((restart.invalid_evaluations == 0).all()),
        "gradient_check": bool(
            (runs.maximum_gradient_relative_error <= cfg["maximum_gradient_relative_error"]).all()
        ),
        "coupling": bool(
            (runs.coupling_log_residual.abs() <= cfg["maximum_coupling_log_residual"]).all()
        ),
        "restart_consistency": bool(
            (runs.maximum_restart_parameter_cv <= cfg["maximum_restart_parameter_cv"]).all()
        ),
        "restart_objective_spread": bool(
            (runs.restart_objective_spread_pct <= cfg["maximum_restart_objective_spread_pct"]).all()
        ),
        "nested_training_objective": bool(
            (
                runs.nested_training_objective_increase
                <= cfg["nested_training_objective_tolerance"]
            ).all()
        ),
    }
    for name, frame in (("local_profile", profiles), ("rear_gain_profile", absolute)):
        feasible = frame[frame.feasible]
        checks[f"{name}_stationarity"] = bool(
            feasible.success.all()
            and (feasible.kkt_residual <= cfg["maximum_kkt_residual"]).all()
            and (feasible.constraint_violation <= cfg["maximum_constraint_violation"]).all()
            and (feasible.invalid_evaluations == 0).all()
        )
    forecast_summary = (
        frames["forecasts"]
        .groupby(["process_noise_scale", "box", "model", "mode", "horizon_samples"], sort=False)
        .sensor_normalized_mse.agg(["mean", "min", "max"])
        .reset_index()
    )
    forecast_summary.to_csv(OUT / f"{PREFIX}_forecast_summary.csv", index=False)
    conclusions = {
        "diagnostic_only": True,
        "numerical_checks": checks,
        "physical_bounds_calibrated": False,
        "physical_readiness_gate_introduced": False,
        "historical_gates_unchanged": True,
        "bound_or_covariance_selected": False,
        "confirmation_run": False,
        "reserved_confirmation_seeds": cfg["reserved_confirmation_seeds"],
        "fitting_restarts": len(restart),
        "stationary_fitting_restarts": int(
            (restart.kkt_residual <= cfg["maximum_kkt_residual"]).sum()
        ),
        "limitations": "Opened development data; heterogeneous steady Gaussian working model and quiet-bias uncertainty. Profiles are not calibrated confidence intervals. Diagnostic boxes are not engineering priors. No labels, latent states or client truths are accessed by fitting or output validation. Correlations are descriptive. No state-estimation or closed-loop claim.",
        "source_sha256": expected,
    }
    (OUT / f"{PREFIX}_conclusions.json").write_text(json.dumps(conclusions, indent=2) + "\n")
    (OUT / f"{PREFIX}_execution_manifest.json").write_text(json.dumps(manifests, indent=2) + "\n")
    plot(frames)
    print(summary.to_string(index=False), flush=True)
    print(json.dumps(checks, indent=2), flush=True)


def plot(frames):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    names = ["original", "wider", "widest"]
    for row, scale in enumerate((0.01, 1.0)):
        runs = frames["runs"][frames["runs"].process_noise_scale == scale]
        parameters = frames["parameters"]
        for seed in sorted(runs.seed.unique()):
            local = runs[runs.seed == seed].set_index("box").loc[names]
            rear = (
                parameters[
                    (parameters.seed == seed)
                    & (parameters.process_noise_scale == scale)
                    & (parameters.parameter == "rear_beta")
                ]
                .set_index("box")
                .loc[names]
            )
            axes[row, 0].plot(names, rear.fitted / rear.nominal, "o-", label=str(seed))
            axes[row, 1].plot(names, local.heldout_sensor_mse, "o-")
        axes[row, 0].set(title=f"Q scale {scale:g}: rear gain", ylabel="b_r / b_r nominal")
        axes[row, 1].set(title="Held-out one-step prediction", ylabel="Fixed-R normalized MSE")
        forecast = frames["forecasts"]
        forecast = forecast[(forecast.process_noise_scale == scale) & (forecast.mode == "forecast")]
        for name in names:
            curve = (
                forecast[(forecast.box == name) & (forecast.model == "fitted")]
                .groupby("horizon_samples")
                .sensor_normalized_mse.mean()
            )
            axes[row, 2].plot(curve.index * 0.01, curve, "o-", label=name)
        nominal = (
            forecast[(forecast.box == "original") & (forecast.model == "nominal")]
            .groupby("horizon_samples")
            .sensor_normalized_mse.mean()
        )
        axes[row, 2].plot(nominal.index * 0.01, nominal, "k--", label="nominal")
        axes[row, 2].set(
            title="Common-origin held-out forecasts",
            xlabel="Horizon (s)",
            ylabel="Fixed-R normalized MSE",
            yscale="log",
        )
        axes[row, 0].legend(fontsize=7)
        axes[row, 2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(ROOT / "results/figures/experiment_10hj_bound_sensitivity.pdf")
    plt.close(fig)


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
    if any(s not in cfg["development_seeds"] for s in seeds):
        parser.error("Only declared opened development seeds may execute")
    if any(q not in cfg["process_noise_scales"] for q in scales):
        parser.error("Only declared fixed Q panels may execute")
    jobs = [
        (s, q)
        for s in seeds
        for q in scales
        if not (args.resume and (OUT / f"{job_prefix(s, q)}_complete.json").exists())
    ]
    failures = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(evaluate_seed, *job): job for job in jobs}
        for future in as_completed(futures):
            error = future.exception()
            if error is None:
                print(f"Completed 10H-J {future.result()}", flush=True)
            else:
                failures.append(futures[future])
                print(f"Failed 10H-J {futures[future]}: {error!r}", flush=True)
    if failures:
        raise RuntimeError(f"Incomplete jobs {failures}; address errors then use --resume")
    if args.seed is None and args.q_scale is None:
        summarize()


if __name__ == "__main__":
    main()
