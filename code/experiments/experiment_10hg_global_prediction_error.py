"""10H-G: leakage-safe global structured prediction-error optimizer gate."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from experiment_10h_higher_order_lpv_gate import sample_fleet
from experiment_10hf_output_identifiability import (
    PARAMETER_NAMES,
    C,
    measurement_covariance,
    nominal_effective_parameters,
    simulate_measurements,
    structured_discrete,
)
from scipy.linalg import solve_discrete_are
from scipy.optimize import minimize
from scipy.signal import dlsim

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "code/config/experiment_10hg.json"
CONFIG_10H = ROOT / "code/config/experiment_10h.json"
CONFIG_10HA = ROOT / "code/config/experiment_10ha.json"
CONFIG_10HF = ROOT / "code/config/experiment_10hf.json"
OUT = ROOT / "results/tables"
FIGURE = ROOT / "results/figures/experiment_10hg_global_prediction_error.pdf"


def effective_truth(client):
    """Retrospective effective coordinates; never used by the estimator."""
    p = client.parameters
    return np.asarray([
        p.mass * p.front_length / p.yaw_inertia,
        p.mass * p.rear_length / p.yaw_inertia,
        p.front_stiffness / (p.mass * client.front_relaxation_length),
        p.front_length * p.front_stiffness / (p.mass * client.front_relaxation_length),
        p.rear_stiffness / (p.mass * client.rear_relaxation_length),
        p.rear_length * p.rear_stiffness / (p.mass * client.rear_relaxation_length),
        1 / client.front_relaxation_length,
        1 / client.rear_relaxation_length,
        1 / client.actuator_time_constant,
    ])


def build_segments(seed, cfg, cfg10h, cfg10ha, cfg10hf):
    clients = sample_fleet(seed, cfg10h)
    rng = np.random.default_rng(seed + 1_700_000)
    segments = {"train": [], "heldout": []}
    truth = {"train": [], "heldout": []}
    for index, client in enumerate(clients):
        block = index % len(cfg10h["coverage_blocks"])
        band_index = (index // len(cfg10h["coverage_blocks"])) % len(
            cfg10h["frequency_bands_hz"]
        )
        split = (
            "heldout" if index % cfg["heldout_client_modulus"] ==
            cfg["heldout_client_remainder"] else "train"
        )
        truth[split].append(effective_truth(client))
        for speed in cfg10h["coverage_blocks"][block]:
            commands, measurements = simulate_measurements(
                client, speed, cfg10h["frequency_bands_hz"][band_index],
                cfg10hf, cfg10h, cfg10ha, rng,
            )
            count = cfg["samples_per_segment"]
            segments[split].append((speed, commands[:count], measurements[:count]))
    return segments, truth


def innovation_system(log_parameters, speed, cfg10h, cfg10ha):
    """Construct the steady-state innovation filter for one unique speed."""
    a, b = structured_discrete(log_parameters, speed, cfg10h["sample_time"])
    scale = json.loads(
        (OUT / "experiment_10ha_frozen_selection.json").read_text()
    )["process_noise_scale"]
    q = np.diag(np.asarray(cfg10ha["base_process_noise_diagonal"]) * scale)
    r = measurement_covariance(cfg10ha)
    try:
        p = solve_discrete_are(a.T, C.T, q, r)
    except np.linalg.LinAlgError:
        return None
    predicted_covariance = a @ p @ a.T + q
    innovation_covariance = C @ predicted_covariance @ C.T + r
    gain = np.linalg.solve(innovation_covariance, C @ predicted_covariance).T
    correction = np.eye(5) - gain @ C
    closed = correction @ a
    input_matrix = np.column_stack([correction @ b, gain])
    output_matrix = -C @ a
    feedthrough = np.column_stack([-C @ b, np.eye(3)])
    return (closed, input_matrix, output_matrix, feedthrough,
            cfg10h["sample_time"]), np.linalg.cholesky(innovation_covariance)


def steady_state_innovations(system, chol, commands, measurements):
    inputs = np.column_stack([commands, measurements])
    _, innovations, _ = dlsim(
        system, inputs,
    )
    return np.linalg.solve(chol, innovations.T).T


def prediction_objective(log_parameters, segments, cfg10h, cfg10ha):
    total = 0.0
    count = 0
    systems = {}
    for speed, commands, measurements in segments:
        if speed not in systems:
            systems[speed] = innovation_system(
                log_parameters, speed, cfg10h, cfg10ha
            )
        if systems[speed] is None:
            return 1e12
        innovations = steady_state_innovations(
            *systems[speed], commands, measurements
        )
        if innovations is None or not np.isfinite(innovations).all():
            return 1e12
        total += float(np.sum(innovations ** 2))
        count += innovations.size
    return total / max(count, 1)


def optimize_restart(initial, bounds, segments, cfg, cfg10h, cfg10ha):
    result = minimize(
        prediction_objective, initial,
        args=(segments, cfg10h, cfg10ha), method="L-BFGS-B", bounds=bounds,
        options={"maxiter": cfg["optimizer_max_iterations"], "ftol": 1e-10,
                 "gtol": 1e-6, "maxls": 30},
    )
    return result


def evaluate_seed(seed):
    cfg = json.loads(CONFIG.read_text())
    cfg10h = json.loads(CONFIG_10H.read_text())
    cfg10ha = json.loads(CONFIG_10HA.read_text())
    cfg10hf = json.loads(CONFIG_10HF.read_text())
    segments, truth = build_segments(seed, cfg, cfg10h, cfg10ha, cfg10hf)
    nominal = np.log(nominal_effective_parameters(cfg10hf))
    lower = nominal + np.log(cfg["parameter_lower_multiplier"])
    upper = nominal + np.log(cfg["parameter_upper_multiplier"])
    bounds = list(zip(lower, upper))
    rng = np.random.default_rng(seed + 1_900_000)
    initials = [nominal]
    for _ in range(cfg["restart_count"] - 1):
        initials.append(np.clip(
            nominal + rng.normal(0, cfg["restart_log_standard_deviation"], len(nominal)),
            lower, upper,
        ))
    results = [
        optimize_restart(initial, bounds, segments["train"], cfg, cfg10h, cfg10ha)
        for initial in initials
    ]
    order = np.argsort([result.fun for result in results])
    best = results[int(order[0])]
    fitted = best.x
    nominal_train = prediction_objective(nominal, segments["train"], cfg10h, cfg10ha)
    nominal_test = prediction_objective(nominal, segments["heldout"], cfg10h, cfg10ha)
    fitted_test = prediction_objective(fitted, segments["heldout"], cfg10h, cfg10ha)
    near = [results[i] for i in order[:min(3, len(results))]]
    objectives = np.asarray([item.fun for item in near])
    parameters = np.exp(np.asarray([item.x for item in near]))
    spread = 100 * (objectives.max() - objectives.min()) / objectives.min()
    cvs = np.std(parameters, axis=0) / np.maximum(np.mean(parameters, axis=0), 1e-12)
    profile_rows = []
    edge_increases = []
    for parameter, name in enumerate(PARAMETER_NAMES):
        values = []
        for offset in cfg["profile_log_offsets"]:
            candidate = fitted.copy()
            candidate[parameter] = np.clip(fitted[parameter] + offset, lower[parameter], upper[parameter])
            value = prediction_objective(candidate, segments["train"], cfg10h, cfg10ha)
            values.append(value)
            profile_rows.append({
                "seed": seed, "parameter": name, "log_offset": offset,
                "objective": value,
            })
        center = values[len(values) // 2]
        edge_increases.append(100 * (min(values[0], values[-1]) - center) / center)
    truth_mean = np.mean(truth["train"], axis=0)
    retrospective_error = np.linalg.norm(np.log(np.exp(fitted) / truth_mean)) / np.sqrt(len(fitted))
    boundary = np.isclose(fitted, lower, atol=1e-5) | np.isclose(fitted, upper, atol=1e-5)
    summary = {
        "seed": seed, "optimizer_success": bool(best.success),
        "train_objective_nominal": nominal_train,
        "train_objective_fitted": float(best.fun),
        "heldout_objective_nominal": nominal_test,
        "heldout_objective_fitted": fitted_test,
        "heldout_improvement_pct": 100 * (nominal_test - fitted_test) / nominal_test,
        "restart_objective_spread_pct": float(spread),
        "maximum_restart_parameter_cv": float(cvs.max()),
        "minimum_profile_edge_increase_pct": float(min(edge_increases)),
        "boundary_fraction": float(np.mean(boundary)),
        "retrospective_log_rmse_to_train_truth_mean": float(retrospective_error),
        "iterations": int(best.nit), "function_evaluations": int(best.nfev),
    }
    parameter_rows = [{
        "seed": seed, "parameter": name, "nominal": float(np.exp(nominal[j])),
        "fitted": float(np.exp(fitted[j])), "retrospective_truth_mean": float(truth_mean[j]),
        "best_restart_cv": float(cvs[j]), "at_boundary": bool(boundary[j]),
    } for j, name in enumerate(PARAMETER_NAMES)]
    restart_rows = [{
        "seed": seed, "restart": j, "objective": float(result.fun),
        "success": bool(result.success), "iterations": int(result.nit),
    } for j, result in enumerate(results)]
    return (pd.DataFrame([summary]), pd.DataFrame(parameter_rows),
            pd.DataFrame(restart_rows), pd.DataFrame(profile_rows))


def summarize(pieces, cfg):
    runs = pd.concat([piece[0] for piece in pieces], ignore_index=True)
    parameters = pd.concat([piece[1] for piece in pieces], ignore_index=True)
    restarts = pd.concat([piece[2] for piece in pieces], ignore_index=True)
    profiles = pd.concat([piece[3] for piece in pieces], ignore_index=True)
    aggregate = {
        "optimizer_success_rate": float(runs.optimizer_success.mean()),
        "mean_heldout_improvement_pct": float(runs.heldout_improvement_pct.mean()),
        "minimum_heldout_improvement_pct": float(runs.heldout_improvement_pct.min()),
        "maximum_restart_objective_spread_pct": float(runs.restart_objective_spread_pct.max()),
        "maximum_restart_parameter_cv": float(runs.maximum_restart_parameter_cv.max()),
        "minimum_profile_edge_increase_pct": float(runs.minimum_profile_edge_increase_pct.min()),
        "maximum_boundary_fraction": float(runs.boundary_fraction.max()),
        "mean_retrospective_log_rmse": float(runs.retrospective_log_rmse_to_train_truth_mean.mean()),
    }
    gates = {
        "optimizer_success_gate": bool(runs.optimizer_success.all()),
        "heldout_improvement_gate": aggregate["minimum_heldout_improvement_pct"] >= cfg["minimum_heldout_nll_improvement_pct"],
        "restart_objective_gate": aggregate["maximum_restart_objective_spread_pct"] <= cfg["maximum_restart_objective_spread_pct"],
        "restart_parameter_gate": aggregate["maximum_restart_parameter_cv"] <= cfg["maximum_restart_parameter_cv"],
        "profile_curvature_gate": aggregate["minimum_profile_edge_increase_pct"] >= cfg["minimum_profile_edge_increase_pct"],
        "boundary_gate": aggregate["maximum_boundary_fraction"] <= cfg["maximum_boundary_fraction"],
    }
    gates["development_gate_pass"] = all(gates.values())
    OUT.mkdir(parents=True, exist_ok=True)
    runs.to_csv(OUT / "experiment_10hg_run_summary.csv", index=False)
    parameters.to_csv(OUT / "experiment_10hg_parameters.csv", index=False)
    restarts.to_csv(OUT / "experiment_10hg_restarts.csv", index=False)
    profiles.to_csv(OUT / "experiment_10hg_profiles.csv", index=False)
    (OUT / "experiment_10hg_conclusions.json").write_text(json.dumps({
        "aggregate": aggregate, "development_gates": gates,
        "confirmation_run": False,
        "reserved_confirmation_seeds": cfg["reserved_confirmation_seeds"],
        "leakage_statement": "No true client physical quantity, state, matrix, class center, or label enters fitting, restart selection, held-out scoring, or gate decisions.",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
    }, indent=2) + "\n")
    plot(runs, parameters, profiles)
    return aggregate, gates


def plot(runs, parameters, profiles):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.7))
    axes[0].bar(runs.seed.astype(str), runs.heldout_improvement_pct, color="#0072B2")
    axes[0].axhline(2.0, color="black", linestyle="--")
    axes[0].set(title="Held-out prediction improvement", xlabel="Fleet seed", ylabel=r"Improvement (\%)")
    pivot = parameters.pivot(index="parameter", columns="seed", values="fitted")
    ratio = pivot.div(parameters.groupby("parameter").retrospective_truth_mean.mean(), axis=0)
    axes[1].boxplot([ratio.loc[name].values for name in PARAMETER_NAMES], tick_labels=range(1, 10))
    axes[1].axhline(1, color="black", linestyle="--")
    axes[1].set(title="Retrospective fitted/truth mean", xlabel="Effective coordinate", ylabel="Ratio")
    mean_profiles = profiles.groupby(["parameter", "log_offset"]).objective.mean().reset_index()
    for name in PARAMETER_NAMES:
        local = mean_profiles[mean_profiles.parameter == name]
        center = local.loc[local.log_offset == 0, "objective"].iloc[0]
        axes[2].plot(local.log_offset, 100 * (local.objective / center - 1), alpha=0.75)
    axes[2].set(title="Conditional likelihood profiles", xlabel="Log-parameter offset", ylabel=r"Objective increase (\%)")
    fig.tight_layout()
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE)
    plt.close(fig)


def main():
    cfg = json.loads(CONFIG.read_text())
    with ProcessPoolExecutor(max_workers=5) as executor:
        pieces = list(executor.map(evaluate_seed, cfg["development_seeds"]))
    aggregate, gates = summarize(pieces, cfg)
    print(json.dumps(aggregate, indent=2))
    print(json.dumps(gates, indent=2))


if __name__ == "__main__":
    main()
