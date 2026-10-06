"""10H-A: tire-force observer feasibility and output-feedback control audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from experiment_10h_higher_order_lpv_gate import (
    architecture_matrices,
    design_schedule,
    interpolate,
    sample_fleet,
    scenarios,
)
from scipy.linalg import solve_discrete_are
from scipy.signal import savgol_filter

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "code/config/experiment_10ha.json"
CONFIG_10H = ROOT / "code/config/experiment_10h.json"
OUT = ROOT / "results/tables"
FIGURE = ROOT / "results/figures/experiment_10ha_tire_force_observer.pdf"
SELECTION = OUT / "experiment_10ha_frozen_selection.json"
C = np.array([
    [0.0, 1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 1.0],
])


def measurement_covariance(cfg):
    noise = cfg["measurement_noise_std"]
    values = [np.deg2rad(noise["yaw_rate_deg_s"]),
              noise["lateral_acceleration_mps2"],
              np.deg2rad(noise["steering_deg"])]
    return np.diag(np.square(values))


def sample_bias(cfg, rng):
    bias = cfg["measurement_bias_std"]
    return np.asarray([
        rng.normal(scale=np.deg2rad(bias["yaw_rate_deg_s"])),
        rng.normal(scale=bias["lateral_acceleration_mps2"]),
        rng.normal(scale=np.deg2rad(bias["steering_deg"])),
    ])


def observer_gain(a, process_covariance, measurement_covariance_matrix):
    p = solve_discrete_are(a.T, C.T, process_covariance, measurement_covariance_matrix)
    predicted = a @ p @ a.T + process_covariance
    innovation = C @ predicted @ C.T + measurement_covariance_matrix
    return np.linalg.solve(innovation, C @ predicted).T


def design_observer_schedule(matrices, scale, cfg):
    base = np.asarray(cfg["base_process_noise_diagonal"], dtype=float)
    q = np.diag(base * scale)
    r = measurement_covariance(cfg)
    return np.asarray([observer_gain(a, q, r) for a, _ in matrices])


def interpolate_matrices(matrices, grid, speed):
    flat_a = np.asarray([a.ravel() for a, _ in matrices])
    flat_b = np.asarray([b.ravel() for _, b in matrices])
    a = interpolate(flat_a, grid, speed).reshape(5, 5)
    b = interpolate(flat_b, grid, speed).reshape(5, 1)
    return a, b


def excitation(cfg, cfg10h):
    time = np.arange(0, cfg["duration_s"] + cfg10h["sample_time"], cfg10h["sample_time"])
    envelope = np.sin(np.pi * time / time[-1]) ** 2
    speed = 20 + 9 * np.sin(2 * np.pi * time / time[-1] - np.pi / 2)
    command = envelope * (
        0.010 * np.sin(2 * np.pi * 0.22 * time)
        + 0.006 * np.sin(2 * np.pi * 0.63 * time + 0.4)
        + 0.004 * np.sin(2 * np.pi * 1.10 * time + 0.9)
    )
    return time, speed, command[:-1]


def simulate_open_loop(client, true_matrices, observer_matrices, exact_matrices,
                       observer_gains, exact_gains, cfg, cfg10h, rng):
    time, speed, command = excitation(cfg, cfg10h)
    grid = np.asarray(cfg10h["speed_grid"], dtype=float)
    states = np.zeros((len(time), 5))
    global_estimate = np.zeros_like(states)
    exact_estimate = np.zeros_like(states)
    measurements = np.zeros((len(time), 3))
    bias = sample_bias(cfg, rng)
    noise_covariance = measurement_covariance(cfg)
    measurements[0] = C @ states[0] + bias + rng.multivariate_normal(np.zeros(3), noise_covariance)
    global_estimate[0] = observer_gains[0] @ measurements[0]
    exact_estimate[0] = exact_gains[0] @ measurements[0]
    for k, local_speed in enumerate(speed[:-1]):
        true_a, true_b = interpolate_matrices(true_matrices, grid, local_speed)
        global_a, global_b = interpolate_matrices(observer_matrices, grid, local_speed)
        exact_a, exact_b = interpolate_matrices(exact_matrices, grid, local_speed)
        states[k + 1] = true_a @ states[k] + true_b[:, 0] * command[k]
        measurements[k + 1] = (
            C @ states[k + 1] + bias
            + rng.multivariate_normal(np.zeros(3), noise_covariance)
        )
        gain_global = np.asarray([
            [np.interp(local_speed, grid, observer_gains[:, row, column])
             for column in range(3)] for row in range(5)
        ])
        gain_exact = np.asarray([
            [np.interp(local_speed, grid, exact_gains[:, row, column])
             for column in range(3)] for row in range(5)
        ])
        predicted_global = global_a @ global_estimate[k] + global_b[:, 0] * command[k]
        predicted_exact = exact_a @ exact_estimate[k] + exact_b[:, 0] * command[k]
        global_estimate[k + 1] = predicted_global + gain_global @ (
            measurements[k + 1] - C @ predicted_global
        )
        exact_estimate[k + 1] = predicted_exact + gain_exact @ (
            measurements[k + 1] - C @ predicted_exact
        )
    yaw_derivative = savgol_filter(
        measurements[:, 0], cfg["algebraic_savgol_window"],
        cfg["algebraic_savgol_order"], deriv=1, delta=cfg10h["sample_time"],
    )
    p = client.parameters
    kz_squared = p.yaw_inertia / p.mass
    length = p.front_length + p.rear_length
    algebraic_front = (p.rear_length * measurements[:, 1] + kz_squared * yaw_derivative) / length
    algebraic_rear = (p.front_length * measurements[:, 1] - kz_squared * yaw_derivative) / length
    return time, states, global_estimate, exact_estimate, np.column_stack([algebraic_front, algebraic_rear])


def estimation_metrics(truth, estimate, start, method):
    truth, estimate = truth[start:], estimate[start:]
    error = estimate - truth
    force_scale = np.maximum(np.sqrt(np.mean(truth[:, 2:4] ** 2, axis=0)), 1e-6)
    force_rmse = np.sqrt(np.mean(error[:, 2:4] ** 2, axis=0))
    return {
        "method": method,
        "beta_rmse_deg": float(np.rad2deg(np.sqrt(np.mean(error[:, 0] ** 2)))),
        "front_force_rmse": float(force_rmse[0]),
        "rear_force_rmse": float(force_rmse[1]),
        "front_force_nrmse_pct": float(100 * force_rmse[0] / force_scale[0]),
        "rear_force_nrmse_pct": float(100 * force_rmse[1] / force_scale[1]),
        "force_sum_rmse": float(np.sqrt(np.mean(np.sum(error[:, 2:4], axis=1) ** 2))),
    }


def algebraic_metrics(truth, estimate, start):
    force_truth = truth[start:, 2:4]
    force_error = estimate[start:] - force_truth
    scale = np.maximum(np.sqrt(np.mean(force_truth**2, axis=0)), 1e-6)
    rmse = np.sqrt(np.mean(force_error**2, axis=0))
    return {
        "method": "Algebraic",
        "beta_rmse_deg": np.nan,
        "front_force_rmse": float(rmse[0]), "rear_force_rmse": float(rmse[1]),
        "front_force_nrmse_pct": float(100 * rmse[0] / scale[0]),
        "rear_force_nrmse_pct": float(100 * rmse[1] / scale[1]),
        "force_sum_rmse": float(np.sqrt(np.mean(np.sum(force_error, axis=1) ** 2))),
    }


def closed_loop(client, true_matrices, observer_matrices, observer_gains,
                controller_schedule, scenario, cfg, cfg10h, rng, mode):
    grid = np.asarray(cfg10h["speed_grid"], dtype=float)
    gains, prefilters = controller_schedule
    state = np.zeros((len(scenario["time"]), 5))
    estimate = np.zeros_like(state)
    integral = np.zeros(len(scenario["time"]))
    command = np.zeros(len(scenario["time"]) - 1)
    bias = sample_bias(cfg, rng) if mode != "FullState" else np.zeros(3)
    noise_covariance = measurement_covariance(cfg)
    measurement = C @ state[0] + bias
    if mode != "FullState":
        measurement += rng.multivariate_normal(np.zeros(3), noise_covariance)
        estimate[0] = observer_gains[0] @ measurement
    for k, speed in enumerate(scenario["speed"][:-1]):
        feedback_state = state[k] if mode == "FullState" else estimate[k]
        gain = interpolate(gains, grid, speed)
        prefilter = float(np.interp(speed, grid, prefilters))
        command[k] = -gain @ np.r_[feedback_state, integral[k]] + prefilter * scenario["reference"][k]
        true_a, true_b = interpolate_matrices(true_matrices, grid, speed)
        state[k + 1] = true_a @ state[k] + true_b[:, 0] * command[k]
        measurement = C @ state[k + 1] + bias
        if mode != "FullState":
            measurement += rng.multivariate_normal(np.zeros(3), noise_covariance)
            observer_a, observer_b = interpolate_matrices(observer_matrices, grid, speed)
            predicted = observer_a @ estimate[k] + observer_b[:, 0] * command[k]
            observer_gain = np.asarray([
                [np.interp(speed, grid, observer_gains[:, row, column])
                 for column in range(3)] for row in range(5)
            ])
            estimate[k + 1] = predicted + observer_gain @ (measurement - C @ predicted)
            yaw_feedback = measurement[0]
        else:
            yaw_feedback = state[k, 1]
        integral[k + 1] = integral[k] + cfg10h["sample_time"] * (
            yaw_feedback - scenario["reference"][k]
        )
    error = state[:, 1] - scenario["reference"]
    return {
        "tracking_rmse_deg_s": float(np.rad2deg(np.sqrt(np.mean(error**2)))),
        "tracking_peak_deg_s": float(np.rad2deg(np.max(np.abs(error)))),
        "command_peak_deg": float(np.rad2deg(np.max(np.abs(command)))),
        "finite": bool(np.isfinite(state).all()),
    }


def observability_audit(matrices):
    rows = []
    for a, _ in matrices:
        matrix = np.vstack([C @ np.linalg.matrix_power(a, power) for power in range(5)])
        scaled = matrix / np.maximum(np.linalg.norm(matrix, axis=0), 1e-15)
        rows.append((np.linalg.matrix_rank(matrix), np.linalg.cond(scaled)))
    return min(rank for rank, _ in rows), max(condition for _, condition in rows)


def evaluate_seed(seed, cfg, cfg10h, process_scale, phase):
    rng = np.random.default_rng(seed + 900_000)
    clients = sample_fleet(seed, cfg10h)
    grid = np.asarray(cfg10h["speed_grid"], dtype=float)
    exact, global_model, _ = architecture_matrices(clients, grid, cfg10h["sample_time"])
    global_observer_gains = design_observer_schedule(global_model, process_scale, cfg)
    controller_weights = json.loads(
        (OUT / "experiment_10h_frozen_selection.json").read_text()
    )["selected"]
    global_controller = design_schedule(global_model, controller_weights, cfg10h["sample_time"])
    estimation_rows, control_rows, observability_rows = [], [], []
    start = int(cfg["observer_transient_s"] / cfg10h["sample_time"])
    for client in clients:
        exact_model = exact[client.client_id]
        exact_observer_gains = design_observer_schedule(exact_model, process_scale, cfg)
        _, truth, global_estimate, exact_estimate, algebraic = simulate_open_loop(
            client, exact_model, global_model, exact_model,
            global_observer_gains, exact_observer_gains, cfg, cfg10h, rng,
        )
        metrics = [
            estimation_metrics(truth, global_estimate, start, "GlobalKF"),
            estimation_metrics(truth, exact_estimate, start, "ExactKF"),
            algebraic_metrics(truth, algebraic, start),
        ]
        for item in metrics:
            estimation_rows.append({
                "seed": seed, "phase": phase, "client": client.client_id,
                "vehicle_class": client.vehicle_class, "process_scale": process_scale,
            } | item)
        rank, condition = observability_audit(exact_model)
        observability_rows.append({
            "seed": seed, "client": client.client_id, "vehicle_class": client.vehicle_class,
            "minimum_rank": rank, "maximum_scaled_condition": condition,
        })
        if phase == "confirmation":
            for scenario_name, scenario in scenarios(cfg10h).items():
                for mode, observer_model, observer_gains in (
                    ("FullState", exact_model, exact_observer_gains),
                    ("GlobalKF", global_model, global_observer_gains),
                    ("ExactKF", exact_model, exact_observer_gains),
                ):
                    result = closed_loop(
                        client, exact_model, observer_model, observer_gains,
                        global_controller, scenario, cfg, cfg10h, rng, mode,
                    )
                    control_rows.append({
                        "seed": seed, "client": client.client_id,
                        "vehicle_class": client.vehicle_class, "scenario": scenario_name,
                        "method": mode,
                    } | result)
    return pd.DataFrame(estimation_rows), pd.DataFrame(control_rows), pd.DataFrame(observability_rows)


def select_scale(cfg, cfg10h):
    rows = []
    for scale in cfg["process_noise_candidates"]:
        with ProcessPoolExecutor(max_workers=5) as executor:
            pieces = list(executor.map(
                evaluate_seed, cfg["development_seeds"], [cfg] * 5, [cfg10h] * 5,
                [scale] * 5, ["development"] * 5,
            ))
        data = pd.concat([piece[0] for piece in pieces], ignore_index=True)
        global_data = data[data.method == "GlobalKF"]
        score = (
            global_data.beta_rmse_deg.mean() / cfg["maximum_beta_rmse_deg"]
            + global_data[["front_force_nrmse_pct", "rear_force_nrmse_pct"]].to_numpy().mean()
              / cfg["maximum_force_nrmse_pct"]
        )
        rows.append({
            "process_scale": scale, "score": score,
            "beta_rmse_deg": global_data.beta_rmse_deg.mean(),
            "force_nrmse_pct": global_data[["front_force_nrmse_pct", "rear_force_nrmse_pct"]].to_numpy().mean(),
        })
    audit = pd.DataFrame(rows)
    selected = float(audit.loc[audit.score.idxmin(), "process_scale"])
    audit.to_csv(OUT / "experiment_10ha_observer_audit.csv", index=False)
    SELECTION.write_text(json.dumps({
        "process_noise_scale": selected,
        "rule": "minimum normalized GlobalKF beta-plus-force estimation score on development fleets",
    }, indent=2) + "\n")
    return selected


def bootstrap_interval(values, resamples, seed=10_009):
    values = np.asarray(values)
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, (resamples, len(values)), replace=True).mean(axis=1)
    return np.quantile(draws, [0.025, 0.975])


def summarize(estimation, control, observability, cfg):
    estimation_summary = estimation.groupby("method").agg(
        beta_rmse_deg=("beta_rmse_deg", "mean"),
        front_force_nrmse_pct=("front_force_nrmse_pct", "mean"),
        rear_force_nrmse_pct=("rear_force_nrmse_pct", "mean"),
        force_sum_rmse=("force_sum_rmse", "mean"),
    ).reset_index()
    seed_control = control.groupby(["seed", "scenario", "method"]).agg(
        mean_tracking=("tracking_rmse_deg_s", "mean"),
        worst_tracking=("tracking_rmse_deg_s", "max"),
        maximum_command=("command_peak_deg", "max"),
        all_finite=("finite", "all"),
    ).reset_index()
    control_summary = seed_control.groupby(["scenario", "method"]).mean(numeric_only=True).reset_index()
    comparisons = []
    for scenario in sorted(seed_control.scenario.unique()):
        view = seed_control[seed_control.scenario == scenario].pivot(index="seed", columns="method")
        for metric in ("mean_tracking", "worst_tracking"):
            degradation = 100 * (view[metric]["GlobalKF"] - view[metric]["FullState"]) / view[metric]["FullState"]
            interval = bootstrap_interval(degradation, cfg["bootstrap_resamples"])
            comparisons.append({
                "scenario": scenario, "metric": metric,
                "degradation_pct": float(degradation.mean()),
                "ci_low": float(interval[0]), "ci_high": float(interval[1]),
            })
    comparisons = pd.DataFrame(comparisons)
    global_estimation = estimation_summary.set_index("method").loc["GlobalKF"]
    gates = {
        "full_observability_rank": bool((observability.minimum_rank == 5).all()),
        "beta_estimation_gate": bool(global_estimation.beta_rmse_deg <= cfg["maximum_beta_rmse_deg"]),
        "force_estimation_gate": bool(max(global_estimation.front_force_nrmse_pct,
                                           global_estimation.rear_force_nrmse_pct)
                                      <= cfg["maximum_force_nrmse_pct"]),
        "mean_control_gate": bool((comparisons[comparisons.metric == "mean_tracking"].degradation_pct
                                   <= cfg["maximum_mean_control_degradation_pct"]).all()),
        "worst_control_gate": bool((comparisons[comparisons.metric == "worst_tracking"].degradation_pct
                                    <= cfg["maximum_worst_control_degradation_pct"]).all()),
        "all_output_feedback_finite": bool(control[control.method == "GlobalKF"].finite.all()),
    }
    gates["all_gates_pass"] = all(gates.values())
    estimation_summary.to_csv(OUT / "experiment_10ha_estimation_summary.csv", index=False)
    seed_control.to_csv(OUT / "experiment_10ha_seed_control.csv", index=False)
    control_summary.to_csv(OUT / "experiment_10ha_control_summary.csv", index=False)
    comparisons.to_csv(OUT / "experiment_10ha_control_comparisons.csv", index=False)
    observability.to_csv(OUT / "experiment_10ha_observability.csv", index=False)
    (OUT / "experiment_10ha_conclusions.json").write_text(json.dumps({
        "gates": gates,
        "maximum_observability_condition": float(observability.maximum_scaled_condition.max()),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
    }, indent=2) + "\n")
    return estimation_summary, control_summary, comparisons, gates


def plot(estimation, control, comparisons):
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    methods = ["Algebraic", "GlobalKF", "ExactKF"]
    view = estimation.set_index("method").loc[methods]
    x = np.arange(len(methods))
    axes[0].bar(x - 0.18, view.front_force_nrmse_pct, 0.36, label="Front")
    axes[0].bar(x + 0.18, view.rear_force_nrmse_pct, 0.36, label="Rear")
    axes[0].set_xticks(x, methods, rotation=15)
    axes[0].set_ylabel("Force NRMSE [%]")
    axes[0].set_title("Tire-force estimation")
    axes[0].legend()
    for axis, scenario in zip(axes[1:], ("broadband", "transient")):
        local = control[control.scenario == scenario].set_index("method").loc[
            ["FullState", "GlobalKF", "ExactKF"]
        ]
        axis.bar(local.index, local.mean_tracking, color=["#009E73", "#0072B2", "#999999"])
        axis.set_title(scenario.capitalize())
        axis.set_ylabel("Yaw-rate RMSE [deg/s]")
        axis.tick_params(axis="x", rotation=15)
        comp = comparisons[(comparisons.scenario == scenario) &
                           (comparisons.metric == "mean_tracking")].iloc[0]
        axis.text(0.5, 0.95, f"GlobalKF loss: {comp.degradation_pct:.1f}%",
                  transform=axis.transAxes, ha="center", va="top")
    fig.tight_layout()
    fig.savefig(FIGURE)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["select", "confirm", "all"], nargs="?", default="all")
    args = parser.parse_args()
    cfg = json.loads(CONFIG.read_text())
    cfg10h = json.loads(CONFIG_10H.read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    scale = select_scale(cfg, cfg10h) if args.phase in ("select", "all") else json.loads(SELECTION.read_text())["process_noise_scale"]
    if args.phase in ("confirm", "all"):
        with ProcessPoolExecutor(max_workers=5) as executor:
            pieces = list(executor.map(
                evaluate_seed, cfg["confirmation_seeds"], [cfg] * 10, [cfg10h] * 10,
                [scale] * 10, ["confirmation"] * 10,
            ))
        estimation = pd.concat([piece[0] for piece in pieces], ignore_index=True)
        control = pd.concat([piece[1] for piece in pieces], ignore_index=True)
        observability = pd.concat([piece[2] for piece in pieces], ignore_index=True)
        estimation.to_csv(OUT / "experiment_10ha_estimation.csv.gz", index=False, compression="gzip")
        control.to_csv(OUT / "experiment_10ha_control.csv.gz", index=False, compression="gzip")
        estimation_summary, control_summary, comparisons, gates = summarize(
            estimation, control, observability, cfg
        )
        plot(estimation_summary, control_summary, comparisons)
        print(estimation_summary.to_string(index=False))
        print(control_summary.to_string(index=False))
        print(comparisons.to_string(index=False))
        print(json.dumps(gates, indent=2))


if __name__ == "__main__":
    main()
