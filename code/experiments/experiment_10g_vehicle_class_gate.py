"""10G: vehicle-class and steering-actuator control-relevance gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from experiment_10f_control_relevance import scenarios
from scipy.linalg import expm, solve_discrete_are

from federated_lpv import VehicleParameters, nonlinear_bicycle_rhs

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "code/config/experiment_10g.json"
CONFIG_10F = ROOT / "code/config/experiment_10f.json"
OUT = ROOT / "results/tables"
FIGURE = ROOT / "results/figures/experiment_10g_vehicle_class_gate.pdf"
SELECTION = OUT / "experiment_10g_frozen_selection.json"
METHODS = ("Global", "OracleClass", "ClientExact")


@dataclass(frozen=True)
class ExtendedClient:
    client_id: str
    vehicle_class: str
    parameters: VehicleParameters
    actuator_time_constant: float


def sample_extended_fleet(seed: int, cfg: dict) -> list[ExtendedClient]:
    rng = np.random.default_rng(seed)
    scatter = cfg["scatter"]
    clients = []
    for name, center in cfg["classes"].items():
        for index in range(cfg["clients_per_class"]):
            scale = lambda key: rng.uniform(1 - scatter[key], 1 + scatter[key])
            parameters = VehicleParameters(
                mass=center["mass"] * scale("mass"),
                yaw_inertia=center["yaw_inertia"] * scale("yaw_inertia"),
                front_length=center["front_length"] * scale("geometry"),
                rear_length=center["rear_length"] * scale("geometry"),
                front_stiffness=center["front_stiffness"] * scale("stiffness"),
                rear_stiffness=center["rear_stiffness"] * scale("stiffness"),
            )
            clients.append(
                ExtendedClient(
                    f"{name}_{index:02d}", name, parameters,
                    center["actuator_time_constant"] * scale("actuator_time_constant"),
                )
            )
    return clients


def continuous_augmented_matrices(client: ExtendedClient, speed: float):
    """Linearization of [beta,r,delta_a] with commanded steering as input."""
    p, tau = client.parameters, client.actuator_time_constant
    m, iz, lf, lr, cf, cr, vx = (
        p.mass, p.yaw_inertia, p.front_length, p.rear_length,
        p.front_stiffness, p.rear_stiffness, float(speed),
    )
    a = np.array([
        [-(cf + cr) / (m * vx), (cr * lr - cf * lf) / (m * vx**2) - 1, cf / (m * vx)],
        [(cr * lr - cf * lf) / iz, -(cf * lf**2 + cr * lr**2) / (iz * vx), cf * lf / iz],
        [0, 0, -1 / tau],
    ])
    b = np.array([[0.0], [0.0], [1 / tau]])
    return a, b


def discrete_augmented_matrices(client: ExtendedClient, speed: float, dt: float):
    a, b = continuous_augmented_matrices(client, speed)
    block = np.block([[a, b], [np.zeros((1, 4))]])
    discrete = expm(block * dt)
    return discrete[:3, :3], discrete[:3, 3:]


def architecture_matrices(clients, speeds, dt):
    exact = {
        client.client_id: [discrete_augmented_matrices(client, speed, dt) for speed in speeds]
        for client in clients
    }
    global_matrices = [
        (np.mean([exact[c.client_id][j][0] for c in clients], axis=0),
         np.mean([exact[c.client_id][j][1] for c in clients], axis=0))
        for j in range(len(speeds))
    ]
    class_matrices = {}
    for vehicle_class in sorted({client.vehicle_class for client in clients}):
        members = [client for client in clients if client.vehicle_class == vehicle_class]
        class_matrices[vehicle_class] = [
            (np.mean([exact[c.client_id][j][0] for c in members], axis=0),
             np.mean([exact[c.client_id][j][1] for c in members], axis=0))
            for j in range(len(speeds))
        ]
    return exact, global_matrices, class_matrices


def design_lqi(a, b, dt, weights):
    c = np.array([[0.0, 1.0, 0.0]])
    aa = np.block([[a, np.zeros((3, 1))], [dt * c, np.ones((1, 1))]])
    bb = np.vstack([b, [[0.0]]])
    q = np.diag([weights["q_beta"], weights["q_yaw"], weights["q_actuator"], weights["q_integral"]])
    r = np.array([[weights["r"]]])
    if np.linalg.matrix_rank(np.column_stack([bb, aa @ bb, aa @ aa @ bb, aa @ aa @ aa @ bb])) < 4:
        raise ValueError("uncontrollable augmented model")
    p = solve_discrete_are(aa, bb, q, r)
    gain = np.linalg.solve(r + bb.T @ p @ bb, bb.T @ p @ aa).ravel()
    equilibrium = np.linalg.solve(
        np.block([[np.eye(3) - a, -b], [c, np.zeros((1, 1))]]),
        np.array([0.0, 0.0, 0.0, 1.0]),
    )
    prefilter = float(equilibrium[3] + gain[:3] @ equilibrium[:3])
    return gain, prefilter


def design_schedule(matrices, weights, dt):
    try:
        designs = [design_lqi(a, b, dt, weights) for a, b in matrices]
    except (ValueError, np.linalg.LinAlgError):
        return None
    return np.asarray([item[0] for item in designs]), np.asarray([item[1] for item in designs])


def interpolate(schedule, speeds, speed):
    gains, prefilters = schedule
    gain = np.asarray([np.interp(speed, speeds, gains[:, j]) for j in range(gains.shape[1])])
    return gain, float(np.interp(speed, speeds, prefilters))


def nonlinear_rhs(state, command, speed, client, mu):
    beta_yaw = nonlinear_bicycle_rhs(state[:2], state[2], speed, client.parameters, mu)
    actuator = (command - state[2]) / client.actuator_time_constant
    return np.r_[beta_yaw, actuator]


def rk4_step(state, command, speed, client, dt, mu):
    rhs = lambda value: nonlinear_rhs(value, command, speed, client, mu)
    k1 = rhs(state)
    k2 = rhs(state + dt * k1 / 2)
    k3 = rhs(state + dt * k2 / 2)
    k4 = rhs(state + dt * k3)
    return state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6


def simulate(client, schedule, scenario, cfg):
    if schedule is None:
        return None
    time, speed = scenario["time"], scenario["speed"]
    reference = speed * scenario["curvature"]
    state = np.zeros((len(time), 3))
    integral = np.zeros(len(time))
    command = np.zeros(len(time) - 1)
    raw_command = np.zeros(len(time) - 1)
    limit = np.deg2rad(cfg["steering_limit_deg"])
    rate_step = np.deg2rad(cfg["steering_rate_limit_deg_s"]) * cfg["sample_time"]
    grid = np.asarray(cfg["speed_grid"], dtype=float)
    for k in range(len(time) - 1):
        gain, prefilter = interpolate(schedule, grid, speed[k])
        raw = -gain @ np.r_[state[k], integral[k]] + prefilter * reference[k]
        previous = command[k - 1] if k else 0.0
        applied = float(np.clip(np.clip(raw, previous - rate_step, previous + rate_step), -limit, limit))
        raw_command[k], command[k] = raw, applied
        state[k + 1] = rk4_step(state[k], applied, speed[k], client, cfg["sample_time"], cfg["friction_coefficient"])
        error = state[k, 1] - reference[k]
        candidate = np.clip(integral[k] + cfg["sample_time"] * error,
                            -cfg["integral_limit"], cfg["integral_limit"])
        integral[k + 1] = candidate if np.isclose(applied, raw) or np.sign(raw - applied) != np.sign(-gain[3] * error) else integral[k]
        if not np.isfinite(state[k + 1]).all():
            return None
    error = state[:, 1] - reference
    saturated = np.abs(raw_command - command) > 1e-10
    return {
        "tracking_rmse_deg_s": float(np.rad2deg(np.sqrt(np.mean(error**2)))),
        "tracking_peak_deg_s": float(np.rad2deg(np.max(np.abs(error)))),
        "beta_peak_deg": float(np.rad2deg(np.max(np.abs(state[:, 0])))),
        "command_peak_deg": float(np.rad2deg(np.max(np.abs(command)))),
        "actuator_peak_deg": float(np.rad2deg(np.max(np.abs(state[:, 2])))),
        "saturation_fraction": float(np.mean(saturated)),
        "feasible": bool(np.rad2deg(np.max(np.abs(state[:, 0]))) <= cfg["beta_limit_deg"]),
    }


def evaluate_seed(seed, cfg, weights, phase):
    clients = sample_extended_fleet(seed, cfg)
    grid = np.asarray(cfg["speed_grid"], dtype=float)
    exact, global_matrices, class_matrices = architecture_matrices(clients, grid, cfg["sample_time"])
    global_schedule = design_schedule(global_matrices, weights, cfg["sample_time"])
    class_schedules = {key: design_schedule(value, weights, cfg["sample_time"]) for key, value in class_matrices.items()}
    exact_schedules = {key: design_schedule(value, weights, cfg["sample_time"]) for key, value in exact.items()}
    rows = []
    for scenario_name, scenario in scenarios(json.loads(CONFIG_10F.read_text())).items():
        for client in clients:
            schedules = {
                "Global": global_schedule,
                "OracleClass": class_schedules[client.vehicle_class],
                "ClientExact": exact_schedules[client.client_id],
            }
            if phase == "development":
                schedules = {"ClientExact": schedules["ClientExact"]}
            for method, schedule in schedules.items():
                metrics = simulate(client, schedule, scenario, cfg)
                row = {"seed": seed, "phase": phase, "scenario": scenario_name,
                       "client": client.client_id, "vehicle_class": client.vehicle_class,
                       "method": method, "controller": weights["name"]}
                if metrics is None:
                    metrics = {"tracking_rmse_deg_s": np.inf, "tracking_peak_deg_s": np.inf,
                               "beta_peak_deg": np.inf, "command_peak_deg": np.inf,
                               "actuator_peak_deg": np.inf, "saturation_fraction": 1.0,
                               "feasible": False}
                rows.append(row | metrics)
    return pd.DataFrame(rows)


def bootstrap_mean(values, resamples, seed=10_007):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, (resamples, len(values)), replace=True).mean(axis=1)
    return np.quantile(draws, [0.025, 0.975])


def select_controller(cfg):
    rows = []
    for weights in cfg["controller_candidates"]:
        pieces = [evaluate_seed(seed, cfg, weights, "development") for seed in cfg["development_seeds"]]
        data = pd.concat(pieces, ignore_index=True)
        exact = data[data.method == "ClientExact"]
        rows.append({
            "controller": weights["name"],
            "mean_tracking": exact.tracking_rmse_deg_s.mean(),
            "worst_tracking": exact.tracking_rmse_deg_s.max(),
            "mean_saturation": exact.saturation_fraction.mean(),
            "all_feasible": bool(exact.feasible.all()),
        })
    audit = pd.DataFrame(rows)
    eligible = audit[(audit.all_feasible) & (audit.mean_saturation <= cfg["selection_max_mean_saturation_fraction"])]
    if eligible.empty:
        raise RuntimeError("no controller candidate passes the development gates")
    selected_name = eligible.loc[eligible.mean_tracking.idxmin(), "controller"]
    selected = next(item for item in cfg["controller_candidates"] if item["name"] == selected_name)
    OUT.mkdir(parents=True, exist_ok=True)
    audit.to_csv(OUT / "experiment_10g_controller_audit.csv", index=False)
    payload = {"selected": selected, "rule": "minimum ClientExact mean tracking among feasible low-saturation candidates"}
    SELECTION.write_text(json.dumps(payload, indent=2) + "\n")
    return selected


def summarize(data, cfg):
    seed_summary = data.groupby(["seed", "scenario", "method"]).agg(
        mean_tracking=("tracking_rmse_deg_s", "mean"),
        worst_tracking=("tracking_rmse_deg_s", "max"),
        mean_saturation=("saturation_fraction", "mean"),
        feasible_fraction=("feasible", "mean"),
    ).reset_index()
    summary = seed_summary.groupby(["scenario", "method"]).agg(
        mean_tracking=("mean_tracking", "mean"), worst_tracking=("worst_tracking", "mean"),
        mean_saturation=("mean_saturation", "mean"), feasible_fraction=("feasible_fraction", "mean"),
    ).reset_index()
    comparisons = []
    for scenario in sorted(seed_summary.scenario.unique()):
        view = seed_summary[seed_summary.scenario == scenario].pivot(index="seed", columns="method")
        for metric in ("mean_tracking", "worst_tracking"):
            global_values = view[metric]["Global"]
            class_values = view[metric]["OracleClass"]
            exact_values = view[metric]["ClientExact"]
            improvement = 100 * (global_values - class_values) / global_values
            degradation = 100 * (global_values - exact_values) / exact_values
            gap = global_values - exact_values
            # Recovery is meaningful only when ClientExact establishes a
            # positive Global-to-exact gap.
            recovery = 100 * (global_values - class_values) / gap.where(gap > 0)
            interval = bootstrap_mean(improvement, cfg["bootstrap_resamples"])
            comparisons.append({
                "scenario": scenario, "metric": metric,
                "class_improvement_pct": float(improvement.mean()),
                "ci_low": float(interval[0]), "ci_high": float(interval[1]),
                "global_degradation_vs_exact_pct": float(degradation.mean()),
                "class_gap_recovery_pct": float(recovery.mean()),
                "absolute_improvement_deg_s": float((global_values - class_values).mean()),
                "class_win_fraction": float((class_values < global_values).mean()),
            })
    comparisons = pd.DataFrame(comparisons)
    gates = {
        "global_degradation_gate": bool((comparisons.global_degradation_vs_exact_pct >= cfg["minimum_global_degradation_pct"]).all()),
        "class_recovery_gate": bool((comparisons.class_gap_recovery_pct >= cfg["minimum_class_recovery_pct"]).all()),
        "positive_interval_gate": bool((comparisons.ci_low > 0).all()),
        "all_class_trajectories_feasible": bool(data[data.method == "OracleClass"].feasible.all()),
    }
    gates["all_gates_pass"] = all(gates.values())
    seed_summary.to_csv(OUT / "experiment_10g_seed_summary.csv", index=False)
    summary.to_csv(OUT / "experiment_10g_summary.csv", index=False)
    comparisons.to_csv(OUT / "experiment_10g_comparisons.csv", index=False)
    provenance = {
        "gates": gates,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
    }
    (OUT / "experiment_10g_conclusions.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return summary, comparisons, gates


def plot(summary, comparisons):
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.7))
    methods = list(METHODS)
    colors = ["#777777", "#0072B2", "#009E73"]
    for axis, scenario in zip(axes, ("moderate", "hard")):
        view = summary[summary.scenario == scenario].set_index("method").loc[methods]
        axis.bar(methods, view.mean_tracking, color=colors)
        axis.set_title(f"{scenario.capitalize()} maneuver")
        axis.set_ylabel("Mean yaw-rate RMSE [deg/s]")
        axis.tick_params(axis="x", rotation=18)
        comp = comparisons[(comparisons.scenario == scenario) & (comparisons.metric == "mean_tracking")].iloc[0]
        axis.text(0.5, 0.95, f"Class vs Global: {comp.class_improvement_pct:.1f}%",
                  transform=axis.transAxes, ha="center", va="top")
    fig.tight_layout()
    fig.savefig(FIGURE)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["select", "confirm", "diagnostic", "all"], nargs="?", default="all")
    args = parser.parse_args()
    cfg = json.loads(CONFIG.read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    if args.phase in ("select", "all"):
        selected = select_controller(cfg)
    else:
        selected = json.loads(SELECTION.read_text())["selected"]
    if args.phase in ("confirm", "all"):
        with ProcessPoolExecutor(max_workers=min(5, len(cfg["confirmation_seeds"]))) as executor:
            pieces = list(executor.map(
                evaluate_seed,
                cfg["confirmation_seeds"],
                [cfg] * len(cfg["confirmation_seeds"]),
                [selected] * len(cfg["confirmation_seeds"]),
                ["confirmation"] * len(cfg["confirmation_seeds"]),
            ))
        data = pd.concat(pieces, ignore_index=True)
        data.to_csv(OUT / "experiment_10g_confirmation.csv.gz", index=False, compression="gzip")
        summary, comparisons, gates = summarize(data, cfg)
        plot(summary, comparisons)
        print(summary.to_string(index=False))
        print(comparisons.to_string(index=False))
        print(json.dumps(gates, indent=2))
    if args.phase == "diagnostic":
        conservative = next(item for item in cfg["controller_candidates"] if item["name"] == "C04")
        with ProcessPoolExecutor(max_workers=min(5, len(cfg["confirmation_seeds"]))) as executor:
            pieces = list(executor.map(
                evaluate_seed,
                cfg["confirmation_seeds"],
                [cfg] * len(cfg["confirmation_seeds"]),
                [conservative] * len(cfg["confirmation_seeds"]),
                ["posthoc_diagnostic"] * len(cfg["confirmation_seeds"]),
            ))
        diagnostic = pd.concat(pieces, ignore_index=True)
        diagnostic.to_csv(OUT / "experiment_10g_posthoc_c04.csv.gz", index=False, compression="gzip")
        result = diagnostic.groupby(["scenario", "method"]).tracking_rmse_deg_s.agg(["mean", "max"])
        result.to_csv(OUT / "experiment_10g_posthoc_c04_summary.csv")
        print(result.to_string())


if __name__ == "__main__":
    main()
