"""10H: higher-order LPV necessity and oracle LQI headroom audit."""

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
from scipy.linalg import expm, solve_discrete_are

from federated_lpv import VehicleParameters

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "code/config/experiment_10h.json"
OUT = ROOT / "results/tables"
FIGURE = ROOT / "results/figures/experiment_10h_higher_order_lpv_gate.pdf"
SELECTION = OUT / "experiment_10h_frozen_selection.json"
METHODS = ("LocalRestricted", "Global", "OracleClass", "ClientExact")


@dataclass(frozen=True)
class RelaxationClient:
    client_id: str
    vehicle_class: str
    parameters: VehicleParameters
    actuator_time_constant: float
    front_relaxation_length: float
    rear_relaxation_length: float


def sample_fleet(seed: int, cfg: dict) -> list[RelaxationClient]:
    rng = np.random.default_rng(seed)
    scatter = cfg["scatter"]
    clients = []
    for name, center in cfg["classes"].items():
        for index in range(cfg["clients_per_class"]):
            def scale(key):
                return rng.uniform(1 - scatter[key], 1 + scatter[key])

            parameters = VehicleParameters(
                center["mass"] * scale("mass"),
                center["yaw_inertia"] * scale("yaw_inertia"),
                center["front_length"] * scale("geometry"),
                center["rear_length"] * scale("geometry"),
                center["front_stiffness"] * scale("stiffness"),
                center["rear_stiffness"] * scale("stiffness"),
            )
            clients.append(RelaxationClient(
                f"{name}_{index:02d}", name, parameters,
                center["actuator_time_constant"] * scale("actuator_time_constant"),
                center["front_relaxation_length"] * scale("relaxation_length"),
                center["rear_relaxation_length"] * scale("relaxation_length"),
            ))
    return clients


def continuous_matrices(client: RelaxationClient, speed: float):
    """Return matrices for x=[beta,r,Fyf/m,Fyr/m,delta_a]."""
    p, v = client.parameters, float(speed)
    cf_m, cr_m = p.front_stiffness / p.mass, p.rear_stiffness / p.mass
    a = np.zeros((5, 5))
    a[0, 1] = -1.0
    a[0, 2:4] = 1.0 / v
    a[1, 2] = p.mass * p.front_length / p.yaw_inertia
    a[1, 3] = -p.mass * p.rear_length / p.yaw_inertia
    a[2, 0] = -v * cf_m / client.front_relaxation_length
    a[2, 1] = -p.front_length * cf_m / client.front_relaxation_length
    a[2, 2] = -v / client.front_relaxation_length
    a[2, 4] = v * cf_m / client.front_relaxation_length
    a[3, 0] = -v * cr_m / client.rear_relaxation_length
    a[3, 1] = p.rear_length * cr_m / client.rear_relaxation_length
    a[3, 3] = -v / client.rear_relaxation_length
    a[4, 4] = -1.0 / client.actuator_time_constant
    b = np.zeros((5, 1))
    b[4, 0] = 1.0 / client.actuator_time_constant
    return a, b


def discrete_matrices(client: RelaxationClient, speed: float, dt: float):
    a, b = continuous_matrices(client, speed)
    block = np.block([[a, b], [np.zeros((1, 6))]])
    discrete = expm(block * dt)
    return discrete[:5, :5], discrete[:5, 5:]


def architecture_matrices(clients, speeds, dt):
    exact = {
        client.client_id: [discrete_matrices(client, speed, dt) for speed in speeds]
        for client in clients
    }
    global_model = [
        (np.mean([exact[c.client_id][j][0] for c in clients], axis=0),
         np.mean([exact[c.client_id][j][1] for c in clients], axis=0))
        for j in range(len(speeds))
    ]
    class_models = {}
    for name in sorted({client.vehicle_class for client in clients}):
        members = [client for client in clients if client.vehicle_class == name]
        class_models[name] = [
            (np.mean([exact[c.client_id][j][0] for c in members], axis=0),
             np.mean([exact[c.client_id][j][1] for c in members], axis=0))
            for j in range(len(speeds))
        ]
    return exact, global_model, class_models


def raw_basis(speeds):
    speeds = np.asarray(speeds, dtype=float)
    normalized = (speeds - 20.0) / 10.0
    return np.column_stack([np.ones_like(speeds), 1 / speeds, 1 / speeds**2,
                            normalized, normalized**2])


def multisine(length, dt, band, std, rng):
    time = np.arange(length) * dt
    frequencies = np.linspace(band[0], band[1], 4)
    phases = rng.uniform(0, 2 * np.pi, len(frequencies))
    signal = sum(np.sin(2 * np.pi * frequency * time + phase)
                 for frequency, phase in zip(frequencies, phases))
    return signal * std / np.std(signal)


def identification_design(client, speeds, band, cfg, rng):
    rows = []
    phi = raw_basis(speeds)
    for speed_index, speed in enumerate(speeds):
        a, b = discrete_matrices(client, speed, cfg["sample_time"])
        samples = cfg["identification_samples_per_speed"]
        inputs = multisine(samples, cfg["sample_time"], band,
                           cfg["identification_input_std_rad"], rng)
        state = np.zeros(5)
        for command in inputs:
            regressor = np.r_[state, command]
            rows.append(np.kron(phi[speed_index], regressor))
            state = a @ state + b[:, 0] * command
    return np.asarray(rows)


def coverage_audit(seed, cfg):
    rng = np.random.default_rng(seed + 800_000)
    clients = sample_fleet(seed, cfg)
    local_designs = {}
    rows = []
    for index, client in enumerate(clients):
        block_index = index % len(cfg["coverage_blocks"])
        band_index = (index // len(cfg["coverage_blocks"])) % len(cfg["frequency_bands_hz"])
        design = identification_design(
            client, cfg["coverage_blocks"][block_index],
            cfg["frequency_bands_hz"][band_index], cfg, rng,
        )
        local_designs[client.client_id] = design
        scaled = design / np.maximum(np.linalg.norm(design, axis=0), 1e-15)
        singular = np.linalg.svd(scaled, compute_uv=False)
        rows.append({
            "seed": seed, "scope": "local", "client": client.client_id,
            "vehicle_class": client.vehicle_class, "rank": int(np.linalg.matrix_rank(design)),
            "required_rank": design.shape[1],
            "condition": float(singular[0] / singular[-1]) if singular[-1] > 1e-14 else np.inf,
        })
    for name in sorted({client.vehicle_class for client in clients}):
        members = [client for client in clients if client.vehicle_class == name]
        design = np.vstack([local_designs[client.client_id] for client in members])
        scaled = design / np.maximum(np.linalg.norm(design, axis=0), 1e-15)
        singular = np.linalg.svd(scaled, compute_uv=False)
        rows.append({
            "seed": seed, "scope": "class_aggregate", "client": name,
            "vehicle_class": name, "rank": int(np.linalg.matrix_rank(design)),
            "required_rank": design.shape[1], "condition": float(singular[0] / singular[-1]),
        })
    return pd.DataFrame(rows)


def design_lqi(a, b, dt, weights):
    c = np.array([[0.0, 1.0, 0.0, 0.0, 0.0]])
    aa = np.block([[a, np.zeros((5, 1))], [dt * c, np.ones((1, 1))]])
    bb = np.vstack([b, [[0.0]]])
    q = np.diag([
        weights["q_beta"], weights["q_yaw"], weights["q_force"],
        weights["q_force"], weights["q_actuator"], weights["q_integral"],
    ])
    r = np.array([[weights["r"]]])
    controllability = np.column_stack([np.linalg.matrix_power(aa, j) @ bb for j in range(6)])
    if np.linalg.matrix_rank(controllability) < 6:
        raise ValueError("uncontrollable augmented model")
    p = solve_discrete_are(aa, bb, q, r)
    gain = np.linalg.solve(r + bb.T @ p @ bb, bb.T @ p @ aa).ravel()
    equilibrium = np.linalg.solve(
        np.block([[np.eye(5) - a, -b], [c, np.zeros((1, 1))]]),
        np.r_[np.zeros(5), 1.0],
    )
    prefilter = float(equilibrium[5] + gain[:5] @ equilibrium[:5])
    return gain, prefilter


def design_schedule(matrices, weights, dt):
    try:
        designs = [design_lqi(a, b, dt, weights) for a, b in matrices]
    except (ValueError, np.linalg.LinAlgError):
        return None
    return np.asarray([item[0] for item in designs]), np.asarray([item[1] for item in designs])


def interpolate(values, grid, speed):
    return np.asarray([np.interp(speed, grid, values[:, column])
                       for column in range(values.shape[1])])


def scenarios(cfg):
    time = np.arange(0, cfg["duration_s"] + cfg["sample_time"], cfg["sample_time"])
    envelope = np.sin(np.pi * time / time[-1]) ** 2
    speed_a = 20 + 9 * np.sin(2 * np.pi * time / time[-1] - np.pi / 2)
    speed_b = 20 + 8 * np.sin(2 * np.pi * time / (time[-1] / 2) + 0.3)
    reference_a = envelope * (
        0.040 * np.sin(2 * np.pi * 0.35 * time)
        + 0.020 * np.sin(2 * np.pi * 0.9 * time + 0.4)
        + 0.010 * np.sin(2 * np.pi * 1.35 * time + 0.8)
    )
    reference_b = 0.10 * (
        np.exp(-0.5 * ((time - 3.0) / 0.32) ** 2)
        - 1.15 * np.exp(-0.5 * ((time - 6.1) / 0.36) ** 2)
        + 0.85 * np.exp(-0.5 * ((time - 9.2) / 0.40) ** 2)
    )
    return {
        "broadband": {"time": time, "speed": speed_a, "reference": reference_a},
        "transient": {"time": time, "speed": speed_b, "reference": reference_b},
    }


def simulate(client, true_matrices, schedule, scenario, cfg):
    if schedule is None:
        return None
    grid = np.asarray(cfg["speed_grid"], dtype=float)
    gains, prefilters = schedule
    state = np.zeros((len(scenario["time"]), 5))
    integral = np.zeros(len(scenario["time"]))
    command = np.zeros(len(scenario["time"]) - 1)
    matrix_a = np.asarray([item[0].ravel() for item in true_matrices])
    matrix_b = np.asarray([item[1].ravel() for item in true_matrices])
    for k, speed in enumerate(scenario["speed"][:-1]):
        gain = interpolate(gains, grid, speed)
        prefilter = float(np.interp(speed, grid, prefilters))
        command[k] = -gain @ np.r_[state[k], integral[k]] + prefilter * scenario["reference"][k]
        a = interpolate(matrix_a, grid, speed).reshape(5, 5)
        b = interpolate(matrix_b, grid, speed).reshape(5, 1)
        state[k + 1] = a @ state[k] + b[:, 0] * command[k]
        integral[k + 1] = integral[k] + cfg["sample_time"] * (
            state[k, 1] - scenario["reference"][k]
        )
        if not np.isfinite(state[k + 1]).all():
            return None
    error = state[:, 1] - scenario["reference"]
    return {
        "tracking_rmse_deg_s": float(np.rad2deg(np.sqrt(np.mean(error**2)))),
        "tracking_peak_deg_s": float(np.rad2deg(np.max(np.abs(error)))),
        "beta_peak_deg": float(np.rad2deg(np.max(np.abs(state[:, 0])))),
        "command_peak_deg": float(np.rad2deg(np.max(np.abs(command)))),
        "feasible": bool(np.rad2deg(np.max(np.abs(command))) <= cfg["maximum_command_deg"]),
    }


def stability_radius(true_matrices, schedule):
    if schedule is None:
        return np.inf
    gains, _ = schedule
    radii = []
    for (a, b), gain in zip(true_matrices, gains):
        c = np.array([[0.0, 1.0, 0.0, 0.0, 0.0]])
        aa = np.block([[a, np.zeros((5, 1))], [0.01 * c, np.ones((1, 1))]])
        bb = np.vstack([b, [[0.0]]])
        radii.append(max(abs(np.linalg.eigvals(aa - bb @ gain[None, :]))))
    return float(max(radii))


def evaluate_seed(seed, cfg, weights, phase):
    clients = sample_fleet(seed, cfg)
    grid = np.asarray(cfg["speed_grid"], dtype=float)
    exact, global_model, class_models = architecture_matrices(clients, grid, cfg["sample_time"])
    global_schedule = design_schedule(global_model, weights, cfg["sample_time"])
    class_schedules = {name: design_schedule(model, weights, cfg["sample_time"])
                       for name, model in class_models.items()}
    exact_schedules = {name: design_schedule(model, weights, cfg["sample_time"])
                       for name, model in exact.items()}
    rows = []
    for scenario_name, scenario in scenarios(cfg).items():
        for client_index, client in enumerate(clients):
            local_speeds = np.asarray(
                cfg["coverage_blocks"][client_index % len(cfg["coverage_blocks"])], dtype=float
            )
            local_matrices = []
            for speed in grid:
                nearest = int(np.argmin(np.abs(local_speeds - speed)))
                source_speed = local_speeds[nearest]
                source_index = int(np.argmin(np.abs(grid - source_speed)))
                local_matrices.append(exact[client.client_id][source_index])
            local_schedule = design_schedule(local_matrices, weights, cfg["sample_time"])
            schedules = {
                "LocalRestricted": local_schedule,
                "Global": global_schedule,
                "OracleClass": class_schedules[client.vehicle_class],
                "ClientExact": exact_schedules[client.client_id],
            }
            if phase == "development":
                schedules = {"ClientExact": schedules["ClientExact"]}
            for method, schedule in schedules.items():
                metrics = simulate(client, exact[client.client_id], schedule, scenario, cfg)
                if metrics is None:
                    metrics = {"tracking_rmse_deg_s": np.inf, "tracking_peak_deg_s": np.inf,
                               "beta_peak_deg": np.inf, "command_peak_deg": np.inf,
                               "feasible": False}
                rows.append({
                    "seed": seed, "phase": phase, "scenario": scenario_name,
                    "client": client.client_id, "vehicle_class": client.vehicle_class,
                    "method": method, "controller": weights["name"],
                    "spectral_radius": stability_radius(exact[client.client_id], schedule),
                } | metrics)
    return pd.DataFrame(rows)


def bootstrap_interval(values, resamples, seed=10_008):
    values = np.asarray(values)
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, (resamples, len(values)), replace=True).mean(axis=1)
    return np.quantile(samples, [0.025, 0.975])


def select_controller(cfg):
    rows = []
    for weights in cfg["controller_candidates"]:
        with ProcessPoolExecutor(max_workers=5) as executor:
            pieces = list(executor.map(
                evaluate_seed, cfg["development_seeds"], [cfg] * 5,
                [weights] * 5, ["development"] * 5,
            ))
        data = pd.concat(pieces, ignore_index=True)
        rows.append({
            "controller": weights["name"],
            "mean_tracking": data.tracking_rmse_deg_s.mean(),
            "worst_tracking": data.tracking_rmse_deg_s.max(),
            "maximum_command_deg": data.command_peak_deg.max(),
            "maximum_spectral_radius": data.spectral_radius.max(),
            "all_feasible": bool(data.feasible.all() and (data.spectral_radius < 1).all()),
        })
    audit = pd.DataFrame(rows)
    eligible = audit[audit.all_feasible]
    if eligible.empty:
        raise RuntimeError("no controller passes the development gates")
    selected_name = eligible.loc[eligible.mean_tracking.idxmin(), "controller"]
    selected = next(item for item in cfg["controller_candidates"] if item["name"] == selected_name)
    audit.to_csv(OUT / "experiment_10h_controller_audit.csv", index=False)
    SELECTION.write_text(json.dumps({
        "selected": selected,
        "rule": "minimum ClientExact mean tracking among feasible stable candidates",
    }, indent=2) + "\n")
    return selected


def summarize(data, coverage, cfg):
    seed_summary = data.groupby(["seed", "scenario", "method"]).agg(
        mean_tracking=("tracking_rmse_deg_s", "mean"),
        worst_tracking=("tracking_rmse_deg_s", "max"),
        maximum_command=("command_peak_deg", "max"),
        maximum_radius=("spectral_radius", "max"),
        feasible_fraction=("feasible", "mean"),
    ).reset_index()
    summary = seed_summary.groupby(["scenario", "method"]).mean(numeric_only=True).reset_index()
    comparisons = []
    for scenario in sorted(seed_summary.scenario.unique()):
        view = seed_summary[seed_summary.scenario == scenario].pivot(index="seed", columns="method")
        for metric in ("mean_tracking", "worst_tracking"):
            for method, baseline in (("Global", "LocalRestricted"),
                                     ("OracleClass", "Global")):
                method_values = view[metric][method]
                baseline_values = view[metric][baseline]
                improvement = 100 * (baseline_values - method_values) / baseline_values
                interval = bootstrap_interval(improvement, cfg["bootstrap_resamples"])
                comparisons.append({
                    "scenario": scenario, "metric": metric, "method": method,
                    "baseline": baseline, "improvement_pct": float(improvement.mean()),
                    "ci_low": float(interval[0]), "ci_high": float(interval[1]),
                    "absolute_improvement_deg_s": float((baseline_values - method_values).mean()),
                    "win_fraction": float((method_values < baseline_values).mean()),
                })
            global_values = view[metric]["Global"]
            exact_values = view[metric]["ClientExact"]
            excess = 100 * (global_values - exact_values) / exact_values
            comparisons.append({
                "scenario": scenario, "metric": metric, "method": "Global",
                "baseline": "ClientExact", "improvement_pct": float(-excess.mean()),
                "ci_low": np.nan, "ci_high": np.nan,
                "absolute_improvement_deg_s": float((exact_values - global_values).mean()),
                "win_fraction": float((global_values < exact_values).mean()),
            })
    comparisons = pd.DataFrame(comparisons)
    local = coverage[coverage.scope == "local"]
    aggregate = coverage[coverage.scope == "class_aggregate"]
    gates = {
        "all_local_designs_rank_deficient": bool((local["rank"] < local.required_rank).all()),
        "all_class_aggregates_full_rank": bool((aggregate["rank"] == aggregate.required_rank).all()),
        "aggregate_condition_gate": bool((aggregate.condition < cfg["maximum_aggregate_condition"]).all()),
        "sharing_improvement_gate": bool((comparisons[
            (comparisons.method == "Global") & (comparisons.baseline == "LocalRestricted")
        ].improvement_pct >= cfg["minimum_sharing_improvement_pct"]).all()),
        "sharing_positive_interval_gate": bool((comparisons[
            (comparisons.method == "Global") & (comparisons.baseline == "LocalRestricted")
        ].ci_low > 0).all()),
        "global_near_exact_gate": bool((comparisons[
            (comparisons.method == "Global") & (comparisons.baseline == "ClientExact")
        ].improvement_pct >= -cfg["maximum_global_excess_vs_exact_pct"]).all()),
        "all_class_controllers_feasible": bool(data[data.method == "OracleClass"].feasible.all()),
        "all_class_controllers_stable": bool((data[data.method == "OracleClass"].spectral_radius < 1).all()),
    }
    gates["all_gates_pass"] = all(gates.values())
    coverage.to_csv(OUT / "experiment_10h_coverage_audit.csv", index=False)
    seed_summary.to_csv(OUT / "experiment_10h_seed_summary.csv", index=False)
    summary.to_csv(OUT / "experiment_10h_summary.csv", index=False)
    comparisons.to_csv(OUT / "experiment_10h_comparisons.csv", index=False)
    (OUT / "experiment_10h_conclusions.json").write_text(json.dumps({
        "gates": gates,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
    }, indent=2) + "\n")
    return summary, comparisons, gates


def plot(summary, coverage, comparisons):
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.7))
    methods = list(METHODS)
    colors = ["#D55E00", "#777777", "#0072B2", "#009E73"]
    for axis, scenario in zip(axes[:2], ("broadband", "transient")):
        view = summary[summary.scenario == scenario].set_index("method").loc[methods]
        axis.bar(methods, view.mean_tracking, color=colors)
        axis.set_title(scenario.capitalize())
        axis.set_ylabel("Mean yaw-rate RMSE [deg/s]")
        axis.tick_params(axis="x", rotation=18)
        comp = comparisons[(comparisons.scenario == scenario) &
                           (comparisons.metric == "mean_tracking") &
                           (comparisons.method == "Global") &
                           (comparisons.baseline == "LocalRestricted")].iloc[0]
        axis.text(0.5, 0.95, f"Sharing gain: {comp.improvement_pct:.1f}%",
                  transform=axis.transAxes, ha="center", va="top")
    grouped = coverage.groupby("scope")["rank"].mean()
    axes[2].bar(["Local", "Class aggregate"],
                [grouped["local"], grouped["class_aggregate"]],
                color=["#D55E00", "#009E73"])
    axes[2].axhline(coverage.required_rank.iloc[0], color="black", linestyle="--", label="Required")
    axes[2].set_ylabel("Mean regression rank")
    axes[2].set_title("Information coverage")
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(FIGURE)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["select", "confirm", "all"], nargs="?", default="all")
    args = parser.parse_args()
    cfg = json.loads(CONFIG.read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    selected = select_controller(cfg) if args.phase in ("select", "all") else json.loads(SELECTION.read_text())["selected"]
    if args.phase in ("confirm", "all"):
        with ProcessPoolExecutor(max_workers=5) as executor:
            pieces = list(executor.map(
                evaluate_seed, cfg["confirmation_seeds"], [cfg] * 10,
                [selected] * 10, ["confirmation"] * 10,
            ))
            coverage_pieces = list(executor.map(
                coverage_audit, cfg["confirmation_seeds"], [cfg] * 10,
            ))
        data = pd.concat(pieces, ignore_index=True)
        coverage = pd.concat(coverage_pieces, ignore_index=True)
        data.to_csv(OUT / "experiment_10h_confirmation.csv.gz", index=False, compression="gzip")
        summary, comparisons, gates = summarize(data, coverage, cfg)
        plot(summary, coverage, comparisons)
        print(summary.to_string(index=False))
        print(comparisons.to_string(index=False))
        print(json.dumps(gates, indent=2))


if __name__ == "__main__":
    main()
