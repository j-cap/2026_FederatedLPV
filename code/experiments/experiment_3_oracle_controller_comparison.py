"""Phase 3: oracle LQI controller comparison on the linear bicycle fleet."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm

from federated_lpv import (
    augmented_tracking_matrices,
    design_oracle_controllers,
    fit_oracle_architectures,
    sample_fleet,
)


ROOT = Path(__file__).resolve().parents[2]
FIGURE = ROOT / "results" / "figures" / "experiment_3_oracle_controllers.pdf"
SUMMARY_TABLE = ROOT / "results" / "tables" / "experiment_3_summary.csv"
FAMILY_TABLE = ROOT / "results" / "tables" / "experiment_3_family_summary.csv"
STABILITY_TABLE = ROOT / "results" / "tables" / "experiment_3_stability.csv"
SUMMARY_JSON = ROOT / "results" / "tables" / "experiment_3_summary.json"
DT = 0.01
DURATION = 12.0
FIT_SPEEDS = np.asarray([12.0, 16.0, 20.0, 24.0, 28.0])
CONTROL_SPEEDS = np.asarray([10.0, 15.0, 20.0, 25.0, 30.0])
DENSE_SPEEDS = np.linspace(10.0, 30.0, 201)
Q = np.diag([100.0, 10.0, 500.0])
R = np.asarray([[10.0]])
METHODS = ("M1", "M2", "M3", "M4", "M5")
FAMILIES = ("nominal", "heavy", "handling")
SEEDS = tuple(range(1, 11))


def smooth_profile(time: np.ndarray, values: tuple[float, float, float]) -> np.ndarray:
    midpoint = DURATION / 2.0
    first, second = np.clip(time / midpoint, 0, 1), np.clip((time - midpoint) / midpoint, 0, 1)
    blend_first, blend_second = 0.5 - 0.5 * np.cos(np.pi * first), 0.5 - 0.5 * np.cos(np.pi * second)
    return np.where(time <= midpoint, values[0] + (values[1] - values[0]) * blend_first,
                    values[1] + (values[2] - values[1]) * blend_second)


def reference_signal(time: np.ndarray, maneuver: str) -> np.ndarray:
    envelope = np.sin(np.pi * time / DURATION) ** 2
    if maneuver == "sine":
        values = 5.0 * envelope * np.sin(2 * np.pi * 0.25 * time)
    elif maneuver == "lane":
        values = 6.0 * (np.exp(-0.5 * ((time - 4.0) / 0.8) ** 2)
                        - np.exp(-0.5 * ((time - 7.0) / 0.8) ** 2))
    elif maneuver == "unseen_s_curve":
        values = 7.0 * (np.exp(-0.5 * ((time - 3.0) / 0.55) ** 2)
                        - 1.25 * np.exp(-0.5 * ((time - 5.3) / 0.7) ** 2)
                        + 0.75 * np.exp(-0.5 * ((time - 8.2) / 0.9) ** 2))
    else:
        raise ValueError(f"unknown maneuver: {maneuver}")
    return np.deg2rad(values)


def exact_matrices(client, speeds: np.ndarray):
    speed = np.asarray(speeds, dtype=float)
    p = client.parameters
    augmented = np.zeros((len(speed), 3, 3))
    augmented[:, 0, 0] = -(p.front_stiffness + p.rear_stiffness) / (p.mass * speed)
    augmented[:, 0, 1] = ((p.rear_stiffness * p.rear_length - p.front_stiffness * p.front_length)
                              / (p.mass * speed**2) - 1.0)
    augmented[:, 1, 0] = ((p.rear_stiffness * p.rear_length - p.front_stiffness * p.front_length)
                              / p.yaw_inertia)
    augmented[:, 1, 1] = -(p.front_stiffness * p.front_length**2
                            + p.rear_stiffness * p.rear_length**2) / (p.yaw_inertia * speed)
    augmented[:, 0, 2] = p.front_stiffness / (p.mass * speed)
    augmented[:, 1, 2] = p.front_stiffness * p.front_length / p.yaw_inertia
    discrete = expm(augmented * DT)
    return discrete[:, :2, :2], discrete[:, :2, 2:]


def scheduled_gains(controller, family: str, speeds: np.ndarray):
    key = "global" if controller.specialization == "global" else family
    if controller.interpolation == "constant":
        return (np.repeat(controller.gains[key][0][None, :], len(speeds), axis=0),
                np.repeat(controller.prefilters[key][0], len(speeds)))
    if controller.interpolation == "nearest":
        indices = np.abs(np.asarray(speeds)[:, None] - controller.speeds[None, :]).argmin(axis=1)
        return controller.gains[key][indices], controller.prefilters[key][indices]
    gains = np.column_stack([np.interp(speeds, controller.speeds, controller.gains[key][:, column])
                             for column in range(3)])
    return gains, np.interp(speeds, controller.speeds, controller.prefilters[key])


def simulate(client, controller, speed: np.ndarray, reference: np.ndarray, a, b):
    gains, prefilters = scheduled_gains(controller, client.family, speed[:-1])
    state = np.zeros((len(speed), 2))
    integral = np.zeros(len(speed))
    steering = np.zeros(len(speed) - 1)
    for index in range(len(speed) - 1):
        augmented_state = np.r_[state[index], integral[index]]
        steering[index] = -gains[index] @ augmented_state + prefilters[index] * reference[index]
        state[index + 1] = a[index] @ state[index] + b[index, :, 0] * steering[index]
        integral[index + 1] = integral[index] + DT * (state[index, 1] - reference[index])
    tracking = state[:, 1] - reference
    steering_rate = np.diff(steering) / DT
    return {
        "tracking_rmse_deg_s": float(np.rad2deg(np.sqrt(np.mean(tracking**2)))),
        "beta_rms_deg": float(np.rad2deg(np.sqrt(np.mean(state[:, 0] ** 2)))),
        "steering_rms_deg": float(np.rad2deg(np.sqrt(np.mean(steering**2)))),
        "peak_steering_deg": float(np.rad2deg(np.max(np.abs(steering)))),
        "peak_steering_rate_deg_s": float(np.rad2deg(np.max(np.abs(steering_rate)))),
    }, state, steering


def stability_audit(clients, controllers):
    rows = []
    for client in clients:
        matrices_a, matrices_b = exact_matrices(client, DENSE_SPEEDS)
        for method, controller in controllers.items():
            gains, _ = scheduled_gains(controller, client.family, DENSE_SPEEDS)
            radii = []
            for a, b, gain in zip(matrices_a, matrices_b, gains):
                augmented_a, augmented_b = augmented_tracking_matrices(a, b, DT)
                radii.append(max(abs(np.linalg.eigvals(augmented_a - augmented_b @ gain[None, :]))))
            rows.append({"client_id": client.client_id, "family": client.family, "method": method,
                         "max_spectral_radius": float(max(radii)),
                         "min_spectral_margin": float(1.0 - max(radii))})
    return rows


def scenario_definitions(time):
    constant = np.full_like(time, 20.0)
    return {
        "C1": [("nominal", smooth_profile(time, (12.0, 28.0, 16.0)), reference_signal(time, "sine"))],
        "C2": [("all", constant, reference_signal(time, "lane"))],
        "C3": [("all", smooth_profile(time, values), reference_signal(time, "lane"))
               for values in ((12.0, 20.0, 15.0), (20.0, 28.0, 18.0), (15.0, 25.0, 12.0))],
        "C4": [("all", smooth_profile(time, (28.0, 16.0, 24.0)), reference_signal(time, "unseen_s_curve"))],
    }


def collect_closed_loop_rows(seed, clients, controllers, scenarios):
    rows, representative = [], None
    for scenario, variants in scenarios.items():
        for variant, (family_filter, speed, reference) in enumerate(variants):
            for client in clients:
                if family_filter != "all" and client.family != family_filter:
                    continue
                matrices_a, matrices_b = exact_matrices(client, speed[:-1])
                for method, controller in controllers.items():
                    metrics, state, steering = simulate(
                        client, controller, speed, reference, matrices_a, matrices_b
                    )
                    rows.append({"seed": seed, "scenario": scenario, "variant": variant,
                                 "client_id": client.client_id, "family": client.family,
                                 "method": method, **metrics})
                    if (seed == 1 and scenario == "C4" and client.client_id == "heavy_00"
                            and method in {"M1", "M3", "M4"}):
                        if representative is None:
                            representative = {"speed": speed, "reference": reference}
                        representative[method] = {"state": state, "steering": steering}
    return rows, representative


def summarize(rows):
    summary, family = [], []
    metrics = ("tracking_rmse_deg_s", "beta_rms_deg", "steering_rms_deg",
               "peak_steering_deg", "peak_steering_rate_deg_s")
    for scenario in ("C1", "C2", "C3", "C4"):
        for method in METHODS:
            selected = [row for row in rows if row["scenario"] == scenario and row["method"] == method]
            item = {"scenario": scenario, "method": method, "runs": len(selected)}
            for metric in metrics:
                values = np.asarray([row[metric] for row in selected])
                item[f"mean_{metric}"] = float(values.mean())
                item[f"worst_{metric}"] = float(values.max())
            family_means = []
            for family_name in FAMILIES:
                subset = [row for row in selected if row["family"] == family_name]
                if not subset:
                    continue
                tracking = np.asarray([row["tracking_rmse_deg_s"] for row in subset])
                family_means.append(float(tracking.mean()))
                family.append({"scenario": scenario, "method": method, "family": family_name,
                               "mean_tracking_rmse_deg_s": float(tracking.mean()),
                               "worst_tracking_rmse_deg_s": float(tracking.max())})
            item["family_gap_deg_s"] = max(family_means) - min(family_means) if len(family_means) > 1 else 0.0
            summary.append(item)
    return summary, family


def paired_seed_reduction(rows, scenario: str, reference_method: str, improved_method: str, metric: str):
    values = []
    for seed in SEEDS:
        means = {method: np.mean([row[metric] for row in rows if row["seed"] == seed
                                  and row["scenario"] == scenario and row["method"] == method])
                 for method in (reference_method, improved_method)}
        values.append(100 * (means[reference_method] - means[improved_method]) / means[reference_method])
    return {"mean_pct": float(np.mean(values)), "min_pct": float(np.min(values)),
            "max_pct": float(np.max(values)), "positive_seeds": int(sum(value > 0 for value in values))}


def make_figure(time, representative, summary, stability):
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 5.6), constrained_layout=True)
    axes[0, 0].plot(time, representative["speed"], color="0.25")
    axes[0, 0].set(xlabel="time [s]", ylabel="speed [m/s]", title="C4 unseen speed sequence")
    for method in ("M1", "M3", "M4"):
        axes[0, 1].plot(time, np.rad2deg(representative[method]["state"][:, 1]), label=method)
    axes[0, 1].plot(time, np.rad2deg(representative["reference"]), "k--", label="reference")
    axes[0, 1].set(xlabel="time [s]", ylabel="yaw rate [deg/s]", title="Representative C4 tracking")
    axes[0, 1].legend(frameon=False, ncol=2, fontsize=8)

    c3 = {row["method"]: row for row in summary if row["scenario"] == "C3"}
    positions = np.arange(len(METHODS))
    axes[1, 0].bar(positions - 0.18, [c3[m]["mean_tracking_rmse_deg_s"] for m in METHODS], 0.36, label="mean")
    axes[1, 0].bar(positions + 0.18, [c3[m]["worst_tracking_rmse_deg_s"] for m in METHODS], 0.36, label="worst")
    axes[1, 0].set(xticks=positions, xticklabels=METHODS, ylabel="yaw RMSE [deg/s]", title="C3 combined variation")
    axes[1, 0].legend(frameon=False)

    worst_radii = [max(row["max_spectral_radius"] for row in stability if row["method"] == method)
                   for method in METHODS]
    axes[1, 1].bar(METHODS, worst_radii)
    axes[1, 1].axhline(1.0, color="tab:red", linestyle="--")
    axes[1, 1].set(ylabel="worst spectral radius", title="Frozen-grid stability audit", ylim=(0.9, 1.005))
    for axis in axes.flat:
        axis.grid(alpha=0.25)
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, bbox_inches="tight")
    plt.close(fig)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    time = np.arange(0.0, DURATION + DT, DT)
    scenarios = scenario_definitions(time)
    rows, stability_rows, representative = [], [], None
    for seed in SEEDS:
        clients = sample_fleet(seed)
        architectures = fit_oracle_architectures(clients, FIT_SPEEDS, DT)
        controllers = design_oracle_controllers(architectures, DT, Q, R, CONTROL_SPEEDS, FIT_SPEEDS)
        stability = stability_audit(clients, controllers)
        for row in stability:
            row["seed"] = seed
        stability_rows.extend(stability)
        seed_rows, seed_representative = collect_closed_loop_rows(seed, clients, controllers, scenarios)
        rows.extend(seed_rows)
        if seed == 1:
            representative = seed_representative
    summary, family = summarize(rows)
    conclusions = {
        "C1_M1_to_M3_tracking": paired_seed_reduction(rows, "C1", "M1", "M3", "tracking_rmse_deg_s"),
        "C2_M3_to_M4_tracking": paired_seed_reduction(rows, "C2", "M3", "M4", "tracking_rmse_deg_s"),
        "C3_M2_to_M4_tracking": paired_seed_reduction(rows, "C3", "M2", "M4", "tracking_rmse_deg_s"),
        "C3_M3_to_M4_tracking": paired_seed_reduction(rows, "C3", "M3", "M4", "tracking_rmse_deg_s"),
        "C4_M3_to_M4_tracking": paired_seed_reduction(rows, "C4", "M3", "M4", "tracking_rmse_deg_s"),
        "C4_M2_to_M4_tracking": paired_seed_reduction(rows, "C4", "M2", "M4", "tracking_rmse_deg_s"),
        "C3_M5_to_M4_tracking": paired_seed_reduction(rows, "C3", "M5", "M4", "tracking_rmse_deg_s"),
        "worst_spectral_radius": {method: max(row["max_spectral_radius"] for row in stability_rows
                                                if row["method"] == method) for method in METHODS},
    }
    write_csv(SUMMARY_TABLE, summary)
    write_csv(FAMILY_TABLE, family)
    write_csv(STABILITY_TABLE, stability_rows)
    SUMMARY_JSON.write_text(json.dumps({"summary": summary, "conclusions": conclusions}, indent=2) + "\n")
    make_figure(time, representative, summary, stability_rows)
    print(json.dumps(conclusions, indent=2))
    print("Wrote Phase-3 oracle controller comparison")


if __name__ == "__main__":
    main()
