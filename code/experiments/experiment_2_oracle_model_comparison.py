"""Phase 2: compare global, family-specific, LTI, LPV, and gridded oracle models."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm

from federated_lpv import fit_oracle_architectures, sample_fleet


ROOT = Path(__file__).resolve().parents[2]
FIGURE = ROOT / "results" / "figures" / "experiment_2_oracle_models.pdf"
SUMMARY_TABLE = ROOT / "results" / "tables" / "experiment_2_summary.csv"
FAMILY_TABLE = ROOT / "results" / "tables" / "experiment_2_family_summary.csv"
SUMMARY_JSON = ROOT / "results" / "tables" / "experiment_2_summary.json"
SAMPLE_TIME = 0.01
DURATION = 12.0
FIT_SPEEDS = np.asarray([12.0, 16.0, 20.0, 24.0, 28.0])
DENSE_SPEEDS = np.linspace(10.0, 30.0, 201)
SEEDS = tuple(range(1, 11))
METHODS = ("M1", "M2", "M3", "M4", "M5")
FAMILIES = ("nominal", "heavy", "handling")
PROFILE_VALUES = {
    "12-20-15": (12.0, 20.0, 15.0),
    "20-28-18": (20.0, 28.0, 18.0),
    "15-25-12": (15.0, 25.0, 12.0),
}


def smooth_profile(time: np.ndarray, values: tuple[float, float, float]) -> np.ndarray:
    midpoint = DURATION / 2.0
    first = np.clip(time / midpoint, 0.0, 1.0)
    second = np.clip((time - midpoint) / midpoint, 0.0, 1.0)
    blend_first = 0.5 - 0.5 * np.cos(np.pi * first)
    blend_second = 0.5 - 0.5 * np.cos(np.pi * second)
    return np.where(time <= midpoint, values[0] + (values[1] - values[0]) * blend_first,
                    values[1] + (values[2] - values[1]) * blend_second)


def steering_signal(time: np.ndarray) -> np.ndarray:
    envelope = np.sin(np.pi * time / DURATION) ** 2
    degrees = envelope * (1.2 * np.sin(2 * np.pi * 0.35 * time)
                          + 0.65 * np.sin(2 * np.pi * 0.73 * time + 0.4)
                          + 0.35 * np.sin(2 * np.pi * 1.17 * time + 1.1))
    return np.deg2rad(degrees)


def truth_matrices(client, speeds: np.ndarray):
    """Vectorized exact ZOH discretization for one client over many speeds."""
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
    discrete = expm(augmented * SAMPLE_TIME)
    return discrete[:, :2, :2], discrete[:, :2, 2:]


def combined_matrix_error(true_a, true_b, predicted_a, predicted_b):
    error_a = np.linalg.norm(predicted_a - true_a, axis=(1, 2)) / np.linalg.norm(true_a, axis=(1, 2))
    error_b = np.linalg.norm(predicted_b - true_b, axis=(1, 2)) / np.linalg.norm(true_b, axis=(1, 2))
    return np.sqrt(0.5 * (error_a**2 + error_b**2))


def propagate(a: np.ndarray, b: np.ndarray, steering: np.ndarray) -> np.ndarray:
    states = np.zeros((len(steering), 2))
    for index in range(len(steering) - 1):
        states[index + 1] = a[index] @ states[index] + b[index, :, 0] * steering[index]
    return states


def collect_matrix_rows(seed, clients, architectures):
    rows = []
    for client in clients:
        true_a, true_b = truth_matrices(client, DENSE_SPEEDS)
        for method in METHODS:
            predicted_a, predicted_b = architectures[method].predict(client.family, DENSE_SPEEDS)
            errors = combined_matrix_error(true_a, true_b, predicted_a, predicted_b)
            rows.extend({"seed": seed, "client_id": client.client_id, "family": client.family,
                         "method": method, "speed": float(speed), "matrix_error": float(error)}
                        for speed, error in zip(DENSE_SPEEDS, errors))
    return rows


def collect_trajectory_rows(seed, clients, architectures, profiles, steering):
    rows = []
    for client in clients:
        for profile_name, speed in profiles.items():
            true_a, true_b = truth_matrices(client, speed[:-1])
            reference = propagate(true_a, true_b, steering)
            reference_rms = np.sqrt(np.mean(reference[1:] ** 2))
            for method in METHODS:
                predicted_a, predicted_b = architectures[method].predict(client.family, speed[:-1])
                prediction = propagate(predicted_a, predicted_b, steering)
                one_step = (np.einsum("kij,kj->ki", predicted_a, reference[:-1])
                            + predicted_b[:, :, 0] * steering[:-1, None])
                one_error = one_step - reference[1:]
                free_error = prediction[1:] - reference[1:]
                rows.append({
                    "seed": seed, "client_id": client.client_id, "family": client.family,
                    "profile": profile_name, "method": method,
                    "one_step_relative_rmse": float(np.sqrt(np.mean(one_error**2)) / reference_rms),
                    "free_relative_rmse": float(np.sqrt(np.mean(free_error**2)) / reference_rms),
                    "beta_rmse_deg": float(np.rad2deg(np.sqrt(np.mean(free_error[:, 0] ** 2)))),
                    "yaw_rmse_deg_s": float(np.rad2deg(np.sqrt(np.mean(free_error[:, 1] ** 2)))),
                })
    return rows


def summarize(matrix_rows, trajectory_rows, architectures):
    summary, family_summary = [], []
    for method in METHODS:
        matrix = np.asarray([row["matrix_error"] for row in matrix_rows if row["method"] == method])
        fixed = np.asarray([row["matrix_error"] for row in matrix_rows
                            if row["method"] == method and np.isclose(row["speed"], 20.0)])
        trajectories = [row for row in trajectory_rows if row["method"] == method]
        item = {
            "method": method,
            "model_count": architectures[method].model_count,
            "dense_matrix_mean": float(matrix.mean()),
            "dense_matrix_worst": float(matrix.max()),
            "fixed20_matrix_mean": float(fixed.mean()),
        }
        for metric in ("one_step_relative_rmse", "free_relative_rmse", "beta_rmse_deg", "yaw_rmse_deg_s"):
            values = np.asarray([row[metric] for row in trajectories])
            item[f"trajectory_mean_{metric}"] = float(values.mean())
            item[f"trajectory_worst_{metric}"] = float(values.max())
        family_means = []
        for family in FAMILIES:
            family_matrix = np.asarray([row["matrix_error"] for row in matrix_rows
                                        if row["method"] == method and row["family"] == family])
            family_trajectories = [row for row in trajectories if row["family"] == family]
            family_free = np.asarray([row["free_relative_rmse"] for row in family_trajectories])
            family_means.append(float(family_free.mean()))
            family_summary.append({"method": method, "family": family,
                                   "matrix_mean": float(family_matrix.mean()),
                                   "matrix_worst": float(family_matrix.max()),
                                   "trajectory_free_mean": float(family_free.mean()),
                                   "trajectory_free_worst": float(family_free.max())})
        item["trajectory_family_gap"] = max(family_means) - min(family_means)
        summary.append(item)
    return summary, family_summary


def make_figure(matrix_rows, summary, family_summary):
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 5.6), constrained_layout=True)
    for method in METHODS:
        method_errors = np.asarray([row["matrix_error"] for row in matrix_rows
                                    if row["method"] == method]).reshape(-1, len(DENSE_SPEEDS))
        means = method_errors.mean(axis=0)
        axes[0, 0].semilogy(DENSE_SPEEDS, means, label=method)
    axes[0, 0].set(xlabel="speed [m/s]", ylabel="mean relative matrix error",
                   title="D1/D3: error versus speed")
    axes[0, 0].legend(ncol=3, frameon=False, fontsize=8)

    positions = np.arange(len(FAMILIES))
    for offset, method in ((-0.18, "M3"), (0.18, "M4")):
        values = [100 * np.mean([row["matrix_error"] for row in matrix_rows
                                 if row["method"] == method and row["family"] == family
                                 and np.isclose(row["speed"], 20.0)]) for family in FAMILIES]
        axes[0, 1].bar(positions + offset, values, 0.36, label=method)
    axes[0, 1].set(xticks=positions, xticklabels=FAMILIES, ylabel="dense-grid matrix error [%]",
                   title="D2: specialization at 20 m/s")
    axes[0, 1].legend(frameon=False)

    values = {row["method"]: row for row in summary}
    mean_free = [100 * values[method]["trajectory_mean_free_relative_rmse"] for method in METHODS]
    worst_free = [100 * values[method]["trajectory_worst_free_relative_rmse"] for method in METHODS]
    positions = np.arange(len(METHODS))
    axes[1, 0].bar(positions - 0.18, mean_free, 0.36, label="mean")
    axes[1, 0].bar(positions + 0.18, worst_free, 0.36, label="worst")
    axes[1, 0].set(xticks=positions, xticklabels=METHODS, ylabel="free-run RMSE [%]",
                   title="D3: varying-speed prediction")
    axes[1, 0].set_yscale("log")
    axes[1, 0].legend(frameon=False)

    counts = [values[method]["model_count"] for method in METHODS]
    for method, count, error in zip(METHODS, counts, mean_free):
        axes[1, 1].scatter(count, error, s=55, label=method)
        axes[1, 1].annotate(method, (count, error), xytext=(3, 3), textcoords="offset points")
    axes[1, 1].set(xlabel="model families / prototypes", ylabel="mean free-run RMSE [%]",
                   title="Accuracy--complexity view")
    axes[1, 1].set_yscale("log")
    for axis in axes.flat:
        axis.grid(alpha=0.25)
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, bbox_inches="tight")
    plt.close(fig)


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def percentage_reduction(reference: float, improved: float) -> float:
    return 100.0 * (reference - improved) / reference


def seed_reductions(trajectory_rows, reference_method: str, improved_method: str):
    reductions = []
    for seed in SEEDS:
        means = {}
        for method in (reference_method, improved_method):
            values = [row["free_relative_rmse"] for row in trajectory_rows
                      if row["seed"] == seed and row["method"] == method]
            means[method] = float(np.mean(values))
        reductions.append(percentage_reduction(means[reference_method], means[improved_method]))
    return {"min_pct": min(reductions), "max_pct": max(reductions),
            "positive_seeds": sum(value > 0.0 for value in reductions), "total_seeds": len(reductions)}


def main():
    time = np.arange(0.0, DURATION + SAMPLE_TIME, SAMPLE_TIME)
    steering = steering_signal(time)
    profiles = {name: smooth_profile(time, values) for name, values in PROFILE_VALUES.items()}
    matrix_rows, trajectory_rows = [], []
    final_architectures = None
    for seed in SEEDS:
        clients = sample_fleet(seed)
        architectures = fit_oracle_architectures(clients, FIT_SPEEDS, SAMPLE_TIME)
        matrix_rows.extend(collect_matrix_rows(seed, clients, architectures))
        trajectory_rows.extend(collect_trajectory_rows(seed, clients, architectures, profiles, steering))
        final_architectures = architectures
    summary, family_summary = summarize(matrix_rows, trajectory_rows, final_architectures)
    indexed = {row["method"]: row for row in summary}
    conclusions = {
        "lpv_relevance_global_free_reduction_pct": percentage_reduction(
            indexed["M1"]["trajectory_mean_free_relative_rmse"], indexed["M3"]["trajectory_mean_free_relative_rmse"]),
        "lpv_relevance_clustered_free_reduction_pct": percentage_reduction(
            indexed["M2"]["trajectory_mean_free_relative_rmse"], indexed["M4"]["trajectory_mean_free_relative_rmse"]),
        "specialization_relevance_lpv_free_reduction_pct": percentage_reduction(
            indexed["M3"]["trajectory_mean_free_relative_rmse"], indexed["M4"]["trajectory_mean_free_relative_rmse"]),
        "m4_vs_m5_free_reduction_pct": percentage_reduction(
            indexed["M5"]["trajectory_mean_free_relative_rmse"], indexed["M4"]["trajectory_mean_free_relative_rmse"]),
        "fixed20_specialization_m3_to_m4_reduction_pct": percentage_reduction(
            indexed["M3"]["fixed20_matrix_mean"], indexed["M4"]["fixed20_matrix_mean"]),
        "seed_consistency_m1_to_m3": seed_reductions(trajectory_rows, "M1", "M3"),
        "seed_consistency_m2_to_m4": seed_reductions(trajectory_rows, "M2", "M4"),
        "seed_consistency_m3_to_m4": seed_reductions(trajectory_rows, "M3", "M4"),
        "seed_consistency_m5_to_m4": seed_reductions(trajectory_rows, "M5", "M4"),
    }
    write_csv(SUMMARY_TABLE, summary)
    write_csv(FAMILY_TABLE, family_summary)
    SUMMARY_JSON.write_text(json.dumps({"methods": indexed, "conclusions": conclusions}, indent=2) + "\n")
    make_figure(matrix_rows, summary, family_summary)
    print(json.dumps({"methods": indexed, "conclusions": conclusions}, indent=2))
    print("Wrote Phase-2 oracle comparison")


if __name__ == "__main__":
    main()
