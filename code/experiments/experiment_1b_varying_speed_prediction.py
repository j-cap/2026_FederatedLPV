"""Experiment 1B: validate the selected LPV basis on varying-speed trajectories."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from federated_lpv import (
    continuous_bicycle_matrices,
    discrete_bicycle_matrices,
    fit_lpv_matrix_model,
    sample_fleet,
)


ROOT = Path(__file__).resolve().parents[2]
FIGURE = ROOT / "results" / "figures" / "experiment_1b_trajectory_validation.pdf"
SUMMARY_TABLE = ROOT / "results" / "tables" / "experiment_1b_summary.csv"
SUMMARY_JSON = ROOT / "results" / "tables" / "experiment_1b_summary.json"
SAMPLE_TIME = 0.01
DURATION = 12.0
FIT_SPEEDS = np.asarray([12.0, 16.0, 20.0, 24.0, 28.0])
SEEDS = tuple(range(1, 11))
PROFILE_VALUES = {
    "12-20-15": (12.0, 20.0, 15.0),
    "20-28-18": (20.0, 28.0, 18.0),
    "15-25-12": (15.0, 25.0, 12.0),
}


def smooth_transition(start: float, stop: float, fraction: np.ndarray) -> np.ndarray:
    blend = 0.5 - 0.5 * np.cos(np.pi * fraction)
    return start + (stop - start) * blend


def speed_profile(time: np.ndarray, values: tuple[float, float, float]) -> np.ndarray:
    midpoint = DURATION / 2.0
    first_fraction = np.clip(time / midpoint, 0.0, 1.0)
    second_fraction = np.clip((time - midpoint) / midpoint, 0.0, 1.0)
    return np.where(
        time <= midpoint,
        smooth_transition(values[0], values[1], first_fraction),
        smooth_transition(values[1], values[2], second_fraction),
    )


def steering_signal(time: np.ndarray) -> np.ndarray:
    """Independent, smooth, moderate multisine excitation with a fade envelope."""
    envelope = np.sin(np.pi * time / DURATION) ** 2
    steering_deg = envelope * (
        1.2 * np.sin(2.0 * np.pi * 0.35 * time)
        + 0.65 * np.sin(2.0 * np.pi * 0.73 * time + 0.4)
        + 0.35 * np.sin(2.0 * np.pi * 1.17 * time + 1.1)
    )
    return np.deg2rad(steering_deg)


def rk4_reference(client, time: np.ndarray, speed: np.ndarray, steering: np.ndarray) -> np.ndarray:
    states = np.zeros((len(time), 2))

    def derivative(state: np.ndarray, local_speed: float, local_input: float) -> np.ndarray:
        a, b = continuous_bicycle_matrices(local_speed, client.parameters)
        return a @ state + b[:, 0] * local_input

    for index in range(len(time) - 1):
        state = states[index]
        speed_left, speed_right = speed[index], speed[index + 1]
        speed_mid = 0.5 * (speed_left + speed_right)
        control = steering[index]
        k1 = derivative(state, speed_left, control)
        k2 = derivative(state + 0.5 * SAMPLE_TIME * k1, speed_mid, control)
        k3 = derivative(state + 0.5 * SAMPLE_TIME * k2, speed_mid, control)
        k4 = derivative(state + SAMPLE_TIME * k3, speed_right, control)
        states[index + 1] = state + SAMPLE_TIME * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    return states


def fitted_models(client):
    pairs = [discrete_bicycle_matrices(speed, client.parameters, SAMPLE_TIME) for speed in FIT_SPEEDS]
    matrices_a = np.asarray([pair[0] for pair in pairs])
    matrices_b = np.asarray([pair[1] for pair in pairs])
    model_a, _ = fit_lpv_matrix_model(FIT_SPEEDS, matrices_a, "reciprocal")
    model_b, _ = fit_lpv_matrix_model(FIT_SPEEDS, matrices_b, "reciprocal")
    return model_a, model_b


def scheduled_matrices(client, speed: np.ndarray, model_a, model_b):
    exact = [discrete_bicycle_matrices(value, client.parameters, SAMPLE_TIME) for value in speed[:-1]]
    exact_a = np.asarray([pair[0] for pair in exact])
    exact_b = np.asarray([pair[1] for pair in exact])
    return exact_a, exact_b, model_a.predict(speed[:-1]), model_b.predict(speed[:-1])


def propagate(a: np.ndarray, b: np.ndarray, steering: np.ndarray) -> np.ndarray:
    states = np.zeros((len(steering), 2))
    for index in range(len(steering) - 1):
        states[index + 1] = a[index] @ states[index] + b[index, :, 0] * steering[index]
    return states


def trajectory_metrics(reference, prediction, one_step_prediction) -> dict[str, float]:
    free_error = prediction - reference
    one_step_error = one_step_prediction - reference[1:]
    reference_rms = float(np.sqrt(np.mean(reference[1:] ** 2)))
    return {
        "one_step_state_rmse": float(np.sqrt(np.mean(one_step_error**2))),
        "one_step_relative_rmse": float(np.sqrt(np.mean(one_step_error**2)) / reference_rms),
        "free_state_rmse": float(np.sqrt(np.mean(free_error[1:] ** 2))),
        "free_relative_rmse": float(np.sqrt(np.mean(free_error[1:] ** 2)) / reference_rms),
        "free_beta_rmse_deg": float(np.rad2deg(np.sqrt(np.mean(free_error[1:, 0] ** 2)))),
        "free_yaw_rmse_deg_s": float(np.rad2deg(np.sqrt(np.mean(free_error[1:, 1] ** 2)))),
        "free_terminal_state_error": float(np.linalg.norm(free_error[-1])),
        "free_peak_state_error": float(np.max(np.linalg.norm(free_error, axis=1))),
    }


def evaluate(seed: int, client, profile_name: str, time, speed, steering):
    reference = rk4_reference(client, time, speed, steering)
    model_a, model_b = fitted_models(client)
    exact_a, exact_b, lpv_a, lpv_b = scheduled_matrices(client, speed, model_a, model_b)
    predictions = {}
    rows = []
    for method, a, b in (("exact_frozen", exact_a, exact_b), ("reciprocal_lpv", lpv_a, lpv_b)):
        prediction = propagate(a, b, steering)
        one_step = np.einsum("kij,kj->ki", a, reference[:-1]) + b[:, :, 0] * steering[:-1, None]
        metrics = trajectory_metrics(reference, prediction, one_step)
        rows.append({"seed": seed, "client_id": client.client_id, "family": client.family,
                     "profile": profile_name, "method": method, **metrics})
        predictions[method] = prediction
    lpv_from_exact = (
        np.einsum("kij,kj->ki", lpv_a, predictions["exact_frozen"][:-1])
        + lpv_b[:, :, 0] * steering[:-1, None]
    )
    basis_metrics = trajectory_metrics(
        predictions["exact_frozen"], predictions["reciprocal_lpv"], lpv_from_exact
    )
    rows.append({"seed": seed, "client_id": client.client_id, "family": client.family,
                 "profile": profile_name, "method": "lpv_vs_exact", **basis_metrics})
    return rows, reference, predictions


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    result = []
    metrics = [key for key in rows[0] if key not in {"seed", "client_id", "family", "profile", "method"}]
    for method in ("exact_frozen", "reciprocal_lpv", "lpv_vs_exact"):
        for family in ("all", "nominal", "heavy", "handling"):
            selected = [row for row in rows if row["method"] == method and (family == "all" or row["family"] == family)]
            item: dict[str, object] = {"method": method, "family": family, "trajectories": len(selected)}
            for metric in metrics:
                values = np.asarray([row[metric] for row in selected], dtype=float)
                item[f"mean_{metric}"] = float(np.mean(values))
                item[f"worst_{metric}"] = float(np.max(values))
            result.append(item)
    return result


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def make_figure(time, speed, steering, reference, predictions, summary) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 5.2), constrained_layout=True)
    axes[0, 0].plot(time, speed, color="tab:blue")
    twin = axes[0, 0].twinx()
    twin.plot(time, np.rad2deg(steering), color="tab:orange", alpha=0.8)
    axes[0, 0].set(ylabel="speed [m/s]", title="Representative excitation")
    twin.set_ylabel("steering [deg]")
    axes[0, 0].set_xlabel("time [s]")

    labels = (("RK4 reference", reference), ("exact frozen", predictions["exact_frozen"]),
              ("reciprocal LPV", predictions["reciprocal_lpv"]))
    for label, states in labels:
        axes[0, 1].plot(time, np.rad2deg(states[:, 1]), label=label)
    axes[0, 1].set(xlabel="time [s]", ylabel="yaw rate [deg/s]", title="Prediction overlay")
    axes[0, 1].legend(frameon=False, fontsize=8)

    for method, color in (("exact_frozen", "tab:blue"), ("reciprocal_lpv", "tab:orange")):
        error = np.linalg.norm(predictions[method] - reference, axis=1)
        axes[1, 0].semilogy(time, np.maximum(error, 1e-16), label=method, color=color)
    axes[1, 0].set(xlabel="time [s]", ylabel="state-error norm", title="Representative free-run error")
    axes[1, 0].legend(frameon=False, fontsize=8)

    all_rows = {row["method"]: row for row in summary if row["family"] == "all"}
    methods = ("exact_frozen", "reciprocal_lpv", "lpv_vs_exact")
    mean = [100 * all_rows[method]["mean_free_relative_rmse"] for method in methods]
    worst = [100 * all_rows[method]["worst_free_relative_rmse"] for method in methods]
    positions = np.arange(3)
    axes[1, 1].bar(positions - 0.18, mean, 0.36, label="mean")
    axes[1, 1].bar(positions + 0.18, worst, 0.36, label="worst")
    axes[1, 1].set(xticks=positions, xticklabels=("exact vs\ncontinuous", "LPV vs\ncontinuous", "LPV vs\nexact"),
                   ylabel="relative free-run RMSE [%]", title="All 900 trajectories")
    axes[1, 1].set_yscale("log")
    axes[1, 1].legend(frameon=False, fontsize=8)
    for axis in axes.flat:
        axis.grid(alpha=0.25)
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    time = np.arange(0.0, DURATION + SAMPLE_TIME, SAMPLE_TIME)
    steering = steering_signal(time)
    profiles = {name: speed_profile(time, values) for name, values in PROFILE_VALUES.items()}
    rows = []
    representative = None
    for seed in SEEDS:
        for client in sample_fleet(seed):
            for profile_name, speed in profiles.items():
                result, reference, predictions = evaluate(seed, client, profile_name, time, speed, steering)
                rows.extend(result)
                if seed == 1 and client.client_id == "nominal_00" and profile_name == "20-28-18":
                    representative = (speed, reference, predictions)
    summary = summarize(rows)
    write_csv(SUMMARY_TABLE, summary)
    compact = {row["method"]: row for row in summary if row["family"] == "all"}
    SUMMARY_JSON.write_text(json.dumps(compact, indent=2) + "\n", encoding="utf-8")
    if representative is None:
        raise RuntimeError("representative trajectory was not collected")
    make_figure(time, representative[0], steering, representative[1], representative[2], summary)
    print(json.dumps(compact, indent=2))
    print("Wrote Experiment 1B tables and figure")


if __name__ == "__main__":
    main()
