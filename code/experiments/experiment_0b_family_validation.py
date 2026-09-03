"""Experiment 0B: validate the three structural vehicle-family centers."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import chirp

from federated_lpv import (
    continuous_bicycle_matrices,
    discrete_bicycle_matrices,
    family_centers,
)


ROOT = Path(__file__).resolve().parents[2]
FIGURE_CHARACTERISTICS = ROOT / "results" / "figures" / "experiment_0b_family_characteristics.pdf"
FIGURE_RESPONSES = ROOT / "results" / "figures" / "experiment_0b_family_responses.pdf"
TABLE_GRID = ROOT / "results" / "tables" / "experiment_0b_family_speed_grid.csv"
TABLE_RESPONSES = ROOT / "results" / "tables" / "experiment_0b_response_metrics.csv"
SUMMARY = ROOT / "results" / "tables" / "experiment_0b_summary.json"
SAMPLE_TIME = 0.01
SPEEDS = np.linspace(10.0, 30.0, 201)
TEST_SPEEDS = (12.0, 20.0, 28.0)


def understeer_coefficient(name: str) -> float:
    p = family_centers()[name]
    length = p.front_length + p.rear_length
    return p.mass * (p.rear_stiffness * p.rear_length - p.front_stiffness * p.front_length) / (
        p.front_stiffness * p.rear_stiffness * length
    )


def frozen_metrics(name: str, speed: float) -> dict[str, float | str | bool]:
    p = family_centers()[name]
    a, b = continuous_bicycle_matrices(speed, p)
    ad, bd = discrete_bicycle_matrices(speed, p, SAMPLE_TIME)
    poles = np.linalg.eigvals(a)
    discrete_poles = np.linalg.eigvals(ad)
    dc_gain = -np.linalg.solve(a, b).ravel()
    controllability = np.column_stack((b, a @ b))
    return {
        "family": name,
        "speed_mps": speed,
        "stable_continuous": bool(np.all(poles.real < 0.0)),
        "max_pole_real_per_s": float(np.max(poles.real)),
        "pole_imag_magnitude_per_s": float(np.max(np.abs(poles.imag))),
        "discrete_spectral_radius": float(np.max(np.abs(discrete_poles))),
        "controllability_rank": int(np.linalg.matrix_rank(controllability)),
        "controllability_sigma_min": float(np.linalg.svd(controllability, compute_uv=False)[-1]),
        "yaw_rate_dc_gain_per_s": float(dc_gain[1]),
        "beta_dc_gain": float(dc_gain[0]),
        "understeer_coefficient_s2_per_m": understeer_coefficient(name),
        "a_norm": float(np.linalg.norm(ad, ord="fro")),
        "b_norm": float(np.linalg.norm(bd, ord="fro")),
    }


def normalized_model_distance(name: str, speed: float) -> float:
    nominal = family_centers()["nominal"]
    candidate = family_centers()[name]
    a0, b0 = discrete_bicycle_matrices(speed, nominal, SAMPLE_TIME)
    a1, b1 = discrete_bicycle_matrices(speed, candidate, SAMPLE_TIME)
    relative_a = np.linalg.norm(a1 - a0, ord="fro") / np.linalg.norm(a0, ord="fro")
    relative_b = np.linalg.norm(b1 - b0, ord="fro") / np.linalg.norm(b0, ord="fro")
    return float(np.sqrt(0.5 * (relative_a**2 + relative_b**2)))


def steering_inputs(duration: float = 12.0) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    time = np.arange(0.0, duration + SAMPLE_TIME, SAMPLE_TIME)
    one_degree = np.deg2rad(1.0)
    step = one_degree * (time >= 1.0)
    sweep = one_degree * chirp(time, f0=0.15, f1=1.5, t1=duration, method="linear")
    # Smooth positive/negative pulse pair: a lane-change-like steering command.
    smooth = lambda center: 0.5 * (1.0 + np.tanh((time - center) / 0.12))
    lane_change = 1.5 * one_degree * (
        smooth(2.0) - 2.0 * smooth(3.2) + 2.0 * smooth(4.4) - smooth(5.6)
    )
    return time, {"step": step, "sine_sweep": sweep, "lane_change": lane_change}


def simulate(name: str, speed: float, steering: np.ndarray) -> np.ndarray:
    ad, bd = discrete_bicycle_matrices(speed, family_centers()[name], SAMPLE_TIME)
    states = np.zeros((len(steering), 2))
    for index in range(len(steering) - 1):
        states[index + 1] = ad @ states[index] + bd[:, 0] * steering[index]
    return states


def response_metrics(
    name: str, speed: float, maneuver: str, time: np.ndarray, steering: np.ndarray
) -> dict[str, float | str]:
    states = simulate(name, speed, steering)
    beta_deg = np.rad2deg(states[:, 0])
    yaw_deg_s = np.rad2deg(states[:, 1])
    return {
        "family": name,
        "speed_mps": speed,
        "maneuver": maneuver,
        "peak_abs_beta_deg": float(np.max(np.abs(beta_deg))),
        "peak_abs_yaw_rate_deg_s": float(np.max(np.abs(yaw_deg_s))),
        "rms_beta_deg": float(np.sqrt(np.mean(beta_deg**2))),
        "rms_yaw_rate_deg_s": float(np.sqrt(np.mean(yaw_deg_s**2))),
        "final_beta_deg": float(beta_deg[-1]),
        "final_yaw_rate_deg_s": float(yaw_deg_s[-1]),
        "duration_s": float(time[-1]),
    }


def make_characteristics_figure(rows: list[dict[str, float | str | bool]]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), constrained_layout=True)
    colors = {"nominal": "C0", "heavy": "C1", "handling": "C2"}
    for name in family_centers():
        selected = [row for row in rows if row["family"] == name]
        speed = np.asarray([row["speed_mps"] for row in selected])
        axes[0, 0].plot(speed, [row["max_pole_real_per_s"] for row in selected], label=name, color=colors[name])
        axes[0, 1].plot(speed, [row["yaw_rate_dc_gain_per_s"] for row in selected], label=name, color=colors[name])
        axes[1, 0].plot(speed, [row["beta_dc_gain"] for row in selected], label=name, color=colors[name])
        if name != "nominal":
            axes[1, 1].plot(
                speed,
                [normalized_model_distance(name, float(value)) for value in speed],
                label=name,
                color=colors[name],
            )
    axes[0, 0].axhline(0.0, color="black", linewidth=0.8)
    axes[0, 0].set(title="Least-negative pole part", ylabel="real part [1/s]")
    axes[0, 1].set(title="Steady-state yaw gain", ylabel=r"$r/\delta$ [1/s]")
    axes[1, 0].axhline(0.0, color="black", linewidth=0.8)
    axes[1, 0].set(title="Steady-state sideslip gain", ylabel=r"$\beta/\delta$ [rad/rad]")
    axes[1, 1].set(title="Distance from nominal", ylabel="normalized model distance")
    for axis in axes.flat:
        axis.set_xlabel("speed [m/s]")
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    FIGURE_CHARACTERISTICS.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_CHARACTERISTICS, bbox_inches="tight")
    plt.close(fig)


def make_response_figure(time: np.ndarray, inputs: dict[str, np.ndarray]) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(7.2, 7.4), sharex=True, constrained_layout=True)
    for row_index, speed in enumerate(TEST_SPEEDS):
        steering = inputs["lane_change"]
        for name in family_centers():
            states = simulate(name, speed, steering)
            axes[row_index, 0].plot(time, np.rad2deg(states[:, 1]), label=name)
            axes[row_index, 1].plot(time, np.rad2deg(states[:, 0]), label=name)
        axes[row_index, 0].set_ylabel(f"{speed:.0f} m/s\n$r$ [deg/s]")
        axes[row_index, 1].set_ylabel(f"{speed:.0f} m/s\n$\\beta$ [deg]")
        axes[row_index, 0].grid(alpha=0.25)
        axes[row_index, 1].grid(alpha=0.25)
    axes[0, 0].set_title("Yaw-rate response")
    axes[0, 1].set_title("Sideslip response")
    axes[0, 0].legend(frameon=False, ncol=3)
    axes[-1, 0].set_xlabel("time [s]")
    axes[-1, 1].set_xlabel("time [s]")
    FIGURE_RESPONSES.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_RESPONSES, bbox_inches="tight")
    plt.close(fig)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    grid_rows = [frozen_metrics(name, float(speed)) for name in family_centers() for speed in SPEEDS]
    time, inputs = steering_inputs()
    response_rows = [
        response_metrics(name, speed, maneuver, time, steering)
        for name in family_centers()
        for speed in TEST_SPEEDS
        for maneuver, steering in inputs.items()
    ]
    summary = {
        name: {
            "mass_kg": p.mass,
            "yaw_inertia_kg_m2": p.yaw_inertia,
            "front_stiffness_n_per_rad": p.front_stiffness,
            "rear_stiffness_n_per_rad": p.rear_stiffness,
            "understeer_coefficient_s2_per_m": understeer_coefficient(name),
            "stable_all_speeds": all(row["stable_continuous"] for row in grid_rows if row["family"] == name),
            "max_discrete_spectral_radius": max(
                row["discrete_spectral_radius"] for row in grid_rows if row["family"] == name
            ),
            "min_controllability_sigma": min(
                row["controllability_sigma_min"] for row in grid_rows if row["family"] == name
            ),
            "distance_from_nominal_min": min(normalized_model_distance(name, float(v)) for v in SPEEDS),
            "distance_from_nominal_max": max(normalized_model_distance(name, float(v)) for v in SPEEDS),
        }
        for name, p in family_centers().items()
    }
    write_csv(TABLE_GRID, grid_rows)
    write_csv(TABLE_RESPONSES, response_rows)
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    make_characteristics_figure(grid_rows)
    make_response_figure(time, inputs)
    print(json.dumps(summary, indent=2))
    print("Wrote Experiment 0B tables and figures")


if __name__ == "__main__":
    main()
