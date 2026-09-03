"""Experiment 0A: validate the nominal frozen-speed bicycle model."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from federated_lpv import (
    continuous_bicycle_matrices,
    discrete_bicycle_matrices,
    family_centers,
)


ROOT = Path(__file__).resolve().parents[2]
FIGURE = ROOT / "results" / "figures" / "experiment_0a_nominal_validation.pdf"
TABLE = ROOT / "results" / "tables" / "experiment_0a_speed_grid.csv"
SUMMARY = ROOT / "results" / "tables" / "experiment_0a_summary.json"
SAMPLE_TIME = 0.01
SPEEDS = np.linspace(10.0, 30.0, 201)


def _controllability_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.column_stack((b, a @ b))


def evaluate_speed(speed: float) -> dict[str, float]:
    parameters = family_centers()["nominal"]
    a, b = continuous_bicycle_matrices(speed, parameters)
    ad, bd = discrete_bicycle_matrices(speed, parameters, SAMPLE_TIME)
    continuous_poles = np.linalg.eigvals(a)
    discrete_poles = np.linalg.eigvals(ad)
    dc_gain = -np.linalg.solve(a, b).ravel()
    ctrb = _controllability_matrix(a, b)
    singular_values = np.linalg.svd(ctrb, compute_uv=False)
    return {
        "speed_mps": speed,
        "a00": a[0, 0],
        "a01": a[0, 1],
        "a10": a[1, 0],
        "a11": a[1, 1],
        "b00": b[0, 0],
        "b10": b[1, 0],
        "continuous_pole_1_real": continuous_poles[0].real,
        "continuous_pole_1_imag": continuous_poles[0].imag,
        "continuous_pole_2_real": continuous_poles[1].real,
        "continuous_pole_2_imag": continuous_poles[1].imag,
        "discrete_pole_radius_max": float(np.max(np.abs(discrete_poles))),
        "beta_dc_gain_rad_per_rad": dc_gain[0],
        "yaw_rate_dc_gain_per_s": dc_gain[1],
        "kinematic_yaw_gain_per_s": speed / (parameters.front_length + parameters.rear_length),
        "controllability_sigma_min": singular_values[-1],
        "controllability_condition": singular_values[0] / singular_values[-1],
        "a_condition": np.linalg.cond(a),
        "ad_condition": np.linalg.cond(ad),
    }


def summarize(rows: list[dict[str, float]]) -> dict[str, float | bool]:
    def values(key: str) -> np.ndarray:
        return np.asarray([row[key] for row in rows])

    parameters = family_centers()["nominal"]
    length = parameters.front_length + parameters.rear_length
    understeer_coefficient = (
        parameters.mass
        * (
            parameters.rear_stiffness * parameters.rear_length
            - parameters.front_stiffness * parameters.front_length
        )
        / (parameters.front_stiffness * parameters.rear_stiffness * length)
    )
    beta_gain = values("beta_dc_gain_rad_per_rad")
    sign_change = np.flatnonzero(np.signbit(beta_gain[:-1]) != np.signbit(beta_gain[1:]))
    beta_zero_speed = float("nan")
    if len(sign_change):
        index = sign_change[0]
        beta_zero_speed = float(
            np.interp(0.0, beta_gain[index : index + 2][::-1], SPEEDS[index : index + 2][::-1])
        )
    return {
        "sample_time_s": SAMPLE_TIME,
        "speed_min_mps": float(SPEEDS.min()),
        "speed_max_mps": float(SPEEDS.max()),
        "grid_points": len(SPEEDS),
        "continuous_stable_all": bool(
            np.all(values("continuous_pole_1_real") < 0)
            and np.all(values("continuous_pole_2_real") < 0)
        ),
        "max_continuous_pole_real": float(
            max(values("continuous_pole_1_real").max(), values("continuous_pole_2_real").max())
        ),
        "max_discrete_spectral_radius": float(values("discrete_pole_radius_max").max()),
        "min_controllability_sigma": float(values("controllability_sigma_min").min()),
        "max_controllability_condition": float(values("controllability_condition").max()),
        "max_a_condition": float(values("a_condition").max()),
        "max_ad_condition": float(values("ad_condition").max()),
        "yaw_gain_min_per_s": float(values("yaw_rate_dc_gain_per_s").min()),
        "yaw_gain_max_per_s": float(values("yaw_rate_dc_gain_per_s").max()),
        "beta_gain_min": float(values("beta_dc_gain_rad_per_rad").min()),
        "beta_gain_max": float(values("beta_dc_gain_rad_per_rad").max()),
        "beta_gain_zero_speed_mps": beta_zero_speed,
        "understeer_coefficient_s2_per_m": float(understeer_coefficient),
        "yaw_gain_ratio_to_kinematic_min": float(
            np.min(values("yaw_rate_dc_gain_per_s") / values("kinematic_yaw_gain_per_s"))
        ),
        "yaw_gain_ratio_to_kinematic_max": float(
            np.max(values("yaw_rate_dc_gain_per_s") / values("kinematic_yaw_gain_per_s"))
        ),
    }


def make_figure(rows: list[dict[str, float]]) -> None:
    speed = np.asarray([row["speed_mps"] for row in rows])
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), constrained_layout=True)

    axes[0, 0].plot(speed, [row["continuous_pole_1_real"] for row in rows], label="real part")
    axes[0, 0].plot(
        speed,
        np.abs([row["continuous_pole_1_imag"] for row in rows]),
        label="imaginary magnitude",
    )
    axes[0, 0].axhline(0.0, color="black", linewidth=0.8)
    axes[0, 0].set(ylabel="pole component [1/s]", title="Continuous-time pole pair")
    axes[0, 0].legend(frameon=False)

    axes[0, 1].plot(speed, [row["discrete_pole_radius_max"] for row in rows])
    axes[0, 1].axhline(1.0, color="black", linewidth=0.8)
    axes[0, 1].set(ylabel="spectral radius", title="Exact ZOH model")

    axes[1, 0].plot(speed, [row["yaw_rate_dc_gain_per_s"] for row in rows], label="dynamic")
    axes[1, 0].plot(
        speed,
        [row["kinematic_yaw_gain_per_s"] for row in rows],
        linestyle="--",
        label="kinematic $v_x/L$",
    )
    axes[1, 0].set(ylabel=r"$r/\delta$ [1/s]", title="Steady-state yaw gain")
    axes[1, 0].legend(frameon=False)

    axes[1, 1].plot(speed, [row["beta_dc_gain_rad_per_rad"] for row in rows])
    axes[1, 1].axhline(0.0, color="black", linewidth=0.8)
    axes[1, 1].set(ylabel=r"$\beta/\delta$ [rad/rad]", title="Steady-state sideslip gain")

    for axis in axes.flat:
        axis.set_xlabel("speed [m/s]")
        axis.grid(alpha=0.25)
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    rows = [evaluate_speed(float(speed)) for speed in SPEEDS]
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    with TABLE.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(rows)
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    make_figure(rows)
    print(json.dumps(summary, indent=2))
    print(f"Wrote {TABLE.relative_to(ROOT)}, {SUMMARY.relative_to(ROOT)}, and {FIGURE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
