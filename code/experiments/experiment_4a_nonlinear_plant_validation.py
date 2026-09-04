"""Experiment 4A: validate the nonlinear tanh-tire bicycle plant."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from federated_lpv import (
    continuous_bicycle_matrices,
    family_centers,
    nonlinear_bicycle_rhs,
    nonlinear_tire_forces,
    static_axle_loads,
    tire_saturation_angles,
)


ROOT = Path(__file__).resolve().parents[2]
FIGURE = ROOT / "results" / "figures" / "experiment_4a_nonlinear_validation.pdf"
SUMMARY_TABLE = ROOT / "results" / "tables" / "experiment_4a_summary.csv"
SUMMARY_JSON = ROOT / "results" / "tables" / "experiment_4a_summary.json"
MU = 0.9
DURATION = 8.0
SPEEDS = (12.0, 20.0, 28.0)
AMPLITUDES_DEG = {"small": 0.1, "moderate": 2.0, "strong": 6.0}
STEPS = (0.04, 0.02, 0.01, 0.005)
REFERENCE_STEP = 0.00125


def steering_signal(time: float | np.ndarray, amplitude_deg: float) -> float | np.ndarray:
    """Smooth zero-net steering doublet used only for plant validation."""
    amplitude = np.deg2rad(amplitude_deg)
    return amplitude * (
        np.exp(-0.5 * ((np.asarray(time) - 2.4) / 0.55) ** 2)
        - np.exp(-0.5 * ((np.asarray(time) - 5.0) / 0.70) ** 2)
    )


def rk4_simulation(parameters, speed: float, amplitude_deg: float, step: float, nonlinear: bool):
    time = np.arange(0.0, DURATION + step / 2.0, step)
    state = np.zeros((len(time), 2))
    for index, start in enumerate(time[:-1]):
        def rhs(local_time, local_state):
            steering = float(steering_signal(local_time, amplitude_deg))
            if nonlinear:
                return nonlinear_bicycle_rhs(local_state, steering, speed, parameters, MU)
            a, b = continuous_bicycle_matrices(speed, parameters)
            return a @ local_state + b[:, 0] * steering

        current = state[index]
        k1 = rhs(start, current)
        k2 = rhs(start + step / 2.0, current + step * k1 / 2.0)
        k3 = rhs(start + step / 2.0, current + step * k2 / 2.0)
        k4 = rhs(start + step, current + step * k3)
        state[index + 1] = current + step * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    return time, state


def trajectory_diagnostics(parameters, speed: float, amplitude_deg: float, step: float = 0.01):
    time, nonlinear_state = rk4_simulation(parameters, speed, amplitude_deg, step, True)
    _, linear_state = rk4_simulation(parameters, speed, amplitude_deg, step, False)
    steering = np.asarray(steering_signal(time, amplitude_deg))
    front_load, rear_load = static_axle_loads(parameters)
    front_limit, rear_limit = MU * front_load, MU * rear_load
    front_force, rear_force, front_slip, rear_slip = [], [], [], []
    for state, command in zip(nonlinear_state, steering):
        values = nonlinear_tire_forces(state, float(command), speed, parameters, MU)
        front_force.append(values[0])
        rear_force.append(values[1])
        front_slip.append(values[2])
        rear_slip.append(values[3])
    front_force, rear_force = np.asarray(front_force), np.asarray(rear_force)
    front_slip, rear_slip = np.asarray(front_slip), np.asarray(rear_slip)
    acceleration = (front_force + rear_force) / parameters.mass
    difference = nonlinear_state - linear_state
    denominator = np.sqrt(np.mean(np.sum(linear_state**2, axis=1)))
    relative_rmse = 100.0 * np.sqrt(np.mean(np.sum(difference**2, axis=1))) / denominator
    return {
        "time": time,
        "nonlinear_state": nonlinear_state,
        "linear_state": linear_state,
        "steering": steering,
        "relative_state_rmse_pct": float(relative_rmse),
        "peak_front_slip_deg": float(np.rad2deg(np.max(np.abs(front_slip)))),
        "peak_rear_slip_deg": float(np.rad2deg(np.max(np.abs(rear_slip)))),
        "peak_lateral_accel_mps2": float(np.max(np.abs(acceleration))),
        "peak_force_utilization": float(
            max(np.max(np.abs(front_force)) / front_limit, np.max(np.abs(rear_force)) / rear_limit)
        ),
    }


def jacobian_error(parameters, speed: float) -> float:
    a, b = continuous_bicycle_matrices(speed, parameters)
    epsilon = 1e-7
    numerical_a = np.column_stack(
        [
            (
                nonlinear_bicycle_rhs(np.eye(2)[j] * epsilon, 0.0, speed, parameters, MU)
                - nonlinear_bicycle_rhs(-np.eye(2)[j] * epsilon, 0.0, speed, parameters, MU)
            )
            / (2.0 * epsilon)
            for j in range(2)
        ]
    )
    numerical_b = (
        nonlinear_bicycle_rhs(np.zeros(2), epsilon, speed, parameters, MU)
        - nonlinear_bicycle_rhs(np.zeros(2), -epsilon, speed, parameters, MU)
    )[:, None] / (2.0 * epsilon)
    exact = np.column_stack((a, b))
    numerical = np.column_stack((numerical_a, numerical_b))
    return float(np.linalg.norm(numerical - exact) / np.linalg.norm(exact))


def convergence_study(parameters):
    reference_time, reference = rk4_simulation(parameters, 20.0, 6.0, REFERENCE_STEP, True)
    rows = []
    for step in STEPS:
        time, state = rk4_simulation(parameters, 20.0, 6.0, step, True)
        indices = np.rint(time / REFERENCE_STEP).astype(int)
        sampled_reference = reference[indices]
        absolute = np.sqrt(np.mean(np.sum((state - sampled_reference) ** 2, axis=1)))
        scale = np.sqrt(np.mean(np.sum(sampled_reference**2, axis=1)))
        rows.append({"step_s": step, "relative_rmse_pct": 100.0 * absolute / scale})
    return rows, reference_time, reference


def create_figure(rows, representative_small, representative_strong, convergence):
    nominal = family_centers()["nominal"]
    front_sat, rear_sat = tire_saturation_angles(nominal, MU)
    slip = np.linspace(-0.25, 0.25, 501)
    front_force = nominal.front_stiffness * front_sat * np.tanh(slip / front_sat)
    rear_force = nominal.rear_stiffness * rear_sat * np.tanh(slip / rear_sat)

    fig, axes = plt.subplots(2, 2, figsize=(8.3, 5.8), constrained_layout=True)
    axes[0, 0].plot(np.rad2deg(slip), front_force / 1000.0, label="front tanh")
    axes[0, 0].plot(np.rad2deg(slip), rear_force / 1000.0, label="rear tanh")
    axes[0, 0].plot(np.rad2deg(slip), nominal.front_stiffness * slip / 1000.0,
                    color="0.5", linestyle="--", label="linear slope")
    axes[0, 0].set(xlabel="slip angle [deg]", ylabel="lateral force [kN]",
                   title="Nominal tire law", ylim=(-10, 10))
    axes[0, 0].legend(frameon=False, fontsize=8)

    for label, style in (("linear_state", "--"), ("nonlinear_state", "-")):
        axes[0, 1].plot(representative_small["time"],
                        np.rad2deg(representative_small[label][:, 1]), style,
                        label=label.replace("_state", ""))
    axes[0, 1].set(xlabel="time [s]", ylabel="yaw rate [deg/s]",
                   title="Small-signal agreement")
    axes[0, 1].legend(frameon=False)

    axes[1, 0].plot(representative_strong["time"],
                    np.rad2deg(representative_strong["linear_state"][:, 1]), "--", label="linear")
    axes[1, 0].plot(representative_strong["time"],
                    np.rad2deg(representative_strong["nonlinear_state"][:, 1]), label="tanh")
    axes[1, 0].set(xlabel="time [s]", ylabel="yaw rate [deg/s]",
                   title="Strong-input departure")
    axes[1, 0].legend(frameon=False)

    axes[1, 1].loglog([row["step_s"] for row in convergence],
                      [row["relative_rmse_pct"] for row in convergence], "o-")
    axes[1, 1].set(xlabel="RK4 step [s]", ylabel="relative trajectory RMSE [%]",
                   title="Integration convergence")
    for axis in axes.flat:
        axis.grid(alpha=0.25)
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, bbox_inches="tight")
    plt.close(fig)


def main():
    centers = family_centers()
    rows = []
    for family, parameters in centers.items():
        front_load, rear_load = static_axle_loads(parameters)
        front_sat, rear_sat = tire_saturation_angles(parameters, MU)
        for speed in SPEEDS:
            for severity, amplitude in AMPLITUDES_DEG.items():
                diagnostics = trajectory_diagnostics(parameters, speed, amplitude)
                rows.append(
                    {
                        "family": family,
                        "speed_mps": speed,
                        "severity": severity,
                        "steering_amplitude_deg": amplitude,
                        "relative_state_rmse_pct": diagnostics["relative_state_rmse_pct"],
                        "peak_front_slip_deg": diagnostics["peak_front_slip_deg"],
                        "peak_rear_slip_deg": diagnostics["peak_rear_slip_deg"],
                        "peak_lateral_accel_mps2": diagnostics["peak_lateral_accel_mps2"],
                        "peak_force_utilization": diagnostics["peak_force_utilization"],
                        "front_saturation_angle_deg": float(np.rad2deg(front_sat)),
                        "rear_saturation_angle_deg": float(np.rad2deg(rear_sat)),
                        "front_force_limit_n": MU * front_load,
                        "rear_force_limit_n": MU * rear_load,
                    }
                )

    jacobian_errors = {
        family: max(jacobian_error(parameters, speed) for speed in SPEEDS)
        for family, parameters in centers.items()
    }
    convergence, _, _ = convergence_study(centers["nominal"])
    representative_small = trajectory_diagnostics(centers["nominal"], 20.0, 0.1)
    representative_strong = trajectory_diagnostics(centers["nominal"], 20.0, 6.0)
    by_severity = {
        severity: {
            "max_relative_state_rmse_pct": max(
                row["relative_state_rmse_pct"] for row in rows if row["severity"] == severity
            ),
            "max_force_utilization": max(
                row["peak_force_utilization"] for row in rows if row["severity"] == severity
            ),
            "max_lateral_accel_mps2": max(
                row["peak_lateral_accel_mps2"] for row in rows if row["severity"] == severity
            ),
        }
        for severity in AMPLITUDES_DEG
    }
    summary = {
        "friction_coefficient": MU,
        "maximum_relative_jacobian_error": max(jacobian_errors.values()),
        "family_jacobian_errors": jacobian_errors,
        "severity_summary": by_severity,
        "integration_convergence": convergence,
        "integration_error_at_0p01_pct": next(
            row["relative_rmse_pct"] for row in convergence if row["step_s"] == 0.01
        ),
    }

    SUMMARY_TABLE.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY_TABLE.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    create_figure(rows, representative_small, representative_strong, convergence)
    print(json.dumps(summary, indent=2))
    print("Wrote Experiment 4A nonlinear-plant validation artifacts")


if __name__ == "__main__":
    main()
