"""Experiment 4B: calibrate reproducible nonlinear maneuver-severity tiers."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from federated_lpv import (
    VehicleClient,
    design_oracle_controllers,
    family_centers,
    fit_oracle_architectures,
    nonlinear_bicycle_rhs,
    nonlinear_tire_forces,
    static_axle_loads,
)


ROOT = Path(__file__).resolve().parents[2]
FIGURE = ROOT / "results" / "figures" / "experiment_4b_maneuver_severity.pdf"
SUMMARY_TABLE = ROOT / "results" / "tables" / "experiment_4b_summary.csv"
TRAJECTORY_TABLE = ROOT / "results" / "tables" / "experiment_4b_trajectories.csv"
SUMMARY_JSON = ROOT / "results" / "tables" / "experiment_4b_summary.json"
DT, DURATION, MU = 0.01, 12.0, 0.9
BASE_REFERENCE_DEG_S = 8.5
SEVERITIES = {"near_linear": 0.5, "moderate": 1.0, "strong": 1.5}
SPEED_PROFILES = {"low_to_high": (12.0, 28.0, 16.0), "high_to_low": (28.0, 16.0, 24.0)}
FIT_SPEEDS = np.asarray([12.0, 16.0, 20.0, 24.0, 28.0])
CONTROL_SPEEDS = np.asarray([10.0, 15.0, 20.0, 25.0, 30.0])
Q, R = np.diag([100.0, 10.0, 500.0]), np.asarray([[10.0]])
STEERING_LIMIT_DEG, ACCELERATION_LIMIT = 12.0, 0.9 * 9.81
SLOPE_LOSS_THRESHOLD = 0.05


def smooth_profile(time: np.ndarray, values: tuple[float, float, float]) -> np.ndarray:
    midpoint = DURATION / 2.0
    first = np.clip(time / midpoint, 0.0, 1.0)
    second = np.clip((time - midpoint) / midpoint, 0.0, 1.0)
    blend_first = 0.5 - 0.5 * np.cos(np.pi * first)
    blend_second = 0.5 - 0.5 * np.cos(np.pi * second)
    return np.where(
        time <= midpoint,
        values[0] + (values[1] - values[0]) * blend_first,
        values[1] + (values[2] - values[1]) * blend_second,
    )


def base_reference(time: np.ndarray) -> np.ndarray:
    """Unseen smooth S-curve shape, normalized to BASE_REFERENCE_DEG_S."""
    shape = (
        np.exp(-0.5 * ((time - 3.0) / 0.60) ** 2)
        - 1.15 * np.exp(-0.5 * ((time - 5.8) / 0.78) ** 2)
        + 0.70 * np.exp(-0.5 * ((time - 8.8) / 0.95) ** 2)
    )
    return np.deg2rad(BASE_REFERENCE_DEG_S * shape)


def rk4_step(state, steering, speed, parameters):
    rhs = lambda value: nonlinear_bicycle_rhs(value, steering, speed, parameters, MU)
    k1 = rhs(state)
    k2 = rhs(state + DT * k1 / 2.0)
    k3 = rhs(state + DT * k2 / 2.0)
    k4 = rhs(state + DT * k3)
    return state + DT * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def simulate(client, controller, time, speed, reference):
    state = np.zeros((len(time), 2))
    integral = np.zeros(len(time))
    steering = np.zeros(len(time) - 1)
    front_force = np.zeros(len(time) - 1)
    rear_force = np.zeros(len(time) - 1)
    front_slip = np.zeros(len(time) - 1)
    rear_slip = np.zeros(len(time) - 1)
    for index in range(len(time) - 1):
        gain, prefilter = controller.evaluate(client.family, float(speed[index]))
        steering[index] = -gain @ np.r_[state[index], integral[index]] + prefilter * reference[index]
        front_force[index], rear_force[index], front_slip[index], rear_slip[index] = nonlinear_tire_forces(
            state[index], steering[index], float(speed[index]), client.parameters, MU
        )
        state[index + 1] = rk4_step(
            state[index], steering[index], float(speed[index]), client.parameters
        )
        integral[index + 1] = integral[index] + DT * (state[index, 1] - reference[index])

    front_load, rear_load = static_axle_loads(client.parameters)
    utilization = np.maximum(np.abs(front_force) / (MU * front_load), np.abs(rear_force) / (MU * rear_load))
    # The normalized tanh tangent is sech^2(alpha/alpha_sat). A 5% slope loss
    # therefore marks the boundary of the declared near-linear tire region.
    front_scale = MU * front_load / client.parameters.front_stiffness
    rear_scale = MU * rear_load / client.parameters.rear_stiffness
    tangent_ratio = np.minimum(1.0 / np.cosh(front_slip / front_scale) ** 2,
                               1.0 / np.cosh(rear_slip / rear_scale) ** 2)
    acceleration = (front_force + rear_force) / client.parameters.mass
    tracking = state[:, 1] - reference
    metrics = {
        "tracking_rmse_deg_s": float(np.rad2deg(np.sqrt(np.mean(tracking**2)))),
        "peak_front_slip_deg": float(np.rad2deg(np.max(np.abs(front_slip)))),
        "peak_rear_slip_deg": float(np.rad2deg(np.max(np.abs(rear_slip)))),
        "peak_lateral_accel_mps2": float(np.max(np.abs(acceleration))),
        "peak_force_utilization": float(np.max(utilization)),
        "fraction_outside_near_linear": float(np.mean(tangent_ratio < 1.0 - SLOPE_LOSS_THRESHOLD)),
        "peak_steering_deg": float(np.rad2deg(np.max(np.abs(steering)))),
        "finite": bool(np.all(np.isfinite(state))),
    }
    return metrics, state, steering, utilization, tangent_ratio


def classify(summary):
    near = summary["near_linear"]
    moderate = summary["moderate"]
    strong = summary["strong"]
    return {
        "near_linear_pass": near["max_force_utilization"] <= 0.35
        and near["max_outside_fraction"] <= 0.10,
        "moderate_pass": 0.35 <= moderate["max_force_utilization"] <= 0.75,
        "strong_pass": 0.75 <= strong["max_force_utilization"] <= 0.98,
        "feasibility_pass": strong["max_steering_deg"] <= STEERING_LIMIT_DEG
        and strong["max_lateral_accel_mps2"] <= ACCELERATION_LIMIT
        and strong["all_finite"],
    }


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def make_figure(time, representative, summary):
    fig, axes = plt.subplots(2, 2, figsize=(8.3, 5.8), constrained_layout=True)
    axes[0, 0].plot(time, representative["speed"], color="0.25")
    axes[0, 0].set(xlabel="time [s]", ylabel="speed [m/s]", title="Calibration speed profile")
    for severity in SEVERITIES:
        item = representative[severity]
        axes[0, 1].plot(time, np.rad2deg(item["state"][:, 1]), label=severity.replace("_", " "))
    axes[0, 1].plot(time, np.rad2deg(representative["strong"]["reference"]), "k--", label="strong ref.")
    axes[0, 1].set(xlabel="time [s]", ylabel="yaw rate [deg/s]", title="Nonlinear closed-loop response")
    axes[0, 1].legend(frameon=False, fontsize=8)
    names = list(SEVERITIES)
    labels = [name.replace("_", " ") for name in names]
    axes[1, 0].bar(labels, [summary[name]["max_force_utilization"] for name in names])
    axes[1, 0].axhline(0.35, color="0.4", linestyle="--")
    axes[1, 0].axhline(0.75, color="tab:red", linestyle="--")
    axes[1, 0].set(ylabel="maximum tire utilization", title="Distinct force regimes", ylim=(0, 1.05))
    axes[1, 0].tick_params(axis="x", rotation=15)
    axes[1, 1].bar(labels, [100 * summary[name]["max_outside_fraction"] for name in names])
    axes[1, 1].set(ylabel="time outside near-linear region [%]", title="Tire nonlinearity exposure")
    axes[1, 1].tick_params(axis="x", rotation=15)
    for axis in axes.flat:
        axis.grid(alpha=0.25)
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, bbox_inches="tight")
    plt.close(fig)


def main():
    time = np.arange(0.0, DURATION + DT, DT)
    clients = [VehicleClient(f"{family}_center", family, parameters)
               for family, parameters in family_centers().items()]
    architectures = fit_oracle_architectures(clients, FIT_SPEEDS, DT)
    controllers = design_oracle_controllers(
        architectures, DT, Q, R, CONTROL_SPEEDS, FIT_SPEEDS
    )
    controller = controllers["M4"]
    rows, representative = [], {"speed": smooth_profile(time, SPEED_PROFILES["high_to_low"])}
    for profile_name, profile_values in SPEED_PROFILES.items():
        speed = smooth_profile(time, profile_values)
        for severity, scale in SEVERITIES.items():
            reference = scale * base_reference(time)
            for client in clients:
                metrics, state, steering, utilization, tangent_ratio = simulate(
                    client, controller, time, speed, reference
                )
                rows.append({"profile": profile_name, "family": client.family, "severity": severity,
                             "scale": scale, "reference_peak_deg_s": float(np.rad2deg(np.max(np.abs(reference)))),
                             **metrics})
                if profile_name == "high_to_low" and client.family == "heavy":
                    representative[severity] = {"state": state, "steering": steering,
                                                "utilization": utilization, "tangent_ratio": tangent_ratio,
                                                "reference": reference}

    summary_rows, summary = [], {}
    for severity in SEVERITIES:
        selected = [row for row in rows if row["severity"] == severity]
        item = {
            "severity": severity,
            "scale": SEVERITIES[severity],
            "reference_peak_deg_s": max(row["reference_peak_deg_s"] for row in selected),
            "max_force_utilization": max(row["peak_force_utilization"] for row in selected),
            "max_outside_fraction": max(row["fraction_outside_near_linear"] for row in selected),
            "max_lateral_accel_mps2": max(row["peak_lateral_accel_mps2"] for row in selected),
            "max_steering_deg": max(row["peak_steering_deg"] for row in selected),
            "mean_tracking_rmse_deg_s": float(np.mean([row["tracking_rmse_deg_s"] for row in selected])),
            "all_finite": all(row["finite"] for row in selected),
        }
        summary[severity] = item
        summary_rows.append(item)
    gates = classify(summary)
    gates["experiment_pass"] = all(gates.values())
    write_csv(TRAJECTORY_TABLE, rows)
    write_csv(SUMMARY_TABLE, summary_rows)
    SUMMARY_JSON.write_text(json.dumps({"summary": summary, "gates": gates}, indent=2) + "\n", encoding="utf-8")
    make_figure(time, representative, summary)
    print(json.dumps({"summary": summary, "gates": gates}, indent=2))
    if not gates["experiment_pass"]:
        raise RuntimeError("Experiment 4B calibration gates failed; adjust base amplitude once globally")
    print("Wrote Experiment 4B maneuver-severity calibration artifacts")


if __name__ == "__main__":
    main()
