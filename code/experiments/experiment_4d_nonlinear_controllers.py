"""Experiment 4D: transfer and redesign oracle controllers on the nonlinear plant."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from federated_lpv import (
    ScheduledController,
    augmented_tracking_matrices,
    design_lqi_gain,
    design_oracle_controllers,
    discrete_bicycle_matrices,
    fit_oracle_architectures,
    nonlinear_tire_forces,
    sample_fleet,
    static_axle_loads,
)
from experiment_4c_nonlinear_oracle_models import (
    CONTROL_SPEEDS,
    DT,
    DURATION,
    FAMILIES,
    FIT_SPEEDS,
    METHODS,
    MU,
    Q,
    R,
    SEVERITIES,
    TEST_PROFILE,
    TRAIN_PROFILES,
    fit_models,
    reference_signal,
    simulate_nonlinear,
    smooth_profile,
)


ROOT = Path(__file__).resolve().parents[2]
FIGURE = ROOT / "results" / "figures" / "experiment_4d_nonlinear_controllers.pdf"
SUMMARY_TABLE = ROOT / "results" / "tables" / "experiment_4d_summary.csv"
FAMILY_TABLE = ROOT / "results" / "tables" / "experiment_4d_family_summary.csv"
SUMMARY_JSON = ROOT / "results" / "tables" / "experiment_4d_summary.json"
STABILITY_TABLE = ROOT / "results" / "tables" / "experiment_4d_stability.csv"
DESIGNS = ("phase3_transfer", "nonlinear_data_redesign")
STEERING_LIMIT_DEG, ACCELERATION_LIMIT = 12.0, 0.9 * 9.81


def redesign_controllers(models):
    controllers = {}
    for method, model in models.items():
        if method in {"M1", "M2"}:
            speeds, interpolation = np.asarray([20.0]), "constant"
        elif method in {"M3", "M4"}:
            # Data-driven models are not extrapolated beyond their 12--28 m/s
            # identification support during safety-relevant synthesis.
            speeds, interpolation = FIT_SPEEDS, "linear"
        else:
            speeds, interpolation = FIT_SPEEDS, "nearest"
        keys = ("global",) if model.specialization == "global" else FAMILIES
        gains, prefilters = {}, {}
        for key in keys:
            family = "nominal" if key == "global" else key
            designs = [design_lqi_gain(*model.matrix(family, float(speed)), DT, Q, R)
                       for speed in speeds]
            gains[key] = np.asarray([item[0] for item in designs])
            prefilters[key] = np.asarray([item[1] for item in designs])
        controllers[method] = ScheduledController(
            f"{method} nonlinear-data redesign", model.specialization,
            interpolation, speeds.copy(), gains, prefilters
        )
    return controllers


def trajectory_metrics(client, state, steering, speed, reference):
    front_load, rear_load = static_axle_loads(client.parameters)
    acceleration, utilization = [], []
    for index, command in enumerate(steering):
        front, rear, _, _ = nonlinear_tire_forces(
            state[index], float(command), float(speed[index]), client.parameters, MU
        )
        acceleration.append((front + rear) / client.parameters.mass)
        utilization.append(max(abs(front) / (MU * front_load), abs(rear) / (MU * rear_load)))
    tracking = state[:, 1] - reference
    steering_rate = np.diff(steering) / DT
    peak_steering = float(np.rad2deg(np.max(np.abs(steering))))
    peak_acceleration = float(np.max(np.abs(acceleration)))
    finite = bool(np.all(np.isfinite(state)) and np.all(np.isfinite(steering)))
    return {
        "tracking_rmse_deg_s": float(np.rad2deg(np.sqrt(np.mean(tracking**2)))),
        "beta_rms_deg": float(np.rad2deg(np.sqrt(np.mean(state[:, 0] ** 2)))),
        "steering_rms_deg": float(np.rad2deg(np.sqrt(np.mean(steering**2)))),
        "peak_steering_deg": peak_steering,
        "peak_steering_rate_deg_s": float(np.rad2deg(np.max(np.abs(steering_rate)))),
        "peak_lateral_accel_mps2": peak_acceleration,
        "peak_force_utilization": float(np.max(utilization)),
        "finite": finite,
        "feasible": finite and peak_steering <= STEERING_LIMIT_DEG
        and peak_acceleration <= ACCELERATION_LIMIT,
    }


def reduction(rows, design, severity, baseline, improved, metric="tracking_rmse_deg_s"):
    means = {method: np.mean([row[metric] for row in rows if row["design"] == design
                              and row["severity"] == severity and row["method"] == method])
             for method in (baseline, improved)}
    return float(100.0 * (means[baseline] - means[improved]) / means[baseline])


def stability_audit(clients, controller_sets):
    rows = []
    speeds = np.linspace(12.0, 28.0, 161)
    for design, controllers in controller_sets.items():
        for client in clients:
            for method, controller in controllers.items():
                radii = []
                for speed in speeds:
                    a, b = discrete_bicycle_matrices(float(speed), client.parameters, DT)
                    augmented_a, augmented_b = augmented_tracking_matrices(a, b, DT)
                    gain, _ = controller.evaluate(client.family, float(speed))
                    radii.append(float(np.max(np.abs(np.linalg.eigvals(
                        augmented_a - augmented_b @ gain[None, :]
                    )))))
                rows.append({"design": design, "client_id": client.client_id,
                             "family": client.family, "method": method,
                             "max_small_signal_spectral_radius": max(radii),
                             "small_signal_stable": max(radii) < 1.0})
    return rows


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def make_figure(time, representative, summary):
    fig, axes = plt.subplots(2, 2, figsize=(8.3, 5.8), constrained_layout=True)
    moderate = {(row["design"], row["method"]): row for row in summary
                if row["severity"] == "moderate"}
    positions = np.arange(len(METHODS))
    axes[0, 0].bar(positions - 0.18,
                   [moderate[(DESIGNS[0], method)]["mean_tracking_rmse_deg_s"] for method in METHODS],
                   0.36, label="Phase-3 transfer")
    axes[0, 0].bar(positions + 0.18,
                   [moderate[(DESIGNS[1], method)]["mean_tracking_rmse_deg_s"] for method in METHODS],
                   0.36, label="nonlinear-data redesign")
    axes[0, 0].set(xticks=positions, xticklabels=METHODS, ylabel="yaw RMSE [deg/s]",
                   title="Moderate tracking")
    axes[0, 0].set_yscale("log")
    axes[0, 0].legend(frameon=False, fontsize=8)
    for design, style in zip(DESIGNS, ("--", "-")):
        item = representative[(design, "M4")]
        axes[0, 1].plot(time, np.rad2deg(item["state"][:, 1]), style,
                        label=design.replace("_", " "))
    axes[0, 1].plot(time, np.rad2deg(representative["reference"]), "k:", label="reference")
    axes[0, 1].set(xlabel="time [s]", ylabel="yaw rate [deg/s]", title="Representative M4 response")
    axes[0, 1].legend(frameon=False, fontsize=7)
    for severity in SEVERITIES:
        selected = [row for row in summary if row["design"] == DESIGNS[1]
                    and row["severity"] == severity]
        axes[1, 0].plot(METHODS, [next(row["mean_tracking_rmse_deg_s"] for row in selected
                                      if row["method"] == method) for method in METHODS],
                        "o-", label=severity.replace("_", " "))
    axes[1, 0].set(ylabel="yaw RMSE [deg/s]", title="Redesign across severity")
    axes[1, 0].set_yscale("log")
    axes[1, 0].legend(frameon=False, fontsize=8)
    axes[1, 1].bar(METHODS, [moderate[(DESIGNS[1], method)]["worst_family_tracking_rmse_deg_s"]
                             for method in METHODS])
    axes[1, 1].set(ylabel="worst-family yaw RMSE [deg/s]", title="Redesign family consistency")
    for axis in axes.flat:
        axis.grid(alpha=0.25)
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, bbox_inches="tight")
    plt.close(fig)


def main():
    time = np.arange(0.0, DURATION + DT, DT)
    clients = sample_fleet(seed=1)
    linear_architectures = fit_oracle_architectures(clients, FIT_SPEEDS, DT)
    transfer = design_oracle_controllers(
        linear_architectures, DT, Q, R, CONTROL_SPEEDS, FIT_SPEEDS
    )
    data_controller = transfer["M4"]
    training = []
    for profile, maneuver in zip(TRAIN_PROFILES, ("lane", "sine")):
        speed = smooth_profile(time, profile)
        for severity, scale in SEVERITIES.items():
            reference = scale * reference_signal(time, maneuver)
            for client in clients:
                state, steering = simulate_nonlinear(client, data_controller, speed, reference)
                training.append({"family": client.family, "speed": speed,
                                 "state": state, "input": steering})
    redesigned = redesign_controllers(fit_models(training))
    controller_sets = {DESIGNS[0]: transfer, DESIGNS[1]: redesigned}
    stability_rows = stability_audit(clients, controller_sets)

    speed = smooth_profile(time, TEST_PROFILE)
    rows, representative = [], {"reference": reference_signal(time, "unseen_s_curve")}
    for severity, scale in SEVERITIES.items():
        reference = scale * reference_signal(time, "unseen_s_curve")
        for client in clients:
            for design, controllers in controller_sets.items():
                for method, controller in controllers.items():
                    state, steering = simulate_nonlinear(client, controller, speed, reference)
                    metrics = trajectory_metrics(client, state, steering, speed, reference)
                    rows.append({"design": design, "severity": severity, "client_id": client.client_id,
                                 "family": client.family, "method": method, **metrics})
                    if severity == "moderate" and client.client_id == "heavy_00" and method == "M4":
                        representative[(design, method)] = {"state": state, "steering": steering}
    representative["reference"] = reference_signal(time, "unseen_s_curve")

    summary, family_rows = [], []
    metrics = ("tracking_rmse_deg_s", "beta_rms_deg", "steering_rms_deg",
               "peak_steering_deg", "peak_steering_rate_deg_s",
               "peak_lateral_accel_mps2", "peak_force_utilization")
    for design in DESIGNS:
        for severity in SEVERITIES:
            for method in METHODS:
                selected = [row for row in rows if row["design"] == design
                            and row["severity"] == severity and row["method"] == method]
                item = {"design": design, "severity": severity, "method": method, "runs": len(selected)}
                for metric in metrics:
                    item[f"mean_{metric}"] = float(np.mean([row[metric] for row in selected]))
                    item[f"worst_{metric}"] = float(np.max([row[metric] for row in selected]))
                item["feasible_fraction"] = float(np.mean([row["feasible"] for row in selected]))
                family_means = []
                for family in FAMILIES:
                    subset = [row for row in selected if row["family"] == family]
                    family_mean = float(np.mean([row["tracking_rmse_deg_s"] for row in subset]))
                    family_means.append(family_mean)
                    family_rows.append({"design": design, "severity": severity, "method": method,
                                        "family": family, "mean_tracking_rmse_deg_s": family_mean,
                                        "feasible_fraction": float(np.mean([row["feasible"] for row in subset]))})
                item["worst_family_tracking_rmse_deg_s"] = max(family_means)
                item["family_gap_tracking_rmse_deg_s"] = max(family_means) - min(family_means)
                summary.append(item)

    conclusions = {
        "redesign_m3_to_m4_moderate_tracking_reduction_pct": reduction(
            rows, DESIGNS[1], "moderate", "M3", "M4"),
        "redesign_m2_to_m4_moderate_tracking_reduction_pct": reduction(
            rows, DESIGNS[1], "moderate", "M2", "M4"),
        "transfer_m3_to_m4_moderate_tracking_reduction_pct": reduction(
            rows, DESIGNS[0], "moderate", "M3", "M4"),
        "transfer_m2_to_m4_moderate_tracking_reduction_pct": reduction(
            rows, DESIGNS[0], "moderate", "M2", "M4"),
        "worst_small_signal_spectral_radius": {
            design: {method: max(row["max_small_signal_spectral_radius"] for row in stability_rows
                                 if row["design"] == design and row["method"] == method)
                     for method in METHODS}
            for design in DESIGNS
        },
    }
    m4_moderate = next(row for row in summary if row["design"] == DESIGNS[1]
                       and row["severity"] == "moderate" and row["method"] == "M4")
    conclusions["redesign_m4_moderate_feasible_fraction"] = m4_moderate["feasible_fraction"]
    conclusions["experiment_pass"] = (
        conclusions["redesign_m3_to_m4_moderate_tracking_reduction_pct"] > 5.0
        and conclusions["redesign_m2_to_m4_moderate_tracking_reduction_pct"] > 5.0
        and conclusions["redesign_m4_moderate_feasible_fraction"] == 1.0
    )
    write_csv(SUMMARY_TABLE, summary)
    write_csv(FAMILY_TABLE, family_rows)
    write_csv(STABILITY_TABLE, stability_rows)
    SUMMARY_JSON.write_text(json.dumps({"summary": summary, "conclusions": conclusions}, indent=2) + "\n")
    make_figure(time, representative, summary)
    print(json.dumps(conclusions, indent=2))
    outcome = "passed" if conclusions["experiment_pass"] else "failed primary scientific gate"
    print(f"Wrote Experiment 4D nonlinear controller artifacts: {outcome}")


if __name__ == "__main__":
    main()
