"""Experiment 4C: oracle linear model architectures on nonlinear trajectories."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from federated_lpv import (
    design_oracle_controllers,
    fit_oracle_architectures,
    nonlinear_bicycle_rhs,
    sample_fleet,
)


ROOT = Path(__file__).resolve().parents[2]
FIGURE = ROOT / "results" / "figures" / "experiment_4c_nonlinear_oracle_models.pdf"
SUMMARY_TABLE = ROOT / "results" / "tables" / "experiment_4c_summary.csv"
FAMILY_TABLE = ROOT / "results" / "tables" / "experiment_4c_family_summary.csv"
SUMMARY_JSON = ROOT / "results" / "tables" / "experiment_4c_summary.json"
DT, DURATION, MU = 0.01, 12.0, 0.9
SEVERITIES = {"near_linear": 0.5, "moderate": 1.0, "strong": 1.5}
FIT_SPEEDS = np.asarray([12.0, 16.0, 20.0, 24.0, 28.0])
CONTROL_SPEEDS = np.asarray([10.0, 15.0, 20.0, 25.0, 30.0])
TRAIN_PROFILES = ((12.0, 20.0, 15.0), (20.0, 28.0, 18.0))
TEST_PROFILE = (28.0, 16.0, 24.0)
Q, R = np.diag([100.0, 10.0, 500.0]), np.asarray([[10.0]])
METHODS = ("M1", "M2", "M3", "M4", "M5")
FAMILIES = ("nominal", "heavy", "handling")
RIDGE = 1e-10


def smooth_profile(time: np.ndarray, values: tuple[float, float, float]) -> np.ndarray:
    midpoint = DURATION / 2.0
    first = np.clip(time / midpoint, 0.0, 1.0)
    second = np.clip((time - midpoint) / midpoint, 0.0, 1.0)
    blend_first = 0.5 - 0.5 * np.cos(np.pi * first)
    blend_second = 0.5 - 0.5 * np.cos(np.pi * second)
    return np.where(time <= midpoint,
                    values[0] + (values[1] - values[0]) * blend_first,
                    values[1] + (values[2] - values[1]) * blend_second)


def reference_signal(time: np.ndarray, maneuver: str) -> np.ndarray:
    envelope = np.sin(np.pi * time / DURATION) ** 2
    if maneuver == "sine":
        values = 6.0 * envelope * np.sin(2.0 * np.pi * 0.25 * time)
    elif maneuver == "lane":
        values = 7.0 * (np.exp(-0.5 * ((time - 4.0) / 0.8) ** 2)
                        - np.exp(-0.5 * ((time - 7.0) / 0.8) ** 2))
    elif maneuver == "unseen_s_curve":
        values = 8.5 * (np.exp(-0.5 * ((time - 3.0) / 0.60) ** 2)
                        - 1.15 * np.exp(-0.5 * ((time - 5.8) / 0.78) ** 2)
                        + 0.70 * np.exp(-0.5 * ((time - 8.8) / 0.95) ** 2))
    else:
        raise ValueError(f"unknown maneuver: {maneuver}")
    return np.deg2rad(values)


def rk4_step(state, steering, speed, parameters):
    rhs = lambda value: nonlinear_bicycle_rhs(value, steering, speed, parameters, MU)
    k1 = rhs(state)
    k2 = rhs(state + DT * k1 / 2.0)
    k3 = rhs(state + DT * k2 / 2.0)
    k4 = rhs(state + DT * k3)
    return state + DT * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def simulate_nonlinear(client, controller, speed, reference):
    state = np.zeros((len(speed), 2))
    integral = np.zeros(len(speed))
    steering = np.zeros(len(speed) - 1)
    for index in range(len(speed) - 1):
        gain, prefilter = controller.evaluate(client.family, float(speed[index]))
        steering[index] = -gain @ np.r_[state[index], integral[index]] + prefilter * reference[index]
        state[index + 1] = rk4_step(state[index], steering[index], float(speed[index]), client.parameters)
        integral[index + 1] = integral[index] + DT * (state[index, 1] - reference[index])
    return state, steering


@dataclass(frozen=True)
class DataModel:
    scheduling: str
    specialization: str
    coefficients: dict[str, np.ndarray]

    def matrix(self, family: str, speed: float):
        key = "global" if self.specialization == "global" else family
        if self.scheduling == "constant":
            theta = self.coefficients[key]
        elif self.scheduling == "lpv":
            basis = np.asarray([1.0, 1.0 / speed, 1.0 / speed**2])
            theta = np.tensordot(basis, self.coefficients[key], axes=1)
        else:
            anchors, matrices = self.coefficients[key]
            theta = matrices[int(np.argmin(np.abs(anchors - speed)))]
        return theta[:, :2], theta[:, 2:]


def solve_regression(features, targets):
    features, targets = np.asarray(features), np.asarray(targets)
    gram = features.T @ features + RIDGE * np.eye(features.shape[1])
    return np.linalg.solve(gram, features.T @ targets).T


def fit_constant(records):
    features = np.concatenate([np.column_stack((row["state"][:-1], row["input"])) for row in records])
    targets = np.concatenate([row["state"][1:] for row in records])
    return solve_regression(features, targets)


def fit_lpv(records):
    features, targets = [], []
    for row in records:
        base = np.column_stack((row["state"][:-1], row["input"]))
        basis = np.column_stack((np.ones(len(row["input"])), 1.0 / row["speed"][:-1],
                                 1.0 / row["speed"][:-1] ** 2))
        features.append(np.einsum("ni,nj->nij", basis, base).reshape(len(base), -1))
        targets.append(row["state"][1:])
    raw = solve_regression(np.concatenate(features), np.concatenate(targets))
    return raw.reshape(2, 3, 3).transpose(1, 0, 2)


def fit_grid(records):
    matrices = []
    for anchor in FIT_SPEEDS:
        features, targets = [], []
        for row in records:
            mask = np.abs(row["speed"][:-1] - anchor) <= 2.0
            if np.any(mask):
                features.append(np.column_stack((row["state"][:-1][mask], row["input"][mask])))
                targets.append(row["state"][1:][mask])
        matrices.append(solve_regression(np.concatenate(features), np.concatenate(targets)))
    return FIT_SPEEDS.copy(), np.asarray(matrices)


def fit_models(records):
    by_family = {family: [row for row in records if row["family"] == family] for family in FAMILIES}
    return {
        "M1": DataModel("constant", "global", {"global": fit_constant(records)}),
        "M2": DataModel("constant", "family", {family: fit_constant(rows) for family, rows in by_family.items()}),
        "M3": DataModel("lpv", "global", {"global": fit_lpv(records)}),
        "M4": DataModel("lpv", "family", {family: fit_lpv(rows) for family, rows in by_family.items()}),
        "M5": DataModel("grid", "family", {family: fit_grid(rows) for family, rows in by_family.items()}),
    }


def evaluate_record(record, model):
    true_state, inputs, speeds = record["state"], record["input"], record["speed"]
    one_step = np.zeros_like(true_state)
    free_run = np.zeros_like(true_state)
    free_run[0] = true_state[0]
    for index, command in enumerate(inputs):
        a, b = model.matrix(record["family"], float(speeds[index]))
        one_step[index + 1] = a @ true_state[index] + b[:, 0] * command
        free_run[index + 1] = a @ free_run[index] + b[:, 0] * command
    one_error = one_step[1:] - true_state[1:]
    free_error = free_run - true_state
    scale = np.std(true_state, axis=0)
    scale = np.maximum(scale, np.asarray([np.deg2rad(0.05), np.deg2rad(0.5)]))
    return {
        "one_step_nrmse": float(np.sqrt(np.mean((one_error / scale) ** 2))),
        "free_run_nrmse": float(np.sqrt(np.mean((free_error / scale) ** 2))),
        "one_step_beta_rmse_deg": float(np.rad2deg(np.sqrt(np.mean(one_error[:, 0] ** 2)))),
        "one_step_yaw_rmse_deg_s": float(np.rad2deg(np.sqrt(np.mean(one_error[:, 1] ** 2)))),
        "free_run_beta_rmse_deg": float(np.rad2deg(np.sqrt(np.mean(free_error[:, 0] ** 2)))),
        "free_run_yaw_rmse_deg_s": float(np.rad2deg(np.sqrt(np.mean(free_error[:, 1] ** 2)))),
    }, one_step, free_run


def reduction(rows, severity, baseline, improved, metric):
    means = {method: np.mean([row[metric] for row in rows if row["severity"] == severity
                              and row["method"] == method]) for method in (baseline, improved)}
    return float(100.0 * (means[baseline] - means[improved]) / means[baseline])


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def make_figure(time, representative, summary):
    fig, axes = plt.subplots(2, 2, figsize=(8.3, 5.8), constrained_layout=True)
    for severity in SEVERITIES:
        selected = [row for row in summary if row["severity"] == severity]
        axes[0, 0].plot(METHODS, [next(row["mean_one_step_nrmse"] for row in selected
                                      if row["method"] == method) for method in METHODS], "o-", label=severity.replace("_", " "))
        axes[0, 1].plot(METHODS, [next(row["mean_free_run_nrmse"] for row in selected
                                      if row["method"] == method) for method in METHODS], "o-")
    axes[0, 0].set(ylabel="one-step NRMSE", title="Held-out one-step prediction")
    axes[0, 0].legend(frameon=False, fontsize=8)
    axes[0, 1].set(ylabel="free-run NRMSE", title="Held-out input replay")
    axes[1, 0].plot(time, np.rad2deg(representative["true"][:, 1]), "k", label="nonlinear plant")
    for method in ("M2", "M3", "M4"):
        axes[1, 0].plot(time, np.rad2deg(representative[method][:, 1]), label=method)
    axes[1, 0].set(xlabel="time [s]", ylabel="yaw rate [deg/s]", title="Moderate unseen S-curve free run")
    axes[1, 0].legend(frameon=False, fontsize=8)
    moderate = {row["method"]: row for row in summary if row["severity"] == "moderate"}
    axes[1, 1].bar(METHODS, [moderate[m]["worst_family_free_run_nrmse"] for m in METHODS])
    axes[1, 1].set(ylabel="worst-family free-run NRMSE", title="Moderate family consistency")
    for axis in axes.flat:
        axis.grid(alpha=0.25)
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, bbox_inches="tight")
    plt.close(fig)


def main():
    time = np.arange(0.0, DURATION + DT, DT)
    clients = sample_fleet(seed=1)
    linear_architectures = fit_oracle_architectures(clients, FIT_SPEEDS, DT)
    controller = design_oracle_controllers(linear_architectures, DT, Q, R,
                                            CONTROL_SPEEDS, FIT_SPEEDS)["M4"]
    training = []
    for profile, maneuver in zip(TRAIN_PROFILES, ("lane", "sine")):
        speed = smooth_profile(time, profile)
        for severity, scale in SEVERITIES.items():
            reference = scale * reference_signal(time, maneuver)
            for client in clients:
                state, steering = simulate_nonlinear(client, controller, speed, reference)
                training.append({"family": client.family, "speed": speed,
                                 "state": state, "input": steering})
    models = fit_models(training)

    test, rows, representative = [], [], None
    speed = smooth_profile(time, TEST_PROFILE)
    for severity, scale in SEVERITIES.items():
        reference = scale * reference_signal(time, "unseen_s_curve")
        for client in clients:
            state, steering = simulate_nonlinear(client, controller, speed, reference)
            record = {"client_id": client.client_id, "family": client.family,
                      "severity": severity, "speed": speed, "state": state, "input": steering}
            test.append(record)
            for method, model in models.items():
                metrics, _, free_run = evaluate_record(record, model)
                rows.append({"client_id": client.client_id, "family": client.family,
                             "severity": severity, "method": method, **metrics})
                if severity == "moderate" and client.client_id == "heavy_00":
                    if representative is None:
                        representative = {"true": state}
                    representative[method] = free_run

    metrics = ("one_step_nrmse", "free_run_nrmse", "one_step_beta_rmse_deg",
               "one_step_yaw_rmse_deg_s", "free_run_beta_rmse_deg", "free_run_yaw_rmse_deg_s")
    summary, family_rows = [], []
    for severity in SEVERITIES:
        for method in METHODS:
            selected = [row for row in rows if row["severity"] == severity and row["method"] == method]
            item = {"severity": severity, "method": method, "runs": len(selected)}
            for metric in metrics:
                item[f"mean_{metric}"] = float(np.mean([row[metric] for row in selected]))
            family_means = []
            for family in FAMILIES:
                subset = [row for row in selected if row["family"] == family]
                values = [row["free_run_nrmse"] for row in subset]
                family_means.append(float(np.mean(values)))
                family_rows.append({"severity": severity, "method": method, "family": family,
                                    "mean_one_step_nrmse": float(np.mean([row["one_step_nrmse"] for row in subset])),
                                    "mean_free_run_nrmse": float(np.mean(values))})
            item["worst_family_free_run_nrmse"] = max(family_means)
            item["family_gap_free_run_nrmse"] = max(family_means) - min(family_means)
            summary.append(item)

    conclusions = {
        "moderate_m3_to_m4_one_step_reduction_pct": reduction(rows, "moderate", "M3", "M4", "one_step_nrmse"),
        "moderate_m2_to_m4_one_step_reduction_pct": reduction(rows, "moderate", "M2", "M4", "one_step_nrmse"),
        "moderate_m3_to_m4_free_run_reduction_pct": reduction(rows, "moderate", "M3", "M4", "free_run_nrmse"),
        "moderate_m2_to_m4_free_run_reduction_pct": reduction(rows, "moderate", "M2", "M4", "free_run_nrmse"),
    }
    conclusions["experiment_pass"] = (
        conclusions["moderate_m3_to_m4_one_step_reduction_pct"] > 5.0
        and conclusions["moderate_m2_to_m4_one_step_reduction_pct"] > 5.0
        and conclusions["moderate_m3_to_m4_free_run_reduction_pct"] > 5.0
    )
    write_csv(SUMMARY_TABLE, summary)
    write_csv(FAMILY_TABLE, family_rows)
    SUMMARY_JSON.write_text(json.dumps({"summary": summary, "conclusions": conclusions}, indent=2) + "\n")
    make_figure(time, representative, summary)
    print(json.dumps(conclusions, indent=2))
    if not conclusions["experiment_pass"]:
        raise RuntimeError("Experiment 4C decision gate failed")
    print("Wrote Experiment 4C nonlinear oracle-model comparison artifacts")


if __name__ == "__main__":
    main()
