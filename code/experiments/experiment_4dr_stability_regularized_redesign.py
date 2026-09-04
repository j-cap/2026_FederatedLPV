"""Experiment 4D-R: stability-screened physics regularization for redesign."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from federated_lpv import design_oracle_controllers, fit_oracle_architectures, sample_fleet
from experiment_4c_nonlinear_oracle_models import (
    DT,
    DURATION,
    FIT_SPEEDS,
    METHODS,
    SEVERITIES,
    TEST_PROFILE,
    TRAIN_PROFILES,
    fit_models,
    reference_signal,
    simulate_nonlinear,
    smooth_profile,
)
from experiment_4d_nonlinear_controllers import redesign_controllers, stability_audit, trajectory_metrics


ROOT = Path(__file__).resolve().parents[2]
FIGURE = ROOT / "results" / "figures" / "experiment_4dr_stability_regularized_redesign.pdf"
SUMMARY_TABLE = ROOT / "results" / "tables" / "experiment_4dr_summary.csv"
SELECTION_TABLE = ROOT / "results" / "tables" / "experiment_4dr_selection.csv"
FAMILY_TABLE = ROOT / "results" / "tables" / "experiment_4dr_family_summary.csv"
SUMMARY_JSON = ROOT / "results" / "tables" / "experiment_4dr_summary.json"
CANDIDATE_WEIGHTS = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
RADIUS_LIMIT = 0.99


@dataclass(frozen=True)
class RegularizedModel:
    """Convex matrix blend of nonlinear-data and validated physics models."""

    data_model: object
    physics_model: object
    physics_weight: float

    @property
    def specialization(self):
        return self.data_model.specialization

    @property
    def scheduling(self):
        return self.data_model.scheduling

    def matrix(self, family: str, speed: float):
        data_a, data_b = self.data_model.matrix(family, speed)
        physics_a, physics_b = self.physics_model.predict(family, np.asarray([speed]))
        weight = self.physics_weight
        return ((1.0 - weight) * data_a + weight * physics_a[0],
                (1.0 - weight) * data_b + weight * physics_b[0])


def select_regularized_controllers(data_models, physics_models, clients):
    selection_rows = []
    for weight in CANDIDATE_WEIGHTS:
        candidates, all_valid = {}, True
        for method in METHODS:
            model = RegularizedModel(data_models[method], physics_models[method], weight)
            try:
                controller = redesign_controllers({method: model})[method]
                audit = stability_audit(clients, {"candidate": {method: controller}})
                radius = max(row["max_small_signal_spectral_radius"] for row in audit)
                valid = bool(np.isfinite(radius) and radius <= RADIUS_LIMIT)
            except (ValueError, np.linalg.LinAlgError):
                controller, radius, valid = None, float("inf"), False
            selection_rows.append({"method": method, "physics_weight": weight,
                                   "max_spectral_radius": radius, "passes": valid})
            candidates[method] = controller
            all_valid = all_valid and valid
        if all_valid:
            return candidates, selection_rows, weight
    raise RuntimeError("no common stable regularization candidate")


def reduction(rows, severity, baseline, improved):
    means = {method: np.mean([row["tracking_rmse_deg_s"] for row in rows
                              if row["severity"] == severity and row["method"] == method])
             for method in (baseline, improved)}
    return float(100.0 * (means[baseline] - means[improved]) / means[baseline])


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def make_figure(time, representative, summary, selection_rows):
    fig, axes = plt.subplots(2, 2, figsize=(8.3, 5.8), constrained_layout=True)
    for method in METHODS:
        rows = [row for row in selection_rows if row["method"] == method]
        axes[0, 0].plot([row["physics_weight"] for row in rows],
                        [row["max_spectral_radius"] for row in rows], "o-", label=method)
    axes[0, 0].axhline(RADIUS_LIMIT, color="k", linestyle="--", label="gate")
    axes[0, 0].set(xlabel="physics regularization weight", ylabel="worst spectral radius",
                   title="Training-fleet stability screen", ylim=(0.85, 25.0))
    axes[0, 0].set_yscale("log")
    axes[0, 0].legend(frameon=False, fontsize=7, ncol=2)
    moderate = {row["method"]: row for row in summary if row["severity"] == "moderate"}
    axes[0, 1].bar(METHODS, [moderate[method]["mean_tracking_rmse_deg_s"] for method in METHODS])
    axes[0, 1].set(ylabel="yaw RMSE [deg/s]", title="Regularized moderate tracking")
    for severity in SEVERITIES:
        selected = [row for row in summary if row["severity"] == severity]
        axes[1, 0].plot(METHODS, [next(row["mean_tracking_rmse_deg_s"] for row in selected
                                      if row["method"] == method) for method in METHODS],
                        "o-", label=severity.replace("_", " "))
    axes[1, 0].set(ylabel="yaw RMSE [deg/s]", title="Regularized redesign across severity")
    axes[1, 0].legend(frameon=False, fontsize=8)
    axes[1, 1].plot(time, np.rad2deg(representative["reference"]), "k--", label="reference")
    for method in ("M2", "M3", "M4"):
        axes[1, 1].plot(time, np.rad2deg(representative[method][:, 1]), label=method)
    axes[1, 1].set(xlabel="time [s]", ylabel="yaw rate [deg/s]", title="Moderate held-out response")
    axes[1, 1].legend(frameon=False, fontsize=8)
    for axis in axes.flat:
        axis.grid(alpha=0.25)
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, bbox_inches="tight")
    plt.close(fig)


def main():
    time = np.arange(0.0, DURATION + DT, DT)
    clients = sample_fleet(seed=1)
    physics_models = fit_oracle_architectures(clients, FIT_SPEEDS, DT)
    data_collection_controller = design_oracle_controllers(
        physics_models, DT,
        np.diag([100.0, 10.0, 500.0]), np.asarray([[10.0]]),
        np.asarray([10.0, 15.0, 20.0, 25.0, 30.0]), FIT_SPEEDS
    )["M4"]

    training = []
    for profile, maneuver in zip(TRAIN_PROFILES, ("lane", "sine")):
        speed = smooth_profile(time, profile)
        for severity, scale in SEVERITIES.items():
            reference = scale * reference_signal(time, maneuver)
            for client in clients:
                state, steering = simulate_nonlinear(
                    client, data_collection_controller, speed, reference
                )
                training.append({"family": client.family, "speed": speed,
                                 "state": state, "input": steering})
    data_models = fit_models(training)
    controllers, selection_rows, common_weight = select_regularized_controllers(
        data_models, physics_models, clients
    )

    speed = smooth_profile(time, TEST_PROFILE)
    rows, representative = [], {}
    for severity, scale in SEVERITIES.items():
        reference = scale * reference_signal(time, "unseen_s_curve")
        for client in clients:
            for method, controller in controllers.items():
                state, steering = simulate_nonlinear(client, controller, speed, reference)
                metrics = trajectory_metrics(client, state, steering, speed, reference)
                rows.append({"severity": severity, "client_id": client.client_id,
                             "family": client.family, "method": method, **metrics})
                if severity == "moderate" and client.client_id == "heavy_00" and method in {"M2", "M3", "M4"}:
                    representative[method] = state
        if severity == "moderate":
            representative["reference"] = reference

    summary, family_rows = [], []
    metrics = ("tracking_rmse_deg_s", "beta_rms_deg", "steering_rms_deg",
               "peak_steering_deg", "peak_steering_rate_deg_s",
               "peak_lateral_accel_mps2", "peak_force_utilization")
    selected_weights = {method: common_weight for method in METHODS}
    selected_radii = {method: next(row["max_spectral_radius"] for row in selection_rows
                                   if row["method"] == method
                                   and row["physics_weight"] == common_weight)
                      for method in METHODS}
    for severity in SEVERITIES:
        for method in METHODS:
            selected = [row for row in rows if row["severity"] == severity and row["method"] == method]
            item = {"severity": severity, "method": method,
                    "physics_weight": selected_weights[method],
                    "training_max_spectral_radius": selected_radii[method], "runs": len(selected)}
            for metric in metrics:
                item[f"mean_{metric}"] = float(np.mean([row[metric] for row in selected]))
                item[f"worst_{metric}"] = float(np.max([row[metric] for row in selected]))
            item["feasible_fraction"] = float(np.mean([row["feasible"] for row in selected]))
            family_means = []
            for family in ("nominal", "heavy", "handling"):
                subset = [row for row in selected if row["family"] == family]
                value = float(np.mean([row["tracking_rmse_deg_s"] for row in subset]))
                family_means.append(value)
                family_rows.append({"severity": severity, "method": method, "family": family,
                                    "mean_tracking_rmse_deg_s": value,
                                    "feasible_fraction": float(np.mean([row["feasible"] for row in subset]))})
            item["worst_family_tracking_rmse_deg_s"] = max(family_means)
            item["family_gap_tracking_rmse_deg_s"] = max(family_means) - min(family_means)
            summary.append(item)

    m4_moderate = next(row for row in summary if row["severity"] == "moderate" and row["method"] == "M4")
    conclusions = {
        "selected_physics_weights": selected_weights,
        "selected_training_spectral_radii": selected_radii,
        "m3_to_m4_moderate_tracking_reduction_pct": reduction(rows, "moderate", "M3", "M4"),
        "m2_to_m4_moderate_tracking_reduction_pct": reduction(rows, "moderate", "M2", "M4"),
        "m4_moderate_feasible_fraction": m4_moderate["feasible_fraction"],
    }
    conclusions["experiment_pass"] = (
        selected_weights["M4"] < 1.0
        and conclusions["m3_to_m4_moderate_tracking_reduction_pct"] > 5.0
        and conclusions["m2_to_m4_moderate_tracking_reduction_pct"] > 5.0
        and conclusions["m4_moderate_feasible_fraction"] == 1.0
    )
    write_csv(SUMMARY_TABLE, summary)
    write_csv(SELECTION_TABLE, selection_rows)
    write_csv(FAMILY_TABLE, family_rows)
    SUMMARY_JSON.write_text(json.dumps({"summary": summary, "conclusions": conclusions}, indent=2) + "\n")
    make_figure(time, representative, summary, selection_rows)
    print(json.dumps(conclusions, indent=2))
    outcome = "passed" if conclusions["experiment_pass"] else "failed primary scientific gate"
    print(f"Wrote Experiment 4D-R artifacts: {outcome}")


if __name__ == "__main__":
    main()
