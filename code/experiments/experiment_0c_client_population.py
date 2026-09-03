"""Experiment 0C: validate within-family scatter and population structure."""

from __future__ import annotations

import csv
import json
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from federated_lpv import (
    VehicleClient,
    continuous_bicycle_matrices,
    discrete_bicycle_matrices,
    family_centers,
    sample_fleet,
)


ROOT = Path(__file__).resolve().parents[2]
FIGURE_ENVELOPES = ROOT / "results" / "figures" / "experiment_0c_family_envelopes.pdf"
FIGURE_PCA = ROOT / "results" / "figures" / "experiment_0c_model_pca.pdf"
CLIENT_TABLE = ROOT / "results" / "tables" / "experiment_0c_clients_seed1.csv"
SEED_TABLE = ROOT / "results" / "tables" / "experiment_0c_seed_metrics.csv"
SUMMARY = ROOT / "results" / "tables" / "experiment_0c_summary.json"
SAMPLE_TIME = 0.01
SPEEDS = np.linspace(10.0, 30.0, 101)
FEATURE_SPEEDS = (10.0, 15.0, 20.0, 25.0, 30.0)
SEEDS = tuple(range(1, 11))


def model_distance(first: VehicleClient, second: VehicleClient) -> float:
    terms = []
    for speed in FEATURE_SPEEDS:
        a1, b1 = discrete_bicycle_matrices(speed, first.parameters, SAMPLE_TIME)
        a2, b2 = discrete_bicycle_matrices(speed, second.parameters, SAMPLE_TIME)
        scale_a = 0.5 * (np.linalg.norm(a1, ord="fro") + np.linalg.norm(a2, ord="fro"))
        scale_b = 0.5 * (np.linalg.norm(b1, ord="fro") + np.linalg.norm(b2, ord="fro"))
        terms.append(0.5 * ((np.linalg.norm(a1 - a2) / scale_a) ** 2 + (np.linalg.norm(b1 - b2) / scale_b) ** 2))
    return float(np.sqrt(np.mean(terms)))


def center_client(family: str) -> VehicleClient:
    return VehicleClient(f"{family}_center", family, family_centers()[family])


def feature_vector(client: VehicleClient) -> np.ndarray:
    values = []
    for speed in FEATURE_SPEEDS:
        a, b = discrete_bicycle_matrices(speed, client.parameters, SAMPLE_TIME)
        values.extend(np.concatenate((a.ravel(), b.ravel())))
    return np.asarray(values)


def distance_matrix(fleet: list[VehicleClient]) -> np.ndarray:
    result = np.zeros((len(fleet), len(fleet)))
    for first, second in combinations(range(len(fleet)), 2):
        value = model_distance(fleet[first], fleet[second])
        result[first, second] = result[second, first] = value
    return result


def true_label_silhouette(fleet: list[VehicleClient], distances: np.ndarray) -> float:
    values = []
    for index, client in enumerate(fleet):
        same = [j for j, other in enumerate(fleet) if other.family == client.family and j != index]
        own = float(np.mean(distances[index, same]))
        alternatives = []
        for family in family_centers():
            if family == client.family:
                continue
            members = [j for j, other in enumerate(fleet) if other.family == family]
            alternatives.append(float(np.mean(distances[index, members])))
        nearest_other = min(alternatives)
        values.append((nearest_other - own) / max(own, nearest_other))
    return float(np.mean(values))


def seed_metrics(seed: int) -> dict[str, float | int | bool]:
    fleet = sample_fleet(seed)
    distances = distance_matrix(fleet)
    within = []
    between = []
    for first, second in combinations(range(len(fleet)), 2):
        target = within if fleet[first].family == fleet[second].family else between
        target.append(distances[first, second])

    predictions = []
    margins = []
    centers = {family: center_client(family) for family in family_centers()}
    for client in fleet:
        center_distances = {family: model_distance(client, center) for family, center in centers.items()}
        ordered = sorted(center_distances.items(), key=lambda item: item[1])
        predictions.append(ordered[0][0] == client.family)
        true_distance = center_distances[client.family]
        nearest_wrong = min(value for family, value in center_distances.items() if family != client.family)
        margins.append(nearest_wrong - true_distance)

    stable_and_controllable = True
    for client in fleet:
        for speed in SPEEDS:
            a, b = continuous_bicycle_matrices(float(speed), client.parameters)
            stable_and_controllable &= bool(np.all(np.linalg.eigvals(a).real < 0.0))
            stable_and_controllable &= np.linalg.matrix_rank(np.column_stack((b, a @ b))) == 2

    return {
        "seed": seed,
        "stable_and_controllable_all": stable_and_controllable,
        "nearest_center_accuracy": float(np.mean(predictions)),
        "minimum_center_margin": float(np.min(margins)),
        "mean_center_margin": float(np.mean(margins)),
        "mean_within_distance": float(np.mean(within)),
        "max_within_distance": float(np.max(within)),
        "mean_between_distance": float(np.mean(between)),
        "min_between_distance": float(np.min(between)),
        "between_to_within_mean_ratio": float(np.mean(between) / np.mean(within)),
        "true_label_silhouette": true_label_silhouette(fleet, distances),
    }


def client_rows(fleet: list[VehicleClient]) -> list[dict[str, float | str]]:
    rows = []
    centers = {family: center_client(family) for family in family_centers()}
    for client in fleet:
        p = client.parameters
        distances = {family: model_distance(client, center) for family, center in centers.items()}
        predicted = min(distances, key=distances.get)
        rows.append(
            {
                "client_id": client.client_id,
                "family": client.family,
                "mass_kg": p.mass,
                "yaw_inertia_kg_m2": p.yaw_inertia,
                "front_stiffness_n_per_rad": p.front_stiffness,
                "rear_stiffness_n_per_rad": p.rear_stiffness,
                "nearest_center_family": predicted,
                "distance_to_own_center": distances[client.family],
                "distance_to_nearest_wrong_center": min(
                    value for family, value in distances.items() if family != client.family
                ),
            }
        )
    return rows


def characteristic(client: VehicleClient, speed: float) -> tuple[float, float, float]:
    a, b = continuous_bicycle_matrices(speed, client.parameters)
    poles = np.linalg.eigvals(a)
    gain = -np.linalg.solve(a, b).ravel()
    return float(np.max(poles.real)), float(gain[1]), float(gain[0])


def make_envelope_figure(fleet: list[VehicleClient]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 3.0), constrained_layout=True)
    colors = {"nominal": "C0", "heavy": "C1", "handling": "C2"}
    labels = ((0, "least-negative pole [1/s]"), (1, r"$r/\delta$ [1/s]"), (2, r"$\beta/\delta$ [rad/rad]"))
    for family, color in colors.items():
        members = [client for client in fleet if client.family == family]
        values = np.asarray([[characteristic(client, float(speed)) for speed in SPEEDS] for client in members])
        center_values = np.asarray([characteristic(center_client(family), float(speed)) for speed in SPEEDS])
        for axis, (column, ylabel) in zip(axes, labels, strict=True):
            axis.fill_between(SPEEDS, values[:, :, column].min(axis=0), values[:, :, column].max(axis=0), color=color, alpha=0.18)
            axis.plot(SPEEDS, center_values[:, column], color=color, label=family)
            axis.set(xlabel="speed [m/s]", ylabel=ylabel)
            axis.grid(alpha=0.25)
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[2].axhline(0.0, color="black", linewidth=0.8)
    axes[0].legend(frameon=False)
    FIGURE_ENVELOPES.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_ENVELOPES, bbox_inches="tight")
    plt.close(fig)


def make_pca_figure(fleet: list[VehicleClient]) -> float:
    features = np.vstack([feature_vector(client) for client in fleet])
    scale = features.std(axis=0)
    standardized = (features - features.mean(axis=0)) / np.where(scale > 1e-12, scale, 1.0)
    _, singular_values, right_vectors = np.linalg.svd(standardized, full_matrices=False)
    scores = standardized @ right_vectors[:2].T
    explained = singular_values**2 / np.sum(singular_values**2)
    fig, axis = plt.subplots(figsize=(5.2, 3.8), constrained_layout=True)
    markers = {"nominal": "o", "heavy": "s", "handling": "^"}
    for family, marker in markers.items():
        indices = [index for index, client in enumerate(fleet) if client.family == family]
        axis.scatter(scores[indices, 0], scores[indices, 1], marker=marker, label=family, s=38)
    axis.set(
        xlabel=f"PC1 ({100 * explained[0]:.1f}% variance)",
        ylabel=f"PC2 ({100 * explained[1]:.1f}% variance)",
        title="Seed-1 client models",
    )
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    FIGURE_PCA.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PCA, bbox_inches="tight")
    plt.close(fig)
    return float(explained[:2].sum())


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    development_fleet = sample_fleet(seed=1)
    per_seed = [seed_metrics(seed) for seed in SEEDS]
    explained_two_pc = make_pca_figure(development_fleet)
    make_envelope_figure(development_fleet)
    summary = {
        "clients_per_family": 10,
        "development_seed": 1,
        "robustness_seeds": list(SEEDS),
        "mass_inertia_half_range": 0.025,
        "stiffness_half_range": 0.05,
        "all_clients_stable_and_controllable": all(row["stable_and_controllable_all"] for row in per_seed),
        "nearest_center_accuracy_min": min(row["nearest_center_accuracy"] for row in per_seed),
        "nearest_center_accuracy_mean": float(np.mean([row["nearest_center_accuracy"] for row in per_seed])),
        "minimum_center_margin_all_seeds": min(row["minimum_center_margin"] for row in per_seed),
        "between_to_within_ratio_min": min(row["between_to_within_mean_ratio"] for row in per_seed),
        "between_to_within_ratio_mean": float(np.mean([row["between_to_within_mean_ratio"] for row in per_seed])),
        "silhouette_min": min(row["true_label_silhouette"] for row in per_seed),
        "silhouette_mean": float(np.mean([row["true_label_silhouette"] for row in per_seed])),
        "development_seed_two_pc_explained_variance": explained_two_pc,
    }
    write_csv(CLIENT_TABLE, client_rows(development_fleet))
    write_csv(SEED_TABLE, per_seed)
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("Wrote Experiment 0C tables and figures")


if __name__ == "__main__":
    main()
