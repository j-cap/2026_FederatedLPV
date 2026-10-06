"""10H-E: leakage-safe clustering of measured-output innovation signatures."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from experiment_10h_higher_order_lpv_gate import (
    RelaxationClient,
    discrete_matrices,
    multisine,
    sample_fleet,
)
from experiment_10ha_tire_force_observer import (
    C,
    design_observer_schedule,
    measurement_covariance,
    sample_bias,
)
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, confusion_matrix
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from federated_lpv import VehicleParameters

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "code/config/experiment_10he.json"
CONFIG_10H = ROOT / "code/config/experiment_10h.json"
CONFIG_10HA = ROOT / "code/config/experiment_10ha.json"
OUT = ROOT / "results/tables"
FIGURE = ROOT / "results/figures/experiment_10he_innovation_separability.pdf"


def nominal_client(cfg):
    p = cfg["nominal_design"]
    return RelaxationClient(
        "fixed_nominal", "fixed_nominal",
        VehicleParameters(
            p["mass"], p["yaw_inertia"], p["front_length"], p["rear_length"],
            p["front_stiffness"], p["rear_stiffness"],
        ),
        p["actuator_time_constant"], p["front_relaxation_length"],
        p["rear_relaxation_length"],
    )


def innovation_trajectory(client, speed, band, nominal_matrix, nominal_gain,
                          cfg10h, cfg10ha, rng):
    samples = cfg10h["identification_samples_per_speed"]
    commands = multisine(
        samples, cfg10h["sample_time"], band,
        cfg10h["identification_input_std_rad"], rng,
    )
    true_a, true_b = discrete_matrices(client, speed, cfg10h["sample_time"])
    state = np.zeros(5)
    estimate = np.zeros(5)
    bias = sample_bias(cfg10ha, rng)
    covariance = measurement_covariance(cfg10ha)
    innovations = []
    for command in commands:
        state = true_a @ state + true_b[:, 0] * command
        measurement = C @ state + bias + rng.multivariate_normal(np.zeros(3), covariance)
        predicted = nominal_matrix[0] @ estimate + nominal_matrix[1][:, 0] * command
        innovation = measurement - C @ predicted
        estimate = predicted + nominal_gain @ innovation
        innovations.append(innovation)
    return np.asarray(innovations), commands


def signature(innovations, commands, covariance):
    centered = innovations - np.mean(innovations, axis=0)
    whitening = np.diag(1 / np.sqrt(np.diag(covariance)))
    white = centered @ whitening
    covariance_features = np.cov(white, rowvar=False)
    upper = covariance_features[np.triu_indices(3)]
    lag = np.asarray([
        np.corrcoef(white[:-1, j], white[1:, j])[0, 1] for j in range(3)
    ])
    lag = np.nan_to_num(lag)
    nis = np.sum(white**2, axis=1)
    command_scale = max(float(np.sqrt(np.mean(commands**2))), 1e-12)
    cross = np.asarray([
        np.mean(white[:, j] * commands) / command_scale for j in range(3)
    ])
    frequency = np.fft.rfft(white, axis=0)
    power = np.abs(frequency) ** 2
    splits = np.array_split(np.arange(1, len(power)), 3)
    spectral = np.asarray([
        np.log(np.maximum(np.mean(power[index], axis=0), 1e-12))
        for index in splits
    ]).ravel()
    return np.r_[upper, lag, np.mean(nis), np.std(nis), cross, spectral]


def residualize_blocks(features, blocks):
    result = np.zeros_like(features)
    for block in np.unique(blocks):
        indices = np.flatnonzero(blocks == block)
        local = features[indices]
        scale = np.maximum(np.std(local, axis=0), 1e-8)
        result[indices] = (local - np.mean(local, axis=0)) / scale
    return result


def best_label_mapping(truth, labels, classes, clusters):
    mapping = {}
    for cluster in range(clusters):
        local = truth[labels == cluster]
        mapping[cluster] = int(np.bincount(local, minlength=classes).argmax())
    predicted = np.asarray([mapping[label] for label in labels])
    return float(np.mean(predicted == truth)), predicted


def cluster_signatures(features, truth, cfg, seed):
    standardized = StandardScaler().fit_transform(features)
    maximum = min(cfg["maximum_pca_components"], standardized.shape[0] - 1,
                  standardized.shape[1])
    pca = PCA(n_components=maximum, random_state=seed).fit(standardized)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    components = min(maximum, int(np.searchsorted(cumulative, 0.95) + 1))
    reduced = PCA(n_components=components, random_state=seed).fit_transform(standardized)
    candidates, models = [], {}
    for clusters in cfg["candidate_clusters"]:
        model = GaussianMixture(
            n_components=clusters, covariance_type="diag", n_init=cfg["gmm_restarts"],
            reg_covar=1e-4, random_state=seed + 100 * clusters,
        ).fit(reduced)
        labels = model.predict(reduced)
        counts = np.bincount(labels, minlength=clusters)
        diagnostic_ari = adjusted_rand_score(truth, labels)
        _, diagnostic_mapped = best_label_mapping(
            truth, labels, len(np.unique(truth)), clusters
        )
        diagnostic_matrix = confusion_matrix(
            truth, diagnostic_mapped, labels=np.arange(len(np.unique(truth)))
        )
        diagnostic_recall = np.diag(diagnostic_matrix) / np.maximum(
            diagnostic_matrix.sum(axis=1), 1
        )
        admissible = bool(counts.min() >= cfg["minimum_cluster_size"])
        candidates.append({
            "k": clusters, "bic": float(model.bic(reduced)),
            "minimum_size": int(counts.min()), "admissible": admissible,
            "diagnostic_ari": float(diagnostic_ari),
            "diagnostic_balanced_accuracy": float(np.mean(diagnostic_recall)),
            "diagnostic_minimum_recall": float(np.min(diagnostic_recall)),
        })
        models[clusters] = model
    admissible = [item for item in candidates if item["admissible"]]
    if not admissible:
        raise RuntimeError("no admissible innovation mixture")
    selected = min(admissible, key=lambda item: (item["bic"], item["k"]))
    model = models[selected["k"]]
    labels = model.predict(reduced)
    probabilities = model.predict_proba(reduced)
    entropy = -np.sum(probabilities * np.log(np.maximum(probabilities, 1e-15)), axis=1)
    normalized_entropy = entropy / max(np.log(selected["k"]), 1.0)
    ari = adjusted_rand_score(truth, labels)
    _, mapped = best_label_mapping(truth, labels, len(np.unique(truth)), selected["k"])
    matrix = confusion_matrix(truth, mapped, labels=np.arange(len(np.unique(truth))))
    recall = np.diag(matrix) / np.maximum(matrix.sum(axis=1), 1)
    return {
        "selected_k": selected["k"], "labels": labels,
        "ari": float(ari), "balanced_accuracy": float(np.mean(recall)),
        "minimum_class_recall": float(np.min(recall)),
        "mean_entropy": float(np.mean(normalized_entropy)),
        "pca_components": components, "recall": recall,
    }, pd.DataFrame(candidates)


def evaluate_seed(seed):
    cfg = json.loads(CONFIG.read_text())
    cfg10h = json.loads(CONFIG_10H.read_text())
    cfg10ha = json.loads(CONFIG_10HA.read_text())
    clients = sample_fleet(seed, cfg10h)
    grid = np.asarray(cfg10h["speed_grid"], dtype=float)
    nominal = nominal_client(cfg)
    nominal_model = [discrete_matrices(nominal, speed, cfg10h["sample_time"])
                     for speed in grid]
    process_scale = json.loads(
        (OUT / "experiment_10ha_frozen_selection.json").read_text()
    )["process_noise_scale"]
    nominal_gains = design_observer_schedule(nominal_model, process_scale, cfg10ha)
    covariance = measurement_covariance(cfg10ha)
    rng = np.random.default_rng(seed + 1_200_000)
    features, block_ids, rows = [], [], []
    blocks = cfg10h["coverage_blocks"]
    bands = cfg10h["frequency_bands_hz"]
    for index, client in enumerate(clients):
        block = index % len(blocks)
        band_index = (index // len(blocks)) % len(bands)
        local = []
        for speed in blocks[block]:
            grid_index = int(np.argmin(np.abs(grid - speed)))
            innovations, commands = innovation_trajectory(
                client, speed, bands[band_index], nominal_model[grid_index],
                nominal_gains[grid_index], cfg10h, cfg10ha, rng,
            )
            local.append(signature(innovations, commands, covariance))
        features.append(np.mean(local, axis=0))
        block_ids.append(block)
        rows.append({
            "seed": seed, "client": client.client_id,
            "vehicle_class": client.vehicle_class, "block": block,
            "frequency_band": band_index,
        })
    residualized = residualize_blocks(np.asarray(features), np.asarray(block_ids))
    names = sorted({client.vehicle_class for client in clients})
    truth = np.asarray([names.index(client.vehicle_class) for client in clients])
    result, candidates = cluster_signatures(residualized, truth, cfg, seed)
    clients_frame = pd.DataFrame(rows)
    clients_frame["learned_cluster"] = result["labels"]
    clients_frame["selected_k"] = result["selected_k"]
    candidates = candidates.assign(seed=seed, selected_k=result["selected_k"])
    summary = {
        "seed": seed, "selected_k": result["selected_k"], "ari": result["ari"],
        "balanced_accuracy": result["balanced_accuracy"],
        "minimum_class_recall": result["minimum_class_recall"],
        "mean_entropy": result["mean_entropy"],
        "pca_components": result["pca_components"],
        **{f"recall_{name}": result["recall"][j] for j, name in enumerate(names)},
    }
    return clients_frame, candidates, pd.DataFrame([summary])


def summarize(client_frames, candidates, seed_results, cfg):
    summary = pd.concat(seed_results, ignore_index=True)
    candidate_frame = pd.concat(candidates, ignore_index=True)
    clients = pd.concat(client_frames, ignore_index=True)
    aggregate = {
        "modal_k": int(summary.selected_k.mode().iloc[0]),
        "three_group_rate": float(np.mean(summary.selected_k == 3)),
        "mean_ari": float(summary.ari.mean()),
        "minimum_ari": float(summary.ari.min()),
        "mean_balanced_accuracy": float(summary.balanced_accuracy.mean()),
        "minimum_seed_balanced_accuracy": float(summary.balanced_accuracy.min()),
        "mean_minimum_class_recall": float(summary.minimum_class_recall.mean()),
        "minimum_class_recall": float(summary.minimum_class_recall.min()),
        "mean_membership_entropy": float(summary.mean_entropy.mean()),
        "fixed_k3_mean_ari": float(
            candidate_frame[candidate_frame.k == 3].diagnostic_ari.mean()
        ),
        "fixed_k3_mean_balanced_accuracy": float(
            candidate_frame[candidate_frame.k == 3].diagnostic_balanced_accuracy.mean()
        ),
        "fixed_k3_minimum_class_recall": float(
            candidate_frame[candidate_frame.k == 3].diagnostic_minimum_recall.min()
        ),
    }
    gates = {
        "modal_three_groups": aggregate["modal_k"] == 3,
        "ari_gate": aggregate["mean_ari"] >= cfg["minimum_ari"],
        "balanced_accuracy_gate": aggregate["mean_balanced_accuracy"] >=
                                  cfg["minimum_balanced_accuracy"],
        "class_recall_gate": aggregate["minimum_class_recall"] >=
                             cfg["minimum_class_recall"],
        "entropy_gate": aggregate["mean_membership_entropy"] <=
                        cfg["maximum_membership_entropy"],
    }
    gates["development_gate_pass"] = all(gates.values())
    clients.to_csv(OUT / "experiment_10he_development_clients.csv", index=False)
    candidate_frame.to_csv(OUT / "experiment_10he_candidate_scores.csv", index=False)
    summary.to_csv(OUT / "experiment_10he_seed_results.csv", index=False)
    (OUT / "experiment_10he_conclusions.json").write_text(json.dumps({
        "aggregate": aggregate, "development_gates": gates,
        "confirmation_run": False,
        "reserved_confirmation_seeds": cfg["reserved_confirmation_seeds"],
        "leakage_statement": "No true client parameter, state, matrix, or class label enters feature construction, PCA, GMM fitting, BIC selection, or assignment.",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
    }, indent=2) + "\n")
    plot(summary, candidate_frame)
    return summary, aggregate, gates


def plot(summary, candidates):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.7))
    axes[0].bar(summary.seed.astype(str), summary.ari, color="#0072B2")
    axes[0].axhline(0.8, color="black", linestyle="--")
    axes[0].set(title="Innovation clustering", ylabel="ARI", ylim=(0, 1.05))
    axes[0].tick_params(axis="x", rotation=45)
    axes[1].bar(summary.seed.astype(str), summary.balanced_accuracy, color="#009E73")
    axes[1].axhline(0.85, color="black", linestyle="--")
    axes[1].set(title="Permutation-matched assignment", ylabel="Balanced accuracy", ylim=(0, 1.05))
    axes[1].tick_params(axis="x", rotation=45)
    selected = candidates[candidates.k == candidates.selected_k]
    counts = selected.selected_k.value_counts().sort_index()
    axes[2].bar(counts.index.astype(str), counts.values, color="#E69F00")
    axes[2].set(title="BIC-selected cardinality", xlabel="K", ylabel="Fleets")
    fig.tight_layout()
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE)
    plt.close(fig)


def main():
    cfg = json.loads(CONFIG.read_text())
    with ProcessPoolExecutor(max_workers=5) as executor:
        pieces = list(executor.map(evaluate_seed, cfg["development_seeds"]))
    summary, aggregate, gates = summarize(
        [piece[0] for piece in pieces], [piece[1] for piece in pieces],
        [piece[2] for piece in pieces], cfg,
    )
    print(summary.to_string(index=False))
    print(json.dumps(aggregate, indent=2))
    print(json.dumps(gates, indent=2))


if __name__ == "__main__":
    main()
