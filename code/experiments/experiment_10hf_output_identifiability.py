"""10H-F: measured-output identifiability and physics-score clustering audit."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from experiment_10h_higher_order_lpv_gate import discrete_matrices, multisine, sample_fleet
from experiment_10ha_tire_force_observer import C, measurement_covariance, sample_bias
from experiment_10he_innovation_separability import best_label_mapping
from scipy.linalg import expm, solve_discrete_are
from sklearn.metrics import adjusted_rand_score, confusion_matrix
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "code/config/experiment_10hf.json"
CONFIG_10H = ROOT / "code/config/experiment_10h.json"
CONFIG_10HA = ROOT / "code/config/experiment_10ha.json"
OUT = ROOT / "results/tables"
FIGURE = ROOT / "results/figures/experiment_10hf_output_identifiability.pdf"
PARAMETER_NAMES = (
    "yaw_front", "yaw_rear", "front_beta", "front_yaw", "rear_beta",
    "rear_yaw", "front_decay", "rear_decay", "actuator_rate",
)


def nominal_effective_parameters(cfg):
    p = cfg["nominal_design"]
    return np.asarray([
        p["mass"] * p["front_length"] / p["yaw_inertia"],
        p["mass"] * p["rear_length"] / p["yaw_inertia"],
        p["front_stiffness"] / (p["mass"] * p["front_relaxation_length"]),
        p["front_length"] * p["front_stiffness"] /
        (p["mass"] * p["front_relaxation_length"]),
        p["rear_stiffness"] / (p["mass"] * p["rear_relaxation_length"]),
        p["rear_length"] * p["rear_stiffness"] /
        (p["mass"] * p["rear_relaxation_length"]),
        1 / p["front_relaxation_length"],
        1 / p["rear_relaxation_length"],
        1 / p["actuator_time_constant"],
    ])


def structured_discrete(log_parameters, speed, dt):
    values = np.exp(log_parameters)
    yf, yr, qfb, qfy, qrb, qry, sf, sr, actuator = values
    a = np.zeros((5, 5))
    a[0, 1] = -1.0
    a[0, 2:4] = 1.0 / speed
    a[1, 2] = yf
    a[1, 3] = -yr
    a[2, 0] = -speed * qfb
    a[2, 1] = -qfy
    a[2, 2] = -speed * sf
    a[2, 4] = speed * qfb
    a[3, 0] = -speed * qrb
    a[3, 1] = qry
    a[3, 3] = -speed * sr
    a[4, 4] = -actuator
    b = np.zeros((5, 1))
    b[4, 0] = actuator
    block = np.block([[a, b], [np.zeros((1, 6))]])
    discrete = expm(block * dt)
    return discrete[:5, :5], discrete[:5, 5:]


def simulate_measurements(client, speed, band, cfg, cfg10h, cfg10ha, rng):
    quiet = cfg["quiet_bias_samples"]
    excited = cfg10h["identification_samples_per_speed"]
    commands = np.r_[
        np.zeros(quiet),
        multisine(excited, cfg10h["sample_time"], band,
                  cfg10h["identification_input_std_rad"], rng),
    ]
    a, b = discrete_matrices(client, speed, cfg10h["sample_time"])
    state = np.zeros(5)
    bias = sample_bias(cfg10ha, rng)
    covariance = measurement_covariance(cfg10ha)
    measurements = []
    for command in commands:
        state = a @ state + b[:, 0] * command
        measurements.append(
            C @ state + bias + rng.multivariate_normal(np.zeros(3), covariance)
        )
    measurements = np.asarray(measurements)
    bias_estimate = np.mean(measurements[:quiet], axis=0)
    return commands[quiet:], measurements[quiet:] - bias_estimate


def filter_innovations(log_parameters, speed, commands, measurements, cfg10h, cfg10ha):
    a, b = structured_discrete(log_parameters, speed, cfg10h["sample_time"])
    scale = json.loads(
        (OUT / "experiment_10ha_frozen_selection.json").read_text()
    )["process_noise_scale"]
    q = np.diag(np.asarray(cfg10ha["base_process_noise_diagonal"]) * scale)
    r = measurement_covariance(cfg10ha)
    try:
        p = solve_discrete_are(a.T, C.T, q, r)
    except np.linalg.LinAlgError:
        return None
    estimate = np.zeros(5)
    innovations = []
    for command, measurement in zip(commands, measurements):
        predicted = a @ estimate + b[:, 0] * command
        predicted_covariance = a @ p @ a.T + q
        innovation_covariance = C @ predicted_covariance @ C.T + r
        gain = np.linalg.solve(innovation_covariance, C @ predicted_covariance).T
        innovation = measurement - C @ predicted
        estimate = predicted + gain @ innovation
        p = (np.eye(5) - gain @ C) @ predicted_covariance
        innovations.append(np.linalg.solve(np.linalg.cholesky(innovation_covariance), innovation))
    return np.asarray(innovations)


def segment_information(log_nominal, speed, commands, measurements, cfg, cfg10h, cfg10ha):
    baseline = filter_innovations(
        log_nominal, speed, commands, measurements, cfg10h, cfg10ha
    )
    if baseline is None:
        raise RuntimeError("nominal likelihood filter failed")
    columns = []
    step = cfg["finite_difference_log_step"]
    for parameter in range(len(log_nominal)):
        offset = np.zeros_like(log_nominal)
        offset[parameter] = step
        plus = filter_innovations(
            log_nominal + offset, speed, commands, measurements, cfg10h, cfg10ha
        )
        minus = filter_innovations(
            log_nominal - offset, speed, commands, measurements, cfg10h, cfg10ha
        )
        columns.append(((plus - minus) / (2 * step)).ravel())
    jacobian = np.column_stack(columns)
    residual = baseline.ravel()
    information = jacobian.T @ jacobian
    score = jacobian.T @ residual
    return information, score


def information_diagnostics(matrix, tolerance):
    eigenvalues = np.linalg.eigvalsh((matrix + matrix.T) / 2)
    maximum = max(float(eigenvalues[-1]), 1e-15)
    retained = eigenvalues[eigenvalues > tolerance * maximum]
    rank = len(retained)
    condition = np.inf if rank == 0 else float(maximum / retained[0])
    return rank, condition, eigenvalues


def residualize_blocks(features, blocks):
    result = np.zeros_like(features)
    for block in np.unique(blocks):
        indices = np.flatnonzero(blocks == block)
        local = features[indices]
        scale = np.maximum(np.std(local, axis=0), 1e-8)
        result[indices] = (local - np.mean(local, axis=0)) / scale
    return result


def cluster_scores(features, truth, cfg, seed):
    standardized = StandardScaler().fit_transform(features)
    candidates, models = [], {}
    for clusters in cfg["candidate_clusters"]:
        model = GaussianMixture(
            n_components=clusters, covariance_type="diag", n_init=cfg["gmm_restarts"],
            reg_covar=1e-4, random_state=seed + 100 * clusters,
        ).fit(standardized)
        labels = model.predict(standardized)
        counts = np.bincount(labels, minlength=clusters)
        _, mapped = best_label_mapping(truth, labels, len(np.unique(truth)), clusters)
        matrix = confusion_matrix(truth, mapped, labels=np.arange(len(np.unique(truth))))
        recall = np.diag(matrix) / np.maximum(matrix.sum(axis=1), 1)
        candidates.append({
            "k": clusters, "bic": float(model.bic(standardized)),
            "minimum_size": int(counts.min()),
            "admissible": bool(counts.min() >= cfg["minimum_cluster_size"]),
            "diagnostic_ari": float(adjusted_rand_score(truth, labels)),
            "diagnostic_balanced_accuracy": float(np.mean(recall)),
            "diagnostic_minimum_recall": float(np.min(recall)),
        })
        models[clusters] = model
    valid = [item for item in candidates if item["admissible"]]
    selected = min(valid, key=lambda item: (item["bic"], item["k"]))
    labels = models[selected["k"]].predict(standardized)
    return selected, labels, pd.DataFrame(candidates)


def evaluate_seed(seed):
    cfg = json.loads(CONFIG.read_text())
    cfg10h = json.loads(CONFIG_10H.read_text())
    cfg10ha = json.loads(CONFIG_10HA.read_text())
    clients = sample_fleet(seed, cfg10h)
    nominal = np.log(nominal_effective_parameters(cfg))
    blocks = cfg10h["coverage_blocks"]
    bands = cfg10h["frequency_bands_hz"]
    rng = np.random.default_rng(seed + 1_300_000)
    information, scores, block_ids, client_rows = [], [], [], []
    for index, client in enumerate(clients):
        block = index % len(blocks)
        band_index = (index // len(blocks)) % len(bands)
        local_information = np.zeros((len(nominal), len(nominal)))
        local_score = np.zeros(len(nominal))
        for speed in blocks[block]:
            commands, measurements = simulate_measurements(
                client, speed, bands[band_index], cfg, cfg10h, cfg10ha, rng
            )
            segment_fisher, segment_score = segment_information(
                nominal, speed, commands, measurements, cfg, cfg10h, cfg10ha
            )
            local_information += segment_fisher
            local_score += segment_score
        rank, condition, _ = information_diagnostics(
            local_information, cfg["information_relative_tolerance"]
        )
        information.append(local_information)
        scores.append(local_score)
        block_ids.append(block)
        client_rows.append({
            "seed": seed, "client": client.client_id,
            "vehicle_class": client.vehicle_class, "block": block,
            "frequency_band": band_index, "local_rank": rank,
            "local_condition": condition,
        })
    fleet_information = np.sum(information, axis=0)
    eigenvalues, eigenvectors = np.linalg.eigh(fleet_information)
    floor = max(float(eigenvalues[-1]) * cfg["information_relative_tolerance"], 1e-12)
    inverse_sqrt = eigenvectors @ np.diag(1 / np.sqrt(np.maximum(eigenvalues, floor))) @ eigenvectors.T
    normalized_scores = np.asarray(scores) @ inverse_sqrt
    normalized_scores = residualize_blocks(normalized_scores, np.asarray(block_ids))
    names = sorted({client.vehicle_class for client in clients})
    truth = np.asarray([names.index(client.vehicle_class) for client in clients])
    selected, labels, candidates = cluster_scores(normalized_scores, truth, cfg, seed)
    rows = pd.DataFrame(client_rows)
    rows["learned_cluster"] = labels
    rows["selected_k"] = selected["k"]
    aggregate_rows = []
    for scope, indices in [
        ("fleet", np.arange(len(clients))),
        *[(f"class:{name}", np.flatnonzero(truth == j)) for j, name in enumerate(names)],
    ]:
        matrix = np.sum([information[index] for index in indices], axis=0)
        rank, condition, eigen = information_diagnostics(
            matrix, cfg["information_relative_tolerance"]
        )
        aggregate_rows.append({
            "seed": seed, "scope": scope, "rank": rank, "condition": condition,
            "minimum_eigenvalue": float(eigen[0]),
            "maximum_eigenvalue": float(eigen[-1]),
        })
    candidates = candidates.assign(seed=seed, selected_k=selected["k"])
    return rows, candidates, pd.DataFrame(aggregate_rows)


def summarize(client_frames, candidate_frames, information_frames, cfg):
    clients = pd.concat(client_frames, ignore_index=True)
    candidates = pd.concat(candidate_frames, ignore_index=True)
    information = pd.concat(information_frames, ignore_index=True)
    selected = candidates[candidates.k == candidates.selected_k]
    fixed = candidates[candidates.k == 3]
    class_information = information[information.scope.str.startswith("class:")]
    fleet_information = information[information.scope == "fleet"]
    aggregate = {
        "all_local_rank_deficient": bool((clients.local_rank < len(PARAMETER_NAMES)).all()),
        "local_full_rank_fraction": float(np.mean(clients.local_rank == len(PARAMETER_NAMES))),
        "median_local_condition": float(clients.local_condition.median()),
        "maximum_local_condition": float(clients.local_condition.max()),
        "all_class_aggregates_full_rank": bool((class_information["rank"] == len(PARAMETER_NAMES)).all()),
        "all_fleet_aggregates_full_rank": bool((fleet_information["rank"] == len(PARAMETER_NAMES)).all()),
        "maximum_class_condition": float(class_information.condition.max()),
        "modal_selected_k": int(selected.selected_k.mode().iloc[0]),
        "three_group_rate": float(np.mean(selected.selected_k == 3)),
        "fixed_k3_mean_ari": float(fixed.diagnostic_ari.mean()),
        "fixed_k3_mean_balanced_accuracy": float(fixed.diagnostic_balanced_accuracy.mean()),
        "fixed_k3_minimum_recall": float(fixed.diagnostic_minimum_recall.min()),
    }
    gates = {
        "complementary_information_gate": aggregate["all_local_rank_deficient"] and
                                          aggregate["all_class_aggregates_full_rank"],
        "information_condition_gate": aggregate["maximum_class_condition"] <=
                                      cfg["maximum_group_information_condition"],
        "score_ari_gate": aggregate["fixed_k3_mean_ari"] >= cfg["minimum_fixed_k3_ari"],
        "score_accuracy_gate": aggregate["fixed_k3_mean_balanced_accuracy"] >=
                               cfg["minimum_fixed_k3_balanced_accuracy"],
        "score_recall_gate": aggregate["fixed_k3_minimum_recall"] >=
                             cfg["minimum_fixed_k3_class_recall"],
    }
    gates["development_gate_pass"] = all(gates.values())
    clients.to_csv(OUT / "experiment_10hf_client_information.csv", index=False)
    candidates.to_csv(OUT / "experiment_10hf_candidate_scores.csv", index=False)
    information.to_csv(OUT / "experiment_10hf_aggregate_information.csv", index=False)
    (OUT / "experiment_10hf_conclusions.json").write_text(json.dumps({
        "aggregate": aggregate, "development_gates": gates,
        "confirmation_run": False,
        "reserved_confirmation_seeds": cfg["reserved_confirmation_seeds"],
        "leakage_statement": "No true client physical quantity, state, matrix, or label enters likelihood filtering, finite differences, information matrices, physical scores, clustering, BIC, or assignment.",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
    }, indent=2) + "\n")
    plot(clients, information, fixed, aggregate)
    return aggregate, gates


def plot(clients, information, fixed, aggregate):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.7))
    axes[0].hist(np.log10(clients.local_condition), bins=18, color="#D55E00")
    axes[0].set(title="Local output information", xlabel=r"$\log_{10}$ condition", ylabel="Clients")
    class_info = information[information.scope.str.startswith("class:")]
    axes[1].hist(np.log10(class_info.condition), bins=12, color="#009E73")
    axes[1].set(title="Compatible pooled information", xlabel=r"$\log_{10}$ condition", ylabel="Class fleets")
    axes[2].bar(["ARI", "Balanced accuracy"],
                [fixed.diagnostic_ari.mean(), fixed.diagnostic_balanced_accuracy.mean()],
                color=["#0072B2", "#56B4E9"])
    axes[2].axhline(0.8, color="black", linestyle="--")
    axes[2].set(title=f"Fixed K=3; modal BIC K={aggregate['modal_selected_k']}", ylim=(0, 1.05))
    fig.tight_layout()
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE)
    plt.close(fig)


def main():
    cfg = json.loads(CONFIG.read_text())
    with ProcessPoolExecutor(max_workers=5) as executor:
        pieces = list(executor.map(evaluate_seed, cfg["development_seeds"]))
    aggregate, gates = summarize(
        [piece[0] for piece in pieces], [piece[1] for piece in pieces],
        [piece[2] for piece in pieces], cfg,
    )
    print(json.dumps(aggregate, indent=2))
    print(json.dumps(gates, indent=2))


if __name__ == "__main__":
    main()
