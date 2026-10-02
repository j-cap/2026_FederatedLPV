"""10E: federated latent-group LPV identification and equivalence gate."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from experiment_10a_basis_coverage_audit import collect_targets, raw_basis
from experiment_10b_oracle_complementary_gate import (
    bootstrap_interval,
    estimate_matrix,
    lift_prior,
    predict,
    recovery_metric,
    ridge_fit,
)
from experiment_10d_learned_compatible_groups import (
    balanced_categories,
    fit_mixture,
)
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

from federated_lpv import sample_fleet

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "code/config/experiment_10e.json"
CONFIG_10A = ROOT / "code/config/experiment_10a.json"
OUT = ROOT / "results/tables"
FIGURE = ROOT / "results/figures/experiment_10e_federated_latent_groups.pdf"


def label_free_categories(clients, rng, blocks):
    """Globally balance blocks without inspecting client family labels."""
    order = rng.permutation(len(clients))
    return {clients[index].client_id: rank % len(blocks) for rank, index in enumerate(order)}


def client_statistics(speeds, targets, cfg10a):
    design = raw_basis(np.asarray(speeds, dtype=float), 7, cfg10a)
    targets = np.asarray(targets, dtype=float)
    return {
        "gram": design.T @ design,
        "right": design.T @ targets,
        "sum": targets.sum(axis=0),
        "square_sum": np.square(targets).sum(axis=0),
        "rows": len(targets),
    }


def solve_statistics(statistics, indices, strength):
    gram = sum((statistics[index]["gram"] for index in indices), start=np.zeros((7, 7)))
    right = sum((statistics[index]["right"] for index in indices), start=np.zeros((7, 6)))
    return np.linalg.solve(gram + (strength + 1e-12) * np.eye(7), right)


def aggregate_scale(statistics):
    count = sum(item["rows"] for item in statistics)
    total = sum((item["sum"] for item in statistics), start=np.zeros(6))
    square = sum((item["square_sum"] for item in statistics), start=np.zeros(6))
    variance = np.maximum(square / count - np.square(total / count), 0.0)
    scale = np.sqrt(variance)
    return np.maximum(scale, np.median(scale) * 0.05)


def local_assignment_scores(models, client_speeds, client_targets, scale, cfg10a):
    scores = np.empty((len(client_speeds), len(models)))
    for client, (speeds, targets) in enumerate(zip(client_speeds, client_targets)):
        for cluster, model in enumerate(models):
            residual = (targets - predict(model, speeds, cfg10a)) / scale
            scores[client, cluster] = np.sum(np.square(residual))
    return scores


def federated_initial_summaries(client_speeds, client_targets, global_model, cfg10a):
    messages = []
    for speeds, targets in zip(client_speeds, client_targets):
        messages.append((targets - predict(global_model, speeds, cfg10a)).mean(axis=0))
    return StandardScaler().fit_transform(np.asarray(messages))


def fit_federated_mixture(
    client_speeds,
    client_targets,
    candidates,
    restarts,
    iterations,
    minimum_size,
    seed,
    cfg10a,
    strength=1e-8,
):
    """Fit a mixture through sufficient statistics and client-side scores only."""
    clients = len(client_speeds)
    statistics = [
        client_statistics(speeds, targets, cfg10a)
        for speeds, targets in zip(client_speeds, client_targets)
    ]
    all_indices = np.arange(clients)
    global_model = solve_statistics(statistics, all_indices, strength)
    scale = aggregate_scale(statistics)
    summaries = federated_initial_summaries(client_speeds, client_targets, global_model, cfg10a)
    scalar_observations = sum(item["rows"] * 6 for item in statistics)

    # Symmetric 7x7 Gram (28), 7x6 right-hand side (42), plus group tag.
    statistic_payload = 28 + 42 + 1
    communication = {
        "uplink_scalars": clients * (28 + 42 + 13 + 6),
        "downlink_scalars": clients * 42,
        "uplink_messages": clients * 3,
        "downlink_messages": clients,
        "model_rounds": 0,
        "assignment_rounds": 0,
    }
    candidate_rows, best_by_k = [], {}
    for clusters in candidates:
        best = None
        if clusters == 1:
            initializations = [np.zeros(clients, dtype=int)]
        else:
            initializations = [
                KMeans(
                    n_clusters=clusters,
                    n_init=1,
                    random_state=seed + 1000 * clusters + restart,
                ).fit_predict(summaries)
                for restart in range(restarts)
            ]
        for restart, initial in enumerate(initializations):
            assignments = initial.copy()
            models = None
            scores = None
            model_matches_assignment = False
            for _ in range(iterations):
                counts = np.bincount(assignments, minlength=clusters)
                if counts.min() < minimum_size:
                    break
                models = [
                    solve_statistics(statistics, np.flatnonzero(assignments == cluster), strength)
                    for cluster in range(clusters)
                ]
                communication["uplink_scalars"] += clients * statistic_payload
                communication["uplink_messages"] += clients
                communication["downlink_scalars"] += clients * clusters * 42
                communication["downlink_messages"] += clients
                communication["model_rounds"] += 1
                scores = local_assignment_scores(
                    models, client_speeds, client_targets, scale, cfg10a
                )
                communication["uplink_scalars"] += clients * clusters
                communication["uplink_messages"] += clients
                communication["assignment_rounds"] += 1
                updated = np.argmin(scores, axis=1)
                model_matches_assignment = np.array_equal(updated, assignments)
                assignments = updated
                if model_matches_assignment:
                    break
            counts = np.bincount(assignments, minlength=clusters)
            if len(counts) < clusters or counts.min() < minimum_size:
                continue
            if not model_matches_assignment:
                models = [
                    solve_statistics(statistics, np.flatnonzero(assignments == cluster), strength)
                    for cluster in range(clusters)
                ]
                communication["uplink_scalars"] += clients * statistic_payload
                communication["uplink_messages"] += clients
                communication["downlink_scalars"] += clients * clusters * 42
                communication["downlink_messages"] += clients
                communication["model_rounds"] += 1
                scores = local_assignment_scores(
                    models, client_speeds, client_targets, scale, cfg10a
                )
                communication["uplink_scalars"] += clients * clusters
                communication["uplink_messages"] += clients
                communication["assignment_rounds"] += 1
            sse = float(sum(scores[index, assignments[index]] for index in range(clients)))
            parameters = clusters * 7 * 6 + clusters - 1
            bic = scalar_observations * np.log(
                max(sse / scalar_observations, 1e-12)
            ) + parameters * np.log(scalar_observations)
            result = {
                "k": clusters,
                "assignments": assignments.copy(),
                "models": models,
                "sse": sse,
                "bic": float(bic),
                "restart": restart,
                "counts": counts,
            }
            if best is None or result["bic"] < best["bic"]:
                best = result
        if best is not None:
            best_by_k[clusters] = best
            candidate_rows.append(
                {
                    "k": clusters,
                    "bic": best["bic"],
                    "sse": best["sse"],
                    "minimum_size": int(best["counts"].min()),
                    "restart": best["restart"],
                }
            )
    if not best_by_k:
        raise RuntimeError("no feasible federated mixture candidate")
    selected = min(best_by_k.values(), key=lambda item: item["bic"])
    communication["downlink_scalars"] += clients
    communication["downlink_messages"] += clients
    return selected, pd.DataFrame(candidate_rows), global_model, communication


def aligned_model_error(central, federated):
    if central["k"] != federated["k"]:
        return np.inf, False
    mapping = {}
    for central_group in range(central["k"]):
        members = np.flatnonzero(central["assignments"] == central_group)
        labels, counts = np.unique(federated["assignments"][members], return_counts=True)
        mapping[central_group] = int(labels[np.argmax(counts)])
    same = (
        np.array_equal(
            np.asarray([mapping[value] for value in central["assignments"]]),
            federated["assignments"],
        )
        and len(set(mapping.values())) == central["k"]
    )
    if not same:
        return np.inf, False
    difference = max(
        np.max(np.abs(central["models"][group] - federated["models"][mapping[group]]))
        for group in range(central["k"])
    )
    return float(difference), True


def run_seed_protocol(seed, protocol):
    cfg = json.loads(CONFIG.read_text())
    cfg10a = json.loads(CONFIG_10A.read_text())
    clients = sample_fleet(seed)
    grid = np.asarray(cfg["speed_grid"], dtype=float)
    rng = np.random.default_rng(seed + (400000 if protocol == "controlled" else 500000))
    if protocol == "controlled":
        categories = balanced_categories(clients, rng, cfg["coverage_blocks"])
    else:
        categories = label_free_categories(clients, rng, cfg["coverage_blocks"])
    client_speeds, client_targets, exact = [], [], {}
    for client in clients:
        speeds = np.asarray(cfg["coverage_blocks"][categories[client.client_id]], dtype=float)
        targets = np.asarray(
            [
                estimate_matrix(client, speed, cfg["samples_per_local_speed"], cfg, rng)
                for speed in speeds
            ]
        )
        client_speeds.append(speeds)
        client_targets.append(targets)
        _, exact[client.client_id] = collect_targets(client, cfg["curvature_per_m"], cfg10a)

    args = (
        client_speeds,
        client_targets,
        cfg["candidate_clusters"],
        cfg["mixture_restarts"],
        cfg["mixture_iterations"],
        cfg["minimum_cluster_size"],
        seed,
        cfg10a,
    )
    central, central_candidates, central_global = fit_mixture(*args)
    federated, fed_candidates, fed_global, communication = fit_federated_mixture(
        *args, strength=cfg["ridge_strength"]
    )
    coefficient_error, same_partition = aligned_model_error(central, federated)
    global_error = float(np.max(np.abs(central_global - fed_global)))
    candidate_error = float(
        np.max(
            np.abs(
                central_candidates.set_index("k").loc[fed_candidates.k, "bic"].to_numpy()
                - fed_candidates.bic.to_numpy()
            )
        )
    )

    families = sorted({client.family for client in clients})
    truth = np.asarray([families.index(client.family) for client in clients])
    ari = adjusted_rand_score(truth, federated["assignments"])
    family_models = {}
    for family in families:
        indices = [index for index, client in enumerate(clients) if client.family == family]
        family_models[family] = ridge_fit(
            np.concatenate([client_speeds[index] for index in indices]),
            np.vstack([client_targets[index] for index in indices]),
            7,
            cfg10a,
            strength=cfg["ridge_strength"],
        )

    rows = []
    for index, client in enumerate(clients):
        local3 = ridge_fit(client_speeds[index], client_targets[index], 3, cfg10a, strength=1e-8)
        local_model = lift_prior(local3, cfg10a)
        exact_model = ridge_fit(grid, exact[client.client_id], 7, cfg10a, strength=1e-10)
        models = {
            "Local": local_model,
            "Global": fed_global,
            "CentralLearned": central["models"][central["assignments"][index]],
            "FederatedLearned": federated["models"][federated["assignments"][index]],
            "OracleFamily": family_models[client.family],
            "ExactLPV": exact_model,
        }
        local_set = set(client_speeds[index])
        for method, model in models.items():
            predictions = predict(model, grid, cfg10a)
            for speed_index, speed in enumerate(grid):
                error = np.linalg.norm(
                    predictions[speed_index] - exact[client.client_id][speed_index]
                ) / np.linalg.norm(exact[client.client_id][speed_index])
                control_matrix = predictions[speed_index]
                if method == "Local" and speed not in local_set:
                    nearest = min(local_set, key=lambda value: abs(value - speed))
                    control_matrix = predict(model, [nearest], cfg10a)[0]
                recovery, feasible, steering = recovery_metric(client, speed, control_matrix, cfg)
                rows.append(
                    {
                        "seed": seed,
                        "protocol": protocol,
                        "client": client.client_id,
                        "family": client.family,
                        "block": categories[client.client_id],
                        "method": method,
                        "speed": speed,
                        "prediction_error": error,
                        "recovery": recovery,
                        "steering_rms": steering,
                        "feasible": feasible,
                    }
                )
    diagnostic = {
        "seed": seed,
        "protocol": protocol,
        "selected_k": federated["k"],
        "central_selected_k": central["k"],
        "ari": ari,
        "same_partition": same_partition,
        "coefficient_error": coefficient_error,
        "global_error": global_error,
        "candidate_bic_error": candidate_error,
        **communication,
    }
    diagnostics = pd.DataFrame([diagnostic])
    candidates = fed_candidates.assign(seed=seed, protocol=protocol, selected_k=federated["k"])
    for suffix, frame in (
        ("clients", pd.DataFrame(rows)),
        ("diagnostics", diagnostics),
        ("candidates", candidates),
    ):
        (OUT / f"experiment_10e_seed{seed}_{protocol}_{suffix}.csv.gz").write_bytes(
            gzip.compress(frame.to_csv(index=False).encode(), mtime=0)
        )
    print(seed, protocol, flush=True)


def run_task(task):
    run_seed_protocol(*task)


def summarize():
    cfg = json.loads(CONFIG.read_text())
    tasks = [(seed, protocol) for seed in cfg["seeds"] for protocol in cfg["coverage_protocols"]]
    raw = pd.concat(
        [
            pd.read_csv(OUT / f"experiment_10e_seed{seed}_{protocol}_clients.csv.gz")
            for seed, protocol in tasks
        ],
        ignore_index=True,
    )
    diagnostics = pd.concat(
        [
            pd.read_csv(OUT / f"experiment_10e_seed{seed}_{protocol}_diagnostics.csv.gz")
            for seed, protocol in tasks
        ],
        ignore_index=True,
    )
    candidates = pd.concat(
        [
            pd.read_csv(OUT / f"experiment_10e_seed{seed}_{protocol}_candidates.csv.gz")
            for seed, protocol in tasks
        ],
        ignore_index=True,
    )
    units = (
        raw.groupby(["seed", "protocol", "method"])
        .agg(
            prediction_error=("prediction_error", "mean"),
            recovery=("recovery", "mean"),
            feasible=("feasible", "min"),
        )
        .reset_index()
    )
    summary = (
        units.groupby(["protocol", "method"])
        .agg(
            prediction_error=("prediction_error", "mean"),
            prediction_std=("prediction_error", "std"),
            recovery=("recovery", "mean"),
            recovery_std=("recovery", "std"),
            feasible_rate=("feasible", "mean"),
        )
        .reset_index()
    )
    comparisons = []
    for protocol in cfg["coverage_protocols"]:
        pivot = units[units.protocol == protocol].pivot(index="seed", columns="method")
        for method, baseline in (
            ("FederatedLearned", "Local"),
            ("FederatedLearned", "Global"),
            ("FederatedLearned", "OracleFamily"),
        ):
            for metric in ("prediction_error", "recovery"):
                method_values = pivot[metric][method]
                baseline_values = pivot[metric][baseline]
                finite = np.isfinite(method_values) & np.isfinite(baseline_values)
                improvement = 100 * (1 - method_values[finite] / baseline_values[finite])
                low, high = bootstrap_interval(
                    improvement,
                    cfg["bootstrap_resamples"],
                    3000 + len(comparisons),
                )
                comparisons.append(
                    {
                        "protocol": protocol,
                        "method": method,
                        "baseline": baseline,
                        "metric": metric,
                        "improvement_pct": float(improvement.mean()),
                        "ci_low": low,
                        "ci_high": high,
                        "wins": int((improvement > 0).sum()),
                        "valid_pairs": int(finite.sum()),
                        "feasibility_wins": int(
                            (np.isfinite(method_values) & ~np.isfinite(baseline_values)).sum()
                        ),
                        "feasibility_losses": int(
                            (~np.isfinite(method_values) & np.isfinite(baseline_values)).sum()
                        ),
                    }
                )
    comparisons = pd.DataFrame(comparisons)
    diagnostics["total_scalars"] = diagnostics.uplink_scalars + diagnostics.downlink_scalars
    diagnostics["total_megabytes"] = diagnostics.total_scalars * cfg["bytes_per_scalar"] / 1e6
    tolerance = cfg["equivalence_tolerance"]
    label_gate = comparisons[
        (comparisons.protocol == "label_free")
        & (comparisons.method == "FederatedLearned")
        & (comparisons.baseline == "Local")
    ].set_index("metric")
    conclusions = {
        "equivalence_gate": bool(
            diagnostics.same_partition.all()
            and (diagnostics.coefficient_error <= tolerance).all()
            and (diagnostics.global_error <= tolerance).all()
            and (diagnostics.candidate_bic_error <= tolerance).all()
        ),
        "label_free_prediction_gate": bool(label_gate.loc["prediction_error", "ci_low"] > 0),
        "label_free_control_gate": bool(label_gate.loc["recovery", "ci_low"] > 0),
        "all_federated_feasible": bool(
            summary[summary.method == "FederatedLearned"].feasible_rate.min() == 1
        ),
        "maximum_coefficient_error": float(diagnostics.coefficient_error.max()),
        "maximum_bic_error": float(diagnostics.candidate_bic_error.max()),
        "modal_selected_k": {
            protocol: int(diagnostics[diagnostics.protocol == protocol].selected_k.mode().iloc[0])
            for protocol in cfg["coverage_protocols"]
        },
        "mean_ari": {
            protocol: float(diagnostics[diagnostics.protocol == protocol].ari.mean())
            for protocol in cfg["coverage_protocols"]
        },
        "mean_total_megabytes_per_fleet": {
            protocol: float(diagnostics[diagnostics.protocol == protocol].total_megabytes.mean())
            for protocol in cfg["coverage_protocols"]
        },
        "interpretation": "Passing 10E establishes numerical federated equivalence and label-free complementary-sharing benefit. It does not establish privacy or control superiority over Global.",
        "provenance": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [
                CONFIG,
                CONFIG_10A,
                ROOT / "code/experiments/experiment_10e_federated_latent_groups.py",
            ]
        },
    }
    units.to_csv(OUT / "experiment_10e_seed_summary.csv", index=False)
    summary.to_csv(OUT / "experiment_10e_summary.csv", index=False)
    comparisons.to_csv(OUT / "experiment_10e_comparisons.csv", index=False)
    diagnostics.to_csv(OUT / "experiment_10e_diagnostics.csv", index=False)
    candidates.to_csv(OUT / "experiment_10e_candidate_scores.csv", index=False)
    (OUT / "experiment_10e_conclusions.json").write_text(json.dumps(conclusions, indent=2) + "\n")
    make_figure(summary, diagnostics)
    print(json.dumps(conclusions, indent=2))
    print(summary.to_string(index=False))
    print(comparisons.to_string(index=False))
    print(diagnostics.to_string(index=False))


def make_figure(summary, diagnostics):
    methods = ["Local", "Global", "FederatedLearned", "OracleFamily", "ExactLPV"]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.5), constrained_layout=True)
    x = np.arange(len(methods))
    width = 0.38
    for offset, protocol in zip((-width / 2, width / 2), ("controlled", "label_free")):
        view = summary[summary.protocol == protocol].set_index("method").loc[methods]
        axes[0].bar(x + offset, 100 * view.prediction_error, width, label=protocol)
        exact = view.loc["ExactLPV", "recovery"]
        recovery_excess = 100 * (view.recovery / exact - 1)
        axes[1].bar(x + offset, recovery_excess.where(np.isfinite(recovery_excess)), width)
    axes[0].set_ylabel("Full-envelope matrix error [%]")
    axes[1].set_ylabel("Recovery excess over Exact [%]")
    axes[0].legend(frameon=False)
    for axis in axes[:2]:
        axis.set_xticks(x, ["Local", "Global", "FedLearned", "Oracle", "Exact"], rotation=28)
        axis.grid(axis="y", alpha=0.25)
    for protocol, marker in (("controlled", "o"), ("label_free", "s")):
        view = diagnostics[diagnostics.protocol == protocol]
        axes[2].scatter(view.selected_k, view.total_megabytes, marker=marker, label=protocol)
    axes[2].set(
        xlabel="Selected groups",
        ylabel="Full search communication [MB]",
        xticks=range(1, 6),
    )
    axes[2].grid(alpha=0.25)
    axes[2].legend(frameon=False)
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="*")
    parser.add_argument("--protocols", nargs="*")
    args = parser.parse_args()
    cfg = json.loads(CONFIG.read_text())
    seeds = args.seeds or cfg["seeds"]
    protocols = args.protocols or cfg["coverage_protocols"]
    tasks = [(seed, protocol) for seed in seeds for protocol in protocols]
    if not args.summarize_only:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(run_task, tasks))
    if set(tasks) == {
        (seed, protocol) for seed in cfg["seeds"] for protocol in cfg["coverage_protocols"]
    }:
        summarize()


if __name__ == "__main__":
    main()
