"""10D: learn compatible LPV-sharing groups from partial client coverage."""

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
from experiment_10a_basis_coverage_audit import collect_targets
from experiment_10b_oracle_complementary_gate import (
    bootstrap_interval,
    estimate_matrix,
    lift_prior,
    predict,
    recovery_metric,
    ridge_fit,
)
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

from federated_lpv import sample_fleet

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "code/config/experiment_10d.json"
CONFIG_10A = ROOT / "code/config/experiment_10a.json"
OUT = ROOT / "results/tables"
FIGURE = ROOT / "results/figures/experiment_10d_learned_groups.pdf"


def balanced_categories(clients, rng, blocks):
    categories = {}
    for family in sorted({client.family for client in clients}):
        family_clients = [client for client in clients if client.family == family]
        for j, index in enumerate(rng.permutation(len(family_clients))):
            categories[family_clients[index].client_id] = j % len(blocks)
    return categories


def observation_scale(targets):
    scale = np.std(np.vstack(targets), axis=0)
    return np.maximum(scale, np.median(scale) * 0.05)


def client_summaries(client_speeds, client_targets, global_model, cfg10a):
    rows = []
    for speeds, targets in zip(client_speeds, client_targets):
        residual = targets - predict(global_model, speeds, cfg10a)
        mean = residual.mean(axis=0)
        rows.append(mean)
    return StandardScaler().fit_transform(np.asarray(rows))


def fit_cluster_models(assignments, client_speeds, client_targets, clusters, cfg10a):
    models = []
    for cluster in range(clusters):
        indices = np.flatnonzero(assignments == cluster)
        speeds = np.concatenate([client_speeds[index] for index in indices])
        targets = np.vstack([client_targets[index] for index in indices])
        models.append(ridge_fit(speeds, targets, 7, cfg10a, strength=1e-8))
    return models


def assignment_cost(models, client_speeds, client_targets, scale, cfg10a):
    cost = np.empty((len(client_speeds), len(models)))
    for index, (speeds, targets) in enumerate(zip(client_speeds, client_targets)):
        for cluster, model in enumerate(models):
            residual = (targets - predict(model, speeds, cfg10a)) / scale
            cost[index, cluster] = np.sum(residual**2)
    return cost


def fit_mixture(
    client_speeds, client_targets, candidates, restarts, iterations, minimum_size, seed, cfg10a
):
    all_speeds = np.concatenate(client_speeds)
    all_targets = np.vstack(client_targets)
    global_model = ridge_fit(all_speeds, all_targets, 7, cfg10a, strength=1e-8)
    summaries = client_summaries(client_speeds, client_targets, global_model, cfg10a)
    scale = observation_scale(client_targets)
    candidate_rows, best_by_k = [], {}
    scalar_observations = sum(len(targets) * targets.shape[1] for targets in client_targets)
    for clusters in candidates:
        best = None
        if clusters == 1:
            initializations = [np.zeros(len(client_speeds), dtype=int)]
        else:
            initializations = [
                KMeans(
                    n_clusters=clusters, n_init=1, random_state=seed + 1000 * clusters + restart
                ).fit_predict(summaries)
                for restart in range(restarts)
            ]
        for restart, assignments in enumerate(initializations):
            assignments = assignments.copy()
            for _ in range(iterations):
                counts = np.bincount(assignments, minlength=clusters)
                if counts.min() < minimum_size:
                    break
                models = fit_cluster_models(
                    assignments, client_speeds, client_targets, clusters, cfg10a
                )
                cost = assignment_cost(models, client_speeds, client_targets, scale, cfg10a)
                updated = np.argmin(cost, axis=1)
                if np.array_equal(updated, assignments):
                    break
                assignments = updated
            counts = np.bincount(assignments, minlength=clusters)
            if len(counts) < clusters or counts.min() < minimum_size:
                continue
            models = fit_cluster_models(
                assignments, client_speeds, client_targets, clusters, cfg10a
            )
            cost = assignment_cost(models, client_speeds, client_targets, scale, cfg10a)
            sse = float(sum(cost[index, assignments[index]] for index in range(len(assignments))))
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
        raise RuntimeError("no feasible mixture candidate")
    selected = min(best_by_k.values(), key=lambda item: item["bic"])
    return selected, pd.DataFrame(candidate_rows), global_model


def run_seed(seed):
    cfg = json.loads(CONFIG.read_text())
    cfg10a = json.loads(CONFIG_10A.read_text())
    clients = sample_fleet(seed)
    grid = np.asarray(cfg["speed_grid"], dtype=float)
    rng = np.random.default_rng(seed + 300000)
    categories = balanced_categories(clients, rng, cfg["coverage_blocks"])
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

    learned, candidates, global_model = fit_mixture(
        client_speeds,
        client_targets,
        cfg["candidate_clusters"],
        cfg["mixture_restarts"],
        cfg["mixture_iterations"],
        cfg["minimum_cluster_size"],
        seed,
        cfg10a,
    )
    truth_names = sorted({client.family for client in clients})
    truth = np.asarray([truth_names.index(client.family) for client in clients])
    ari = adjusted_rand_score(truth, learned["assignments"])

    family_models = {}
    for family in truth_names:
        indices = [index for index, client in enumerate(clients) if client.family == family]
        family_models[family] = ridge_fit(
            np.concatenate([client_speeds[index] for index in indices]),
            np.vstack([client_targets[index] for index in indices]),
            7,
            cfg10a,
            strength=1e-8,
        )

    rows, fit_rows = [], []
    family_size = {
        family: sum(client.family == family for client in clients) for family in truth_names
    }
    for index, client in enumerate(clients):
        local3 = ridge_fit(client_speeds[index], client_targets[index], 3, cfg10a, strength=1e-8)
        local_model = lift_prior(local3, cfg10a)
        total_family_transitions = (
            family_size[client.family]
            * np.mean(
                [
                    len(client_speeds[j])
                    for j, other in enumerate(clients)
                    if other.family == client.family
                ]
            )
            * cfg["samples_per_local_speed"]
        )
        per_speed = max(3, round(total_family_transitions / len(grid)))
        full_targets = np.asarray(
            [estimate_matrix(client, speed, per_speed, cfg, rng) for speed in grid]
        )
        exact_model = ridge_fit(grid, exact[client.client_id], 7, cfg10a, strength=1e-10)
        models = {
            "Local": local_model,
            "Global": global_model,
            "LearnedCluster": learned["models"][learned["assignments"][index]],
            "OracleFamily": family_models[client.family],
            "LocalFull": ridge_fit(grid, full_targets, 7, cfg10a, strength=1e-8),
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
                control, feasible, steering = recovery_metric(client, speed, control_matrix, cfg)
                rows.append(
                    {
                        "seed": seed,
                        "client": client.client_id,
                        "family": client.family,
                        "block": categories[client.client_id],
                        "learned_cluster": int(learned["assignments"][index]),
                        "method": method,
                        "speed": speed,
                        "prediction_error": error,
                        "recovery": control,
                        "steering_rms": steering,
                        "feasible": feasible,
                    }
                )
        fit_rows.append(
            {
                "seed": seed,
                "client": client.client_id,
                "family": client.family,
                "block": categories[client.client_id],
                "learned_cluster": int(learned["assignments"][index]),
                "selected_k": learned["k"],
                "ari": ari,
                "local_full_transitions": per_speed * len(grid),
            }
        )
    candidates = candidates.assign(seed=seed, selected_k=learned["k"], ari=ari)
    for suffix, frame in (
        ("clients", pd.DataFrame(rows)),
        ("fits", pd.DataFrame(fit_rows)),
        ("candidates", candidates),
    ):
        (OUT / f"experiment_10d_seed{seed}_{suffix}.csv.gz").write_bytes(
            gzip.compress(frame.to_csv(index=False).encode(), mtime=0)
        )
    print(seed, flush=True)


def summarize():
    cfg = json.loads(CONFIG.read_text())
    raw = pd.concat(
        [
            pd.read_csv(OUT / f"experiment_10d_seed{seed}_clients.csv.gz")
            for seed in cfg["development_seeds"]
        ]
    )
    fits = pd.concat(
        [
            pd.read_csv(OUT / f"experiment_10d_seed{seed}_fits.csv.gz")
            for seed in cfg["development_seeds"]
        ]
    )
    candidates = pd.concat(
        [
            pd.read_csv(OUT / f"experiment_10d_seed{seed}_candidates.csv.gz")
            for seed in cfg["development_seeds"]
        ]
    )
    units = (
        raw.groupby(["seed", "method"])
        .agg(
            prediction_error=("prediction_error", "mean"),
            recovery=("recovery", "mean"),
            feasible=("feasible", "min"),
        )
        .reset_index()
    )
    summary = (
        units.groupby("method")
        .agg(
            prediction_error=("prediction_error", "mean"),
            prediction_std=("prediction_error", "std"),
            recovery=("recovery", "mean"),
            recovery_std=("recovery", "std"),
            feasible_rate=("feasible", "mean"),
        )
        .reset_index()
    )
    pivot = units.pivot(index="seed", columns="method")
    comparisons = []
    for method, baseline in (
        ("LearnedCluster", "Local"),
        ("LearnedCluster", "Global"),
        ("LearnedCluster", "OracleFamily"),
        ("OracleFamily", "Global"),
    ):
        for metric in ("prediction_error", "recovery"):
            improvement = 100 * (1 - pivot[metric][method] / pivot[metric][baseline])
            low, high = bootstrap_interval(
                improvement, cfg["bootstrap_resamples"], 2000 + len(comparisons)
            )
            comparisons.append(
                {
                    "method": method,
                    "baseline": baseline,
                    "metric": metric,
                    "improvement_pct": float(improvement.mean()),
                    "ci_low": low,
                    "ci_high": high,
                    "wins": int((improvement > 0).sum()),
                }
            )
    comparisons = pd.DataFrame(comparisons)
    selection = fits.groupby("seed").first().reset_index()[["seed", "selected_k", "ari"]]
    learned_vs_local = comparisons[
        (comparisons.method == "LearnedCluster") & (comparisons.baseline == "Local")
    ].set_index("metric")
    summary_by_method = summary.set_index("method")
    retained_benefit = {}
    for metric in ("prediction_error", "recovery"):
        local = summary_by_method.loc["Local", metric]
        learned = summary_by_method.loc["LearnedCluster", metric]
        oracle = summary_by_method.loc["OracleFamily", metric]
        retained_benefit[metric] = float((local - learned) / (local - oracle))
    conclusions = {
        "modal_selected_k": int(selection.selected_k.mode().iloc[0]),
        "three_group_rate": float(np.mean(selection.selected_k == 3)),
        "mean_ari": float(selection.ari.mean()),
        "oracle_benefit_retained": retained_benefit,
        "prediction_gate_pass": bool(learned_vs_local.loc["prediction_error", "ci_low"] > 0),
        "control_gate_pass": bool(learned_vs_local.loc["recovery", "ci_low"] > 0),
        "all_learned_feasible": bool(
            summary[summary.method == "LearnedCluster"].feasible_rate.iloc[0] == 1
        ),
        "interpretation": "Learned clustering is successful only if it retains most of the oracle-family benefit without using family labels; ARI is diagnostic rather than an optimization target.",
        "provenance": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [
                CONFIG,
                CONFIG_10A,
                ROOT / "code/experiments/experiment_10d_learned_compatible_groups.py",
            ]
        },
    }
    units.to_csv(OUT / "experiment_10d_seed_summary.csv", index=False)
    summary.to_csv(OUT / "experiment_10d_summary.csv", index=False)
    comparisons.to_csv(OUT / "experiment_10d_comparisons.csv", index=False)
    selection.to_csv(OUT / "experiment_10d_cluster_selection.csv", index=False)
    candidates.to_csv(OUT / "experiment_10d_candidate_scores.csv", index=False)
    fits.to_csv(OUT / "experiment_10d_fit_summary.csv", index=False)
    (OUT / "experiment_10d_conclusions.json").write_text(json.dumps(conclusions, indent=2) + "\n")
    make_figure(summary, selection)
    print(json.dumps(conclusions, indent=2))
    print(summary.to_string(index=False))
    print(comparisons.to_string(index=False))
    print(selection.to_string(index=False))


def make_figure(summary, selection):
    order = ["Local", "Global", "LearnedCluster", "OracleFamily", "LocalFull", "ExactLPV"]
    view = summary.set_index("method").loc[order]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.5), constrained_layout=True)
    x = np.arange(len(order))
    axes[0].bar(x, 100 * view.prediction_error)
    axes[0].set(ylabel="Full-envelope matrix error [%]")
    exact = view.loc["ExactLPV", "recovery"]
    axes[1].bar(x, 100 * (view.recovery / exact - 1))
    axes[1].set(ylabel="Recovery excess over Exact [%]")
    axes[2].scatter(selection.selected_k, selection.ari)
    axes[2].axvline(3, color="k", linestyle="--", linewidth=1)
    axes[2].set(xlabel="BIC-selected groups", ylabel="ARI", xticks=range(1, 6), ylim=(-0.05, 1.05))
    for axis in axes[:2]:
        axis.set_xticks(
            x, ["Local", "Global", "Learned", "Oracle", "LocalFull", "Exact"], rotation=28
        )
        axis.grid(axis="y", alpha=0.25)
    axes[2].grid(alpha=0.25)
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="*")
    args = parser.parse_args()
    cfg = json.loads(CONFIG.read_text())
    seeds = args.seeds or cfg["development_seeds"]
    if not args.summarize_only:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(run_seed, seeds))
    if set(seeds) == set(cfg["development_seeds"]):
        summarize()


if __name__ == "__main__":
    main()
