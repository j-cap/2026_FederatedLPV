"""10H-C: label-free, unknown-cardinality grouping of observer-model updates."""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from experiment_10h_higher_order_lpv_gate import (
    architecture_matrices,
    design_schedule,
    raw_basis,
    sample_fleet,
    scenarios,
)
from experiment_10ha_tire_force_observer import closed_loop, design_observer_schedule
from experiment_10hb_observer_mechanism import paired_rng
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "code/config/experiment_10hc.json"
CONFIG_10H = ROOT / "code/config/experiment_10h.json"
CONFIG_10HA = ROOT / "code/config/experiment_10ha.json"
OUT = ROOT / "results/tables"
FIGURE = ROOT / "results/figures/experiment_10hc_learned_observer_groups.pdf"


def flatten_schedule(matrices):
    return np.asarray([np.column_stack([a, b]).ravel() for a, b in matrices])


def unflatten_schedule(values):
    schedules = []
    for row in values:
        matrix = row.reshape(5, 6)
        schedules.append((matrix[:, :5], matrix[:, 5:]))
    return schedules


def fit_schedule(speeds, targets, ridge):
    phi = raw_basis(speeds)
    normal = phi.T @ phi + ridge * np.eye(phi.shape[1])
    return np.linalg.solve(normal, phi.T @ targets)


def predict_schedule(coefficients, speeds):
    return raw_basis(speeds) @ coefficients


def client_updates(clients, exact, cfg, cfg10h, seed):
    rng = np.random.default_rng(seed + 1_000_000)
    grid = np.asarray(cfg10h["speed_grid"], dtype=float)
    blocks = cfg10h["coverage_blocks"]
    speeds, targets = [], []
    all_values = np.vstack([flatten_schedule(exact[c.client_id]) for c in clients])
    scale = np.maximum(np.std(all_values, axis=0), 1e-7)
    for index, client in enumerate(clients):
        local_speeds = np.asarray(blocks[index % len(blocks)], dtype=float)
        indices = [int(np.argmin(np.abs(grid - speed))) for speed in local_speeds]
        truth = flatten_schedule(exact[client.client_id])[indices]
        noise = rng.normal(size=truth.shape) * scale * cfg["matrix_update_noise_relative_std"]
        speeds.append(local_speeds)
        targets.append(truth + noise)
    return speeds, targets, scale


def fit_models(assignments, speeds, targets, clusters, cfg):
    models = []
    for cluster in range(clusters):
        members = np.flatnonzero(assignments == cluster)
        models.append(fit_schedule(
            np.concatenate([speeds[index] for index in members]),
            np.vstack([targets[index] for index in members]),
            cfg["ridge_strength"],
        ))
    return models


def assignment_cost(models, speeds, targets, scale):
    costs = np.empty((len(speeds), len(models)))
    for index, (local_speeds, local_targets) in enumerate(zip(speeds, targets)):
        for cluster, model in enumerate(models):
            residual = (local_targets - predict_schedule(model, local_speeds)) / scale
            costs[index, cluster] = np.sum(residual**2)
    return costs


def initial_features(speeds, targets, cfg):
    global_model = fit_schedule(
        np.concatenate(speeds), np.vstack(targets), cfg["ridge_strength"]
    )
    residuals = [
        np.mean(target - predict_schedule(global_model, local_speed), axis=0)
        for local_speed, target in zip(speeds, targets)
    ]
    return StandardScaler().fit_transform(np.asarray(residuals))


def fit_unknown_groups(speeds, targets, scale, cfg, seed):
    features = initial_features(speeds, targets, cfg)
    candidates, best_by_k = [], {}
    observations = sum(target.size for target in targets)
    for clusters in cfg["candidate_clusters"]:
        best = None
        for restart in range(1 if clusters == 1 else cfg["mixture_restarts"]):
            if clusters == 1:
                assignments = np.zeros(len(speeds), dtype=int)
            else:
                assignments = KMeans(
                    n_clusters=clusters, n_init=1,
                    random_state=seed + clusters * 1000 + restart,
                ).fit_predict(features)
            for _ in range(cfg["mixture_iterations"]):
                counts = np.bincount(assignments, minlength=clusters)
                if counts.min() < cfg["minimum_cluster_size"]:
                    break
                models = fit_models(assignments, speeds, targets, clusters, cfg)
                updated = np.argmin(assignment_cost(models, speeds, targets, scale), axis=1)
                if np.array_equal(updated, assignments):
                    break
                assignments = updated
            counts = np.bincount(assignments, minlength=clusters)
            if len(counts) < clusters or counts.min() < cfg["minimum_cluster_size"]:
                continue
            models = fit_models(assignments, speeds, targets, clusters, cfg)
            costs = assignment_cost(models, speeds, targets, scale)
            sse = float(sum(costs[j, assignments[j]] for j in range(len(speeds))))
            parameters = clusters * 5 * 30 + clusters - 1
            bic = observations * np.log(max(sse / observations, 1e-15))
            bic += parameters * np.log(observations)
            result = {
                "k": clusters, "assignments": assignments.copy(), "models": models,
                "sse": sse, "bic": float(bic), "minimum_size": int(counts.min()),
                "restart": restart,
            }
            if best is None or result["bic"] < best["bic"]:
                best = result
        if best is not None:
            best_by_k[clusters] = best
            candidates.append({key: best[key] for key in
                               ("k", "sse", "bic", "minimum_size", "restart")})
    if not best_by_k:
        raise RuntimeError("no admissible group model")
    selected = min(best_by_k.values(), key=lambda item: (item["bic"], item["k"]))
    return selected, pd.DataFrame(candidates)


def biased_config(cfg10ha):
    return json.loads(json.dumps(cfg10ha))


def evaluate_seed(task):
    phase, seed = task
    cfg = json.loads(CONFIG.read_text())
    cfg10h = json.loads(CONFIG_10H.read_text())
    cfg10ha = json.loads(CONFIG_10HA.read_text())
    clients = sample_fleet(seed, cfg10h)
    grid = np.asarray(cfg10h["speed_grid"], dtype=float)
    exact, reference_global_model, _ = architecture_matrices(
        clients, grid, cfg10h["sample_time"]
    )
    speeds, targets, scale = client_updates(clients, exact, cfg, cfg10h, seed)
    learned, candidates = fit_unknown_groups(speeds, targets, scale, cfg, seed)
    class_names = sorted({client.vehicle_class for client in clients})
    truth = np.asarray([class_names.index(client.vehicle_class) for client in clients])
    ari = adjusted_rand_score(truth, learned["assignments"])
    global_coefficients = fit_schedule(
        np.concatenate(speeds), np.vstack(targets), cfg["ridge_strength"]
    )
    class_coefficients = {}
    for name in class_names:
        members = [j for j, client in enumerate(clients) if client.vehicle_class == name]
        class_coefficients[name] = fit_schedule(
            np.concatenate([speeds[j] for j in members]),
            np.vstack([targets[j] for j in members]), cfg["ridge_strength"],
        )
    global_model = unflatten_schedule(predict_schedule(global_coefficients, grid))
    class_models = {
        name: unflatten_schedule(predict_schedule(model, grid))
        for name, model in class_coefficients.items()
    }
    learned_models = [
        unflatten_schedule(predict_schedule(model, grid)) for model in learned["models"]
    ]
    process_scale = json.loads(
        (OUT / "experiment_10ha_frozen_selection.json").read_text()
    )["process_noise_scale"]
    global_gains = design_observer_schedule(global_model, process_scale, cfg10ha)
    class_gains = {
        name: design_observer_schedule(model, process_scale, cfg10ha)
        for name, model in class_models.items()
    }
    learned_gains = [
        design_observer_schedule(model, process_scale, cfg10ha) for model in learned_models
    ]
    controller_weights = json.loads(
        (OUT / "experiment_10h_frozen_selection.json").read_text()
    )["selected"]
    controller = design_schedule(
        reference_global_model, controller_weights, cfg10h["sample_time"]
    )
    if controller is None:
        raise RuntimeError("identified global controller schedule is infeasible")
    rows = []
    for client_index, client in enumerate(clients):
        exact_model = exact[client.client_id]
        exact_gains = design_observer_schedule(exact_model, process_scale, cfg10ha)
        cluster = int(learned["assignments"][client_index])
        observers = {
            "ExactKF": (exact_model, exact_gains),
            "OracleClassKF": (class_models[client.vehicle_class],
                              class_gains[client.vehicle_class]),
            "LearnedClusterKF": (learned_models[cluster], learned_gains[cluster]),
            "GlobalKF": (global_model, global_gains),
        }
        for scenario_index, (scenario_name, scenario) in enumerate(scenarios(cfg10h).items()):
            for method, (model, gains) in observers.items():
                metrics = closed_loop(
                    client, exact_model, model, gains, controller, scenario,
                    biased_config(cfg10ha), cfg10h,
                    paired_rng(seed, client_index, scenario_index), method,
                )
                rows.append({
                    "phase": phase, "seed": seed, "client": client.client_id,
                    "vehicle_class": client.vehicle_class, "scenario": scenario_name,
                    "method": method, "selected_k": learned["k"], "ari": ari,
                    "learned_cluster": cluster,
                } | metrics)
    candidates = candidates.assign(
        phase=phase, seed=seed, selected_k=learned["k"], ari=ari
    )
    return pd.DataFrame(rows), candidates


def bootstrap_interval(values, resamples, seed=10_011):
    values = np.asarray(values)
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, (resamples, len(values)), replace=True).mean(axis=1)
    return np.quantile(draws, [0.025, 0.975])


def summarize(data, candidates, cfg):
    seeds = data.groupby(["phase", "seed", "scenario", "method"]).agg(
        mean_tracking=("tracking_rmse_deg_s", "mean"),
        worst_tracking=("tracking_rmse_deg_s", "max"),
        maximum_command=("command_peak_deg", "max"),
        all_finite=("finite", "all"),
        selected_k=("selected_k", "first"), ari=("ari", "first"),
    ).reset_index()
    summary = seeds.groupby(["phase", "scenario", "method"]).mean(numeric_only=True).reset_index()
    comparisons = []
    recovery_rows = []
    for phase in ("development", "confirmation"):
        for scenario in sorted(data.scenario.unique()):
            local = seeds[(seeds.phase == phase) & (seeds.scenario == scenario)]
            pivot = local.pivot(index="seed", columns="method")
            for metric in ("mean_tracking", "worst_tracking"):
                for method, baseline in (
                    ("LearnedClusterKF", "GlobalKF"),
                    ("LearnedClusterKF", "OracleClassKF"),
                    ("OracleClassKF", "GlobalKF"),
                ):
                    values = 100 * (pivot[metric][baseline] - pivot[metric][method]) / pivot[metric][baseline]
                    interval = bootstrap_interval(values, cfg["bootstrap_resamples"])
                    comparisons.append({
                        "phase": phase, "scenario": scenario, "metric": metric,
                        "method": method, "baseline": baseline,
                        "improvement_pct": float(values.mean()),
                        "ci_low": float(interval[0]), "ci_high": float(interval[1]),
                    })
                denominator = pivot[metric].GlobalKF - pivot[metric].ExactKF
                retention = 100 * (pivot[metric].GlobalKF - pivot[metric].LearnedClusterKF) / denominator
                interval = bootstrap_interval(retention, cfg["bootstrap_resamples"])
                recovery_rows.append({
                    "phase": phase, "scenario": scenario, "metric": metric,
                    "oracle_gap_retention_pct": float(retention.mean()),
                    "ci_low": float(interval[0]), "ci_high": float(interval[1]),
                })
    comparisons = pd.DataFrame(comparisons)
    recovery = pd.DataFrame(recovery_rows)
    selected = candidates[candidates.k == candidates.selected_k].groupby("phase").agg(
        modal_k=("selected_k", lambda x: int(x.mode().iloc[0])),
        three_group_rate=("selected_k", lambda x: float(np.mean(x == 3))),
        mean_ari=("ari", "mean"), minimum_ari=("ari", "min"),
    ).reset_index()
    confirmation_recovery = recovery[recovery.phase == "confirmation"]
    confirmation_comparison = comparisons[
        (comparisons.phase == "confirmation") &
        (comparisons.method == "LearnedClusterKF") &
        (comparisons.baseline == "OracleClassKF")
    ]
    gates = {
        "unknown_cardinality_selects_three_modal": bool(
            selected.set_index("phase").loc["confirmation", "modal_k"] == 3
        ),
        "positive_cluster_recovery_gate": bool((confirmation_recovery.ci_low > 0).all()),
        "oracle_gap_retention_gate": bool((confirmation_recovery.oracle_gap_retention_pct >=
                                             cfg["minimum_oracle_gap_retention_pct"]).all()),
        "near_oracle_tracking_gate": bool((confirmation_comparison.improvement_pct >=
                                             -cfg["maximum_tracking_excess_vs_oracle_pct"]).all()),
        "all_learned_trajectories_finite": bool(
            data[(data.phase == "confirmation") &
                 (data.method == "LearnedClusterKF")].finite.all()
        ),
    }
    gates["all_gates_pass"] = all(gates.values())
    data.to_csv(OUT / "experiment_10hc_control.csv.gz", index=False, compression="gzip")
    candidates.to_csv(OUT / "experiment_10hc_candidate_scores.csv", index=False)
    seeds.to_csv(OUT / "experiment_10hc_seed_summary.csv", index=False)
    summary.to_csv(OUT / "experiment_10hc_summary.csv", index=False)
    comparisons.to_csv(OUT / "experiment_10hc_comparisons.csv", index=False)
    recovery.to_csv(OUT / "experiment_10hc_gap_retention.csv", index=False)
    selected.to_csv(OUT / "experiment_10hc_cluster_selection.csv", index=False)
    (OUT / "experiment_10hc_conclusions.json").write_text(json.dumps({
        "gates": gates,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
    }, indent=2) + "\n")
    return summary, comparisons, recovery, selected, gates


def plot(summary, recovery, selected):
    confirmation = summary[summary.phase == "confirmation"]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
    methods = ["ExactKF", "OracleClassKF", "LearnedClusterKF", "GlobalKF"]
    colors = ["#009E73", "#0072B2", "#56B4E9", "#D55E00"]
    for axis, scenario in zip(axes[:2], ("broadband", "transient")):
        view = confirmation[confirmation.scenario == scenario].set_index("method").loc[methods]
        axis.bar(methods, view.mean_tracking, color=colors)
        axis.set_title(scenario.capitalize())
        axis.set_ylabel("Mean yaw-rate RMSE [deg/s]")
        axis.tick_params(axis="x", rotation=18)
    view = recovery[(recovery.phase == "confirmation") & (recovery.metric == "mean_tracking")]
    axes[2].bar(view.scenario, view.oracle_gap_retention_pct, color="#56B4E9")
    axes[2].axhline(75, color="black", linestyle="--", label="Gate")
    axes[2].set_ylabel("Global-to-Exact gap retained [%]")
    axes[2].set_title(
        f"Learned groups: modal K={selected.set_index('phase').loc['confirmation', 'modal_k']:.0f}"
    )
    axes[2].legend()
    fig.tight_layout()
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--phase", choices=["development", "confirmation", "all"], default="all")
    args = parser.parse_args()
    cfg = json.loads(CONFIG.read_text())
    phases = ["development", "confirmation"] if args.phase == "all" else [args.phase]
    tasks = [(phase, seed) for phase in phases for seed in cfg[f"{phase}_seeds"]]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        pieces = list(executor.map(evaluate_seed, tasks))
    data = pd.concat([piece[0] for piece in pieces], ignore_index=True)
    candidates = pd.concat([piece[1] for piece in pieces], ignore_index=True)
    if args.phase == "all":
        summary, comparisons, recovery, selected, gates = summarize(data, candidates, cfg)
        plot(summary, recovery, selected)
        print(summary.to_string(index=False))
        print(comparisons.to_string(index=False))
        print(recovery.to_string(index=False))
        print(selected.to_string(index=False))
        print(json.dumps(gates, indent=2))


if __name__ == "__main__":
    main()
