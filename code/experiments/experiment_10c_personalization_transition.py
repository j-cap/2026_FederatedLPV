"""10C: calibration-coverage transition from a shared backbone to personalization."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from federated_lpv import sample_fleet
from experiment_10a_basis_coverage_audit import collect_targets
from experiment_10b_oracle_complementary_gate import (
    bootstrap_interval,
    estimate_matrix,
    lift_prior,
    predict,
    recovery_metric,
    ridge_fit,
    select_strength,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "code/config/experiment_10c.json"
CONFIG_10A = ROOT / "code/config/experiment_10a.json"
OUT = ROOT / "results/tables"
FIGURE = ROOT / "results/figures/experiment_10c_personalization_transition.pdf"


def progressive_speeds(initial, grid, stage):
    """Expand an initial block by repeatedly adding the nearest missing speed."""
    selected = list(initial)
    target = len(selected) if stage == "base" else int(stage)
    while len(selected) < target:
        missing = [speed for speed in grid if speed not in selected]
        chosen = min(missing, key=lambda value: (min(abs(value - x) for x in selected), value))
        selected.append(chosen)
    return np.asarray(sorted(selected), dtype=float)


def run_seed(seed):
    cfg = json.loads(CONFIG.read_text())
    cfg10a = json.loads(CONFIG_10A.read_text())
    clients = sample_fleet(seed)
    grid = np.asarray(cfg["speed_grid"], dtype=float)
    rng = np.random.default_rng(seed + 200000)
    categories = {}
    for family in sorted({client.family for client in clients}):
        family_clients = [client for client in clients if client.family == family]
        order = rng.permutation(len(family_clients))
        for j, index in enumerate(order):
            categories[family_clients[index].client_id] = j % len(cfg["initial_coverage_blocks"])

    observed, exact = {}, {}
    for client in clients:
        observed[client.client_id] = {
            speed: estimate_matrix(client, speed, cfg["samples_per_speed"], cfg, rng)
            for speed in grid
        }
        _, exact[client.client_id] = collect_targets(client, cfg["curvature_per_m"], cfg10a)

    family_models = {}
    for family in sorted({client.family for client in clients}):
        family_speeds, family_targets = [], []
        for client in clients:
            if client.family != family:
                continue
            initial = cfg["initial_coverage_blocks"][categories[client.client_id]]
            family_speeds.extend(initial)
            family_targets.extend(observed[client.client_id][float(speed)] for speed in initial)
        family_models[family] = ridge_fit(family_speeds, family_targets, 7, cfg10a, strength=1e-8)

    rows, fit_rows = [], []
    for client in clients:
        initial = cfg["initial_coverage_blocks"][categories[client.client_id]]
        family = family_models[client.family]
        exact_coefficients = ridge_fit(grid, exact[client.client_id], 7, cfg10a, strength=1e-10)
        control_cache = {}
        for stage_index, stage in enumerate(cfg["coverage_stages"]):
            local_speeds = progressive_speeds(initial, grid, stage)
            targets = np.asarray([observed[client.client_id][float(speed)] for speed in local_speeds])
            local3 = ridge_fit(local_speeds, targets, 3, cfg10a, strength=1e-8)
            local_prior = lift_prior(local3, cfg10a)
            local_strength = select_strength(
                local_speeds, targets, cfg, cfg10a,
                lambda keep: lift_prior(ridge_fit(local_speeds[keep], targets[keep], 3,
                                                   cfg10a, strength=1e-6), cfg10a),
            )
            personalized_strength = select_strength(
                local_speeds, targets, cfg, cfg10a, lambda keep: family
            )
            models = {
                "FamilyPool": family,
                "FamilyPersonalized": ridge_fit(local_speeds, targets, 7, cfg10a,
                                                 family, personalized_strength),
                "LocalRegularized": ridge_fit(local_speeds, targets, 7, cfg10a,
                                               local_prior, local_strength),
                "ExactLPV": exact_coefficients,
            }
            local_prior_predictions = predict(local_prior, grid, cfg10a)
            family_predictions = predict(family, grid, cfg10a)
            fit_rows.append(dict(
                seed=seed, client=client.client_id, family=client.family,
                stage=stage_index, speed_count=len(local_speeds),
                transitions=len(local_speeds) * cfg["samples_per_speed"],
                personalized_strength=personalized_strength,
                local_strength=local_strength,
                personalization_distance=float(np.linalg.norm(models["FamilyPersonalized"] - family) /
                                               np.linalg.norm(family)),
                personalization_active=personalized_strength <= cfg["personalization_activation_max_strength"],
            ))
            for method, coefficients in models.items():
                predictions = predict(coefficients, grid, cfg10a)
                for index, speed in enumerate(grid):
                    error = np.linalg.norm(predictions[index] - exact[client.client_id][index]) / np.linalg.norm(exact[client.client_id][index])
                    cache_stage = -1 if method in ("FamilyPool", "ExactLPV") else stage_index
                    cache_key = (cache_stage, method, float(speed))
                    if cache_key not in control_cache:
                        primary = recovery_metric(client, speed, predictions[index], cfg)
                        fallback_used = False
                        if not primary[1] and method == "FamilyPersonalized":
                            primary = recovery_metric(client, speed, family_predictions[index], cfg)
                            fallback_used = True
                        elif not primary[1] and method == "LocalRegularized":
                            nearest_index = int(np.argmin(np.abs(local_speeds - speed)))
                            nearest_speed = local_speeds[nearest_index]
                            grid_index = int(np.flatnonzero(grid == nearest_speed)[0])
                            primary = recovery_metric(client, speed, local_prior_predictions[grid_index], cfg)
                            fallback_used = True
                        control_cache[cache_key] = (*primary, fallback_used)
                    control, feasible, steering, fallback_used = control_cache[cache_key]
                    rows.append(dict(
                        seed=seed, client=client.client_id, family=client.family,
                        stage=stage_index, speed_count=len(local_speeds),
                        transitions=len(local_speeds) * cfg["samples_per_speed"],
                        method=method, speed=speed, prediction_error=error,
                        recovery=control, steering_rms=steering, feasible=feasible,
                        fallback_used=fallback_used,
                    ))
    for suffix, frame in (("clients", pd.DataFrame(rows)), ("fits", pd.DataFrame(fit_rows))):
        (OUT / f"experiment_10c_seed{seed}_{suffix}.csv.gz").write_bytes(
            gzip.compress(frame.to_csv(index=False).encode(), mtime=0))
    print(seed, flush=True)


def summarize():
    cfg = json.loads(CONFIG.read_text())
    raw = pd.concat([pd.read_csv(OUT / f"experiment_10c_seed{seed}_clients.csv.gz")
                     for seed in cfg["development_seeds"]])
    fits = pd.concat([pd.read_csv(OUT / f"experiment_10c_seed{seed}_fits.csv.gz")
                      for seed in cfg["development_seeds"]])
    units = raw.groupby(["seed", "stage", "method"]).agg(
        speed_count=("speed_count", "mean"), transitions=("transitions", "mean"),
        prediction_error=("prediction_error", "mean"), recovery=("recovery", "mean"),
        feasible=("feasible", "min"), fallback_rate=("fallback_used", "mean")
    ).reset_index()
    summary = units.groupby(["stage", "method"]).agg(
        speed_count=("speed_count", "mean"), transitions=("transitions", "mean"),
        prediction_error=("prediction_error", "mean"), prediction_std=("prediction_error", "std"),
        recovery=("recovery", "mean"), recovery_std=("recovery", "std"),
        feasible_rate=("feasible", "mean"), fallback_rate=("fallback_rate", "mean")
    ).reset_index()
    fit_summary = fits.groupby("stage").agg(
        speed_count=("speed_count", "mean"), transitions=("transitions", "mean"),
        median_strength=("personalized_strength", "median"),
        active_fraction=("personalization_active", "mean"),
        personalization_distance=("personalization_distance", "mean")
    ).reset_index()
    comparisons = []
    for stage, group in units.groupby("stage"):
        pivot = group.pivot(index="seed", columns="method")
        for method, baseline in (("FamilyPersonalized", "FamilyPool"),
                                 ("LocalRegularized", "FamilyPool"),
                                 ("FamilyPersonalized", "LocalRegularized")):
            for metric in ("prediction_error", "recovery"):
                improvement = 100 * (1 - pivot[metric][method] / pivot[metric][baseline])
                low, high = bootstrap_interval(improvement, cfg["bootstrap_resamples"],
                                               1000 + len(comparisons))
                comparisons.append(dict(
                    stage=stage, speed_count=float(group.speed_count.mean()),
                    transitions=float(group.transitions.mean()), method=method,
                    baseline=baseline, metric=metric,
                    improvement_pct=float(improvement.mean()), ci_low=low, ci_high=high,
                    wins=int((improvement > 0).sum()),
                ))
    comparisons = pd.DataFrame(comparisons)
    pfl = comparisons[(comparisons.method == "FamilyPersonalized") &
                      (comparisons.baseline == "FamilyPool")]
    prediction_crossovers = pfl[(pfl.metric == "prediction_error") & (pfl.ci_low > 0)]
    control_crossovers = pfl[(pfl.metric == "recovery") & (pfl.ci_low > 0)]
    joint = sorted(set(prediction_crossovers.stage) & set(control_crossovers.stage))
    conclusions = {
        "prediction_crossover_stage": None if prediction_crossovers.empty else int(prediction_crossovers.stage.min()),
        "control_crossover_stage": None if control_crossovers.empty else int(control_crossovers.stage.min()),
        "joint_crossover_stage": None if not joint else int(min(joint)),
        "all_personalized_feasible": bool(summary[summary.method == "FamilyPersonalized"].feasible_rate.min() == 1),
        "maximum_personalized_fallback_rate": float(summary[summary.method == "FamilyPersonalized"].fallback_rate.max()),
        "interpretation": "Personalization first improves full-envelope model accuracy at eight locally covered speeds, but never improves nonlinear recovery over the family backbone. At full coverage, LocalRegularized is more accurate than FamilyPersonalized and control remains statistically equivalent.",
        "provenance": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in [CONFIG, CONFIG_10A,
                                    ROOT / "code/experiments/experiment_10c_personalization_transition.py"]},
    }
    units.to_csv(OUT / "experiment_10c_seed_summary.csv", index=False)
    summary.to_csv(OUT / "experiment_10c_summary.csv", index=False)
    fit_summary.to_csv(OUT / "experiment_10c_personalization.csv", index=False)
    comparisons.to_csv(OUT / "experiment_10c_comparisons.csv", index=False)
    fits.to_csv(OUT / "experiment_10c_fit_summary.csv", index=False)
    (OUT / "experiment_10c_conclusions.json").write_text(json.dumps(conclusions, indent=2) + "\n")
    make_figure(summary, fit_summary)
    print(json.dumps(conclusions, indent=2)); print(summary.to_string(index=False)); print(fit_summary.to_string(index=False)); print(comparisons.to_string(index=False))


def make_figure(summary, fit_summary):
    methods = ["FamilyPool", "FamilyPersonalized", "LocalRegularized", "ExactLPV"]
    labels = {"FamilyPool": "Family backbone", "FamilyPersonalized": "Personalized",
              "LocalRegularized": "Local", "ExactLPV": "Exact"}
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), constrained_layout=True)
    for method in methods:
        view = summary[summary.method == method].sort_values("speed_count")
        axes[0].plot(view.speed_count, 100 * view.prediction_error, marker="o", label=labels[method])
        exact = summary[(summary.method == "ExactLPV")].sort_values("speed_count").recovery.to_numpy()
        axes[1].plot(view.speed_count, 100 * (view.recovery.to_numpy() / exact - 1), marker="o", label=labels[method])
    axes[0].set(ylabel="Full-envelope matrix error [%]", yscale="log")
    axes[1].set(ylabel="Recovery excess over Exact [%]")
    axes[2].plot(fit_summary.speed_count, fit_summary.active_fraction, marker="o", label="active fraction")
    axes[2].set(ylabel="Personalization-active clients", ylim=(-.05, 1.05))
    for axis in axes:
        axis.set_xlabel("Locally covered speeds")
        axis.grid(alpha=.25)
    axes[0].legend(frameon=False, fontsize=8)
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, bbox_inches="tight"); plt.close(fig)


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
