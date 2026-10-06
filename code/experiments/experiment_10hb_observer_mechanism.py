"""10H-B: paired observer bias and model-mismatch mechanism audit."""

from __future__ import annotations

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
    sample_fleet,
    scenarios,
)
from experiment_10ha_tire_force_observer import (
    closed_loop,
    design_observer_schedule,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "code/config/experiment_10hb.json"
CONFIG_10H = ROOT / "code/config/experiment_10h.json"
CONFIG_10HA = ROOT / "code/config/experiment_10ha.json"
OUT = ROOT / "results/tables"
FIGURE = ROOT / "results/figures/experiment_10hb_observer_mechanism.pdf"
METHODS = ("FullState", "ExactKF", "OracleClassKF", "GlobalKF")
CONDITIONS = ("noise_only", "noise_plus_bias")


def condition_config(cfg10ha, condition):
    local = json.loads(json.dumps(cfg10ha))
    if condition == "noise_only":
        local["measurement_bias_std"] = {
            "yaw_rate_deg_s": 0.0,
            "lateral_acceleration_mps2": 0.0,
            "steering_deg": 0.0,
        }
    return local


def paired_rng(seed, client_index, scenario_index):
    """Common random numbers across observer models and bias conditions."""
    return np.random.default_rng(seed * 100_000 + client_index * 100 + scenario_index)


def evaluate_seed(seed, cfg, cfg10h, cfg10ha):
    clients = sample_fleet(seed, cfg10h)
    grid = np.asarray(cfg10h["speed_grid"], dtype=float)
    exact, global_model, class_models = architecture_matrices(
        clients, grid, cfg10h["sample_time"]
    )
    scale = json.loads(
        (OUT / "experiment_10ha_frozen_selection.json").read_text()
    )["process_noise_scale"]
    observer_models = {"GlobalKF": global_model}
    observer_gains = {"GlobalKF": design_observer_schedule(global_model, scale, cfg10ha)}
    for name, model in class_models.items():
        observer_models[f"OracleClassKF:{name}"] = model
        observer_gains[f"OracleClassKF:{name}"] = design_observer_schedule(model, scale, cfg10ha)
    controller_weights = json.loads(
        (OUT / "experiment_10h_frozen_selection.json").read_text()
    )["selected"]
    controller = design_schedule(global_model, controller_weights, cfg10h["sample_time"])
    rows = []
    scenario_map = scenarios(cfg10h)
    for client_index, client in enumerate(clients):
        exact_model = exact[client.client_id]
        exact_gains = design_observer_schedule(exact_model, scale, cfg10ha)
        models = {
            "ExactKF": (exact_model, exact_gains),
            "OracleClassKF": (
                observer_models[f"OracleClassKF:{client.vehicle_class}"],
                observer_gains[f"OracleClassKF:{client.vehicle_class}"],
            ),
            "GlobalKF": (global_model, observer_gains["GlobalKF"]),
        }
        for scenario_index, (scenario_name, scenario) in enumerate(scenario_map.items()):
            full = closed_loop(
                client, exact_model, exact_model, exact_gains, controller, scenario,
                condition_config(cfg10ha, "noise_only"), cfg10h,
                paired_rng(seed, client_index, scenario_index), "FullState",
            )
            rows.append({
                "seed": seed, "client": client.client_id,
                "vehicle_class": client.vehicle_class, "scenario": scenario_name,
                "condition": "full_state", "method": "FullState",
            } | full)
            for condition in CONDITIONS:
                local_cfg = condition_config(cfg10ha, condition)
                for method, (model, gains) in models.items():
                    result = closed_loop(
                        client, exact_model, model, gains, controller, scenario,
                        local_cfg, cfg10h,
                        paired_rng(seed, client_index, scenario_index), method,
                    )
                    rows.append({
                        "seed": seed, "client": client.client_id,
                        "vehicle_class": client.vehicle_class, "scenario": scenario_name,
                        "condition": condition, "method": method,
                    } | result)
    return pd.DataFrame(rows)


def bootstrap_interval(values, resamples, seed=10_010):
    values = np.asarray(values)
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, (resamples, len(values)), replace=True).mean(axis=1)
    return np.quantile(draws, [0.025, 0.975])


def seed_summary(data):
    return data.groupby(["seed", "scenario", "condition", "method"]).agg(
        mean_tracking=("tracking_rmse_deg_s", "mean"),
        worst_tracking=("tracking_rmse_deg_s", "max"),
        maximum_command=("command_peak_deg", "max"),
        all_finite=("finite", "all"),
    ).reset_index()


def paired_effect(view, metric, method, condition, baseline_method, baseline_condition):
    left = view[(view.method == method) & (view.condition == condition)].set_index("seed")[metric]
    right = view[(view.method == baseline_method) &
                 (view.condition == baseline_condition)].set_index("seed")[metric]
    common = left.index.intersection(right.index)
    effect = 100 * (left.loc[common] - right.loc[common]) / right.loc[common]
    return effect


def summarize(data, cfg):
    seeds = seed_summary(data)
    summary = seeds.groupby(["scenario", "condition", "method"]).mean(numeric_only=True).reset_index()
    effects = []
    for scenario in sorted(seeds.scenario.unique()):
        local = seeds[seeds.scenario == scenario]
        for metric in ("mean_tracking", "worst_tracking"):
            comparisons = [
                ("ExactKF", "noise_only", "FullState", "full_state", "observer_filtering"),
                ("ExactKF", "noise_plus_bias", "ExactKF", "noise_only", "sensor_bias"),
                ("OracleClassKF", "noise_plus_bias", "ExactKF", "noise_plus_bias", "within_class_mismatch"),
                ("GlobalKF", "noise_plus_bias", "OracleClassKF", "noise_plus_bias", "cross_class_mismatch"),
                ("GlobalKF", "noise_plus_bias", "ExactKF", "noise_plus_bias", "total_model_mismatch"),
            ]
            for method, condition, base_method, base_condition, mechanism in comparisons:
                values = paired_effect(local, metric, method, condition, base_method, base_condition)
                interval = bootstrap_interval(values, cfg["bootstrap_resamples"])
                effects.append({
                    "scenario": scenario, "metric": metric, "mechanism": mechanism,
                    "method": method, "condition": condition,
                    "baseline_method": base_method, "baseline_condition": base_condition,
                    "effect_pct": float(values.mean()),
                    "ci_low": float(interval[0]), "ci_high": float(interval[1]),
                })
    effects = pd.DataFrame(effects)

    recovery_rows = []
    for scenario in sorted(seeds.scenario.unique()):
        local = seeds[(seeds.scenario == scenario) & (seeds.condition == "noise_plus_bias")]
        for metric in ("mean_tracking", "worst_tracking"):
            pivot = local.pivot(index="seed", columns="method", values=metric)
            denominator = pivot.GlobalKF - pivot.ExactKF
            recovery = 100 * (pivot.GlobalKF - pivot.OracleClassKF) / denominator
            recovery = recovery[np.abs(denominator) > 1e-12]
            interval = bootstrap_interval(recovery, cfg["bootstrap_resamples"])
            recovery_rows.append({
                "scenario": scenario, "metric": metric,
                "gap_recovery_pct": float(recovery.mean()),
                "ci_low": float(interval[0]), "ci_high": float(interval[1]),
            })
    recovery = pd.DataFrame(recovery_rows)

    biased = seeds[seeds.condition == "noise_plus_bias"].pivot(
        index=["seed", "scenario"], columns="method", values=["mean_tracking", "worst_tracking"]
    )
    oracle_excess = []
    for metric in ("mean_tracking", "worst_tracking"):
        oracle_excess.extend(
            100 * (biased[metric].OracleClassKF - biased[metric].ExactKF) /
            biased[metric].ExactKF
        )
    gates = {
        "all_output_feedback_finite": bool(data[data.method != "FullState"].finite.all()),
        "oracle_gap_recovery_gate": bool((recovery.gap_recovery_pct >=
                                           cfg["minimum_oracle_gap_recovery_pct"]).all()),
        "oracle_near_exact_gate": bool(np.mean(oracle_excess) <=
                                        cfg["maximum_oracle_excess_vs_exact_pct"]),
        "class_observer_beats_global_gate": bool((recovery.gap_recovery_pct > 0).all()),
    }
    gates["all_gates_pass"] = all(gates.values())
    data.to_csv(OUT / "experiment_10hb_control.csv.gz", index=False, compression="gzip")
    seeds.to_csv(OUT / "experiment_10hb_seed_summary.csv", index=False)
    summary.to_csv(OUT / "experiment_10hb_summary.csv", index=False)
    effects.to_csv(OUT / "experiment_10hb_mechanism_effects.csv", index=False)
    recovery.to_csv(OUT / "experiment_10hb_gap_recovery.csv", index=False)
    (OUT / "experiment_10hb_conclusions.json").write_text(json.dumps({
        "gates": gates,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
    }, indent=2) + "\n")
    return summary, effects, recovery, gates


def plot(summary, effects, recovery):
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
    colors = {"ExactKF": "#009E73", "OracleClassKF": "#0072B2", "GlobalKF": "#D55E00"}
    for axis, scenario in zip(axes[:2], ("broadband", "transient")):
        view = summary[(summary.scenario == scenario) &
                       (summary.condition == "noise_plus_bias")].set_index("method")
        methods = ["ExactKF", "OracleClassKF", "GlobalKF"]
        axis.bar(methods, view.loc[methods].mean_tracking,
                 color=[colors[item] for item in methods])
        axis.set_title(scenario.capitalize())
        axis.set_ylabel("Mean yaw-rate RMSE [deg/s]")
        axis.tick_params(axis="x", rotation=16)
    mechanisms = ["observer_filtering", "sensor_bias", "within_class_mismatch", "cross_class_mismatch"]
    labels = ["Filtering", "Bias", "Within class", "Cross class"]
    local = effects[(effects.scenario == "transient") & (effects.metric == "mean_tracking")]
    values = [local[local.mechanism == item].effect_pct.iloc[0] for item in mechanisms]
    axes[2].bar(labels, values, color=["#999999", "#CC79A7", "#56B4E9", "#E69F00"])
    axes[2].axhline(0, color="black", linewidth=0.8)
    axes[2].set_ylabel("Incremental tracking effect [%]")
    axes[2].set_title("Transient mechanism attribution")
    axes[2].tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(FIGURE)
    plt.close(fig)


def main():
    cfg = json.loads(CONFIG.read_text())
    cfg10h = json.loads(CONFIG_10H.read_text())
    cfg10ha = json.loads(CONFIG_10HA.read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=5) as executor:
        pieces = list(executor.map(
            evaluate_seed, cfg["confirmation_seeds"], [cfg] * 10,
            [cfg10h] * 10, [cfg10ha] * 10,
        ))
    data = pd.concat(pieces, ignore_index=True)
    summary, effects, recovery, gates = summarize(data, cfg)
    plot(summary, effects, recovery)
    print(summary.to_string(index=False))
    print(effects.to_string(index=False))
    print(recovery.to_string(index=False))
    print(json.dumps(gates, indent=2))


if __name__ == "__main__":
    main()
