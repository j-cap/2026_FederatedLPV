"""10H-D: learned observer groups from production-like measured outputs."""

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
    RelaxationClient,
    architecture_matrices,
    design_schedule,
    discrete_matrices,
    multisine,
    sample_fleet,
    scenarios,
)
from experiment_10ha_tire_force_observer import (
    C,
    closed_loop,
    design_observer_schedule,
    measurement_covariance,
    sample_bias,
)
from experiment_10hb_observer_mechanism import paired_rng
from experiment_10hc_learned_observer_groups import (
    bootstrap_interval,
    fit_schedule,
    fit_unknown_groups,
    predict_schedule,
    unflatten_schedule,
)
from sklearn.metrics import adjusted_rand_score

from federated_lpv import VehicleParameters

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "code/config/experiment_10hd.json"
CONFIG_10H = ROOT / "code/config/experiment_10h.json"
CONFIG_10HA = ROOT / "code/config/experiment_10ha.json"
OUT = ROOT / "results/tables"
FIGURE = ROOT / "results/figures/experiment_10hd_measured_output_groups.pdf"


def nominal_client(cfg10h):
    centers = list(cfg10h["classes"].values())
    mean = {key: float(np.mean([item[key] for item in centers])) for key in centers[0]}
    return RelaxationClient(
        "nominal", "nominal",
        VehicleParameters(
            mean["mass"], mean["yaw_inertia"], mean["front_length"],
            mean["rear_length"], mean["front_stiffness"], mean["rear_stiffness"],
        ),
        mean["actuator_time_constant"], mean["front_relaxation_length"],
        mean["rear_relaxation_length"],
    )


def interpolate_gain(gains, grid, speed):
    return np.asarray([
        [np.interp(speed, grid, gains[:, row, column]) for column in range(3)]
        for row in range(5)
    ])


def local_record(client, speed, band, nominal_matrix, nominal_gain, cfg, cfg10h,
                 cfg10ha, rng):
    samples = cfg10h["identification_samples_per_speed"]
    inputs = multisine(
        samples, cfg10h["sample_time"], band,
        cfg10h["identification_input_std_rad"], rng,
    )
    true_a, true_b = discrete_matrices(client, speed, cfg10h["sample_time"])
    state = np.zeros((samples + 1, 5))
    measurement = np.zeros((samples + 1, 3))
    bias = sample_bias(cfg10ha, rng)
    covariance = measurement_covariance(cfg10ha)
    for k, command in enumerate(inputs):
        state[k + 1] = true_a @ state[k] + true_b[:, 0] * command
        measurement[k + 1] = (
            C @ state[k + 1] + bias
            + rng.multivariate_normal(np.zeros(3), covariance)
        )
    corrected = measurement.copy()
    if cfg["offline_bias_centering"]:
        corrected -= np.mean(corrected, axis=0)
    estimate = np.zeros_like(state)
    for k, command in enumerate(inputs):
        predicted = nominal_matrix[0] @ estimate[k] + nominal_matrix[1][:, 0] * command
        estimate[k + 1] = predicted + nominal_gain @ (corrected[k + 1] - C @ predicted)
    start = cfg["local_transient_samples"]
    regressors = np.column_stack([estimate[start:-1], inputs[start:]])
    targets = estimate[start + 1:]
    return regressors, targets, state[start:-1], state[start + 1:]


def prior_regularized_matrix(regressors, targets, prior, strength):
    hessian = regressors.T @ regressors
    floor = max(float(np.max(np.diag(hessian))) * 1e-6, 1e-12)
    diagonal = np.maximum(np.diag(hessian), floor)
    penalty = strength * np.diag(diagonal)
    coefficients = np.linalg.solve(
        hessian + penalty, regressors.T @ targets + penalty @ prior.T
    )
    return coefficients.T


def measured_updates(clients, exact, nominal_model, nominal_gains, cfg, cfg10h,
                     cfg10ha, seed):
    rng = np.random.default_rng(seed + 1_100_000)
    grid = np.asarray(cfg10h["speed_grid"], dtype=float)
    blocks = cfg10h["coverage_blocks"]
    bands = cfg10h["frequency_bands_hz"]
    speeds, targets, diagnostics = [], [], []
    for index, client in enumerate(clients):
        local_speeds = np.asarray(blocks[index % len(blocks)], dtype=float)
        band = bands[(index // len(blocks)) % len(bands)]
        matrices = []
        for speed in local_speeds:
            grid_index = int(np.argmin(np.abs(grid - speed)))
            regressors, outputs, true_x, true_next = local_record(
                client, speed, band, nominal_model[grid_index],
                nominal_gains[grid_index], cfg, cfg10h, cfg10ha, rng,
            )
            prior = np.column_stack(nominal_model[grid_index])
            estimate = prior_regularized_matrix(
                regressors, outputs, prior, cfg["local_prior_strength"]
            )
            truth = np.column_stack(exact[client.client_id][grid_index])
            pseudo_prediction = outputs - regressors @ estimate.T
            true_prediction = true_next - np.column_stack([
                true_x, regressors[:, -1]
            ]) @ estimate.T
            diagnostics.append({
                "client": client.client_id, "vehicle_class": client.vehicle_class,
                "speed": speed,
                "matrix_error": float(np.linalg.norm(estimate - truth) / np.linalg.norm(truth)),
                "pseudo_prediction_rmse": float(np.sqrt(np.mean(pseudo_prediction**2))),
                "true_state_prediction_rmse": float(np.sqrt(np.mean(true_prediction**2))),
            })
            matrices.append(estimate.ravel())
        speeds.append(local_speeds)
        targets.append(np.asarray(matrices))
    scale = np.maximum(np.std(np.vstack(targets), axis=0), 1e-7)
    return speeds, targets, scale, pd.DataFrame(diagnostics)


def evaluate_seed(task):
    phase, seed = task
    cfg = json.loads(CONFIG.read_text())
    cfg10h = json.loads(CONFIG_10H.read_text())
    cfg10ha = json.loads(CONFIG_10HA.read_text())
    clients = sample_fleet(seed, cfg10h)
    grid = np.asarray(cfg10h["speed_grid"], dtype=float)
    exact, reference_global, _ = architecture_matrices(clients, grid, cfg10h["sample_time"])
    nominal = nominal_client(cfg10h)
    nominal_model = [discrete_matrices(nominal, speed, cfg10h["sample_time"])
                     for speed in grid]
    process_scale = json.loads(
        (OUT / "experiment_10ha_frozen_selection.json").read_text()
    )["process_noise_scale"]
    nominal_gains = design_observer_schedule(nominal_model, process_scale, cfg10ha)
    speeds, targets, scale, diagnostics = measured_updates(
        clients, exact, nominal_model, nominal_gains, cfg, cfg10h, cfg10ha, seed
    )
    learned, candidates = fit_unknown_groups(speeds, targets, scale, cfg, seed)
    names = sorted({client.vehicle_class for client in clients})
    truth_labels = np.asarray([names.index(client.vehicle_class) for client in clients])
    ari = adjusted_rand_score(truth_labels, learned["assignments"])
    global_coefficients = fit_schedule(
        np.concatenate(speeds), np.vstack(targets), cfg["ridge_strength"]
    )
    class_coefficients = {}
    for name in names:
        members = [j for j, client in enumerate(clients) if client.vehicle_class == name]
        class_coefficients[name] = fit_schedule(
            np.concatenate([speeds[j] for j in members]),
            np.vstack([targets[j] for j in members]), cfg["ridge_strength"],
        )
    observer_models = {
        "GlobalKF": unflatten_schedule(predict_schedule(global_coefficients, grid)),
        **{
            f"OracleClassKF:{name}": unflatten_schedule(predict_schedule(model, grid))
            for name, model in class_coefficients.items()
        },
    }
    learned_models = [
        unflatten_schedule(predict_schedule(model, grid)) for model in learned["models"]
    ]
    try:
        observer_gains = {
            key: design_observer_schedule(model, process_scale, cfg10ha)
            for key, model in observer_models.items()
        }
        learned_gains = [
            design_observer_schedule(model, process_scale, cfg10ha)
            for model in learned_models
        ]
    except np.linalg.LinAlgError:
        return pd.DataFrame(), candidates.assign(
            phase=phase, seed=seed, selected_k=learned["k"], ari=ari,
            observer_design_success=False,
        ), diagnostics.assign(phase=phase, seed=seed)
    controller_weights = json.loads(
        (OUT / "experiment_10h_frozen_selection.json").read_text()
    )["selected"]
    controller = design_schedule(reference_global, controller_weights, cfg10h["sample_time"])
    rows = []
    for client_index, client in enumerate(clients):
        exact_model = exact[client.client_id]
        exact_gains = design_observer_schedule(exact_model, process_scale, cfg10ha)
        cluster = int(learned["assignments"][client_index])
        models = {
            "ExactKF": (exact_model, exact_gains),
            "OracleClassKF": (
                observer_models[f"OracleClassKF:{client.vehicle_class}"],
                observer_gains[f"OracleClassKF:{client.vehicle_class}"],
            ),
            "LearnedClusterKF": (learned_models[cluster], learned_gains[cluster]),
            "GlobalKF": (observer_models["GlobalKF"], observer_gains["GlobalKF"]),
            "NominalKF": (nominal_model, nominal_gains),
        }
        for scenario_index, (scenario_name, scenario) in enumerate(scenarios(cfg10h).items()):
            for method, (model, gains) in models.items():
                result = closed_loop(
                    client, exact_model, model, gains, controller, scenario,
                    cfg10ha, cfg10h, paired_rng(seed, client_index, scenario_index), method,
                )
                rows.append({
                    "phase": phase, "seed": seed, "client": client.client_id,
                    "vehicle_class": client.vehicle_class, "scenario": scenario_name,
                    "method": method, "selected_k": learned["k"], "ari": ari,
                    "learned_cluster": cluster,
                } | result)
    candidates = candidates.assign(
        phase=phase, seed=seed, selected_k=learned["k"], ari=ari,
        observer_design_success=True,
    )
    return pd.DataFrame(rows), candidates, diagnostics.assign(phase=phase, seed=seed)


def summarize(data, candidates, diagnostics, cfg):
    seeds = data.groupby(["phase", "seed", "scenario", "method"]).agg(
        mean_tracking=("tracking_rmse_deg_s", "mean"),
        worst_tracking=("tracking_rmse_deg_s", "max"),
        maximum_command=("command_peak_deg", "max"),
        all_finite=("finite", "all"), selected_k=("selected_k", "first"),
        ari=("ari", "first"),
    ).reset_index()
    summary = seeds.groupby(["phase", "scenario", "method"]).mean(numeric_only=True).reset_index()
    comparisons, retention_rows = [], []
    for phase in ("development", "confirmation"):
        for scenario in sorted(data.scenario.unique()):
            local = seeds[(seeds.phase == phase) & (seeds.scenario == scenario)]
            pivot = local.pivot(index="seed", columns="method")
            for metric in ("mean_tracking", "worst_tracking"):
                for method, baseline in (
                    ("LearnedClusterKF", "GlobalKF"),
                    ("LearnedClusterKF", "OracleClassKF"),
                    ("LearnedClusterKF", "NominalKF"),
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
                retained = 100 * (pivot[metric].GlobalKF - pivot[metric].LearnedClusterKF) / denominator
                interval = bootstrap_interval(retained, cfg["bootstrap_resamples"])
                retention_rows.append({
                    "phase": phase, "scenario": scenario, "metric": metric,
                    "gap_retention_pct": float(retained.mean()),
                    "ci_low": float(interval[0]), "ci_high": float(interval[1]),
                })
    comparisons = pd.DataFrame(comparisons)
    retention = pd.DataFrame(retention_rows)
    selected = candidates[candidates.k == candidates.selected_k].groupby("phase").agg(
        modal_k=("selected_k", lambda x: int(x.mode().iloc[0])),
        mean_k=("selected_k", "mean"), three_group_rate=("selected_k", lambda x: np.mean(x == 3)),
        mean_ari=("ari", "mean"), minimum_ari=("ari", "min"),
        observer_design_rate=("observer_design_success", "mean"),
    ).reset_index()
    confirm_compare = comparisons[
        (comparisons.phase == "confirmation") &
        (comparisons.method == "LearnedClusterKF")
    ]
    global_comparison = confirm_compare[confirm_compare.baseline == "GlobalKF"]
    oracle_comparison = confirm_compare[confirm_compare.baseline == "OracleClassKF"]
    confirm_retention = retention[retention.phase == "confirmation"]
    gates = {
        "observer_design_gate": bool(
            selected.set_index("phase").loc["confirmation", "observer_design_rate"] == 1
        ),
        "positive_global_improvement_intervals": bool((global_comparison.ci_low > 0).all()),
        "global_improvement_gate": bool((global_comparison.improvement_pct >=
                                          cfg["minimum_global_improvement_pct"]).all()),
        "gap_retention_gate": bool((confirm_retention.gap_retention_pct >=
                                     cfg["minimum_oracle_gap_retention_pct"]).all()),
        "near_oracle_gate": bool((oracle_comparison.improvement_pct >=
                                   -cfg["maximum_tracking_excess_vs_oracle_pct"]).all()),
        "all_learned_finite": bool(data[(data.phase == "confirmation") &
                                         (data.method == "LearnedClusterKF")].finite.all()),
    }
    gates["all_gates_pass"] = all(gates.values())
    data.to_csv(OUT / "experiment_10hd_control.csv.gz", index=False, compression="gzip")
    candidates.to_csv(OUT / "experiment_10hd_candidate_scores.csv", index=False)
    diagnostics.to_csv(OUT / "experiment_10hd_local_fit_diagnostics.csv", index=False)
    seeds.to_csv(OUT / "experiment_10hd_seed_summary.csv", index=False)
    summary.to_csv(OUT / "experiment_10hd_summary.csv", index=False)
    comparisons.to_csv(OUT / "experiment_10hd_comparisons.csv", index=False)
    retention.to_csv(OUT / "experiment_10hd_gap_retention.csv", index=False)
    selected.to_csv(OUT / "experiment_10hd_cluster_selection.csv", index=False)
    (OUT / "experiment_10hd_conclusions.json").write_text(json.dumps({
        "gates": gates,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
    }, indent=2) + "\n")
    return summary, comparisons, retention, selected, gates


def summarize_development(data, candidates, diagnostics, cfg):
    seeds = data.groupby(["seed", "scenario", "method"]).agg(
        mean_tracking=("tracking_rmse_deg_s", "mean"),
        worst_tracking=("tracking_rmse_deg_s", "max"),
        all_finite=("finite", "all"), selected_k=("selected_k", "first"),
        ari=("ari", "first"),
    ).reset_index()
    summary = seeds.groupby(["scenario", "method"]).mean(numeric_only=True).reset_index()
    comparisons = []
    for scenario in sorted(data.scenario.unique()):
        pivot = seeds[seeds.scenario == scenario].pivot(index="seed", columns="method")
        for metric in ("mean_tracking", "worst_tracking"):
            for baseline in ("GlobalKF", "OracleClassKF", "NominalKF"):
                values = 100 * (
                    pivot[metric][baseline] - pivot[metric].LearnedClusterKF
                ) / pivot[metric][baseline]
                interval = bootstrap_interval(values, cfg["bootstrap_resamples"])
                comparisons.append({
                    "scenario": scenario, "metric": metric,
                    "method": "LearnedClusterKF", "baseline": baseline,
                    "improvement_pct": float(values.mean()),
                    "ci_low": float(interval[0]), "ci_high": float(interval[1]),
                })
    comparisons = pd.DataFrame(comparisons)
    selected_rows = candidates[candidates.k == candidates.selected_k]
    selection = pd.DataFrame([{
        "modal_k": int(selected_rows.selected_k.mode().iloc[0]),
        "mean_k": float(selected_rows.selected_k.mean()),
        "three_group_rate": float(np.mean(selected_rows.selected_k == 3)),
        "mean_ari": float(selected_rows.ari.mean()),
        "minimum_ari": float(selected_rows.ari.min()),
        "observer_design_rate": float(selected_rows.observer_design_success.mean()),
    }])
    fit_summary = diagnostics.groupby("phase").agg(
        matrix_error=("matrix_error", "mean"),
        pseudo_prediction_rmse=("pseudo_prediction_rmse", "mean"),
        true_state_prediction_rmse=("true_state_prediction_rmse", "mean"),
    ).reset_index()
    global_comparison = comparisons[comparisons.baseline == "GlobalKF"]
    gates = {
        "modal_three_groups": bool(selection.modal_k.iloc[0] == 3),
        "ari_gate": bool(selection.mean_ari.iloc[0] >= cfg["minimum_development_ari"]),
        "local_matrix_error_gate": bool(
            fit_summary.matrix_error.iloc[0] <= cfg["maximum_development_matrix_error"]
        ),
        "positive_global_intervals": bool((global_comparison.ci_low > 0).all()),
        "global_improvement_gate": bool((global_comparison.improvement_pct >=
                                          cfg["minimum_global_improvement_pct"]).all()),
        "all_learned_finite": bool(data[data.method == "LearnedClusterKF"].finite.all()),
    }
    gates["development_gate_pass"] = all(gates.values())
    data.to_csv(OUT / "experiment_10hd_development_control.csv.gz", index=False,
                compression="gzip")
    candidates.to_csv(OUT / "experiment_10hd_development_candidate_scores.csv", index=False)
    diagnostics.to_csv(OUT / "experiment_10hd_development_local_fit_diagnostics.csv", index=False)
    seeds.to_csv(OUT / "experiment_10hd_development_seed_summary.csv", index=False)
    summary.to_csv(OUT / "experiment_10hd_development_summary.csv", index=False)
    comparisons.to_csv(OUT / "experiment_10hd_development_comparisons.csv", index=False)
    selection.to_csv(OUT / "experiment_10hd_development_cluster_selection.csv", index=False)
    fit_summary.to_csv(OUT / "experiment_10hd_development_fit_summary.csv", index=False)
    (OUT / "experiment_10hd_conclusions.json").write_text(json.dumps({
        "development_gates": gates,
        "confirmation_run": False,
        "reserved_confirmation_seeds": cfg["confirmation_seeds"],
        "decision": "Stop after the failed development gate; do not expose confirmation fleets.",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
    }, indent=2) + "\n")
    plot_development(summary, selection, fit_summary)
    return summary, comparisons, selection, fit_summary, gates


def plot_development(summary, selection, fit_summary):
    methods = ["ExactKF", "OracleClassKF", "LearnedClusterKF", "GlobalKF", "NominalKF"]
    colors = ["#009E73", "#0072B2", "#56B4E9", "#D55E00", "#999999"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    for axis, scenario in zip(axes[:2], ("broadband", "transient")):
        view = summary[summary.scenario == scenario].set_index("method").loc[methods]
        axis.bar(methods, view.mean_tracking, color=colors)
        axis.set_title(scenario.capitalize())
        axis.set_ylabel("Mean yaw-rate RMSE [deg/s]")
        axis.tick_params(axis="x", rotation=20)
    values = [100 * fit_summary.matrix_error.iloc[0], 100 * selection.mean_ari.iloc[0]]
    axes[2].bar(["Matrix accuracy", "Clustering ARI"], values,
                color=["#CC79A7", "#E69F00"])
    axes[2].axhline(100, color="black", linewidth=0.8)
    axes[2].set_ylabel("Score [%]; 100 is ideal")
    axes[2].set_title(f"Modal K={selection.modal_k.iloc[0]:.0f}")
    fig.tight_layout()
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE)
    plt.close(fig)


def plot(summary, retention, selected):
    confirmation = summary[summary.phase == "confirmation"]
    methods = ["ExactKF", "OracleClassKF", "LearnedClusterKF", "GlobalKF", "NominalKF"]
    colors = ["#009E73", "#0072B2", "#56B4E9", "#D55E00", "#999999"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    for axis, scenario in zip(axes[:2], ("broadband", "transient")):
        view = confirmation[confirmation.scenario == scenario].set_index("method").loc[methods]
        axis.bar(methods, view.mean_tracking, color=colors)
        axis.set_title(scenario.capitalize())
        axis.set_ylabel("Mean yaw-rate RMSE [deg/s]")
        axis.tick_params(axis="x", rotation=20)
    view = retention[(retention.phase == "confirmation") &
                     (retention.metric == "mean_tracking")]
    axes[2].bar(view.scenario, view.gap_retention_pct, color="#56B4E9")
    axes[2].axhline(50, color="black", linestyle="--", label="Gate")
    axes[2].set_ylabel("Global-to-Exact gap retained [%]")
    axes[2].set_title(
        f"Measured-output groups: modal K={selected.set_index('phase').loc['confirmation', 'modal_k']:.0f}"
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
    data = pd.concat([piece[0] for piece in pieces if not piece[0].empty], ignore_index=True)
    candidates = pd.concat([piece[1] for piece in pieces], ignore_index=True)
    diagnostics = pd.concat([piece[2] for piece in pieces], ignore_index=True)
    if args.phase == "all":
        summary, comparisons, retention, selected, gates = summarize(
            data, candidates, diagnostics, cfg
        )
        plot(summary, retention, selected)
        print(summary.to_string(index=False))
        print(comparisons.to_string(index=False))
        print(retention.to_string(index=False))
        print(selected.to_string(index=False))
        print(diagnostics.groupby("phase").mean(numeric_only=True).to_string())
        print(json.dumps(gates, indent=2))
    elif args.phase == "development":
        summary, comparisons, selection, fit_summary, gates = summarize_development(
            data, candidates, diagnostics, cfg
        )
        print(summary.to_string(index=False))
        print(comparisons.to_string(index=False))
        print(selection.to_string(index=False))
        print(fit_summary.to_string(index=False))
        print(json.dumps(gates, indent=2))


if __name__ == "__main__":
    main()
