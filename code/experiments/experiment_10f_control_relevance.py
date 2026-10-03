"""10F: control-relevance stress test for federated LPV models."""

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
    ridge_fit,
)
from experiment_10e_federated_latent_groups import (
    fit_federated_mixture,
    label_free_categories,
)

from federated_lpv import (
    augmented_tracking_matrices,
    design_lqi_gain,
    nonlinear_bicycle_rhs,
    sample_fleet,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "code/config/experiment_10f.json"
CONFIG_10A = ROOT / "code/config/experiment_10a.json"
OUT = ROOT / "results/tables"
SELECTION = OUT / "experiment_10f_frozen_selection.json"
FIGURE = ROOT / "results/figures/experiment_10f_control_relevance.pdf"
METHODS = ("Local", "Global", "FederatedLearned", "OracleFamily", "ExactLPV")


def smooth_three_point(time, values):
    midpoint = time[-1] / 2
    first = np.clip(time / midpoint, 0, 1)
    second = np.clip((time - midpoint) / midpoint, 0, 1)
    blend_first = 0.5 - 0.5 * np.cos(np.pi * first)
    blend_second = 0.5 - 0.5 * np.cos(np.pi * second)
    return np.where(
        time <= midpoint,
        values[0] + (values[1] - values[0]) * blend_first,
        values[1] + (values[2] - values[1]) * blend_second,
    )


def scenarios(cfg):
    time = np.arange(0, cfg["duration_s"] + cfg["sample_time"], cfg["sample_time"])
    envelope = np.sin(np.pi * time / time[-1]) ** 2
    moderate_speed = smooth_three_point(time, (11.0, 29.0, 15.0))
    hard_speed = smooth_three_point(time, (29.0, 12.0, 28.0))
    # Define severity through requested lateral acceleration, a_y = v^2 kappa,
    # so changing speed does not create an accidental high-speed overload.
    moderate_acceleration = (
        4.0
        * envelope
        * (0.72 * np.sin(2 * np.pi * 0.35 * time) + 0.28 * np.sin(2 * np.pi * 0.72 * time + 0.4))
    )
    hard_acceleration = 7.0 * (
        np.exp(-0.5 * ((time - 3.0) / 0.40) ** 2)
        - 1.10 * np.exp(-0.5 * ((time - 6.0) / 0.48) ** 2)
        + 0.80 * np.exp(-0.5 * ((time - 9.2) / 0.55) ** 2)
    )
    return {
        "moderate": {
            "time": time,
            "speed": moderate_speed,
            "curvature": moderate_acceleration / np.square(moderate_speed),
        },
        "hard": {
            "time": time,
            "speed": hard_speed,
            "curvature": hard_acceleration / np.square(hard_speed),
        },
    }


def build_fleet_models(seed, cfg, cfg10a):
    clients = sample_fleet(seed)
    rng = np.random.default_rng(seed + 600000)
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
    learned, _, global_model, _ = fit_federated_mixture(
        client_speeds,
        client_targets,
        cfg["candidate_clusters"],
        cfg["mixture_restarts"],
        cfg["mixture_iterations"],
        cfg["minimum_cluster_size"],
        seed,
        cfg10a,
        strength=cfg["ridge_strength"],
    )
    family_models = {}
    for family in sorted({client.family for client in clients}):
        indices = [index for index, client in enumerate(clients) if client.family == family]
        family_models[family] = ridge_fit(
            np.concatenate([client_speeds[index] for index in indices]),
            np.vstack([client_targets[index] for index in indices]),
            7,
            cfg10a,
            strength=cfg["ridge_strength"],
        )
    grid = np.asarray(cfg["speed_grid"], dtype=float)
    models = {}
    for index, client in enumerate(clients):
        local3 = ridge_fit(client_speeds[index], client_targets[index], 3, cfg10a, strength=1e-8)
        models[client.client_id] = {
            "Local": lift_prior(local3, cfg10a),
            "Global": global_model,
            "FederatedLearned": learned["models"][learned["assignments"][index]],
            "OracleFamily": family_models[client.family],
            "ExactLPV": ridge_fit(grid, exact[client.client_id], 7, cfg10a, strength=1e-10),
            "local_speeds": client_speeds[index],
        }
    return clients, models, learned["k"]


def design_schedule(model, speeds, weights, cfg10a, local_speeds=None):
    q = np.diag([weights["q_beta"], weights["q_yaw"], weights["q_integral"]])
    r = np.asarray([[weights["r"]]])
    gains, prefilters = [], []
    for speed in speeds:
        evaluation_speed = speed
        if local_speeds is not None and speed not in set(local_speeds):
            evaluation_speed = min(local_speeds, key=lambda value: abs(value - speed))
        matrix = predict(model, [evaluation_speed], cfg10a)[0]
        a, b = matrix[:4].reshape(2, 2), matrix[4:].reshape(2, 1)
        try:
            gain, prefilter = design_lqi_gain(a, b, cfg10a["sample_time"], q, r)
        except (ValueError, np.linalg.LinAlgError):
            return None
        gains.append(gain)
        prefilters.append(prefilter)
    return np.asarray(gains), np.asarray(prefilters)


def interpolate_schedule(schedule, grid, speed):
    gains, prefilters = schedule
    gain = np.asarray([np.interp(speed, grid, gains[:, column]) for column in range(3)])
    return gain, float(np.interp(speed, grid, prefilters))


def rk4_step(state, steering, speed, parameters, dt):
    def rhs(value):
        return nonlinear_bicycle_rhs(value, steering, speed, parameters)

    k1 = rhs(state)
    k2 = rhs(state + dt * k1 / 2)
    k3 = rhs(state + dt * k2 / 2)
    k4 = rhs(state + dt * k3)
    return state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6


def simulate_tracking(client, schedule, scenario, cfg):
    time, speed, curvature = (
        scenario["time"],
        scenario["speed"],
        scenario["curvature"],
    )
    reference = speed * curvature
    state = np.zeros((len(time), 2))
    integral = np.zeros(len(time))
    steering = np.zeros(len(time) - 1)
    raw_steering = np.zeros(len(time) - 1)
    limit = np.deg2rad(cfg["steering_limit_deg"])
    rate_step = np.deg2rad(cfg["steering_rate_limit_deg_s"]) * cfg["sample_time"]
    grid = np.asarray(cfg["speed_grid"], dtype=float)
    if schedule is None:
        return None
    for index in range(len(time) - 1):
        gain, prefilter = interpolate_schedule(schedule, grid, speed[index])
        raw = -gain @ np.r_[state[index], integral[index]] + prefilter * reference[index]
        previous = steering[index - 1] if index else 0.0
        command = np.clip(raw, previous - rate_step, previous + rate_step)
        command = float(np.clip(command, -limit, limit))
        raw_steering[index] = raw
        steering[index] = command
        state[index + 1] = rk4_step(
            state[index], command, float(speed[index]), client.parameters, cfg["sample_time"]
        )
        error = state[index, 1] - reference[index]
        candidate_integral = np.clip(
            integral[index] + cfg["sample_time"] * error,
            -cfg["integral_limit"],
            cfg["integral_limit"],
        )
        # Conditional integration prevents windup when saturation pushes in the
        # same direction as the unconstrained command.
        if np.isclose(command, raw) or np.sign(raw - command) != np.sign(-gain[2] * error):
            integral[index + 1] = candidate_integral
        else:
            integral[index + 1] = integral[index]
        if not np.isfinite(state[index + 1]).all():
            return None
    tracking = state[:, 1] - reference
    steering_rate = np.diff(np.r_[0.0, steering]) / cfg["sample_time"]
    saturated = np.abs(raw_steering - steering) > 1e-10
    metrics = {
        "tracking_rmse_deg_s": float(np.rad2deg(np.sqrt(np.mean(tracking**2)))),
        "tracking_peak_deg_s": float(np.rad2deg(np.max(np.abs(tracking)))),
        "beta_rms_deg": float(np.rad2deg(np.sqrt(np.mean(state[:, 0] ** 2)))),
        "beta_peak_deg": float(np.rad2deg(np.max(np.abs(state[:, 0])))),
        "steering_rms_deg": float(np.rad2deg(np.sqrt(np.mean(steering**2)))),
        "steering_peak_deg": float(np.rad2deg(np.max(np.abs(steering)))),
        "steering_rate_peak_deg_s": float(np.rad2deg(np.max(np.abs(steering_rate)))),
        "saturation_fraction": float(np.mean(saturated)),
    }
    metrics["feasible"] = bool(
        metrics["beta_peak_deg"] <= cfg["beta_limit_deg"]
        and np.isfinite(list(metrics.values())).all()
    )
    return metrics


def stability_radius(schedule, exact_model, cfg, cfg10a):
    if schedule is None:
        return np.inf
    radii = []
    grid = np.asarray(cfg["speed_grid"], dtype=float)
    for speed in np.linspace(grid.min(), grid.max(), 81):
        matrix = predict(exact_model, [speed], cfg10a)[0]
        a, b = matrix[:4].reshape(2, 2), matrix[4:].reshape(2, 1)
        augmented_a, augmented_b = augmented_tracking_matrices(a, b, cfg["sample_time"])
        gain, _ = interpolate_schedule(schedule, grid, speed)
        radii.append(
            float(np.max(np.abs(np.linalg.eigvals(augmented_a - augmented_b @ gain[None, :]))))
        )
    return max(radii)


def evaluate_prebuilt(seed, weights, phase, clients, models, selected_k, cfg, cfg10a):
    grid = np.asarray(cfg["speed_grid"], dtype=float)
    rows = []
    methods = ("ExactLPV",) if phase == "audit" else METHODS
    scenario_set = scenarios(cfg)
    for client in clients:
        for method in methods:
            item = models[client.client_id]
            schedule = design_schedule(
                item[method],
                grid,
                weights,
                cfg10a,
                item["local_speeds"] if method == "Local" else None,
            )
            rho = stability_radius(schedule, item["ExactLPV"], cfg, cfg10a)
            for scenario_name, scenario in scenario_set.items():
                metrics = simulate_tracking(client, schedule, scenario, cfg)
                if metrics is None:
                    metrics = {
                        "tracking_rmse_deg_s": np.inf,
                        "tracking_peak_deg_s": np.inf,
                        "beta_rms_deg": np.inf,
                        "beta_peak_deg": np.inf,
                        "steering_rms_deg": np.inf,
                        "steering_peak_deg": np.inf,
                        "steering_rate_peak_deg_s": np.inf,
                        "saturation_fraction": 1.0,
                        "feasible": False,
                    }
                rows.append(
                    {
                        "seed": seed,
                        "phase": phase,
                        "candidate": weights["name"],
                        "selected_k": selected_k,
                        "client": client.client_id,
                        "family": client.family,
                        "method": method,
                        "scenario": scenario_name,
                        "rho": rho,
                        **metrics,
                    }
                )
    return pd.DataFrame(rows)


def evaluate_seed(seed, weights, phase):
    cfg = json.loads(CONFIG.read_text())
    cfg10a = json.loads(CONFIG_10A.read_text())
    clients, models, selected_k = build_fleet_models(seed, cfg, cfg10a)
    return evaluate_prebuilt(seed, weights, phase, clients, models, selected_k, cfg, cfg10a)


def run_audit_seed(seed):
    cfg = json.loads(CONFIG.read_text())
    cfg10a = json.loads(CONFIG_10A.read_text())
    clients, models, selected_k = build_fleet_models(seed, cfg, cfg10a)
    frames = [
        evaluate_prebuilt(seed, candidate, "audit", clients, models, selected_k, cfg, cfg10a)
        for candidate in cfg["controller_candidates"]
    ]
    frame = pd.concat(frames, ignore_index=True)
    path = OUT / f"experiment_10f_audit_seed{seed}.csv.gz"
    path.write_bytes(gzip.compress(frame.to_csv(index=False).encode(), mtime=0))
    print("audit", seed, flush=True)


def select_controller():
    cfg = json.loads(CONFIG.read_text())
    raw = pd.concat(
        [
            pd.read_csv(OUT / f"experiment_10f_audit_seed{seed}.csv.gz")
            for seed in cfg["development_seeds"]
        ],
        ignore_index=True,
    )
    summary = (
        raw.groupby("candidate")
        .agg(
            tracking=("tracking_rmse_deg_s", "mean"),
            worst_tracking=("tracking_rmse_deg_s", "max"),
            feasible_fraction=("feasible", "mean"),
            saturation_fraction=("saturation_fraction", "mean"),
            beta_peak=("beta_peak_deg", "max"),
            rho=("rho", "max"),
        )
        .reset_index()
    )
    passing = summary[
        (1 - summary.feasible_fraction <= cfg["selection_max_infeasible_fraction"])
        & (summary.saturation_fraction <= cfg["selection_max_mean_saturation_fraction"])
        & (summary.rho < 1)
    ]
    if passing.empty:
        raise RuntimeError("no controller candidate passes the frozen development constraints")
    selected_name = passing.sort_values(["tracking", "worst_tracking"]).iloc[0].candidate
    selected = next(item for item in cfg["controller_candidates"] if item["name"] == selected_name)
    payload = {
        "selection_rule": "minimum ExactLPV mean tracking RMSE among candidates passing feasibility, mean saturation, and dense-grid stability gates",
        "selected": selected,
        "development_seeds": cfg["development_seeds"],
        "federated_or_global_results_used_for_selection": False,
    }
    summary.to_csv(OUT / "experiment_10f_controller_audit.csv", index=False)
    SELECTION.write_text(json.dumps(payload, indent=2) + "\n")
    return selected


def run_confirmation_seed(seed):
    selected = json.loads(SELECTION.read_text())["selected"]
    frame = evaluate_seed(seed, selected, "confirmation")
    path = OUT / f"experiment_10f_confirmation_seed{seed}.csv.gz"
    path.write_bytes(gzip.compress(frame.to_csv(index=False).encode(), mtime=0))
    print("confirmation", seed, flush=True)


def summarize_confirmation():
    cfg = json.loads(CONFIG.read_text())
    raw = pd.concat(
        [
            pd.read_csv(OUT / f"experiment_10f_confirmation_seed{seed}.csv.gz")
            for seed in cfg["confirmation_seeds"]
        ],
        ignore_index=True,
    )
    seed_summary = (
        raw.groupby(["seed", "scenario", "method"])
        .agg(
            tracking=("tracking_rmse_deg_s", "mean"),
            worst_client=("tracking_rmse_deg_s", "max"),
            tracking_peak=("tracking_peak_deg_s", "max"),
            beta_peak=("beta_peak_deg", "max"),
            steering_rms=("steering_rms_deg", "mean"),
            steering_rate_peak=("steering_rate_peak_deg_s", "max"),
            saturation_fraction=("saturation_fraction", "mean"),
            feasible=("feasible", "min"),
            rho=("rho", "max"),
        )
        .reset_index()
    )
    summary = (
        seed_summary.groupby(["scenario", "method"])
        .agg(
            tracking=("tracking", "mean"),
            tracking_std=("tracking", "std"),
            worst_client=("worst_client", "mean"),
            tracking_peak=("tracking_peak", "mean"),
            beta_peak=("beta_peak", "max"),
            steering_rms=("steering_rms", "mean"),
            steering_rate_peak=("steering_rate_peak", "max"),
            saturation_fraction=("saturation_fraction", "mean"),
            feasible_rate=("feasible", "mean"),
            rho=("rho", "max"),
        )
        .reset_index()
    )
    comparisons = []
    for scenario in ("moderate", "hard"):
        pivot = seed_summary[seed_summary.scenario == scenario].pivot(
            index="seed", columns="method"
        )
        for baseline in ("Local", "Global", "OracleFamily"):
            for metric in ("tracking", "worst_client"):
                improvement = 100 * (
                    1 - pivot[metric]["FederatedLearned"] / pivot[metric][baseline]
                )
                absolute = pivot[metric][baseline] - pivot[metric]["FederatedLearned"]
                low, high = bootstrap_interval(
                    improvement, cfg["bootstrap_resamples"], 4000 + len(comparisons)
                )
                absolute_low, absolute_high = bootstrap_interval(
                    absolute, cfg["bootstrap_resamples"], 5000 + len(comparisons)
                )
                comparisons.append(
                    {
                        "scenario": scenario,
                        "method": "FederatedLearned",
                        "baseline": baseline,
                        "metric": metric,
                        "improvement_pct": float(improvement.mean()),
                        "ci_low": low,
                        "ci_high": high,
                        "absolute_improvement_deg_s": float(absolute.mean()),
                        "absolute_ci_low": absolute_low,
                        "absolute_ci_high": absolute_high,
                        "wins": int((improvement > 0).sum()),
                    }
                )
    comparisons = pd.DataFrame(comparisons)
    global_rows = comparisons[comparisons.baseline == "Global"].set_index(["scenario", "metric"])
    summary_index = summary.set_index(["scenario", "method"])
    selected_k = raw.groupby("seed").selected_k.first()
    closed_gap = {}
    for scenario in ("moderate", "hard"):
        closed_gap[scenario] = {}
        for metric in ("tracking", "worst_client"):
            exact = summary_index.loc[(scenario, "ExactLPV"), metric]
            global_value = summary_index.loc[(scenario, "Global"), metric]
            federated = summary_index.loc[(scenario, "FederatedLearned"), metric]
            closed_gap[scenario][metric] = float(
                100 * (global_value - federated) / (global_value - exact)
            )
    conclusions = {
        "selected_controller": json.loads(SELECTION.read_text())["selected"],
        "moderate_global_tracking_gate": bool(
            global_rows.loc[("moderate", "tracking"), "ci_low"] > 0
        ),
        "hard_global_tracking_gate": bool(global_rows.loc[("hard", "tracking"), "ci_low"] > 0),
        "hard_global_worst_client_gate": bool(
            global_rows.loc[("hard", "worst_client"), "ci_low"] > 0
        ),
        "all_federated_feasible": bool(
            summary[summary.method == "FederatedLearned"].feasible_rate.min() == 1
        ),
        "all_federated_small_signal_stable": bool(
            summary[summary.method == "FederatedLearned"].rho.max() < 1
        ),
        "multi_group_fleet_rate": float(np.mean(selected_k > 1)),
        "selected_group_counts": {
            str(int(k)): int(v) for k, v in selected_k.value_counts().items()
        },
        "global_to_exact_gap_closed_pct": closed_gap,
        "interpretation": "A control-relevance claim requires a positive confirmation interval against Global after controller selection using ExactLPV only.",
        "provenance": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [
                CONFIG,
                CONFIG_10A,
                SELECTION,
                ROOT / "code/experiments/experiment_10f_control_relevance.py",
            ]
        },
    }
    seed_summary.to_csv(OUT / "experiment_10f_seed_summary.csv", index=False)
    summary.to_csv(OUT / "experiment_10f_summary.csv", index=False)
    comparisons.to_csv(OUT / "experiment_10f_comparisons.csv", index=False)
    (OUT / "experiment_10f_conclusions.json").write_text(json.dumps(conclusions, indent=2) + "\n")
    make_figure(summary, comparisons)
    print(json.dumps(conclusions, indent=2))
    print(summary.to_string(index=False))
    print(comparisons.to_string(index=False))


def make_figure(summary, comparisons):
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.5), constrained_layout=True)
    x = np.arange(len(METHODS))
    width = 0.38
    for offset, scenario in zip((-width / 2, width / 2), ("moderate", "hard")):
        view = summary[summary.scenario == scenario].set_index("method").loc[list(METHODS)]
        axes[0].bar(x + offset, view.tracking, width, label=scenario)
        axes[1].bar(x + offset, view.worst_client, width)
    axes[0].set_ylabel("Mean yaw tracking RMSE [deg/s]")
    axes[1].set_ylabel("Mean worst-client RMSE [deg/s]")
    axes[0].legend(frameon=False)
    for axis in axes[:2]:
        axis.set_xticks(x, ["Local", "Global", "FedLearned", "Oracle", "Exact"], rotation=28)
        axis.grid(axis="y", alpha=0.25)
    view = comparisons[(comparisons.baseline == "Global") & (comparisons.metric == "tracking")]
    axes[2].bar(view.scenario, view.improvement_pct)
    axes[2].errorbar(
        np.arange(len(view)),
        view.improvement_pct,
        yerr=np.vstack((view.improvement_pct - view.ci_low, view.ci_high - view.improvement_pct)),
        fmt="none",
        color="k",
        capsize=3,
    )
    axes[2].axhline(0, color="k", linewidth=1)
    axes[2].set_ylabel("FedLearned improvement vs Global [%]")
    axes[2].grid(axis="y", alpha=0.25)
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("audit", "confirm", "all"), nargs="?", default="all")
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(CONFIG.read_text())
    if args.phase in ("audit", "all") and not args.summarize_only:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(run_audit_seed, cfg["development_seeds"]))
        select_controller()
    elif args.phase == "audit":
        select_controller()
    if args.phase in ("confirm", "all"):
        if not SELECTION.exists():
            raise RuntimeError("run and freeze the development controller audit first")
        if not args.summarize_only:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                list(pool.map(run_confirmation_seed, cfg["confirmation_seeds"]))
        summarize_confirmation()


if __name__ == "__main__":
    main()
