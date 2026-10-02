"""10B: oracle-family finite-data gate under complementary speed coverage."""

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
from scipy.linalg import solve_discrete_are

from federated_lpv import nonlinear_bicycle_rhs, sample_fleet
from experiment_10a_basis_coverage_audit import collect_targets, equilibrium, raw_basis


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "code/config/experiment_10b.json"
CONFIG_10A = ROOT / "code/config/experiment_10a.json"
OUT = ROOT / "results/tables"
FIGURE = ROOT / "results/figures/experiment_10b_oracle_complementary_gate.pdf"


def rk4_step(parameters, state, steering, speed, dt):
    def rhs(x):
        return nonlinear_bicycle_rhs(x, steering, speed, parameters)

    k1 = rhs(state)
    k2 = rhs(state + dt * k1 / 2)
    k3 = rhs(state + dt * k2 / 2)
    k4 = rhs(state + dt * k3)
    return state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6


def estimate_matrix(client, speed, samples, cfg, rng):
    operating = equilibrium(client.parameters, speed, cfg["curvature_per_m"])
    state_star, steering_star = operating[:2], operating[2]
    x = rng.normal(size=(samples, 2)) * np.asarray(cfg["state_excitation_std"])
    u = rng.normal(size=samples) * cfg["steering_excitation_std"]
    y = []
    for state_delta, input_delta in zip(x, u):
        next_state = rk4_step(client.parameters, state_star + state_delta,
                              steering_star + input_delta, speed, cfg["sample_time"])
        noise = rng.normal(size=2) * np.asarray(cfg["next_state_noise_std"])
        y.append(next_state - state_star + noise)
    regressor = np.column_stack((x, u))
    coefficient, _, rank, _ = np.linalg.lstsq(regressor, np.asarray(y), rcond=None)
    if rank != 3:
        raise RuntimeError("local perturbation regression is rank deficient")
    a = coefficient[:2].T
    b = coefficient[2:].T
    return np.r_[a.ravel(), b.ravel()]


def ridge_fit(speeds, targets, order, cfg10a, prior=None, strength=0.0):
    design = raw_basis(np.asarray(speeds, dtype=float), order, cfg10a)
    gram = design.T @ design
    right = design.T @ np.asarray(targets)
    if prior is not None and strength > 0:
        gram = gram + strength * np.eye(order)
        right = right + strength * prior
    elif strength > 0:
        gram = gram + strength * np.eye(order)
    return np.linalg.solve(gram + 1e-12 * np.eye(order), right)


def predict(coefficients, speeds, cfg10a):
    return raw_basis(np.asarray(speeds, dtype=float), len(coefficients), cfg10a) @ coefficients


def lift_prior(local_l3, cfg10a):
    grid = np.asarray(cfg10a["speed_grid"], dtype=float)
    return ridge_fit(grid, predict(local_l3, grid, cfg10a), 7, cfg10a)


def select_strength(speeds, targets, cfg, cfg10a, prior_builder):
    if len(speeds) < 3:
        return cfg["ridge_candidates"][len(cfg["ridge_candidates"]) // 2]
    scores = []
    for strength in cfg["ridge_candidates"]:
        errors = []
        for heldout in range(len(speeds)):
            keep = np.arange(len(speeds)) != heldout
            prior = prior_builder(keep)
            coefficients = ridge_fit(np.asarray(speeds)[keep], np.asarray(targets)[keep], 7,
                                     cfg10a, prior, strength)
            estimate = predict(coefficients, [speeds[heldout]], cfg10a)[0]
            errors.append(np.linalg.norm(estimate - targets[heldout]) / np.linalg.norm(targets[heldout]))
        scores.append(np.mean(errors))
    return cfg["ridge_candidates"][int(np.argmin(scores))]


def lqr_gain(flat_matrix, cfg):
    a = flat_matrix[:4].reshape(2, 2)
    b = flat_matrix[4:].reshape(2, 1)
    q = np.diag(cfg["lqr_q"])
    r = np.array([[cfg["lqr_r"]]])
    try:
        p = solve_discrete_are(a, b, q, r)
        gain = np.linalg.solve(r + b.T @ p @ b, b.T @ p @ a).ravel()
        return gain, bool(np.isfinite(gain).all())
    except Exception:
        return np.zeros(2), False


def recovery_metric(client, speed, flat_matrix, cfg):
    gain, designed = lqr_gain(flat_matrix, cfg)
    if not designed:
        return np.inf, False, np.inf
    operating = equilibrium(client.parameters, speed, cfg["curvature_per_m"])
    state_star, steering_star = operating[:2], operating[2]
    state = state_star + np.asarray(cfg["initial_perturbation"])
    squared, steering_energy = [], []
    steps = round(cfg["recovery_duration_s"] / cfg["sample_time"])
    for _ in range(steps):
        delta = state - state_star
        command = steering_star - float(gain @ delta)
        state = rk4_step(client.parameters, state, command, speed, cfg["sample_time"])
        squared.append((delta[0] / cfg["initial_perturbation"][0])**2 +
                       (delta[1] / cfg["initial_perturbation"][1])**2)
        steering_energy.append((command - steering_star)**2)
        if not np.isfinite(state).all() or np.max(np.abs(state - state_star)) > 1.0:
            return np.inf, False, np.inf
    final_delta = state - state_star
    feasible = np.linalg.norm(final_delta / np.asarray(cfg["initial_perturbation"])) < 0.1
    return float(np.sqrt(np.mean(squared))), bool(feasible), float(np.sqrt(np.mean(steering_energy)))


def run_seed(seed):
    cfg = json.loads(CONFIG.read_text())
    cfg10a = json.loads(CONFIG_10A.read_text())
    clients = sample_fleet(seed)
    speeds = np.asarray(cfg["speed_grid"], dtype=float)
    rng = np.random.default_rng(seed + 100000)
    order = rng.permutation(len(clients))
    categories = {client.client_id: int(order[j] % len(cfg["coverage_blocks"]))
                  for j, client in enumerate(clients)}
    observed = {}
    exact = {}
    for client in clients:
        block = cfg["coverage_blocks"][categories[client.client_id]]
        observed[client.client_id] = np.asarray([
            estimate_matrix(client, speed, cfg["samples_per_local_speed"], cfg, rng)
            for speed in block
        ])
        _, exact[client.client_id] = collect_targets(client, cfg["curvature_per_m"], cfg10a)

    all_speeds, all_targets = [], []
    for client in clients:
        block = cfg["coverage_blocks"][categories[client.client_id]]
        all_speeds.extend(block); all_targets.extend(observed[client.client_id])
    global_model = ridge_fit(all_speeds, all_targets, 7, cfg10a, strength=1e-8)

    family_models = {}
    for family in sorted({client.family for client in clients}):
        fs, ft = [], []
        for client in clients:
            if client.family != family:
                continue
            block = cfg["coverage_blocks"][categories[client.client_id]]
            fs.extend(block); ft.extend(observed[client.client_id])
        family_models[family] = ridge_fit(fs, ft, 7, cfg10a, strength=1e-8)

    models, fit_rows = {}, []
    family_size = {family: sum(client.family == family for client in clients)
                   for family in family_models}
    for client in clients:
        block = np.asarray(cfg["coverage_blocks"][categories[client.client_id]], dtype=float)
        targets = observed[client.client_id]
        local3 = ridge_fit(block, targets, 3, cfg10a, strength=1e-8)
        local_prior = lift_prior(local3, cfg10a)
        local_strength = select_strength(
            block, targets, cfg, cfg10a,
            lambda keep: lift_prior(ridge_fit(block[keep], targets[keep], 3, cfg10a,
                                               strength=1e-6), cfg10a),
        )
        personalized_strength = select_strength(
            block, targets, cfg, cfg10a, lambda keep: family_models[client.family]
        )
        total_family_samples = family_size[client.family] * np.mean([
            len(cfg["coverage_blocks"][categories[c.client_id]])
            for c in clients if c.family == client.family
        ]) * cfg["samples_per_local_speed"]
        per_speed = max(3, round(total_family_samples / len(speeds)))
        full_targets = np.asarray([
            estimate_matrix(client, speed, per_speed, cfg, rng) for speed in speeds
        ])
        models[client.client_id] = {
            "Local": local_prior,
            "LocalRegularized": ridge_fit(block, targets, 7, cfg10a, local_prior, local_strength),
            "Global": global_model,
            "FamilyPool": family_models[client.family],
            "FamilyPersonalized": ridge_fit(block, targets, 7, cfg10a,
                                             family_models[client.family], personalized_strength),
            "LocalFull": ridge_fit(speeds, full_targets, 7, cfg10a, strength=1e-8),
            "ExactLPV": ridge_fit(speeds, exact[client.client_id], 7, cfg10a, strength=1e-10),
        }
        fit_rows.append(dict(seed=seed, client=client.client_id, family=client.family,
                             block=categories[client.client_id], local_strength=local_strength,
                             personalized_strength=personalized_strength,
                             local_transitions=len(block) * cfg["samples_per_local_speed"],
                             local_full_transitions=per_speed * len(speeds)))

    rows = []
    for client in clients:
        local_block = set(cfg["coverage_blocks"][categories[client.client_id]])
        for method, coefficients in models[client.client_id].items():
            predictions = predict(coefficients, speeds, cfg10a)
            for index, speed in enumerate(speeds):
                region = "seen" if speed in local_block else "unseen"
                error = np.linalg.norm(predictions[index] - exact[client.client_id][index]) / np.linalg.norm(exact[client.client_id][index])
                control_matrix = predictions[index]
                if method in ("Local", "LocalRegularized") and region == "unseen":
                    nearest = min(local_block, key=lambda value: abs(value - speed))
                    control_matrix = predict(coefficients, [nearest], cfg10a)[0]
                control, feasible, steering = recovery_metric(client, speed, control_matrix, cfg)
                rows.append(dict(seed=seed, client=client.client_id, family=client.family,
                                 block=categories[client.client_id], method=method, speed=speed,
                                 region=region, prediction_error=error, recovery=control,
                                 steering_rms=steering, feasible=feasible))
    for suffix, frame in (("clients", pd.DataFrame(rows)), ("fits", pd.DataFrame(fit_rows))):
        (OUT / f"experiment_10b_seed{seed}_{suffix}.csv.gz").write_bytes(
            gzip.compress(frame.to_csv(index=False).encode(), mtime=0))
    print(seed, flush=True)


def bootstrap_interval(values, resamples, seed=10):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(resamples, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(draws, .025)), float(np.quantile(draws, .975))


def summarize():
    cfg = json.loads(CONFIG.read_text())
    raw = pd.concat([pd.read_csv(OUT / f"experiment_10b_seed{seed}_clients.csv.gz")
                     for seed in cfg["development_seeds"]])
    fits = pd.concat([pd.read_csv(OUT / f"experiment_10b_seed{seed}_fits.csv.gz")
                      for seed in cfg["development_seeds"]])
    units = raw.groupby(["seed", "region", "method"]).agg(
        prediction_error=("prediction_error", "mean"), recovery=("recovery", "mean"),
        steering_rms=("steering_rms", "mean"), feasible=("feasible", "min")
    ).reset_index()
    summary = units.groupby(["region", "method"]).agg(
        prediction_error=("prediction_error", "mean"), prediction_std=("prediction_error", "std"),
        recovery=("recovery", "mean"), recovery_std=("recovery", "std"),
        steering_rms=("steering_rms", "mean"), feasible_rate=("feasible", "mean")
    ).reset_index()
    unseen = units[units.region == "unseen"].pivot(index="seed", columns="method")
    comparisons = []
    pairs = [(method, "Local") for method in summary.method.unique() if method != "Local"]
    pairs.extend([("FamilyPersonalized", "FamilyPool"), ("FamilyPool", "Global"),
                  ("LocalFull", "FamilyPool")])
    for method, baseline in pairs:
        for metric in ("prediction_error", "recovery"):
            difference = 100 * (1 - unseen[metric][method] / unseen[metric][baseline])
            low, high = bootstrap_interval(difference, cfg["bootstrap_resamples"], 100 + len(comparisons))
            comparisons.append(dict(method=method, baseline=baseline, metric=metric,
                                    improvement_pct=float(difference.mean()),
                                    ci_low=low, ci_high=high, wins=int((difference > 0).sum())))
    comparisons = pd.DataFrame(comparisons)
    primary = comparisons[(comparisons.method == "FamilyPersonalized") &
                          (comparisons.baseline == "Local")].set_index("metric")
    conclusions = {
        "primary_prediction_pass": bool(primary.loc["prediction_error", "ci_low"] > 0),
        "primary_control_pass": bool(primary.loc["recovery", "ci_low"] > 0),
        "all_primary_feasible": bool(summary[(summary.region == "unseen") &
                                              (summary.method == "FamilyPersonalized")].feasible_rate.iloc[0] == 1),
        "gate_pass": bool((primary.ci_low > 0).all() and
                          summary[(summary.region == "unseen") &
                                  (summary.method == "FamilyPersonalized")].feasible_rate.iloc[0] == 1),
        "interpretation": "Oracle family labels and centralized aggregation isolate the value of complementary compatible coverage; no federated implementation is claimed.",
        "provenance": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in [CONFIG, CONFIG_10A,
                                    ROOT / "code/experiments/experiment_10b_oracle_complementary_gate.py"]},
    }
    units.to_csv(OUT / "experiment_10b_seed_summary.csv", index=False)
    summary.to_csv(OUT / "experiment_10b_summary.csv", index=False)
    comparisons.to_csv(OUT / "experiment_10b_comparisons.csv", index=False)
    pd.DataFrame(fits).to_csv(OUT / "experiment_10b_fit_summary.csv", index=False)
    (OUT / "experiment_10b_conclusions.json").write_text(json.dumps(conclusions, indent=2) + "\n")
    make_figure(summary, comparisons)
    print(json.dumps(conclusions, indent=2)); print(summary.to_string(index=False)); print(comparisons.to_string(index=False))


def make_figure(summary, comparisons):
    order = ["Local", "LocalRegularized", "Global", "FamilyPool",
             "FamilyPersonalized", "LocalFull", "ExactLPV"]
    view = summary[summary.region == "unseen"].set_index("method").loc[order]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.5), constrained_layout=True)
    x = np.arange(len(order))
    axes[0].bar(x, 100 * view.prediction_error)
    axes[0].set(ylabel="Unseen-speed matrix error [%]", title="Finite-data prediction")
    recovery_excess = 100 * (view.recovery / view.loc["ExactLPV", "recovery"] - 1)
    axes[1].bar(x, recovery_excess)
    axes[1].set(ylabel="Recovery excess over Exact [%]", title="Nonlinear closed-loop recovery")
    for axis in axes:
        axis.set_xticks(x, ["Local", "Local reg.", "Global", "Family", "Family PFL", "LocalFull", "Exact"], rotation=28)
        axis.grid(axis="y", alpha=.25)
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
