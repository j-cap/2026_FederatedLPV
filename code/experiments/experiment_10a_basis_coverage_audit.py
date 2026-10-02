"""10A: basis-order and partial-speed-coverage necessity audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import expm
from scipy.optimize import least_squares

from federated_lpv import nonlinear_bicycle_rhs, sample_fleet


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "code/config/experiment_10a.json"
OUT = ROOT / "results/tables"
FIGURE = ROOT / "results/figures/experiment_10a_basis_coverage_audit.pdf"


def equilibrium(parameters, speed: float, curvature: float) -> np.ndarray:
    """Return [beta, yaw rate, steering] at a steady constant-curvature turn."""
    yaw_rate = speed * curvature

    def residual(z):
        beta, steering = z
        return nonlinear_bicycle_rhs(
            np.array([beta, yaw_rate]), steering, speed, parameters
        )

    linear_guess = np.array([0.0, (parameters.front_length + parameters.rear_length) * curvature])
    solution = least_squares(residual, linear_guess, xtol=1e-12, ftol=1e-12, gtol=1e-12)
    if not solution.success or np.linalg.norm(solution.fun) > 1e-8:
        raise RuntimeError("steady-cornering equilibrium did not converge")
    return np.array([solution.x[0], yaw_rate, solution.x[1]])


def discrete_jacobian(parameters, speed: float, curvature: float, sample_time: float):
    operating = equilibrium(parameters, speed, curvature)
    state, steering = operating[:2], operating[2]
    eps_state = np.array([1e-6, 1e-6])
    a = np.column_stack(
        [
            (
                nonlinear_bicycle_rhs(state + np.eye(2)[j] * eps_state[j], steering, speed, parameters)
                - nonlinear_bicycle_rhs(state - np.eye(2)[j] * eps_state[j], steering, speed, parameters)
            )
            / (2 * eps_state[j])
            for j in range(2)
        ]
    )
    eps_input = 1e-6
    b = (
        nonlinear_bicycle_rhs(state, steering + eps_input, speed, parameters)
        - nonlinear_bicycle_rhs(state, steering - eps_input, speed, parameters)
    )[:, None] / (2 * eps_input)
    augmented = np.block([[a, b], [np.zeros((1, 3))]])
    discrete = expm(augmented * sample_time)
    return discrete[:2, :2], discrete[:2, 2:], operating


def raw_basis(speeds: np.ndarray, order: int, cfg: dict) -> np.ndarray:
    """Evaluate a fixed-domain orthogonalized Laurent-polynomial basis."""
    speeds = np.asarray(speeds, dtype=float)
    domain = cfg["orthogonalization_grid"]

    def laurent(values):
        normalized = (values - 20.0) / 10.0
        columns = [np.ones_like(values), 1 / values, 1 / values**2]
        columns.extend(normalized**degree for degree in range(1, order - 2))
        return np.column_stack(columns)

    reference = np.linspace(domain["minimum_mps"], domain["maximum_mps"], domain["points"])
    _, transform = np.linalg.qr(laurent(reference))
    return laurent(speeds) @ np.linalg.inv(transform)


def fit_predict(train_speeds, train_targets, test_speeds, order, cfg):
    train = raw_basis(train_speeds, order, cfg)
    test = raw_basis(test_speeds, order, cfg)
    mean = train[:, 1:].mean(axis=0)
    scale = train[:, 1:].std(axis=0)
    if np.any(scale < 1e-12):
        return None, np.inf, 0
    train[:, 1:] = (train[:, 1:] - mean) / scale
    test[:, 1:] = (test[:, 1:] - mean) / scale
    coefficients, _, rank, _ = np.linalg.lstsq(train, train_targets, rcond=None)
    if rank < order:
        return None, np.inf, rank
    return test @ coefficients, float(np.linalg.cond(train)), rank


def relative_error(truth, prediction):
    return float(np.linalg.norm(prediction - truth) / np.linalg.norm(truth))


def collect_targets(client, curvature, cfg):
    rows, targets = [], []
    for speed in cfg["speed_grid"]:
        a, b, operating = discrete_jacobian(
            client.parameters, speed, curvature, cfg["sample_time"]
        )
        rows.append(
            dict(
                speed=speed,
                beta_deg=float(np.rad2deg(operating[0])),
                yaw_rate_deg_s=float(np.rad2deg(operating[1])),
                steering_deg=float(np.rad2deg(operating[2])),
                lateral_acceleration=speed**2 * curvature,
            )
        )
        targets.append(np.r_[a.ravel(), b.ravel()])
    return pd.DataFrame(rows), np.asarray(targets)


def evaluate(cfg):
    validation_rows, operating_rows, coverage_rows = [], [], []
    speeds = np.asarray(cfg["speed_grid"], dtype=float)
    for seed in cfg["development_seeds"]:
        clients = sample_fleet(seed)
        for client in clients:
            for curvature in cfg["curvatures_per_m"]:
                operating, targets = collect_targets(client, curvature, cfg)
                operating = operating.assign(seed=seed, client=client.client_id, family=client.family, curvature=curvature)
                operating_rows.extend(operating.to_dict("records"))
                for order in cfg["candidate_orders"]:
                    for fold, withheld in enumerate(cfg["blocked_validation"]):
                        test_mask = np.isin(speeds, withheld)
                        prediction, condition, rank = fit_predict(
                            speeds[~test_mask], targets[~test_mask], speeds[test_mask], order, cfg
                        )
                        error = np.inf if prediction is None else relative_error(targets[test_mask], prediction)
                        validation_rows.append(
                            dict(seed=seed, client=client.client_id, family=client.family,
                                 curvature=curvature, order=order, fold=fold,
                                 withheld_min=min(withheld), withheld_max=max(withheld),
                                 relative_error=error, condition=condition, rank=rank)
                        )
                # Rank/conditioning of one client's restricted domain versus the union.
                for block_index, block in enumerate(cfg["partial_coverage_blocks"]):
                    block_speeds = np.asarray(block, dtype=float)
                    for order in cfg["candidate_orders"]:
                        design = raw_basis(block_speeds, order, cfg)
                        coverage_rows.append(
                            dict(seed=seed, client=client.client_id, family=client.family,
                                 curvature=curvature, scope=f"local_block_{block_index}", order=order,
                                 samples=len(block_speeds), rank=int(np.linalg.matrix_rank(design)),
                                 condition=float(np.linalg.cond(design)) if len(block_speeds) >= order else np.inf)
                        )
                for order in cfg["candidate_orders"]:
                    design = raw_basis(speeds, order, cfg)
                    coverage_rows.append(
                        dict(seed=seed, client=client.client_id, family=client.family,
                             curvature=curvature, scope="aggregate_envelope", order=order,
                             samples=len(speeds), rank=int(np.linalg.matrix_rank(design)),
                             condition=float(np.linalg.cond(design)))
                    )
    return pd.DataFrame(validation_rows), pd.DataFrame(operating_rows), pd.DataFrame(coverage_rows)


def summarize(validation, operating, coverage, cfg):
    fleet = validation.groupby(["seed", "curvature", "order"]).agg(
        error=("relative_error", "mean"), worst_error=("relative_error", "max"),
        condition=("condition", "max")
    ).reset_index()
    summary = fleet.groupby(["curvature", "order"]).agg(
        error_mean=("error", "mean"), error_se=("error", "sem"),
        error_worst_client_fold=("worst_error", "max"), condition_worst=("condition", "max")
    ).reset_index()

    selections = []
    for curvature, group in summary.groupby("curvature"):
        eligible = group[group.condition_worst <= cfg["maximum_aggregate_condition"]]
        best = eligible.loc[eligible.error_mean.idxmin()]
        threshold = max(best.error_mean + best.error_se, cfg["practical_relative_error_floor"])
        selected = eligible[eligible.error_mean <= threshold].sort_values("order").iloc[0]
        selections.append(dict(curvature=curvature, best_order=int(best.order),
                               selected_order=int(selected.order), best_error=float(best.error_mean),
                               threshold=float(threshold), selected_error=float(selected.error_mean)))
    selections = pd.DataFrame(selections)
    selected_nonlinear = int(selections.loc[selections.curvature == max(cfg["curvatures_per_m"]), "selected_order"].iloc[0])

    coverage_summary = coverage.groupby(["curvature", "scope", "order"]).agg(
        rank_min=("rank", "min"), rank_max=("rank", "max"),
        condition_median=("condition", "median"), condition_max=("condition", "max")
    ).reset_index()
    target = coverage_summary[(coverage_summary.curvature == max(cfg["curvatures_per_m"])) &
                              (coverage_summary.order == selected_nonlinear)]
    local = target[target.scope.str.startswith("local")]
    aggregate = target[target.scope == "aggregate_envelope"].iloc[0]
    max_ay = float(operating.lateral_acceleration.max())
    conclusions = {
        "selected_order": selected_nonlinear,
        "selection_curvature_per_m": max(cfg["curvatures_per_m"]),
        "straight_line_selected_order": int(selections.loc[selections.curvature == 0, "selected_order"].iloc[0]),
        "all_local_blocks_rank_deficient": bool((local.rank_max < selected_nonlinear).all()),
        "aggregate_full_rank": bool(aggregate.rank_min == selected_nonlinear),
        "aggregate_raw_condition": float(aggregate.condition_max),
        "maximum_lateral_acceleration_mps2": max_ay,
        "interpretation": "10A establishes representation need and aggregate identifiability only; it does not establish a federated performance benefit.",
        "provenance": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in [CONFIG, ROOT / "code/experiments/experiment_10a_basis_coverage_audit.py"]},
    }
    return fleet, summary, selections, coverage_summary, conclusions


def make_figure(summary, selections, coverage_summary, cfg):
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4), constrained_layout=True)
    for curvature in cfg["curvatures_per_m"]:
        view = summary[summary.curvature == curvature]
        axes[0].errorbar(view.order, 100 * view.error_mean, yerr=100 * view.error_se,
                         marker="o", label=fr"$\kappa={curvature:.4f}$ m$^{{-1}}$")
    axes[0].set(xlabel="Basis order $L$", ylabel="Blocked-speed relative error [%]",
                yscale="log", title="Whole-region validation")
    axes[0].grid(alpha=.25); axes[0].legend(frameon=False, fontsize=8)
    selected = int(selections.loc[selections.curvature == max(cfg["curvatures_per_m"]), "selected_order"].iloc[0])
    view = coverage_summary[(coverage_summary.curvature == max(cfg["curvatures_per_m"])) &
                            (coverage_summary.order == selected)]
    labels = [x.replace("local_block_", "Block ").replace("aggregate_envelope", "Fleet union") for x in view.scope]
    axes[1].bar(np.arange(len(view)), view.rank_min)
    axes[1].axhline(selected, color="k", linestyle="--", linewidth=1, label=f"required rank $L={selected}$")
    axes[1].set(xticks=np.arange(len(view)), xticklabels=labels, ylabel="Design rank",
                title="Partial versus aggregate coverage", ylim=(0, selected + 1))
    axes[1].tick_params(axis="x", rotation=25); axes[1].grid(axis="y", alpha=.25); axes[1].legend(frameon=False)
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, bbox_inches="tight"); plt.close(fig)


def main():
    cfg = json.loads(CONFIG.read_text())
    validation, operating, coverage = evaluate(cfg)
    fleet, summary, selections, coverage_summary, conclusions = summarize(validation, operating, coverage, cfg)
    OUT.mkdir(parents=True, exist_ok=True)
    validation.to_csv(OUT / "experiment_10a_validation.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    operating.to_csv(OUT / "experiment_10a_operating_points.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    coverage.to_csv(OUT / "experiment_10a_coverage.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    fleet.to_csv(OUT / "experiment_10a_seed_summary.csv", index=False)
    summary.to_csv(OUT / "experiment_10a_summary.csv", index=False)
    selections.to_csv(OUT / "experiment_10a_selection.csv", index=False)
    coverage_summary.to_csv(OUT / "experiment_10a_coverage_summary.csv", index=False)
    (OUT / "experiment_10a_conclusions.json").write_text(json.dumps(conclusions, indent=2) + "\n")
    make_figure(summary, selections, coverage_summary, cfg)
    print(json.dumps(conclusions, indent=2)); print(summary.to_string(index=False)); print(selections.to_string(index=False))


if __name__ == "__main__":
    main()
