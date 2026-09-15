"""9K: frozen confirmation of the 9J distributed personalized LPV protocol."""
from concurrent.futures import ProcessPoolExecutor
import argparse, hashlib, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import experiment_9j_distributed_backbone as ninej

ROOT = ninej.ROOT
OUT = ninej.OUT
CONFIG = ROOT / "code/config/experiment_9k.json"
NINEJ_CONFIG = ROOT / "code/config/experiment_9j.json"


def assert_frozen(config):
    """Fail before execution if any inherited 9J setting has changed."""
    reference = json.loads(NINEJ_CONFIG.read_text())
    excluded = {"development_seeds", "reserved_confirmation_seeds", "notes"}
    inherited = {key: value for key, value in reference.items() if key not in excluded}
    observed = {key: config[key] for key in inherited}
    if observed != inherited:
        changed = [key for key in inherited if observed.get(key) != inherited[key]]
        raise RuntimeError(f"9K is not frozen against 9J; changed settings: {changed}")
    if config["confirmation_seeds"] != reference["reserved_confirmation_seeds"]:
        raise RuntimeError("9K seeds do not match the seeds reserved by 9J")


def run_seed(seed):
    ninej.CONFIG = CONFIG
    ninej.run_seed(seed)


def summarize():
    cfg = json.loads(CONFIG.read_text())
    assert_frozen(cfg)

    def load(suffix):
        return pd.concat([
            pd.read_csv(OUT / f"experiment_9j_seed{seed}_{suffix}.csv.gz")
            for seed in cfg["confirmation_seeds"]
        ])

    raw = load("clients")
    diagnostics = load("diagnostics")
    fits = load("fits")
    assignments = load("assignments")
    protocol = load("protocol")
    equivalence = load("equivalence")

    seed_summary = raw.groupby(["seed", "budget", "method"]).agg(
        tracking=("tracking", "mean"),
        q95=("tracking", lambda values: np.quantile(values, 0.95)),
        feasible=("feasible", "min"),
    ).reset_index()
    summary = seed_summary.groupby(["budget", "method"]).agg(
        tracking=("tracking", "mean"),
        tracking_std=("tracking", "std"),
        q95=("q95", "mean"),
        feasible_rate=("feasible", "mean"),
    ).reset_index().merge(
        diagnostics.groupby(["budget", "method"]).agg(
            parameter_error=("parameter_error", "mean"),
            gain_error=("gain_error", "mean"),
        ).reset_index(),
        on=["budget", "method"],
    )
    central = summary[summary.method == "Central"][["budget", "tracking"]].rename(
        columns={"tracking": "central"}
    )
    local = summary[summary.method == "Local"][["budget", "tracking"]].rename(
        columns={"tracking": "local"}
    )
    comparisons = summary.merge(central, on="budget").merge(local, on="budget")
    comparisons["vs_central_pct"] = 100 * (comparisons.tracking / comparisons.central - 1)
    comparisons["vs_local_pct"] = 100 * (comparisons.tracking / comparisons.local - 1)
    communication = protocol.groupby("method").agg(
        participation=("participation", "first"),
        selected_k=("selected_k", "mean"),
        ari=("train_ari", "mean"),
        coverage=("coverage", "mean"),
        messages=("client_messages", "mean"),
        upload_bytes=("upload_bytes", "mean"),
        download_bytes=("download_bytes", "mean"),
    ).reset_index()

    gates = cfg["predeclared_gates"]
    degradation = comparisons.set_index(["budget", "method"])["vs_central_pct"]
    tracking = summary.set_index(["budget", "method"])["tracking"]
    gate_results = {
        "fed100_equivalence": bool(degradation.xs("Fed100", level="method").max()
                                   < gates["fed100_max_degradation_vs_central_pct"]),
        "fed50_partial_participation": bool(degradation.xs("Fed50", level="method").max()
                                             < gates["fed50_max_degradation_vs_central_pct"]),
        "fed20_partial_participation": bool(degradation.xs("Fed20", level="method").max()
                                             < gates["fed20_max_degradation_vs_central_pct"]),
        "fed20_cold_start": bool(
            tracking.loc[(gates["fed20_beats_local_at_seconds"], "Fed20")]
            < tracking.loc[(gates["fed20_beats_local_at_seconds"], "Local")]
        ),
        "closed_loop_feasibility": bool(raw.feasible.all()),
        "deterministic_aggregation": bool(
            equivalence.k_equal.all()
            and equivalence.mean_difference.max() < 1e-12
            and equivalence.covariance_difference.max() < 1e-12
        ),
    }
    conclusions = {
        "frozen_settings_verified": True,
        "confirmation_seeds": cfg["confirmation_seeds"],
        "provenance": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [CONFIG, NINEJ_CONFIG,
                         ROOT / "code/src/federated_lpv/federated_manifold.py",
                         ROOT / "code/experiments/experiment_9j_distributed_backbone.py",
                         ROOT / "code/experiments/experiment_9k_blind_confirmation.py"]
        },
        "fits": len(fits),
        "all_fits_converged": bool(fits.success.all() and assignments.success.all()),
        "evaluations": len(raw),
        "gate_results": gate_results,
        "all_predeclared_gates_pass": bool(all(gate_results[key] for key in [
            "fed100_equivalence", "fed50_partial_participation",
            "fed20_partial_participation", "fed20_cold_start",
            "closed_loop_feasibility",
        ])),
    }
    summary.to_csv(OUT / "experiment_9k_summary.csv", index=False)
    comparisons.to_csv(OUT / "experiment_9k_comparisons.csv", index=False)
    communication.to_csv(OUT / "experiment_9k_communication.csv", index=False)
    seed_summary.to_csv(OUT / "experiment_9k_seed_summary.csv", index=False)
    (OUT / "experiment_9k_conclusions.json").write_text(json.dumps(conclusions, indent=2) + "\n")

    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    order = ["Local", "Central", "Fed100", "Fed50", "Fed20"]
    x = np.arange(2)
    width = 0.15
    for index, method in enumerate(order):
        group = summary[summary.method == method].set_index("budget").loc[cfg["calibration_budgets"]]
        ax.bar(x + (index - 2) * width, group.tracking, width, label=method)
    ax.set(xticks=x, xticklabels=["0.75", "1.25"], xlabel="Calibration time [s]",
           ylabel="Tracking RMSE [rad/s]")
    ax.legend(ncol=3, fontsize=8)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(ROOT / "results/figures/experiment_9k_blind_confirmation.pdf")
    plt.close(fig)
    print(json.dumps(conclusions, indent=2))
    print(summary.to_string(index=False))
    print(comparisons[["budget", "method", "vs_central_pct", "vs_local_pct"]].to_string(index=False))
    print(communication.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text())
    assert_frozen(config)
    if not args.summarize_only:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(run_seed, config["confirmation_seeds"]))
    summarize()
