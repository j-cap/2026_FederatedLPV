"""Reanalyse frozen label-free records. No model fitting or plant simulation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/tables"
METHODS = ["Local", "Global", "FederatedLearned"]


def load_records():
    cfg = json.loads((ROOT / "code/config/experiment_10e.json").read_text())
    sources = [OUT / f"experiment_10e_seed{s}_label_free_clients.csv.gz" for s in cfg["seeds"]]
    raw = pd.concat([pd.read_csv(p) for p in sources], ignore_index=True)
    keys = ["seed", "client", "method", "speed"]
    if raw.duplicated(keys).any() or len(raw) != len(cfg["seeds"]) * 30 * 6 * 11:
        raise ValueError("Frozen record set is incomplete or duplicated")
    if not np.isfinite(raw.prediction_error).all():
        raise ValueError("Nonfinite matrix error")
    frozen = pd.read_csv(OUT / "experiment_10e_seed_summary.csv")
    check = raw.groupby(["seed", "method"]).prediction_error.mean()
    expected = frozen[frozen.protocol == "label_free"].set_index(["seed", "method"]).prediction_error
    np.testing.assert_allclose(check.sort_index(), expected.sort_index(), rtol=1e-12, atol=1e-14)
    raw["region"] = ["seen" if v in cfg["coverage_blocks"][int(b)] else "unseen"
                     for v, b in zip(raw.speed, raw.block)]
    return cfg, raw, sources


def main():
    cfg, raw, sources = load_records()
    # Match the frozen endpoint: average all client-speed records within each fleet.
    units = raw.groupby(["seed", "region", "method"]).prediction_error.mean().reset_index()
    full = raw.groupby(["seed", "method"]).prediction_error.mean().reset_index().assign(region="full")
    units = pd.concat([units, full], ignore_index=True)
    summary = units.groupby(["region", "method"]).prediction_error.agg(["mean", "std"]).reset_index()
    rng = np.random.default_rng(20261009)
    comparisons = []
    for region in ["full", "seen", "unseen"]:
        view = units[units.region == region].pivot(index="seed", columns="method", values="prediction_error")
        for baseline in ["Local", "Global"]:
            gains = 100 * (1 - view.FederatedLearned / view[baseline])
            draws = rng.choice(gains, size=(cfg["bootstrap_resamples"], len(gains)), replace=True).mean(axis=1)
            low, high = np.quantile(draws, [.025, .975])
            comparisons.append(dict(region=region, baseline=baseline, improvement_pct=gains.mean(),
                                    ci_low=low, ci_high=high, wins=int((gains > 0).sum())))
    clients = raw.groupby(["seed", "client", "region", "method"]).prediction_error.mean().unstack("method")
    tails = clients.reset_index()[["seed", "client", "region"]].copy()
    tails["grouped_vs_global_improvement_pct"] = (100 * (1 - clients.FederatedLearned / clients.Global)).to_numpy()
    tails["global_error"] = clients.Global.to_numpy()
    tails["grouped_error"] = clients.FederatedLearned.to_numpy()
    units.to_csv(OUT / "paper_identification_regions_seed.csv", index=False)
    summary.to_csv(OUT / "paper_identification_regions_summary.csv", index=False)
    pd.DataFrame(comparisons).to_csv(OUT / "paper_identification_regions_comparisons.csv", index=False)
    tails.to_csv(OUT / "paper_identification_client_tails.csv", index=False)

    macros = []
    for method, short in [("Local", "Local"), ("Global", "Global"), ("FederatedLearned", "Fed")]:
        for region, suffix in [("seen", "Seen"), ("unseen", "Unseen")]:
            row = summary[(summary.method == method) & (summary.region == region)].iloc[0]
            macros.append(f"\\newcommand{{\\Primary{short}{suffix}Error}}{{{100 * row['mean']:.3f}}}")
    unseen = tails[tails.region == "unseen"]
    worse = int(((unseen.grouped_error - unseen.global_error) > 1e-12).sum())
    macros.append(f"\\newcommand{{\\PrimaryUnseenWorseClients}}{{{worse}}}")
    (ROOT / "paper/region_numbers.tex").write_text("% Generated from frozen client records.\n" + "\n".join(macros) + "\n")
    manifest = dict(operation="frozen_record_reanalysis_no_fitting_or_simulation", fleets=10,
                    records=len(raw), source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    (OUT / "paper_identification_records_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.labelsize": 8})
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.0), sharex=True, sharey=True, layout="constrained")
    colors = ["#777777", "#cf741c", "#236b9a"]
    for block, ax in enumerate(axes.flat):
        view = raw[(raw.block == block) & raw.method.isin(METHODS)]
        block_speeds = cfg["coverage_blocks"][block]
        ax.axvspan(min(block_speeds) - .4, max(block_speeds) + .4, color="#dbe9d4", alpha=.7)
        for method, color, label in zip(METHODS, colors, ["Restricted Local", "Global", "Grouped"]):
            fleet = view[view.method == method].groupby(["seed", "speed"]).prediction_error.mean().unstack("speed")
            values = 100 * fleet.to_numpy()
            draws = np.random.default_rng(20261009 + block).integers(0, len(values), size=(20000, len(values)))
            means = values[draws].mean(axis=1)
            lo, hi = np.quantile(means, [.025, .975], axis=0)
            ax.plot(fleet.columns, values.mean(axis=0), color=color, label=label, linewidth=1.4)
            ax.fill_between(fleet.columns, lo, hi, color=color, alpha=.16, linewidth=0)
        ax.set_title(f"Observed block: {min(block_speeds)}--{max(block_speeds)} m/s")
        ax.set_yscale("log")
        ax.set_xticks([10, 15, 20, 25, 30])
        ax.grid(alpha=.2, which="both")
    for ax in axes[1]:
        ax.set_xlabel("Speed [m/s]")
    for ax in axes[:, 0]:
        ax.set_ylabel("Relative matrix error [%]")
    axes[0, 0].legend(frameon=False, fontsize=7, loc="upper left")
    path = ROOT / "results/figures/paper_identification_coverage.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(json.dumps({"status": "verified", "records": len(raw), "fleets": 10}))
    print(summary[summary.method.isin(METHODS)].to_string(index=False))


if __name__ == "__main__":
    main()
