"""Generate manuscript figures from frozen evidence, without fits or simulations.

Run from any directory. --preflight is read-only; --generate writes new figure
files and a provenance manifest. The active manuscript and its figures are not
overwritten. Combined figures and separate panels are exported as PDF and PNG.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results/tables"
DEFAULT_OUTPUT = ROOT / "results/figures"
LOCK = ROOT / "paper/figure_input_lock.json"
COLORS = {"Local": "#777777", "Global": "#cf741c", "FederatedLearned": "#236b9a"}
LABELS = {"Local": "Restricted Local", "Global": "Global", "FederatedLearned": "Grouped"}
STYLES = {"Local": "--", "Global": "-.", "FederatedLearned": "-"}
TIE_TOLERANCE = 1e-12  # Dimensionless matrix error, before conversion to percent.


def digest(path: Path) -> str:
    data = path.read_bytes()
    if path.suffix in {".csv", ".json", ".py"}:
        data = data.replace(b"\r\n", b"\n")  # git autocrlf on Windows.
    return hashlib.sha256(data).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def csv(name):
    return pd.read_csv(TABLES / name)


def load_and_check():
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    for relative, expected in lock["source_sha256"].items():
        path = ROOT / relative
        require(path.is_file(), f"Missing frozen input: {relative}. Pull the repository first.")
        require(digest(path) == expected, f"Frozen input changed: {relative}. Do not regenerate the input lock.")
    cfg = json.loads((ROOT / "code/config/experiment_10e.json").read_text())
    cfg_a = json.loads((ROOT / "code/config/experiment_10a.json").read_text())
    audit = json.loads((TABLES / "paper_review_validation_independent_audit.json").read_text())
    seeds = cfg["seeds"]
    speeds = cfg["speed_grid"]
    require(seeds == list(range(371, 381)), "Expected the ten original primary fleets.")
    raw = pd.concat([csv(f"experiment_10e_seed{s}_label_free_clients.csv.gz") for s in seeds], ignore_index=True)
    require(len(raw) == 19800 and not raw.duplicated(["seed", "client", "method", "speed"]).any(),
            "Incomplete or duplicated primary client-speed records.")
    require(np.isfinite(raw.prediction_error).all() and (raw.prediction_error >= 0).all(), "Invalid matrix errors.")
    require(set(raw.seed) == set(seeds) and set(raw.speed) == set(speeds), "Primary evaluation grid changed.")
    require(raw.groupby(["seed", "method"]).size().eq(30 * len(speeds)).all(), "Unbalanced primary records.")
    require(raw.groupby(["seed", "client"]).block.nunique().eq(1).all(), "A client has multiple coverage blocks.")
    frozen = csv("experiment_10e_seed_summary.csv")
    expected = frozen[frozen.protocol == "label_free"].set_index(["seed", "method"]).prediction_error.sort_index()
    actual = raw.groupby(["seed", "method"]).prediction_error.mean().sort_index()
    require(actual.index.equals(expected.index), "Primary methods changed.")
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-14)
    raw["region"] = ["seen" if speed in cfg["coverage_blocks"][int(block)] else "unseen"
                     for speed, block in zip(raw.speed, raw.block)]
    regions = raw.groupby(["seed", "region", "method"]).prediction_error.mean()
    frozen_regions = csv("paper_identification_regions_seed.csv")
    frozen_regions = frozen_regions[frozen_regions.region.isin(["seen", "unseen"])].set_index(["seed", "region", "method"]).prediction_error
    require(regions.sort_index().index.equals(frozen_regions.sort_index().index), "Regional evaluation units changed.")
    np.testing.assert_allclose(regions.sort_index(), frozen_regions.sort_index(), rtol=1e-12, atol=1e-14)
    fleet = actual.unstack("method")
    gains = 100 * (1 - fleet.FederatedLearned / fleet.Global)
    comparison = csv("experiment_10e_comparisons.csv")
    comparison = comparison[(comparison.protocol == "label_free") & (comparison.method == "FederatedLearned") &
                            (comparison.baseline == "Global") & (comparison.metric == "prediction_error")]
    require(len(comparison) == 1, "Expected one frozen primary paired comparison.")
    comparison = comparison.iloc[0]
    np.testing.assert_allclose(gains.mean(), comparison.improvement_pct, rtol=1e-12, atol=1e-12)
    require(int((fleet.Global - fleet.FederatedLearned > TIE_TOLERANCE).sum()) == 8, "Fleet win count changed.")
    clients = raw.groupby(["seed", "client", "region", "method"]).prediction_error.mean().unstack("method")
    unseen = clients.xs("unseen", level="region").copy()
    require(len(unseen) == 300, "Expected 300 nested client instances.")
    tails = csv("paper_identification_client_tails.csv")
    tails = tails[tails.region == "unseen"].set_index(["seed", "client"]).sort_index()
    np.testing.assert_allclose(unseen.Global.sort_index(), tails.global_error, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(unseen.FederatedLearned.sort_index(), tails.grouped_error, rtol=1e-12, atol=1e-14)
    delta = unseen.FederatedLearned - unseen.Global
    counts = {"better": int((delta < -TIE_TOLERANCE).sum()), "tied": int((delta.abs() <= TIE_TOLERANCE).sum()),
              "worse": int((delta > TIE_TOLERANCE).sum())}
    require(counts == {"better": 169, "tied": 60, "worse": 71}, "Client tradeoff counts changed.")
    groups = csv("paper_review_validation_groups.csv")
    local = csv("paper_review_validation_local_rank_audit.csv")
    require(audit["status"] == "independently_audited" and len(groups) == audit["groups"] == 18,
            "Expected the independently audited 18 learned groups.")
    require(groups["rank"].eq(7).all() and all(json.loads(s) == speeds for s in groups.unique_speeds),
            "Selected group scheduling information changed.")
    k = groups.groupby("seed").size().reindex(seeds)
    expected_k = {row["seed"]: row["selected_k"] for row in audit["fleet_checks"]}
    require(k.to_dict() == expected_k, "Selected group counts changed.")
    require(len(local) == 300 and not local.duplicated(["seed", "client"]).any(), "Local rank records incomplete.")
    require(local.local_rank.value_counts().to_dict() == {3: 230, 2: 70}, "Local rank counts changed.")
    basis = csv("experiment_10a_summary.csv")
    selection = csv("experiment_10a_selection.csv")
    coverage = csv("experiment_10a_coverage_summary.csv")
    require(len(basis) == 15 and not basis.duplicated(["curvature", "order"]).any(), "Basis audit incomplete.")
    for _, selected in selection.iterrows():
        view = basis[basis.curvature == selected.curvature].sort_values("order")
        require(view.order.tolist() == cfg_a["candidate_orders"], "Development basis orders changed.")
        eligible = view[view.condition_worst <= cfg_a["maximum_aggregate_condition"]]
        best = eligible.loc[eligible.error_mean.idxmin()]
        threshold = max(best.error_mean + best.error_se, cfg_a["practical_relative_error_floor"])
        order = int(eligible[eligible.error_mean <= threshold].order.min())
        require(order == int(selected.selected_order), "Frozen basis selection disagrees with its rule.")
        np.testing.assert_allclose(threshold, selected.threshold, rtol=1e-12, atol=1e-14)
    require(selection.set_index("curvature").selected_order.to_dict() == {0.0: 3, 0.0025: 6, 0.005: 7},
            "Development selected orders changed.")
    relevant = coverage[(coverage.curvature == cfg["curvature_per_m"]) & (coverage.order == 7)].set_index("scope")
    expected_ranks = {"aggregate_envelope": 7, **{f"local_block_{i}": r for i, r in enumerate([3, 3, 3, 2])}}
    require(relevant.rank_min.to_dict() == expected_ranks and relevant.rank_max.to_dict() == expected_ranks,
            "Development coverage ranks changed.")
    condition = [float(groups.design_condition.min()), float(groups.design_condition.max())]
    np.testing.assert_allclose(condition, audit["group_design_condition_range"], rtol=1e-12)
    checks = {"primary_fleets": len(seeds), "primary_records": len(raw), "nested_client_instances": len(unseen),
              "selected_groups": len(groups), "local_rank_counts": {"3": 230, "2": 70},
              "selected_k": {str(s): int(value) for s, value in k.items()}, "group_condition_range": condition,
              "client_unobserved_changes": counts,
              "full_envelope_paired_gain_pct": float(gains.mean()),
              "frozen_fleet_bootstrap_ci_pct": [float(comparison.ci_low), float(comparison.ci_high)],
              "basis_selected_orders": {str(r.curvature): int(r.selected_order) for _, r in selection.iterrows()}}
    return dict(cfg=cfg, cfg_a=cfg_a, raw=raw, fleet=fleet, unseen=unseen, counts=counts, groups=groups,
                k=k, basis=basis, selection=selection, coverage=relevant, checks=checks, lock=lock)


def set_style():
    plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 8,
                        "mathtext.fontset": "dejavuserif", "axes.labelsize": 8, "axes.titlesize": 8,
                        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
                        "axes.linewidth": .6, "lines.linewidth": 1.3, "pdf.fonttype": 42,
                        "ps.fonttype": 42, "savefig.facecolor": "white"})


def tidy(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(width=.6, length=3)
    ax.grid(axis="y", color="#dddddd", linewidth=.5)
    ax.set_axisbelow(True)


def basis_panel(ax, data, panel=""):
    for curvature, color, label in [(0.0, COLORS["FederatedLearned"], "Straight line"),
                                    (.005, COLORS["Global"], r"Cornering ($\kappa=0.005$ m$^{-1}$)")]:
        view = data["basis"][data["basis"].curvature == curvature].sort_values("order")
        ax.errorbar(view.order, 100 * view.error_mean, yerr=100 * view.error_se, marker="o", markersize=3.5,
                    color=color, capsize=2, label=label)
        order = int(data["selection"].set_index("curvature").loc[curvature, "selected_order"])
        value = float(view.set_index("order").loc[order, "error_mean"])
        ax.scatter([order], [100 * value], s=65, facecolors="none", edgecolors=color, linewidths=1.2, zorder=5)
        ax.annotate(f"Selected L={order}", (order, 100 * value), xytext=(5, 10) if order == 3 else (-70, -14),
                    textcoords="offset points", fontsize=7, color=color)
    ax.axhline(100 * data["cfg_a"]["practical_relative_error_floor"], color="#888888", linestyle=":", linewidth=1)
    ax.text(3.02, .012, "Selection floor: 0.01%", fontsize=6.5, color="#555555")
    ax.set(xlabel="Basis order L", ylabel="Blocked-speed matrix error [%]", yscale="log",
           title=f"{panel}Development-only basis validation", xticks=range(3, 8), xlim=(2.9, 7.15), ylim=(1e-5, 5))
    tidy(ax)
    ax.legend(frameon=False, loc="upper right")


def information_panel(ax, data, panel=""):
    speeds = data["cfg"]["speed_grid"]
    rows = data["cfg"]["coverage_blocks"] + [speeds, speeds]
    labels = ["Local 10-14", "Local 16-20", "Local 22-26", "Local 28-30", "Fleet union (dev.)", "Learned groups"]
    ranks = [3, 3, 3, 2, 7, 7]
    for row, observed in enumerate(rows):
        color = "#9b9b9b" if row < 4 else ("#84aa7c" if row == 4 else COLORS["FederatedLearned"])
        for col, speed in enumerate(speeds):
            ax.add_patch(Rectangle((col - .43, row - .35), .86, .7,
                                   facecolor=color if speed in observed else "white", edgecolor="#c9c9c9", linewidth=.5))
        ax.text(11.15, row, str(ranks[row]), va="center", ha="center", fontsize=8,
                fontweight="bold" if row >= 4 else "normal")
    ax.text(11.15, -.8, "Rank", ha="center", va="center", fontsize=7)
    ax.axhline(3.6, color="#aaaaaa", linewidth=.6, linestyle=":")
    ax.set(xlim=(-.55, 11.8), ylim=(6.75, -1.0), xticks=range(len(speeds)),
           xticklabels=speeds, yticks=range(6), yticklabels=labels,
           xlabel="Observed grid speed [m/s]", title=f"{panel}Scheduling information (L=7)")
    lo, hi = data["checks"]["group_condition_range"]
    ax.text(4.8, 6.05, f"All 18 learned groups: rank 7; condition {lo:.2f}-{hi:.2f}",
            fontsize=6.5, ha="center", va="center")
    ax.tick_params(length=0, pad=3)
    for spine in ax.spines.values():
        spine.set_visible(False)


def fleet_panel(ax, data, panel=""):
    fleet = data["fleet"]
    x = np.arange(len(fleet))
    for i, (_, row) in enumerate(fleet.iterrows()):
        ax.plot([i, i], 100 * row[["Global", "FederatedLearned"]].to_numpy(), color="#b4b4b4", linewidth=1, zorder=1)
    ax.plot(x, 100 * fleet.Global, linestyle="none", marker="o", markersize=5,
            markerfacecolor="white", color=COLORS["Global"], label="Global", zorder=3)
    ax.plot(x, 100 * fleet.FederatedLearned, linestyle="none", marker="s", markersize=3.5,
            color=COLORS["FederatedLearned"], label="Grouped", zorder=4)
    for i, seed in enumerate(fleet.index):
        if data["k"].loc[seed] == 1:
            ax.annotate("K=1", (i, 100 * fleet.loc[seed, "Global"]), xytext=(0, 7),
                        textcoords="offset points", ha="center", fontsize=6.5)
    ymax = float(100 * fleet[["Global", "FederatedLearned"]].to_numpy().max())
    ax.set(xlabel="Independent fleet seed", ylabel="Full-envelope matrix error [%]",
           title=f"{panel}Paired primary results", xticks=x, xticklabels=fleet.index,
           xlim=(-.5, len(x) - .5), ylim=(0, ymax * 1.23))
    tidy(ax)
    ax.legend(frameon=False, loc="lower left")


def client_panel(ax, data, panel=""):
    delta = (data["unseen"].FederatedLearned - data["unseen"].Global).to_numpy()
    delta[np.abs(delta) <= TIE_TOLERANCE] = 0
    delta = np.sort(100 * delta)
    margin = .08 * (delta.max() - delta.min())
    left, right = delta.min() - margin, delta.max() + margin
    ax.axvspan(left, 0, color=COLORS["FederatedLearned"], alpha=.055)
    ax.axvspan(0, right, color=COLORS["Global"], alpha=.055)
    ax.step(np.r_[left, delta, right], np.r_[0, np.arange(1, len(delta) + 1) / len(delta), 1],
            where="post", color=COLORS["FederatedLearned"], linewidth=1.4)
    ax.axvline(0, color="#555555", linestyle="--", linewidth=.8)
    ax.text(.02, .96, "169 better / 60 tied / 71 worse", transform=ax.transAxes, fontsize=7,
            va="top", bbox={"facecolor": "white", "edgecolor": "none", "alpha": .85, "pad": 2})
    ax.text(.03, .06, "Better", transform=ax.transAxes, fontsize=7, color=COLORS["FederatedLearned"])
    ax.text(.97, .06, "Worse", transform=ax.transAxes, fontsize=7, color=COLORS["Global"], ha="right")
    ax.set(xlabel="Unobserved error change [percentage points]\n(Grouped - Global)",
           ylabel="Fraction of client instances", title=f"{panel}Client tradeoff (descriptive)",
           xlim=(left, right), ylim=(0, 1.04), yticks=[0, .25, .5, .75, 1])
    tidy(ax)


def coverage_statistics(data):
    result = {}
    for block in range(4):
        draws = np.random.default_rng(20261009 + block).integers(0, 10, size=(20000, 10))
        for method in COLORS:
            view = data["raw"][(data["raw"].block == block) & (data["raw"].method == method)]
            values = 100 * view.groupby(["seed", "speed"]).prediction_error.mean().unstack("speed").to_numpy()
            require(values.shape == (10, 11), "Incomplete block-by-fleet curve.")
            low, high = np.quantile(values[draws].mean(axis=1), [.025, .975], axis=0)
            result[(block, method)] = (values.mean(axis=0), low, high)
    return result


def coverage_panel(ax, data, statistics, block, panel="", xlabel=True, ylabel=True):
    observed = data["cfg"]["coverage_blocks"][block]
    speeds = data["cfg"]["speed_grid"]
    ax.axvspan(min(observed) - .35, max(observed) + .35, color="#dbe9d4", alpha=.65, linewidth=0)
    for method in COLORS:
        mean, low, high = statistics[(block, method)]
        ax.plot(speeds, mean, color=COLORS[method], linestyle=STYLES[method], label=LABELS[method])
        ax.fill_between(speeds, low, high, color=COLORS[method], alpha=.13, linewidth=0)
    ax.plot(observed, [.035] * len(observed), marker="|", markersize=7, color="#4b7345", linestyle="none",
            transform=ax.get_xaxis_transform(), zorder=5)
    bounds = [value for _, (_, low, high) in statistics.items() for value in [float(low.min()), float(high.max())]]
    ymin, ymax = 10 ** np.floor(np.log10(min(bounds))), 10 ** np.ceil(np.log10(max(bounds)))
    ax.set(yscale="log", xlim=(9, 31), ylim=(ymin, ymax), xticks=[10, 15, 20, 25, 30],
           title=f"{panel}Local grid: {', '.join(map(str, observed))} m/s")
    if xlabel:
        ax.set_xlabel("Speed [m/s]")
    if ylabel:
        ax.set_ylabel("Relative matrix error [%]")
    tidy(ax)


def save(fig, name, output):
    files = []
    for extension in ["pdf", "png"]:
        path = output / f"{name}.{extension}"
        options = {"metadata": {"Creator": "prepare_identification_figures.py", "CreationDate": None, "ModDate": None}} if extension == "pdf" else {"dpi": 300}
        fig.savefig(path, **options)
        files.append(path)
    plt.close(fig)
    return files


def generate(data, output):
    output.mkdir(parents=True, exist_ok=True)
    set_style()
    files = []
    for name, panels in [("paper_basis_information", [basis_panel, information_panel]),
                         ("paper_fleet_client_tradeoff", [fleet_panel, client_panel])]:
        fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.85), layout="constrained")
        for i, (ax, draw) in enumerate(zip(axes, panels)):
            draw(ax, data, f"({chr(97 + i)}) ")
        files.extend(save(fig, name, output))
        for suffix, draw in zip(["a", "b"], panels):
            fig, ax = plt.subplots(figsize=(3.48, 2.85), layout="constrained")
            draw(ax, data)
            files.extend(save(fig, f"{name}_{suffix}", output))
    statistics = coverage_statistics(data)
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 4.25), sharex=True, sharey=True, layout="constrained")
    for block, ax in enumerate(axes.flat):
        coverage_panel(ax, data, statistics, block, f"({chr(97 + block)}) ", xlabel=block >= 2, ylabel=block % 2 == 0)
    axes[0, 0].legend(frameon=False, fontsize=7, loc="upper left")
    files.extend(save(fig, "paper_identification_coverage_v2", output))
    for block in range(4):
        fig, ax = plt.subplots(figsize=(3.48, 2.2), layout="constrained")
        coverage_panel(ax, data, statistics, block)
        if block == 0:
            ax.legend(frameon=False, loc="upper left")
        files.extend(save(fig, f"paper_identification_coverage_v2_{chr(97 + block)}", output))
    return files


def display_path(path):
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true", help="Verify frozen inputs and endpoints; write nothing.")
    mode.add_argument("--generate", action="store_true", help="Verify inputs, then export all figures and panels.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT, help="Default: results/figures.")
    args = parser.parse_args()
    data = load_and_check()
    if args.preflight:
        print(json.dumps({"status": "preflight_passed", **data["checks"]}, indent=2))
        return
    output = args.output_dir.resolve()
    files = generate(data, output)
    manifest_path = TABLES / "paper_visuals_manifest.json" if output == DEFAULT_OUTPUT else output / "paper_visuals_manifest.json"
    manifest = {"status": "generated_from_verified_frozen_evidence", "operation": "plotting_only_no_fitting_or_simulation",
                "generator": "paper/prepare_identification_figures.py", "generator_sha256": digest(Path(__file__)),
                "input_lock_sha256": digest(LOCK), "source_sha256": data["lock"]["source_sha256"],
                "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                             "matplotlib": matplotlib.__version__}, "checks": data["checks"],
                "conventions": {"matrix_error_display": "percent = 100 * dimensionless error",
                                "client_change": "100 * (Grouped - Global), percentage points; positive means worse",
                                "client_tie_tolerance_dimensionless": TIE_TOLERANCE,
                                "client_ecdf": "300 nested client instances; descriptive; no independent-client confidence interval",
                                "coverage_bands": "pointwise 95% bootstrap across 10 fleets; 20000 draws; seeds 20261009 + block",
                                "basis_error_bars": "one standard error across 10 development fleets",
                                "basis_displayed_curvatures_per_m": [0, .005],
                                "paired_ci": "copied from frozen original full-envelope comparison, not recomputed"},
                "output_sha256": {display_path(path): digest(path) for path in files}}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "pdf_files": len(files) // 2, "png_previews": len(files) // 2,
                      "output_dir": str(output), "manifest": str(manifest_path), **data["checks"]}, indent=2))


if __name__ == "__main__":
    main()
