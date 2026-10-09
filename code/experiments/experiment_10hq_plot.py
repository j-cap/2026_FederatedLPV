"""Plot frozen Q fleet contrasts and common-input command errors."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from experiment_10hk_joint_client_mixture import OUT, ROOT, sha

CASES = [
    "nominal_with_nominal_kf", "nominal_with_true_state", "oracle_with_true_state",
    "learned_with_true_state", "learned_with_learned_kf",
    "learned_with_learned_kf_correct_forces",
]
LABELS = [
    "Nominal\nKF/LQI", "Nominal\ntrue state", "Oracle\ntrue state", "Learned\ntrue state",
    "Both", "Both +\ntrue forces",
]
COLORS = {"broadband": "#315c85", "transient": "#aa5238"}


def style():
    plt.rcParams.update({
        "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 16,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.labelsize": 16, "xtick.labelsize": 13.5, "ytick.labelsize": 14,
        "legend.fontsize": 14, "pdf.fonttype": 42,
    })


def run():
    style()
    figures = ROOT / "results/figures"
    figures.mkdir(parents=True, exist_ok=True)
    fleet = pd.read_csv(OUT / "experiment_10hq_fleets.csv")
    common = pd.read_csv(OUT / "experiment_10hq_common_input_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.8), sharey=True, layout="constrained")
    for ax, scale in zip(axes, [0.01, 1.0]):
        for offset, scenario in [(-0.18, "broadband"), (0.18, "transient")]:
            panel = fleet[(fleet.process_noise_scale == scale) & (fleet.scenario == scenario)]
            values = panel.pivot(index="seed", columns="case", values="tracking_rmse_deg_s")
            ratios = values[CASES].div(values[CASES[0]], axis=0)
            x = np.arange(len(CASES)) + offset
            ax.bar(x, ratios.mean(), width=0.32, color=COLORS[scenario], alpha=0.75,
                   label=scenario.capitalize(), zorder=2)
            for i, case in enumerate(CASES):
                ax.scatter(x[i] + np.linspace(-0.1, 0.1, 5), ratios[case], s=10,
                           color=COLORS[scenario], edgecolor="white", linewidth=0.3, zorder=3)
        ax.axhline(1, color="#555555", linestyle="--", linewidth=0.8, zorder=1)
        ax.set_yscale("log")
        ax.set_ylim(0.4, 16)
        ax.set_yticks([0.5, 1, 2, 4, 8, 16], labels=["0.5", "1", "2", "4", "8", "16"])
        ax.set_xticks(np.arange(len(CASES)), LABELS)
        ax.grid(axis="y", which="major", color="#dddddd", linewidth=0.5, zorder=0)
        ax.set_title(rf"$Q_{{\rm fit}}={scale:g}$")
    axes[0].set_ylabel("Tracking RMSE ratio\n(nominal KF/LQI = 1)")
    axes[0].legend(frameon=False, loc="upper left")
    fig.savefig(figures / "experiment_10hq_tracking_diagnostic.pdf",
                metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.2), sharey=True, layout="constrained")
    observers = ["nominal", "learned", "oracle"]
    for ax, scale in zip(axes, [0.01, 1.0]):
        for offset, scenario in [(-0.18, "broadband"), (0.18, "transient")]:
            selected = common[
                (common.process_noise_scale == scale) & (common.scenario == scenario)
                & (common.case == "learned_with_learned_kf")
            ].set_index("replay_observer")
            ax.bar(np.arange(3) + offset, selected.loc[observers, "command_total_rms_deg"],
                   width=0.32, color=COLORS[scenario], label=scenario.capitalize(), zorder=2)
        ax.set_xticks(np.arange(3), ["Nominal KF", "Learned KF", "True-model KF"])
        ax.set_title(rf"$Q_{{\rm fit}}={scale:g}$")
        ax.grid(axis="y", color="#dddddd", linewidth=0.5, zorder=0)
    axes[0].set_ylabel("Steering command-error RMS (deg)")
    axes[0].legend(frameon=False, loc="upper left")
    fig.savefig(figures / "experiment_10hq_command_projection.pdf",
                metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)
    paths = [
        "results/tables/experiment_10hq_fleets.csv",
        "results/tables/experiment_10hq_common_input_summary.csv",
        "results/figures/experiment_10hq_tracking_diagnostic.pdf",
        "results/figures/experiment_10hq_command_projection.pdf",
    ]
    (OUT / "experiment_10hq_figure_manifest.json").write_text(json.dumps({
        "source_sha256": sha(ROOT / "code/experiments/experiment_10hq_plot.py"),
        "file_sha256": {path: sha(ROOT / path) for path in paths},
        "tracking_statistic": "Mean of five paired fleet RMSE ratios; dots are individual fleets.",
        "command_statistic": "Mean of recipient RMS after averaging donor deployments then fleets.",
    }, indent=2) + "\n")


if __name__ == "__main__":
    run()
