"""Scientific figure for the frozen Q bottleneck diagnostic."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from experiment_10hk_joint_client_mixture import OUT, ROOT


def panel(ax, groups, labels, colors, ylabel):
    for i, (values, color) in enumerate(zip(groups, colors)):
        ax.bar(i, np.mean(values), color=color, width=0.66)
        ax.scatter(i + np.linspace(-0.14, 0.14, len(values)), values,
                   c="black", s=16, zorder=3)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def run():
    fleet = pd.read_csv(OUT / "experiment_10hq_fleets.csv")
    common = pd.read_csv(OUT / "experiment_10hq_common_input_fleets.csv")
    cases = ["nominal_with_nominal_kf", "learned_with_learned_kf", "learned_with_true_state",
             "learned_with_learned_kf_correct_forces", "oracle_with_true_state"]
    labels = ["Nominal", "Both", "Learned\ntrue states", "Both\ntrue forces", "Oracle\ntrue states"]
    colors = ["#89939e", "#2674ad", "#df8e39", "#8970ad", "#4b9b77"]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.7), layout="constrained")
    for col, scene in enumerate(["broadband", "transient"]):
        selected = fleet[(fleet.process_noise_scale == 0.01) & (fleet.scenario == scene)]
        values = [selected[selected.case == case].tracking_rmse_deg_s.to_numpy() for case in cases]
        panel(axes[0, col], values, labels, colors, "Tracking RMSE (deg/s)")
        axes[0, col].set_title(scene.capitalize() + " tracking")
        selected = common[(common.process_noise_scale == 0.01) & (common.scenario == scene)
                          & (common.case == "learned_with_learned_kf")]
        values = [selected[selected.replay_observer == name].command_total_rms_deg.to_numpy()
                  for name in ["nominal", "learned", "oracle"]]
        panel(axes[1, col], values, ["Nominal KF", "Learned KF", "Oracle KF"],
              [colors[0], colors[1], colors[4]], "Projected steering RMS (deg)")
        axes[1, col].set_title(scene.capitalize() + ": common Both input")
    fig.suptitle(r"10H-Q frozen feedback diagnostic: $Q_{fit}=0.01$", fontsize=14)
    destination = ROOT / "results/figures/experiment_10hq_feedback_bottleneck.pdf"
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


if __name__ == "__main__":
    run()
