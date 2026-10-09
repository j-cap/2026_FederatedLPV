"""Summarize prespecified frozen-Q contrasts without choosing a new method."""

import json
import platform

import numpy as np
import pandas as pd
import scipy
from experiment_10hi_physically_coupled_fit import load_configuration, make_coordinates
from experiment_10hk_joint_client_mixture import OUT, ROOT, sha
from experiment_10hp_output_feedback import configuration

from federated_lpv.physical_coupled_fit import PHYSICAL_NAMES

BASELINE = "nominal_with_nominal_kf"
BOTH = "learned_with_learned_kf"
GROUP = ["seed", "process_noise_scale", "scenario"]


def contrasts(cases):
    result = [("vs_nominal_output_feedback", case, BASELINE) for case in cases if case != BASELINE]
    result.extend([
        ("learned_vs_nominal_true_state", "learned_with_true_state", "nominal_with_true_state"),
        ("oracle_vs_nominal_true_state", "oracle_with_true_state", "nominal_with_true_state"),
    ])
    result.extend(
        (f"{controller}_learned_kf_vs_true", f"{controller}_with_learned_kf",
         f"{controller}_with_true_state")
        for controller in ["nominal", "learned", "oracle"]
    )
    for correction in ["beta", "forces", "beta_forces"]:
        target = BOTH + "_correct_" + correction
        result.extend([
            (f"{correction}_vs_both", target, BOTH),
            (f"{correction}_vs_learned_true", target, "learned_with_true_state"),
        ])
    return result


def paired_tables(frame):
    recipients = frame.groupby(GROUP + ["case", "client_position"])[
        ["tracking_rmse_deg_s", "full_path_tracking_delta_deg_s"]
    ].mean().reset_index()
    fleets = recipients.groupby(GROUP + ["case"])[
        ["tracking_rmse_deg_s", "full_path_tracking_delta_deg_s"]
    ].mean().reset_index()
    fleet_rows, recipient_rows = [], []
    for name, target, reference in contrasts(sorted(frame.case.unique())):
        for source, keys, rows in [
            (fleets, GROUP, fleet_rows),
            (recipients, GROUP + ["client_position"], recipient_rows),
        ]:
            left = source[source.case == target].drop(columns="case")
            right = source[source.case == reference].drop(columns="case")
            paired = left.merge(right, on=keys, suffixes=("_target", "_reference"),
                                validate="one_to_one")
            t, r = paired.tracking_rmse_deg_s_target, paired.tracking_rmse_deg_s_reference
            te, re = (paired.full_path_tracking_delta_deg_s_target,
                      paired.full_path_tracking_delta_deg_s_reference)
            paired["gain_pct"] = 100 * (1 - t / r)
            paired["replay_gain_lower_pct"] = 100 * (1 - (t + te) / np.maximum(r - re, 1e-12))
            paired["replay_gain_upper_pct"] = 100 * (1 - np.maximum(t - te, 0) / (r + re))
            paired["contrast"], paired["target"], paired["reference"] = name, target, reference
            rows.extend(paired.to_dict("records"))
    paired_fleets, paired_recipients = pd.DataFrame(fleet_rows), pd.DataFrame(recipient_rows)
    keys = ["process_noise_scale", "scenario", "contrast", "target", "reference"]
    summary = paired_fleets.groupby(keys).agg(
        mean_gain_pct=("gain_pct", "mean"),
        fleets_improved=("gain_pct", lambda s: int((s > 0).sum())),
        mean_replay_gain_lower_pct=("replay_gain_lower_pct", "mean"),
        mean_replay_gain_upper_pct=("replay_gain_upper_pct", "mean"),
    ).reset_index()
    recipient_summary = paired_recipients.groupby(keys).agg(
        recipients_improved=("gain_pct", lambda s: int((s > 0).sum())),
        recipients_degraded=("gain_pct", lambda s: int((s < 0).sum())),
        median_gain_pct=("gain_pct", "median"),
        p10_gain_pct=("gain_pct", lambda s: s.quantile(0.1)),
        worst_gain_pct=("gain_pct", "min"),
    ).reset_index()
    for suffix, table in [
        ("paired_fleets", paired_fleets), ("paired_recipients", paired_recipients),
        ("paired_summary", summary), ("paired_recipient_summary", recipient_summary),
    ]:
        table.to_csv(OUT / f"experiment_10hq_{suffix}.csv", index=False)
    return summary


def common_tables():
    common = pd.read_csv(OUT / "experiment_10hq_common_input.csv")
    measures = [c for c in common if c.endswith(("rmse_deg", "rmse_deg_s", "force_rmse",
                                                "rms_deg", "rad2"))]
    keys = GROUP + ["case", "replay_observer", "client_position"]
    recipient = common.groupby(keys)[measures].mean().reset_index()
    fleet = recipient.groupby(keys[:-1])[measures].mean().reset_index()
    summary = fleet.groupby(keys[1:-1])[measures].mean().reset_index()
    fleet.to_csv(OUT / "experiment_10hq_common_input_fleets.csv", index=False)
    summary.to_csv(OUT / "experiment_10hq_common_input_summary.csv", index=False)


def physical_table():
    cfg, inherited, _ = configuration()
    bounds, _ = load_configuration()
    coordinates = make_coordinates(bounds, inherited["10hf"])
    libraries = json.loads((OUT / "experiment_10hn_libraries.json").read_text())
    rows = []
    for lib in libraries:
        for choice in sorted(set(lib["recipient_choices"])):
            values = np.exp(coordinates.nominal_physical_logs + np.asarray(lib["eta"][choice]))
            rows.append(dict(seed=lib["seed"], process_noise_scale=lib["process_noise_scale"],
                             donor_fold=lib["donor_fold"], selected_plant=choice,
                             **dict(zip(PHYSICAL_NAMES, values))))
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "experiment_10hq_deployed_physical_coordinates.csv", index=False)
    ranges = {}
    for scale in cfg["fitting_panels"]:
        subset = frame[frame.process_noise_scale == scale]
        ranges[str(scale)] = {
            name: {"minimum": float(subset[name].min()), "maximum": float(subset[name].max()),
                   "nominal": float(np.exp(coordinates.nominal_physical_logs[i]))}
            for i, name in enumerate(PHYSICAL_NAMES)
        }
    return ranges


def run():
    keys = GROUP + ["donor_fold", "client_position", "case"]
    metrics = pd.read_csv(OUT / "experiment_10hq_metrics.csv")
    audit = pd.read_csv(OUT / "experiment_10hq_replay_audit.csv")
    frame = metrics.merge(audit[keys + ["full_path_tracking_delta_deg_s"]], on=keys,
                          validate="one_to_one")
    assert len(frame) == 4800
    summary = paired_tables(frame)
    common_tables()
    safety = metrics.groupby(["process_noise_scale", "case"]).agg(
        runs=("finite", "size"), finite=("finite", "sum"),
        design_failures=("design_failed", "sum"), saturation_samples=("saturation_samples", "sum"),
        beta_violation_samples=("beta_violation_samples", "sum"),
        steering_violation_samples=("steering_violation_samples", "sum"),
    ).reset_index()
    safety.to_csv(OUT / "experiment_10hq_safety.csv", index=False)
    radii = pd.concat([pd.read_csv(p) for p in sorted(OUT.glob("experiment_10hq_seed*_frozen_feedback.csv"))])
    radius_summary = radii.groupby(["process_noise_scale", "case"]).agg(
        pairs=("maximum_feedback_radius", "size"),
        unstable_pairs=("maximum_feedback_radius", lambda s: int((s >= 1).sum())),
        maximum_radius=("maximum_feedback_radius", "max"),
    ).reset_index()
    radius_summary.to_csv(OUT / "experiment_10hq_feedback_summary.csv", index=False)
    primary = summary[(summary.process_noise_scale == 0.01) & summary.contrast.isin([
        "learned_vs_nominal_true_state", "oracle_vs_nominal_true_state", "forces_vs_both",
    ])]
    result = {
        "reconstruction": True, "confirmation_run": False,
        "primary_contrasts": primary.to_dict("records"),
        "physical_coordinate_ranges": physical_table(),
        "interpretation": [
            "True states do not rescue the learned controller in mean fleet tracking loss.",
            "Force truth corrections worsen primary-panel Both tracking; observer/controller errors may compensate.",
            "Common-input beta improvement coexists with worse force and steering-relevant reconstruction.",
            "Oracle full-state improvement is modest for these fixed tasks/costs, not an optimal tracking bound.",
            "Physical coupling and heuristic coefficient bounds do not guarantee plausible controller models.",
            "No iterative federated KF/controller improvement or blind confirmation has been demonstrated.",
        ],
        "replay_envelope_definition": "Absolute alternate-path RMSE changes bound paired means; not confidence intervals or a general robustness guarantee.",
        "common_input_summary_definition": "Mean of recipient RMS values after averaging two donor deployments, then ten recipients per fleet, then five fleets; second moments are averaged separately.",
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "scipy": scipy.__version__, "pandas": pd.__version__},
        "source_sha256": sha(ROOT / "code/experiments/experiment_10hq_reporting.py"),
        "audit_sha256": sha(OUT / "experiment_10hq_audit.json"),
        "projection_audit_sha256": sha(OUT / "experiment_10hq_projection_audit.json"),
    }
    (OUT / "experiment_10hq_conclusions.json").write_text(json.dumps(result, indent=2) + "\n")
    print(primary.to_string(index=False))


if __name__ == "__main__":
    run()
