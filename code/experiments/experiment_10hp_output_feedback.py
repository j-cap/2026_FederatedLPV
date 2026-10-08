"""Frozen reconstructed P baseline and shared causal Q rollout adapter."""

from __future__ import annotations

import argparse
import json
import subprocess
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from experiment_10h_higher_order_lpv_gate import discrete_matrices, sample_fleet, scenarios
from experiment_10ha_tire_force_observer import measurement_covariance, sample_bias
from experiment_10hi_physically_coupled_fit import make_coordinates
from experiment_10hk_joint_client_mixture import OUT, ROOT, sha
from experiment_10hl_validated_model_selection import load_configuration

from federated_lpv.feedback_diagnostics import (
    FeedbackCase,
    command_error_components,
    command_error_summary,
    diagnostic_cases,
)
from federated_lpv.innovation_likelihood import structured_matrices
from federated_lpv.output_feedback_control import (
    design_schedule,
    feedback_radius,
    metrics,
    replay_observer,
    rollout,
    schedule_values,
)

BASELINE_CASE = "nominal_with_nominal_kf"
BOTH_CASE = "learned_with_learned_kf"
P_CASES = (
    FeedbackCase(BASELINE_CASE, "nominal", "nominal_kf", "none"),
    FeedbackCase("nominal_with_learned_kf", "nominal", "learned_kf", "none"),
    FeedbackCase("learned_with_nominal_kf", "learned", "nominal_kf", "none"),
    FeedbackCase(BOTH_CASE, "learned", "learned_kf", "none"),
)


def configuration():
    cfg = json.loads((ROOT / "code/config/experiment_10hp.json").read_text())
    _, inherited = load_configuration()
    weights = json.loads((OUT / "experiment_10h_frozen_selection.json").read_text())["selected"]
    return cfg, inherited, weights


def verify_learned_inputs():
    for prefix in ["experiment_10hm", "experiment_10hn"]:
        manifest = json.loads((OUT / f"{prefix}_complete.json").read_text())
        for key in ["source_sha256", "input_sha256"]:
            for name, value in manifest.get(key, {}).items():
                if sha(ROOT / name) != value:
                    raise RuntimeError(f"Changed learned input source {name}")
        for name, value in manifest["output_sha256"].items():
            if sha(OUT / name) != value:
                raise RuntimeError(f"Changed learned result {name}")
        audit = json.loads((OUT / f"{prefix}_audit.json").read_text())
        if not audit["all_checks_pass"] or audit["library_sha256"] != sha(
            OUT / f"{prefix}_libraries.json"
        ):
            raise RuntimeError("Unaudited learned library")


def source_hashes(mode):
    paths = [
        "code/config/experiment_10hp.json",
        "docs/experiment_10hp_protocol.md",
        "code/experiments/experiment_10hp_output_feedback.py",
        "code/experiments/experiment_10hp_result_audit.py",
        "code/src/federated_lpv/output_feedback_control.py",
        "code/src/federated_lpv/feedback_diagnostics.py",
        "code/src/federated_lpv/innovation_likelihood.py",
        "code/src/federated_lpv/physical_coupled_fit.py",
        "code/tests/test_experiment_10hp.py",
        "results/tables/experiment_10hn_libraries.json",
        "results/tables/experiment_10hn_complete.json",
        "results/tables/experiment_10hn_audit.json",
        "results/tables/experiment_10h_frozen_selection.json",
    ]
    paths += [
        f"code/config/experiment_{name}.json" for name in ["10h", "10ha", "10hf", "10hk", "10hl"]
    ]
    paths += [
        f"code/experiments/experiment_{name}.py"
        for name in [
            "10h_higher_order_lpv_gate",
            "10ha_tire_force_observer",
            "10hl_validated_model_selection",
            "10hi_physically_coupled_fit",
        ]
    ]
    if mode == "10hq":
        paths += [
            "code/config/experiment_10hq.json",
            "docs/experiment_10hq_protocol.md",
            "code/experiments/experiment_10hq_feedback_bottleneck.py",
            "results/tables/experiment_10hp_complete.json",
            "results/tables/experiment_10hp_audit.json",
        ]
    return {p: sha(ROOT / p) for p in paths}


def prepare(seed, scale, include_oracle=False):
    cfg, inherited, weights = configuration()
    if seed not in cfg["development_seeds"] or scale not in cfg["fitting_panels"]:
        raise ValueError("Only declared development jobs allowed")
    h, ha = inherited["10h"], inherited["10ha"]
    grid, true_grid = np.asarray(cfg["controller_grid"]), np.asarray(h["speed_grid"])
    q = np.diag(np.asarray(ha["base_process_noise_diagonal"]) * scale)
    r = measurement_covariance(ha)
    bound_cfg, _ = load_configuration()
    nominal = make_coordinates(bound_cfg, inherited["10hf"]).nominal

    def design(z):
        return design_schedule(
            [structured_matrices(z, v, cfg["sample_time"]) for v in grid],
            grid,
            q,
            r,
            weights,
            cfg["sample_time"],
        )

    nominal_schedule = design(nominal)
    libraries = [
        lib
        for lib in json.loads((OUT / "experiment_10hn_libraries.json").read_text())
        if lib["seed"] == seed and lib["process_noise_scale"] == scale
    ]
    if len(libraries) != 2:
        raise RuntimeError("Expected two frozen donor deployments")
    true = {}
    fleet = sample_fleet(seed, h)
    for position in libraries[0]["recipient_clients"]:
        matrices = [discrete_matrices(fleet[position], v, cfg["sample_time"]) for v in true_grid]
        oracle = None
        if include_oracle:
            a = schedule_values(np.asarray([m[0] for m in matrices]), true_grid, grid)
            b = schedule_values(np.asarray([m[1] for m in matrices]), true_grid, grid)
            oracle = design_schedule(list(zip(a, b)), grid, q, r, weights, cfg["sample_time"])
        true[position] = (matrices, oracle)
    designs = {}
    for lib in libraries:
        for choice in set(lib["recipient_choices"]):
            try:
                designs[lib["donor_fold"], choice] = design(lib["log_parameters"][choice])
            except (ValueError, np.linalg.LinAlgError):
                designs[lib["donor_fold"], choice] = None
    return cfg, inherited, libraries, nominal_schedule, designs, true


def sensor_noise(seed, position, scenario_index, count, cfg, ha):
    rng = np.random.default_rng(
        cfg["noise_seed_offset"] + 1000 * seed + 10 * position + scenario_index
    )
    bias = sample_bias(ha, rng)
    r = measurement_covariance(ha)
    quiet = bias + rng.multivariate_normal(np.zeros(3), r, size=cfg["quiet_bias_samples"])
    bias_estimate = quiet.mean(axis=0)
    measurements = bias + rng.multivariate_normal(np.zeros(3), r, size=count)
    return measurements - bias_estimate


def steering_diagnostics(estimate, truth, controller, speeds, start):
    gains = controller.at(speeds[:-1])["feedback"][:, :5]
    values = command_error_components(estimate[:-1], truth[:-1], gains)
    summary = command_error_summary(values, start)
    row = {
        f"command_{k}_rms_deg": float(np.rad2deg(v))
        for k, v in summary["component_rms_rad"].items()
    }
    row.update(
        command_total_rms_deg=float(np.rad2deg(summary["total_rms_rad"])),
        command_joint_force_rms_deg=float(np.rad2deg(summary["joint_force_rms_rad"])),
    )
    row.update(
        {
            f"moment_{i}{j}_rad2": summary["second_moment_rad2"][i][j]
            for i in range(5)
            for j in range(5)
        }
    )
    return row


def execute_job(job):
    seed, scale, mode = job
    prefix = f"experiment_{mode}_seed{seed}_Q{scale:g}"
    complete = OUT / f"{prefix}_complete.json"
    sources = source_hashes(mode)
    if complete.exists():
        saved = json.loads(complete.read_text())
        if saved["source_sha256"] != sources:
            raise RuntimeError("Changed execution sources")
        for path, expected in saved["output_sha256"].items():
            if sha(OUT / path) != expected:
                raise RuntimeError("Changed execution outputs")
        return prefix
    cfg, inherited, libraries, nominal, learned, true = prepare(seed, scale, mode == "10hq")
    h = inherited["10h"] | {"duration_s": cfg["duration_s"]}
    scenarios_by_name = scenarios(h)
    cases = (
        P_CASES
        if mode == "10hp"
        else diagnostic_cases(json.loads((ROOT / "code/config/experiment_10hq.json").read_text()))
    )
    limit, beta_limit = np.deg2rad([cfg["maximum_command_deg"], cfg["maximum_beta_deg"]])
    start = int(cfg["state_burnin_s"] / cfg["sample_time"])
    data_dir = ROOT / f"results/data/experiment_{mode}"
    data_dir.mkdir(parents=True, exist_ok=True)
    rows, radius_rows, replays = [], [], []
    speeds = np.linspace(10, 30, cfg["frozen_speed_audit_points"])
    for lib in libraries:
        fold = lib["donor_fold"]
        for position, choice in zip(lib["recipient_clients"], lib["recipient_choices"]):
            matrices, oracle = true[position]
            designs = {"nominal": nominal, "learned": learned[fold, choice], "oracle": oracle}
            ta = schedule_values(np.asarray([m[0] for m in matrices]), h["speed_grid"], speeds)
            tb = schedule_values(
                np.asarray([m[1].reshape(5) for m in matrices]), h["speed_grid"], speeds
            )
            for case in cases:
                controller = designs[case.controller]
                observer = designs["learned" if case.state_source == "learned_kf" else "nominal"]
                if controller is None or observer is None:
                    continue
                obs, control = observer.at(speeds), controller.at(speeds)
                radii = [
                    feedback_radius(
                        ta[j],
                        tb[j],
                        obs["a"][j],
                        obs["b"][j],
                        obs["observer"][j],
                        control["feedback"][j],
                        cfg["sample_time"],
                        case.state_source == "true_state",
                        correction=case.correction,
                    )
                    for j in range(len(speeds))
                ]
                radius_rows.append(
                    {
                        "seed": seed,
                        "process_noise_scale": scale,
                        "donor_fold": fold,
                        "client_position": position,
                        "case": case.name,
                        "maximum_feedback_radius": max(radii),
                        "unstable_grid_points": sum(v >= 1 for v in radii),
                    }
                )
            for scenario_index, (scenario_name, scenario) in enumerate(scenarios_by_name.items()):
                noise = sensor_noise(
                    seed, position, scenario_index, len(scenario["time"]), cfg, inherited["10ha"]
                )
                for case in cases:
                    controller = designs[case.controller]
                    observer = designs[
                        "learned" if case.state_source == "learned_kf" else "nominal"
                    ]
                    metadata = {
                        "seed": seed,
                        "process_noise_scale": scale,
                        "donor_fold": fold,
                        "client_position": position,
                        "scenario": scenario_name,
                        "case": case.name,
                        "controller": case.controller,
                        "state_source": case.state_source,
                        "correction": case.correction,
                        "selected_plant": choice,
                    }
                    if controller is None or observer is None:
                        rows.append(dict(metadata, finite=False, design_failed=True))
                        continue
                    record = rollout(
                        matrices,
                        h["speed_grid"],
                        observer,
                        controller,
                        scenario,
                        noise,
                        cfg["sample_time"],
                        limit,
                        case.correction,
                    )
                    path = (
                        data_dir
                        / f"{prefix}_fold{fold}_client{position}_{scenario_name}_{case.name}.npz"
                    )
                    np.savez_compressed(path, **record, noise=noise, **scenario)
                    row = dict(
                        metadata,
                        **metrics(record, scenario, start, limit, beta_limit),
                        design_failed=False,
                        record_path=str(path.relative_to(ROOT)),
                        record_sha256=sha(path),
                    )
                    if mode == "10hq":
                        row.update(
                            steering_diagnostics(
                                record["supplied"],
                                record["truth"],
                                controller,
                                scenario["speed"],
                                start,
                            )
                        )
                        if case.name in [BASELINE_CASE, BOTH_CASE]:
                            for observer_name in ["nominal", "learned", "oracle"]:
                                replay = replay_observer(
                                    record, designs[observer_name], scenario["speed"]
                                )
                                replay_record = dict(record, estimate=replay)
                                replays.append(
                                    dict(
                                        metadata,
                                        replay_observer=observer_name,
                                        **metrics(
                                            replay_record, scenario, start, limit, beta_limit
                                        ),
                                        **steering_diagnostics(
                                            replay,
                                            record["truth"],
                                            controller,
                                            scenario["speed"],
                                            start,
                                        ),
                                    )
                                )
                    rows.append(row)
    frames = {"metrics": pd.DataFrame(rows), "frozen_feedback": pd.DataFrame(radius_rows)}
    if mode == "10hq":
        frames["common_input"] = pd.DataFrame(replays)
    for name, frame in frames.items():
        frame.to_csv(OUT / f"{prefix}_{name}.csv", index=False)
    complete.write_text(
        json.dumps(
            {
                "source_sha256": sources,
                "output_sha256": {
                    f"{prefix}_{name}.csv": sha(OUT / f"{prefix}_{name}.csv") for name in frames
                },
                "rollouts": len(rows),
                "finite_rollouts": sum(bool(row["finite"]) for row in rows),
                "reconstruction": True,
                "confirmation_run": False,
                "revision": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], text=True
                ).strip(),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"{mode} complete seed={seed} Q={scale:g} rollouts={len(rows)}", flush=True)
    return prefix


def summarize(mode, names):
    frame = pd.concat(
        [pd.read_csv(OUT / f"{name}_metrics.csv") for name in names], ignore_index=True
    )
    frame.to_csv(OUT / f"experiment_{mode}_metrics.csv", index=False)
    keys = ["seed", "process_noise_scale", "scenario", "case", "client_position"]
    measures = [
        c
        for c in frame.select_dtypes(include=["number"]).columns
        if c not in keys + ["donor_fold", "selected_plant"]
    ]
    recipients = frame.groupby(keys)[measures].mean().reset_index()
    recipients.to_csv(OUT / f"experiment_{mode}_recipients.csv", index=False)
    fleets = recipients.groupby(keys[:-1])[measures].mean().reset_index()
    baseline = fleets[fleets.case == BASELINE_CASE][
        ["seed", "process_noise_scale", "scenario", "tracking_rmse_deg_s"]
    ].rename(columns={"tracking_rmse_deg_s": "nominal_tracking_rmse_deg_s"})
    fleets = fleets.merge(
        baseline, on=["seed", "process_noise_scale", "scenario"], validate="many_to_one"
    )
    fleets["tracking_gain_pct"] = 100 * (
        1 - fleets.tracking_rmse_deg_s / fleets.nominal_tracking_rmse_deg_s
    )
    fleets.to_csv(OUT / f"experiment_{mode}_fleets.csv", index=False)
    summary = (
        fleets.groupby(["process_noise_scale", "scenario", "case"])
        .agg(
            tracking_rmse_deg_s=("tracking_rmse_deg_s", "mean"),
            tracking_gain_pct=("tracking_gain_pct", "mean"),
            fleets_improved=("tracking_gain_pct", lambda s: int((s > 0).sum())),
            beta_rmse_deg=("beta_rmse_deg", "mean"),
            front_force_rmse=("front_force_rmse", "mean"),
            rear_force_rmse=("rear_force_rmse", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(OUT / f"experiment_{mode}_summary.csv", index=False)
    tails = (
        recipients.groupby(["process_noise_scale", "scenario", "case"])
        .tracking_rmse_deg_s.agg(median="median", p90=lambda s: s.quantile(0.9), maximum="max")
        .reset_index()
    )
    tails.to_csv(OUT / f"experiment_{mode}_tails.csv", index=False)
    outputs = [
        f"experiment_{mode}_{suffix}.csv"
        for suffix in ["metrics", "recipients", "fleets", "summary", "tails"]
    ]
    if mode == "10hq":
        replay = pd.concat(
            [pd.read_csv(OUT / f"{name}_common_input.csv") for name in names], ignore_index=True
        )
        replay.to_csv(OUT / f"experiment_{mode}_common_input.csv", index=False)
        outputs.append(f"experiment_{mode}_common_input.csv")
    (OUT / f"experiment_{mode}_complete.json").write_text(
        json.dumps(
            {
                "source_sha256": source_hashes(mode),
                "output_sha256": {p: sha(OUT / p) for p in outputs},
                "job_manifest_sha256": {name: sha(OUT / f"{name}_complete.json") for name in names},
                "rollouts": len(frame),
                "finite_rollouts": int(frame.finite.sum()),
                "design_failures": int(frame.design_failed.sum()),
                "reconstruction": True,
                "confirmation_run": False,
            },
            indent=2,
        )
        + "\n"
    )


def run(mode="10hp", workers=4):
    verify_learned_inputs()
    cfg, _, _ = configuration()
    jobs = [
        (seed, scale, mode) for seed in cfg["development_seeds"] for scale in cfg["fitting_panels"]
    ]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        names = list(pool.map(execute_job, jobs))
    summarize(mode, names)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    run(workers=parser.parse_args().workers)
