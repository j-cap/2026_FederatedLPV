"""Independently audit frozen Q paths, observer replays and command projections."""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from experiment_10hk_joint_client_mixture import OUT, ROOT, sha
from experiment_10hp_output_feedback import configuration, prepare
from experiment_10hp_result_audit import independent_schedule
from experiment_10hp_result_audit import run as audit_paths

from federated_lpv.innovation_likelihood import C

NAMES = ("beta", "yaw_rate", "front_force_per_mass", "rear_force_per_mass", "applied_steering")
KEYS = ["seed", "process_noise_scale", "donor_fold", "client_position", "scenario", "case"]


def projection(estimate, truth, gain, start):
    values = -(estimate[:-1] - truth[:-1]) * gain[:, :5]
    values = values[start:]
    moments = values.T @ values / len(values)
    total = values.sum(axis=1)
    result = {
        f"command_{name}_rms_deg": float(np.rad2deg(np.sqrt(np.mean(values[:, i] ** 2))))
        for i, name in enumerate(NAMES)
    }
    result.update(
        command_total_rms_deg=float(np.rad2deg(np.sqrt(np.mean(total**2)))),
        command_joint_force_rms_deg=float(
            np.rad2deg(np.sqrt(np.mean((values[:, 2] + values[:, 3]) ** 2)))
        ),
    )
    result.update({f"moment_{i}{j}_rad2": moments[i, j] for i in range(5) for j in range(5)})
    return result, float(abs(np.mean(total**2) - moments.sum()))


def replay(record, observer):
    a, b, gain = (
        independent_schedule(getattr(observer, key), observer.grid, record["speed"])
        for key in ["a", "b", "observer"]
    )
    estimate = np.empty_like(record["truth"])
    estimate[0] = gain[0] @ record["measurement"][0]
    for k, command in enumerate(record["command"]):
        prediction = a[k] @ estimate[k] + b[k] * command
        estimate[k + 1] = prediction + gain[k] @ (
            record["measurement"][k + 1] - C @ prediction
        )
    return estimate


def audit_job(job):
    seed, scale = job
    cfg, _, libraries, nominal, learned, true = prepare(seed, scale, True)
    prefix = f"experiment_10hq_seed{seed}_Q{scale:g}"
    closed = pd.read_csv(OUT / f"{prefix}_metrics.csv")
    common = pd.read_csv(OUT / f"{prefix}_common_input.csv")
    choices = {
        lib["donor_fold"]: dict(zip(lib["recipient_clients"], lib["recipient_choices"]))
        for lib in libraries
    }
    start = int(cfg["state_burnin_s"] / cfg["sample_time"])
    rows = []
    for row in closed.itertuples():
        assert not row.design_failed and row.finite
        assert sha(ROOT / row.record_path) == row.record_sha256
        with np.load(ROOT / row.record_path) as saved:
            record = dict(saved)
        designs = {
            "nominal": nominal,
            "learned": learned[row.donor_fold, choices[row.donor_fold][row.client_position]],
            "oracle": true[row.client_position][1],
        }
        controller = designs[row.controller]
        gain = independent_schedule(controller.feedback, controller.grid, record["speed"][:-1])
        scored, moment_error = projection(record["supplied"], record["truth"], gain, start)
        pref = independent_schedule(controller.prefilter, controller.grid, record["speed"][:-1])
        truth_command = (
            -np.einsum("ni,ni->n", gain[:, :5], record["truth"][:-1])
            - gain[:, 5] * record["integral"][:-1]
            + pref * record["reference"][:-1]
        )
        projected = np.sum(-(record["supplied"][:-1] - record["truth"][:-1]) * gain[:, :5], axis=1)
        identity_error = float(np.max(abs(record["raw_command"] - truth_command - projected)))
        metadata = {key: getattr(row, key) for key in KEYS}
        rows.append(dict(
            metadata, kind="closed_loop", replay_observer="", moment_error=moment_error,
            command_identity_error=identity_error,
            metric_error=max(abs(getattr(row, key) - value) for key, value in scored.items()),
        ))
        selected = common
        for key in ["donor_fold", "client_position", "scenario", "case"]:
            selected = selected[selected[key] == getattr(row, key)]
        for item in selected.itertuples():
            estimate = replay(record, designs[item.replay_observer])
            scored, moment_error = projection(estimate, record["truth"], gain, start)
            rmse = np.sqrt(np.mean((estimate[start:] - record["truth"][start:]) ** 2, axis=0))
            scored.update(
                beta_rmse_deg=float(np.rad2deg(rmse[0])),
                yaw_rmse_deg_s=float(np.rad2deg(rmse[1])),
                front_force_rmse=rmse[2], rear_force_rmse=rmse[3],
                applied_steer_rmse_deg=float(np.rad2deg(rmse[4])),
            )
            rows.append(dict(
                metadata, kind="common_input", replay_observer=item.replay_observer,
                moment_error=moment_error, command_identity_error=0.0,
                metric_error=max(abs(getattr(item, key) - value) for key, value in scored.items()),
            ))
    print(f"Q projection audited seed={seed} Q={scale:g} rows={len(rows)}", flush=True)
    return rows


def run(workers=4):
    audit_paths("10hq", workers)
    cfg, _, _ = configuration()
    jobs = [(seed, scale) for seed in cfg["development_seeds"] for scale in cfg["fitting_panels"]]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(audit_job, jobs))
    frame = pd.DataFrame([row for job in rows for row in job])
    assert (frame.kind == "closed_loop").sum() == 4800
    assert (frame.kind == "common_input").sum() == 2400
    assert frame.metric_error.max() <= 1e-10
    assert frame.command_identity_error.max() <= 1e-10
    assert frame.moment_error.max() <= 1e-10
    table = OUT / "experiment_10hq_projection_audit.csv"
    frame.to_csv(table, index=False)
    result = {
        "all_checks_pass": True, "records_audited": len(frame),
        "closed_loop_records": 4800, "common_input_records": 2400,
        "maximum_metric_error": float(frame.metric_error.max()),
        "maximum_command_identity_error": float(frame.command_identity_error.max()),
        "maximum_second_moment_error": float(frame.moment_error.max()),
        "audit_source_sha256": sha(ROOT / "code/experiments/experiment_10hq_result_audit.py"),
        "audit_table_sha256": sha(table),
        "result_manifest_sha256": sha(OUT / "experiment_10hq_complete.json"),
    }
    (OUT / "experiment_10hq_projection_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    run(parser.parse_args().workers)
