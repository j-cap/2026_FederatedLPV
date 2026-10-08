"""Independent saved-equation and full-path audit of P/Q feedback records."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from experiment_10hk_joint_client_mixture import OUT, ROOT, sha
from experiment_10hp_output_feedback import P_CASES, configuration, prepare, source_hashes

from federated_lpv.feedback_diagnostics import diagnostic_cases
from federated_lpv.innovation_likelihood import C

CORRECTIONS = {
    "none": [],
    "beta": [0],
    "forces": [2, 3],
    "beta_forces": [0, 2, 3],
    "all": list(range(5)),
}


def independent_schedule(values, grid, speeds):
    values = np.asarray(values)
    flat = values.reshape(len(grid), -1)
    result = np.column_stack([np.interp(speeds, grid, flat[:, j]) for j in range(flat.shape[1])])
    return result.reshape((len(speeds),) + values.shape[1:])


def audit_record(record, matrices, grid, observer, controller, case, cfg):
    x, estimate, measured, eta, command = (
        record[k] for k in ["truth", "estimate", "measurement", "integral", "command"]
    )
    speed, reference = record["speed"], record["reference"]
    dt, limit = cfg["sample_time"], np.deg2rad(cfg["maximum_command_deg"])
    ta = independent_schedule([m[0] for m in matrices], grid, speed[:-1])
    tb = independent_schedule([m[1].reshape(5) for m in matrices], grid, speed[:-1])
    a, b, l = (
        independent_schedule(getattr(observer, key), observer.grid, speed)
        for key in ["a", "b", "observer"]
    )
    gain, pref = (
        independent_schedule(getattr(controller, key), controller.grid, speed)
        for key in ["feedback", "prefilter"]
    )
    supplied = estimate.copy()
    supplied[:, CORRECTIONS[case.correction]] = x[:, CORRECTIONS[case.correction]]
    raw = (
        -np.einsum("ni,ni->n", gain[:-1], np.column_stack([supplied[:-1], eta[:-1]]))
        + pref[:-1] * reference[:-1]
    )
    predicted = np.einsum("nij,nj->ni", a[:-1], estimate[:-1]) + b[:-1] * command[:, None]
    increment = dt * (measured[:-1, 0] - reference[:-1])
    increment = np.where((abs(raw) > limit) & (raw * (-gain[:-1, 5] * increment) > 0), 0, increment)
    differences = [
        np.max(abs(record["supplied"] - supplied)),
        np.max(abs(record["raw_command"] - raw)),
        np.max(abs(command - np.clip(raw, -limit, limit))),
        np.max(abs(x[1:] - np.einsum("nij,nj->ni", ta, x[:-1]) - tb * command[:, None])),
        np.max(
            abs(
                estimate[1:]
                - predicted
                - np.einsum("nij,nj->ni", l[:-1], measured[1:] - predicted @ C.T)
            )
        ),
        np.max(abs(measured - x @ C.T - record["noise"])),
        np.max(abs(eta[1:] - eta[:-1] - increment)),
        np.max(abs(x[0])),
        abs(eta[0]),
        np.max(abs(estimate[0] - l[0] @ measured[0])),
    ]
    # Independent path: np.interp and a six-term command dot product intentionally
    # differ in floating-point ordering from the production kernel.
    xr, er = np.zeros_like(x), np.zeros_like(estimate)
    yr, ir = np.zeros_like(measured), np.zeros_like(eta)
    ur, rr = np.zeros_like(command), np.zeros_like(command)
    yr[0] = record["noise"][0]
    er[0] = l[0] @ yr[0]
    for k in range(len(command)):
        feedback = er[k].copy()
        feedback[CORRECTIONS[case.correction]] = xr[k, CORRECTIONS[case.correction]]
        rr[k] = -gain[k] @ np.r_[feedback, ir[k]] + pref[k] * reference[k]
        ur[k] = np.clip(rr[k], -limit, limit)
        increment = dt * (yr[k, 0] - reference[k])
        ir[k + 1] = (
            ir[k]
            if abs(rr[k]) > limit and rr[k] * (-gain[k, 5] * increment) > 0
            else ir[k] + increment
        )
        xr[k + 1] = ta[k] @ xr[k] + tb[k] * ur[k]
        yr[k + 1] = C @ xr[k + 1] + record["noise"][k + 1]
        prediction = a[k] @ er[k] + b[k] * ur[k]
        er[k + 1] = prediction + l[k] @ (yr[k + 1] - C @ prediction)
    path_error = max(
        np.max(abs(x - xr)),
        np.max(abs(estimate - er)),
        np.max(abs(eta - ir)),
        np.max(abs(command - ur)),
        np.max(abs(record["raw_command"] - rr)),
    )
    replay_tracking = float(np.rad2deg(np.sqrt(np.mean((xr[:, 1] - reference) ** 2))))
    tracking = float(np.rad2deg(np.sqrt(np.mean((x[:, 1] - reference) ** 2))))
    start = int(cfg["state_burnin_s"] / dt)
    state_rmse = np.sqrt(np.mean((estimate[start:] - x[start:]) ** 2, axis=0))
    independently = {
        "tracking_rmse_deg_s": tracking,
        "beta_rmse_deg": np.rad2deg(state_rmse[0]),
        "yaw_rmse_deg_s": np.rad2deg(state_rmse[1]),
        "front_force_rmse": state_rmse[2],
        "rear_force_rmse": state_rmse[3],
        "applied_steer_rmse_deg": np.rad2deg(state_rmse[4]),
        "command_rms_deg": np.rad2deg(np.sqrt(np.mean(command**2))),
        "saturation_samples": int(np.count_nonzero(abs(record["raw_command"]) > limit)),
    }
    return max(differences), path_error, abs(tracking - replay_tracking), independently


def audit_job(job):
    seed, scale, mode = job
    cfg, inherited, libraries, nominal, learned, true = prepare(seed, scale, mode == "10hq")
    cases = (
        P_CASES
        if mode == "10hp"
        else diagnostic_cases(json.loads((ROOT / "code/config/experiment_10hq.json").read_text()))
    )
    case_lookup = {case.name: case for case in cases}
    prefix = f"experiment_{mode}_seed{seed}_Q{scale:g}"
    manifest = json.loads((OUT / f"{prefix}_complete.json").read_text())
    assert manifest["source_sha256"] == source_hashes(mode)
    for path, expected in manifest["output_sha256"].items():
        assert sha(OUT / path) == expected
    frame = pd.read_csv(OUT / f"{prefix}_metrics.csv")
    rows = []
    choices = {
        lib["donor_fold"]: dict(zip(lib["recipient_clients"], lib["recipient_choices"]))
        for lib in libraries
    }
    for row in frame.itertuples():
        if row.design_failed:
            continue
        path = ROOT / row.record_path
        assert sha(path) == row.record_sha256
        with np.load(path) as saved:
            record = dict(saved)
        case = case_lookup[row.case]
        matrices, oracle = true[row.client_position]
        designs = {
            "nominal": nominal,
            "learned": learned[row.donor_fold, choices[row.donor_fold][row.client_position]],
            "oracle": oracle,
        }
        observer = designs["learned" if case.state_source == "learned_kf" else "nominal"]
        step_error, path_error, tracking_delta, independently = audit_record(
            record,
            matrices,
            inherited["10h"]["speed_grid"],
            observer,
            designs[case.controller],
            case,
            cfg,
        )
        metric_error = max(
            abs(getattr(row, key) - float(value)) for key, value in independently.items()
        )
        overlap_error = 0.0
        if mode == "10hq" and case.name in {item.name for item in P_CASES}:
            p_path = ROOT / row.record_path.replace("experiment_10hq", "experiment_10hp")
            with np.load(p_path) as parent:
                overlap_error = max(np.max(abs(record[key] - parent[key])) for key in record)
        rows.append(
            {
                "seed": seed,
                "process_noise_scale": scale,
                "donor_fold": row.donor_fold,
                "client_position": row.client_position,
                "case": row.case,
                "scenario": row.scenario,
                "saved_step_max_error": step_error,
                "metric_max_error": metric_error,
                "full_path_max_error": path_error,
                "full_path_tracking_delta_deg_s": tracking_delta,
                "parent_overlap_max_error": overlap_error,
                "strict_full_path_pass": bool(path_error <= cfg["full_path_tolerance"]),
            }
        )
    print(f"{mode} audited seed={seed} Q={scale:g} records={len(rows)}", flush=True)
    return rows


def run(mode="10hp", workers=4):
    cfg, _, _ = configuration()
    manifest = json.loads((OUT / f"experiment_{mode}_complete.json").read_text())
    assert manifest["source_sha256"] == source_hashes(mode)
    for path, expected in manifest["output_sha256"].items():
        assert sha(OUT / path) == expected
    for name, expected in manifest["job_manifest_sha256"].items():
        assert sha(OUT / f"{name}_complete.json") == expected
    jobs = [
        (seed, scale, mode) for seed in cfg["development_seeds"] for scale in cfg["fitting_panels"]
    ]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        all_rows = list(pool.map(audit_job, jobs))
    frame = pd.DataFrame([row for rows in all_rows for row in rows])
    frame.to_csv(OUT / f"experiment_{mode}_replay_audit.csv", index=False)
    integrity = bool(
        (frame.saved_step_max_error <= cfg["saved_step_tolerance"]).all()
        and (frame.metric_max_error <= 1e-10).all()
        and (frame.parent_overlap_max_error == 0).all()
    )
    audit = {
        "reconstruction": True,
        "integrity_checks_pass": integrity,
        "records_audited": len(frame),
        "maximum_saved_step_error": float(frame.saved_step_max_error.max()),
        "maximum_metric_error": float(frame.metric_max_error.max()),
        "strict_full_path_passes": int(frame.strict_full_path_pass.sum()),
        "strict_full_path_sensitive": int((~frame.strict_full_path_pass).sum()),
        "maximum_full_path_error": float(frame.full_path_max_error.max()),
        "maximum_full_path_tracking_delta_deg_s": float(frame.full_path_tracking_delta_deg_s.max()),
        "maximum_parent_overlap_error": float(frame.parent_overlap_max_error.max()),
        "audit_source_sha256": sha(ROOT / "code/experiments/experiment_10hp_result_audit.py"),
        "result_manifest_sha256": sha(OUT / f"experiment_{mode}_complete.json"),
        "audit_table_sha256": sha(OUT / f"experiment_{mode}_replay_audit.csv"),
    }
    (OUT / f"experiment_{mode}_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))
    if not integrity:
        raise RuntimeError("Saved-step, metric or parent overlap audit failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["10hp", "10hq"], default="10hp")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    run(args.mode, args.workers)
