"""Verify every P case with the Q design setup before executing new interventions."""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from experiment_10h_higher_order_lpv_gate import scenarios
from experiment_10hk_joint_client_mixture import OUT, ROOT, sha
from experiment_10hp_output_feedback import configuration, prepare, sensor_noise
from experiment_10hq_preflight import parent_preflight

from federated_lpv.output_feedback_control import rollout


def check_job(job):
    seed, scale = job
    cfg, inherited, libraries, nominal, learned, true = prepare(seed, scale, True)
    choices = {
        lib["donor_fold"]: dict(zip(lib["recipient_clients"], lib["recipient_choices"]))
        for lib in libraries
    }
    scene = scenarios(inherited["10h"] | {"duration_s": cfg["duration_s"]})
    frame = pd.read_csv(OUT / f"experiment_10hp_seed{seed}_Q{scale:g}_metrics.csv")
    rows = []
    for row in frame.itertuples():
        if row.design_failed:
            raise RuntimeError("A failed P design cannot be used as the overlap reference")
        path = ROOT / row.record_path
        if sha(path) != row.record_sha256:
            raise RuntimeError(f"Changed parent trajectory: {path}")
        designs = {
            "nominal": nominal,
            "learned": learned[row.donor_fold, choices[row.donor_fold][row.client_position]],
        }
        scenario = scene[row.scenario]
        noise = sensor_noise(
            seed, row.client_position, list(scene).index(row.scenario),
            len(scenario["time"]), cfg, inherited["10ha"],
        )
        matrices, _ = true[row.client_position]
        candidate = rollout(
            matrices, inherited["10h"]["speed_grid"],
            designs["learned" if row.state_source == "learned_kf" else "nominal"],
            designs[row.controller], scenario, noise, cfg["sample_time"],
            np.deg2rad(cfg["maximum_command_deg"]), row.correction,
        ) | {"noise": noise} | scenario
        with np.load(path) as saved:
            equal = set(candidate) == set(saved.files) and all(
                np.array_equal(candidate[key], saved[key]) for key in candidate
            )
            error = max(float(np.max(abs(candidate[key] - saved[key]))) for key in candidate)
        rows.append({
            "seed": seed, "process_noise_scale": scale, "donor_fold": row.donor_fold,
            "client_position": row.client_position, "scenario": row.scenario,
            "case": row.case, "array_equal": equal, "maximum_array_error": error,
        })
    return rows


def run(workers=4):
    config_path = ROOT / "code/config/experiment_10hq.json"
    qcfg = json.loads(config_path.read_text())
    status = parent_preflight(ROOT, qcfg)
    if status["status"] != "ready_for_frozen_diagnostic":
        raise RuntimeError(f"Parent provenance blocked: {status}")
    cfg, _, _ = configuration()
    jobs = [(seed, scale) for seed in cfg["development_seeds"] for scale in cfg["fitting_panels"]]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        records = list(pool.map(check_job, jobs))
    frame = pd.DataFrame([row for rows in records for row in rows])
    table = OUT / "experiment_10hq_parent_overlap_precheck.csv"
    frame.to_csv(table, index=False)
    result = {
        "all_checks_pass": bool(len(frame) == 1600 and frame.array_equal.all()),
        "records_checked": len(frame), "maximum_array_error": float(frame.maximum_array_error.max()),
        "required_parent_revision": qcfg["required_parent_revision"],
        "source_sha256": sha(ROOT / "code/experiments/experiment_10hq_parent_overlap.py"),
        "output_sha256": sha(table), "reconstruction": True, "confirmation_run": False,
    }
    (OUT / "experiment_10hq_parent_overlap_precheck.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2), flush=True)
    if not result["all_checks_pass"]:
        raise RuntimeError("Parent cases differ under the Q design setup")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    run(parser.parse_args().workers)
