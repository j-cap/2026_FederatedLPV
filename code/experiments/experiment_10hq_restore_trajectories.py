"""Restore ignored P/Q records without changing frozen manifests or tables."""

import argparse
import json
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from experiment_10h_higher_order_lpv_gate import scenarios
from experiment_10hk_joint_client_mixture import OUT, ROOT, sha
from experiment_10hp_output_feedback import (
    configuration,
    prepare,
    sensor_noise,
    verify_learned_inputs,
)

from federated_lpv.output_feedback_control import rollout


def restore_job(job):
    seed, scale, mode, verify = job
    cfg, inherited, libraries, nominal, learned, true = prepare(seed, scale, mode == "10hq")
    frame = pd.read_csv(OUT / f"experiment_{mode}_seed{seed}_Q{scale:g}_metrics.csv")
    if verify:
        frame = pd.concat(
            [rows.iloc[[0, -1]] for _, rows in frame.groupby("case")], ignore_index=True
        )
    choices = {
        lib["donor_fold"]: dict(zip(lib["recipient_clients"], lib["recipient_choices"]))
        for lib in libraries
    }
    h = inherited["10h"] | {"duration_s": cfg["duration_s"]}
    scene = scenarios(h)
    restored = 0
    with tempfile.TemporaryDirectory(prefix="10hq-regeneration-") as temporary:
        for row in frame.itertuples():
            if row.design_failed:
                continue
            destination = ROOT / row.record_path
            if not verify and destination.is_file():
                if sha(destination) != row.record_sha256:
                    raise RuntimeError(f"Corrupt cached trajectory: {destination}")
                continue
            matrices, oracle = true[row.client_position]
            designs = {
                "nominal": nominal,
                "learned": learned[row.donor_fold, choices[row.donor_fold][row.client_position]],
                "oracle": oracle,
            }
            observer = designs["learned" if row.state_source == "learned_kf" else "nominal"]
            scenario = scene[row.scenario]
            noise = sensor_noise(
                seed,
                row.client_position,
                list(scene).index(row.scenario),
                len(scenario["time"]),
                cfg,
                inherited["10ha"],
            )
            record = rollout(
                matrices,
                h["speed_grid"],
                observer,
                designs[row.controller],
                scenario,
                noise,
                cfg["sample_time"],
                np.deg2rad(cfg["maximum_command_deg"]),
                row.correction,
            )
            generated = Path(temporary) / "record.npz"
            np.savez_compressed(generated, **record, noise=noise, **scenario)
            if sha(generated) != row.record_sha256:
                raise RuntimeError(
                    "Regeneration differs from frozen bytes; use the recorded numerical environment"
                )
            if not verify:
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(generated.read_bytes())
            restored += 1
    return {
        "seed": seed,
        "process_noise_scale": scale,
        "experiment": mode,
        "verified_or_restored": restored,
    }


def run(workers=4, verify=False, experiment="both"):
    verify_learned_inputs()
    cfg, _, _ = configuration()
    modes = ["10hp", "10hq"] if experiment == "both" else [experiment]
    jobs = [
        (seed, scale, mode, verify)
        for seed in cfg["development_seeds"]
        for scale in cfg["fitting_panels"]
        for mode in modes
    ]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(restore_job, jobs))
    result = {
        "all_checks_pass": True,
        "sampled_regeneration": verify,
        "reconstruction": True,
        "verified_or_restored": sum(row["verified_or_restored"] for row in results),
        "jobs": results,
        "source_sha256": sha(ROOT / "code/experiments/experiment_10hq_restore_trajectories.py"),
    }
    if verify:
        (OUT / "experiment_10hq_regeneration_audit.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
    else:
        (OUT / f"experiment_10hq_restore_{experiment}.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--verify-regeneration", action="store_true")
    parser.add_argument("--experiment", choices=["10hp", "10hq", "both"], default="both")
    args = parser.parse_args()
    run(args.workers, args.verify_regeneration, args.experiment)
