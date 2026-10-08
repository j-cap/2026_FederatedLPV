"""Reconstruct 10H-M from verified actual 10H-L fold-trained models."""

from __future__ import annotations

import json
import subprocess

import numpy as np
import pandas as pd
from experiment_10ha_tire_force_observer import measurement_covariance
from experiment_10hh_single_model_repair import dataset_hash, measured_datasets
from experiment_10hi_physically_coupled_fit import make_coordinates
from experiment_10hk_joint_client_mixture import OUT, ROOT, sha
from experiment_10hl_validated_model_selection import (
    context_models,
    job_prefix,
    load_configuration,
    model_parameters,
    numerical_eligible,
)

from federated_lpv.causal_model_selection import (
    candidate_forecasts,
    deploy_rows,
    forecast_choices,
    prefix_scores,
)

PREFIX = "experiment_10hm"


def verify_manifest(path):
    manifest = json.loads(path.read_text())
    for name, expected in manifest["source_sha256"].items():
        if sha(ROOT / name) != expected:
            raise RuntimeError(f"Changed historical source: {name}")
    for name, expected in manifest["output_sha256"].items():
        if sha(OUT / name) != expected:
            raise RuntimeError(f"Changed historical output: {name}")
    return manifest


def choose_library(scores, eligible, tolerance=1e-12):
    candidates = []
    for (restart, margin), rows in scores.groupby(["restart", "margin"]):
        if eligible[int(restart)]:
            candidates.append(
                {
                    "restart": int(restart),
                    "margin": float(margin),
                    "score": float(rows.selection_score.mean()),
                }
            )
    if not candidates:
        return None
    best = min(row["score"] for row in candidates)
    return min(
        (row for row in candidates if row["score"] <= best + tolerance),
        key=lambda row: (-row["margin"], row["restart"]),
    )


def runtime(seed, scale):
    cfg, inherited = load_configuration()
    if seed not in cfg["development_seeds"] or scale not in cfg["process_noise_scales"]:
        raise ValueError("Only declared development jobs are allowed")
    data, split = measured_datasets(seed, cfg, inherited)
    parent = verify_manifest(OUT / f"{job_prefix(seed, scale)}_complete.json")
    if parent["measured_array_sha256"] != {k: dataset_hash(v) for k, v in data.items()}:
        raise RuntimeError("Historical measured arrays differ")
    positions = split.loc[split.split == "train", "client_position"].to_numpy()
    recipients = split.loc[split.split == "heldout", "client_position"].to_numpy()
    clients = np.unique(positions)
    folds = np.array([np.searchsorted(clients, c) % 2 for c in positions])
    coords = make_coordinates(cfg, inherited["10hf"])
    dt = inherited["10h"]["sample_time"]
    q = np.diag(np.asarray(inherited["10ha"]["base_process_noise_diagonal"]) * scale)
    r = measurement_covariance(inherited["10ha"])
    return cfg, inherited, data, split, positions, recipients, folds, coords, dt, q, r


def evaluate_library(data, positions, parameters, margin, cfg, dt, q, r):
    clients, scores = prefix_scores(
        data,
        positions,
        parameters,
        dt,
        q,
        r,
        cfg["membership_prefix_samples"],
        cfg["calibration_horizons_samples"],
        cfg["calibration_burn_in_samples"],
    )
    choices = forecast_choices(scores, margin)
    _, rows, _, _ = candidate_forecasts(
        data,
        positions,
        parameters,
        dt,
        q,
        r,
        cfg["forecast_horizons_samples"],
        cfg["forecast_burn_in_samples"],
    )
    deployed = deploy_rows(rows, clients, choices)
    baseline = deploy_rows(rows, clients, np.zeros(len(clients), dtype=int))
    return clients, scores, choices, deployed, baseline


def run():
    cfg, _ = load_configuration()
    forecasts, libraries = [], []
    inputs = {}
    for seed in cfg["development_seeds"]:
        for scale in cfg["process_noise_scales"]:
            cfg, _, data, _, pos, recipients, folds, coords, dt, q, r = runtime(seed, scale)
            inner_path = OUT / f"{job_prefix(seed, scale)}_inner_scores.csv"
            inputs[str(inner_path.relative_to(ROOT))] = sha(inner_path)
            inner = pd.read_csv(inner_path)
            for fold in range(2):
                cache = f"{job_prefix(seed, scale)}_fold{fold}"
                path = OUT / f"{cache}_complete.json"
                parent = verify_manifest(path)
                inputs[str(path.relative_to(ROOT))] = sha(path)
                frames = {"fits": pd.read_csv(OUT / f"{cache}_fits.csv")}
                global_eta, models = context_models(frames, cfg)
                fits = frames["fits"].query("model == 'K2'")
                eligible = {
                    int(row.restart): numerical_eligible(row, cfg) for _, row in fits.iterrows()
                }
                selected = choose_library(inner[inner.fold == fold], eligible)
                values = (
                    models[selected["restart"]] if selected else np.r_[global_eta, global_eta, 0.5]
                )
                parameters = model_parameters(global_eta, values, coords)
                clients, scores, choices, rows, baseline = evaluate_library(
                    data["heldout"],
                    recipients,
                    parameters,
                    selected["margin"] if selected else 0.999999,
                    cfg,
                    dt,
                    q,
                    r,
                )
                if selected is None:
                    choices[:] = 0
                    rows = baseline
                library = {
                    "seed": seed,
                    "process_noise_scale": scale,
                    "donor_fold": fold,
                    "reconstruction": True,
                    "selection": selected,
                    "eta": [global_eta.tolist(), values[:8].tolist(), values[8:16].tolist()],
                    "log_parameters": [p.tolist() for p in parameters],
                    "mixing_weights": [float(values[16]), float(1 - values[16])],
                    "donor_clients": np.unique(pos[folds != fold]).tolist(),
                    "validation_clients": np.unique(pos[folds == fold]).tolist(),
                    "recipient_clients": clients.tolist(),
                    "recipient_choices": choices.tolist(),
                    "recipient_prefix_scores": scores.tolist(),
                    "parent_manifest_sha256": sha(path),
                    "training_array_sha256": parent["training_array_sha256"],
                }
                libraries.append(library)
                for method, records in [("frozen_library", rows), ("donor_global", baseline)]:
                    forecasts.extend(
                        dict(
                            row,
                            seed=seed,
                            process_noise_scale=scale,
                            donor_fold=fold,
                            method=method,
                        )
                        for row in records
                    )
                print(f"10H-M seed={seed} Q={scale:g} fold={fold} selected={selected}", flush=True)
    (OUT / f"{PREFIX}_libraries.json").write_text(json.dumps(libraries, indent=2) + "\n")
    frame = pd.DataFrame(forecasts)
    frame.to_csv(OUT / f"{PREFIX}_forecasts.csv", index=False)
    paired = (
        frame[frame.client_position >= 0]
        .groupby(["seed", "process_noise_scale", "method", "horizon_samples", "client_position"])
        .sensor_normalized_mse.mean()
        .unstack("method")
    )
    paired["gain_pct"] = 100 * (1 - paired.frozen_library / paired.donor_global)
    paired.reset_index().to_csv(OUT / f"{PREFIX}_paired_clients.csv", index=False)
    summary = paired.groupby(["seed", "process_noise_scale", "horizon_samples"]).agg(
        mean_gain_pct=("gain_pct", "mean"), worst_gain_pct=("gain_pct", "min")
    )
    summary.reset_index().to_csv(OUT / f"{PREFIX}_summary.csv", index=False)
    sources = [
        "code/experiments/experiment_10hm_frozen_libraries.py",
        "docs/experiment_10hm_protocol.md",
        "code/tests/test_experiment_10hm.py",
    ]
    outputs = [
        f"{PREFIX}_{name}"
        for name in ["libraries.json", "forecasts.csv", "paired_clients.csv", "summary.csv"]
    ]
    manifest = {
        "reconstruction": True,
        "confirmation_run": False,
        "libraries": len(libraries),
        "source_sha256": {p: sha(ROOT / p) for p in sources},
        "input_sha256": inputs,
        "output_sha256": {p: sha(OUT / p) for p in outputs},
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    }
    (OUT / f"{PREFIX}_complete.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    run()
