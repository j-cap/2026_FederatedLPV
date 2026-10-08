"""Fresh 10H-N reconstruction: refine and validate actual frozen bundles."""

from __future__ import annotations

import argparse
import json
import subprocess
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from experiment_10hi_physically_coupled_fit import subset
from experiment_10hk_joint_client_mixture import OUT, ROOT, sha
from experiment_10hl_validated_model_selection import job_prefix
from experiment_10hm_frozen_libraries import evaluate_library, runtime
from scipy.optimize import LinearConstraint, minimize

from federated_lpv.causal_model_selection import (
    candidate_forecasts,
    deploy_rows,
    forecast_choices,
    prefix_scores,
)
from federated_lpv.multihorizon_fit import MultihorizonLoss
from federated_lpv.physical_coupled_fit import LOG_MAP, constraint_audit

PREFIX = "experiment_10hn"
SOURCES = [
    "code/experiments/experiment_10hn_multihorizon_libraries.py",
    "code/src/federated_lpv/multihorizon_fit.py",
    "code/tests/test_experiment_10hn.py",
    "docs/experiment_10hn_protocol.md",
    "results/tables/experiment_10hm_complete.json",
    "results/tables/experiment_10hm_libraries.json",
]


def source_hashes():
    return {p: sha(ROOT / p) for p in SOURCES}


def refine(initial, loss, coordinates):
    initial_value = loss.value_gradient(initial)[0]
    divisor = max(initial_value, 1e-12)

    def objective(eta):
        value, grad = loss.value_gradient(eta)
        return value / divisor, grad / divisor

    result = minimize(
        objective,
        initial,
        jac=True,
        method="SLSQP",
        constraints=[
            LinearConstraint(LOG_MAP, coordinates.relative_lower, coordinates.relative_upper)
        ],
        options={"maxiter": 200, "ftol": 1e-10},
    )
    value, gradient = objective(result.x)
    audit = constraint_audit(result.x, gradient, coordinates)
    eligible = bool(
        result.success
        and audit["kkt_residual"] <= 1e-4
        and audit["constraint_violation"] <= 1e-8
        and value <= 1 + 1e-10
    )
    row = {
        "success": bool(result.success),
        "eligible": eligible,
        "iterations": int(result.nit),
        "evaluations": int(result.nfev),
        "message": str(result.message),
        "initial_loss": initial_value,
        "final_loss": value * divisor,
        "kkt_residual": audit["kkt_residual"],
        "constraint_violation": audit["constraint_violation"],
        "proposed_eta": result.x.tolist(),
        "retained_original": not eligible,
    }
    return result.x if eligible else np.asarray(initial), row


def select_bundle(data, positions, bundles, cfg, dt, q, r):
    forecasts, scores, clients = {}, {}, None
    for name, params in bundles.items():
        clients, scores[name] = prefix_scores(
            data, positions, params, dt, q, r, 100, [5, 20, 50], 20
        )
        _, forecasts[name], _, _ = candidate_forecasts(
            data, positions, params, dt, q, r, [1, 5, 20, 50], 100
        )
    baseline = {
        (row["client_position"], row["horizon_samples"]): row["sensor_normalized_mse"]
        for row in forecasts["original"]
        if row["candidate"] == 0
    }
    candidates = []
    for name in bundles:
        for margin in cfg["fallback_minimum_prefix_gain_candidates"]:
            rows = deploy_rows(forecasts[name], clients, forecast_choices(scores[name], margin))
            logs = [
                np.log(
                    max(row["sensor_normalized_mse"], 1e-12)
                    / max(baseline[row["client_position"], row["horizon_samples"]], 1e-12)
                )
                for row in rows
                if row["client_position"] >= 0 and row["horizon_samples"] in [5, 20, 50]
            ]
            candidates.append(
                {"variant": name, "margin": float(margin), "score": float(np.mean(logs))}
            )
    best = min(row["score"] for row in candidates)
    selected = min(
        (row for row in candidates if row["score"] <= best + 1e-12),
        key=lambda row: (row["variant"] != "original", -row["margin"]),
    )
    return selected, candidates


def run_library(lib):
    seed, scale, fold = lib["seed"], lib["process_noise_scale"], lib["donor_fold"]
    name = f"{PREFIX}_seed{seed}_Q{scale:g}_fold{fold}"
    complete = OUT / f"{name}_complete.json"
    sources = source_hashes()
    if complete.exists():
        saved = json.loads(complete.read_text())
        if saved["source_sha256"] != sources:
            raise RuntimeError("Checkpoint sources changed")
        for path, expected in saved["output_sha256"].items():
            if sha(OUT / path) != expected:
                raise RuntimeError("Checkpoint output changed")
        return name
    cfg, _, data, _, pos, recipients, folds, coords, dt, q, r = runtime(seed, scale)
    mask = folds != fold
    train = subset(data["train"], mask)
    memberships = pd.read_csv(OUT / f"{job_prefix(seed, scale)}_fold{fold}_memberships.csv")
    memberships = memberships[memberships.restart == lib["selection"]["restart"]].set_index(
        "client_position"
    )
    refined, audits = [], []
    for candidate, eta in enumerate(lib["eta"]):
        weights = (
            None
            if candidate == 0
            else memberships.loc[pos[mask], f"weight_{candidate - 1}"].to_numpy()
        )
        loss = MultihorizonLoss(train, coords, dt, q, r, [1, 5, 20, 50], 20, weights)
        fitted, audit = refine(np.asarray(eta), loss, coords)
        refined.append(fitted.tolist())
        audits.append(dict(audit, candidate=candidate))
        print(
            f"10H-N {name} plant={candidate} loss={audit['initial_loss']:.5g}->{audit['final_loss']:.5g} "
            f"eligible={audit['eligible']} KKT={audit['kkt_residual']:.2g}",
            flush=True,
        )
    bundles = {
        "original": np.asarray(lib["log_parameters"]),
        "refined": np.asarray([coords.effective(eta) for eta in refined]),
    }
    selected, candidates = select_bundle(
        subset(data["train"], ~mask), pos[~mask], bundles, cfg, dt, q, r
    )
    parameters = bundles[selected["variant"]]
    frozen = {
        **lib,
        "experiment": "10H-N reconstruction",
        "selection": selected,
        "validation_candidates": candidates,
        "refinement_audits": audits,
        "eta": refined if selected["variant"] == "refined" else lib["eta"],
        "log_parameters": parameters.tolist(),
        "source_sha256": sources,
    }
    # Persist the validation-only decision before inspecting outer forecast errors.
    (OUT / f"{name}_frozen.json").write_text(json.dumps(frozen, indent=2) + "\n")
    clients, scores, choices, rows, baseline = evaluate_library(
        data["heldout"], recipients, parameters, selected["margin"], cfg, dt, q, r
    )
    frozen.update(
        recipient_clients=clients.tolist(),
        recipient_choices=choices.tolist(),
        recipient_prefix_scores=scores.tolist(),
    )
    library_path = OUT / f"{name}_library.json"
    library_path.write_text(json.dumps(frozen, indent=2) + "\n")
    frame = pd.DataFrame(
        [
            dict(row, method=method)
            for method, records in [("multihorizon_library", rows), ("donor_global", baseline)]
            for row in records
        ]
    )
    frame.assign(seed=seed, process_noise_scale=scale, donor_fold=fold).to_csv(
        OUT / f"{name}_forecasts.csv", index=False
    )
    output_paths = [
        f"{name}_{suffix}" for suffix in ["frozen.json", "library.json", "forecasts.csv"]
    ]
    complete.write_text(
        json.dumps(
            {
                "source_sha256": sources,
                "output_sha256": {p: sha(OUT / p) for p in output_paths},
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
    return name


def run(workers):
    libraries = json.loads((OUT / "experiment_10hm_libraries.json").read_text())
    with ProcessPoolExecutor(max_workers=workers) as pool:
        names = list(pool.map(run_library, libraries))
    frozen = [json.loads((OUT / f"{name}_library.json").read_text()) for name in names]
    (OUT / f"{PREFIX}_libraries.json").write_text(json.dumps(frozen, indent=2) + "\n")
    frame = pd.concat(
        [pd.read_csv(OUT / f"{name}_forecasts.csv") for name in names], ignore_index=True
    )
    frame.to_csv(OUT / f"{PREFIX}_forecasts.csv", index=False)
    m = pd.read_csv(OUT / "experiment_10hm_forecasts.csv")
    selected = pd.concat(
        [frame[frame.method == "multihorizon_library"], m[m.method == "frozen_library"]]
    )
    paired = (
        selected[selected.client_position >= 0]
        .groupby(["seed", "process_noise_scale", "horizon_samples", "client_position", "method"])
        .sensor_normalized_mse.mean()
        .unstack("method")
    )
    paired["gain_vs_M_pct"] = 100 * (1 - paired.multihorizon_library / paired.frozen_library)
    paired.reset_index().to_csv(OUT / f"{PREFIX}_paired_clients.csv", index=False)
    paired.groupby(["seed", "process_noise_scale", "horizon_samples"]).gain_vs_M_pct.agg(
        ["mean", "min"]
    ).reset_index().to_csv(OUT / f"{PREFIX}_summary.csv", index=False)
    paths = [
        f"{PREFIX}_{suffix}"
        for suffix in ["libraries.json", "forecasts.csv", "paired_clients.csv", "summary.csv"]
    ]
    (OUT / f"{PREFIX}_complete.json").write_text(
        json.dumps(
            {
                "source_sha256": source_hashes(),
                "output_sha256": {p: sha(OUT / p) for p in paths},
                "reconstruction": True,
                "libraries": len(frozen),
                "job_manifests": {name: sha(OUT / f"{name}_complete.json") for name in names},
                "confirmation_run": False,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    run(parser.parse_args().workers)
