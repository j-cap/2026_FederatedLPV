"""10H-L: client-blocked restart selection and causal global/group fallback."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from experiment_10hf_output_identifiability import measurement_covariance
from experiment_10hh_single_model_repair import dataset_hash, measured_datasets
from experiment_10hi_physically_coupled_fit import (
    make_coordinates,
    make_initials,
    subset,
)
from experiment_10hi_physically_coupled_fit import (
    scalar_audit as global_scalar_audit,
)
from experiment_10hk_joint_client_mixture import (
    OUT,
    ROOT,
    archived_baseline,
    residual_stats,
    scalar_audit,
    sha,
)
from experiment_10hk_joint_client_mixture import (
    source_hashes as previous_sources,
)

from federated_lpv.causal_model_selection import (
    candidate_forecasts,
    deploy_rows,
    first_record_prefix,
    forecast_choices,
    prefix_scores,
    select_inner_candidate,
)
from federated_lpv.client_mixture import (
    ClientLikelihood,
    ClientMixture,
    calibrate_memberships,
    mixture_audit,
    optimize_mixture,
    project_split,
)
from federated_lpv.innovation_likelihood import (
    PARAMETER_NAMES,
    InnovationLikelihood,
    information_scale,
    physical_coupling_diagnostic,
)
from federated_lpv.physical_coupled_fit import CoupledLikelihood, constraint_audit, optimize_coupled

PREFIX = "experiment_10hl"
CONFIG = ROOT / "code/config/experiment_10hl.json"
TABLES = (
    "fits",
    "phases",
    "initials",
    "memberships",
    "parameters",
    "calibration",
    "inner_scores",
    "cv_candidates",
    "selection",
    "forecasts",
    "restart_forecasts",
    "residuals",
    "correlations",
    "splits",
)
FIT_TABLES = ("fits", "phases", "initials", "memberships")


def load_configuration():
    own = json.loads(CONFIG.read_text())
    cfg = json.loads((CONFIG.parent / own["base_config"]).read_text()) | own
    if set(cfg["development_seeds"]) & set(cfg["reserved_confirmation_seeds"]):
        raise ValueError("Development and confirmation overlap")
    inherited = {
        name: json.loads((ROOT / f"code/config/experiment_{name}.json").read_text())
        for name in ("10h", "10ha", "10hf")
    }
    return cfg, inherited


def job_prefix(seed, scale):
    return f"{PREFIX}_seed{seed}_Q{scale:g}"


def source_hashes(seed, scale):
    result = previous_sources(seed, scale)
    paths = [
        CONFIG,
        Path(__file__),
        ROOT / "docs/experiment_10hl_protocol.md",
        ROOT / "code/src/federated_lpv/causal_model_selection.py",
        ROOT / "code/tests/test_experiment_10hl.py",
        *[
            OUT / f"experiment_10hk_seed{seed}_Q{scale:g}_{suffix}"
            for suffix in ("complete.json", "restarts.csv", "initials.csv", "runs.csv")
        ],
    ]
    result.update({str(p.relative_to(ROOT)): sha(p) for p in paths})
    return result


def numerical_eligible(row, cfg):
    return bool(
        row["success"]
        and row["kkt_residual"] <= cfg["maximum_kkt_residual"]
        and row["constraint_violation"] <= cfg["maximum_constraint_violation"]
    )


def archived_mixture(seed, scale):
    prefix = f"experiment_10hk_seed{seed}_Q{scale:g}"
    manifest = json.loads((OUT / f"{prefix}_complete.json").read_text())
    for relative, expected in manifest["source_sha256"].items():
        if sha(ROOT / relative) != expected:
            raise RuntimeError(f"Historical execution source changed: {relative}")
    for name in ("restarts", "initials", "runs"):
        path = OUT / f"{prefix}_{name}.csv"
        if sha(path) != manifest["output_sha256"][path.name]:
            raise RuntimeError(f"Historical output changed: {path}")
    return (
        pd.read_csv(OUT / f"{prefix}_restarts.csv"),
        pd.read_csv(OUT / f"{prefix}_initials.csv"),
        pd.read_csv(OUT / f"{prefix}_runs.csv"),
    )


def fit_context(
    seed, scale, context, data, positions, global_archive, cfg, dt, q, r, coordinates, sources
):
    cache = f"{job_prefix(seed, scale)}_{context}"
    completion = OUT / f"{cache}_complete.json"
    if completion.exists():
        manifest = json.loads(completion.read_text())
        if manifest["source_sha256"] != sources or manifest[
            "training_array_sha256"
        ] != dataset_hash(data):
            raise RuntimeError("Context cache source/data mismatch")
        frames = {}
        for name in FIT_TABLES:
            path = OUT / f"{cache}_{name}.csv"
            if sha(path) != manifest["output_sha256"][path.name]:
                raise RuntimeError("Context cache output changed")
            frames[name] = pd.read_csv(path)
        return frames
    pieces = {name: [] for name in FIT_TABLES}
    evaluator = CoupledLikelihood(InnovationLikelihood(data, dt, q, r), coordinates)
    global_rows = []
    if context == "full":
        global_eta, previous = global_archive
        value, gradient = evaluator.value_gradient(global_eta)
        if abs(value - previous["train_objective"]) > 1e-10:
            raise RuntimeError("Global archived objective mismatch")
        audit = constraint_audit(global_eta, gradient, coordinates, cfg["active_bound_tolerance"])
        global_rows.append(
            dict(
                model="K1",
                restart=0,
                objective=value,
                success=previous["all_restart_success"],
                reused=True,
                evaluations=0,
                invalid_evaluations=0,
                **global_scalar_audit(audit),
                **{f"value_{j}": float(x) for j, x in enumerate(global_eta)},
            )
        )
    else:
        global_cfg = cfg | {"restart_count": cfg["global_restart_count"]}
        for restart, initial in enumerate(make_initials(seed, global_cfg, coordinates)):
            eta, result, phases, calls, invalid, audit = optimize_coupled(
                initial, evaluator, coordinates, cfg
            )
            global_rows.append(
                dict(
                    model="K1",
                    restart=restart,
                    objective=evaluator.value(eta),
                    success=bool(result.success),
                    reused=False,
                    evaluations=calls,
                    invalid_evaluations=invalid,
                    **global_scalar_audit(audit),
                    **{f"value_{j}": float(x) for j, x in enumerate(eta)},
                )
            )
            pieces["phases"].append(pd.DataFrame(phases).assign(model="K1", restart=restart))
        good = [row for row in global_rows if numerical_eligible(row, cfg)]
        if not good:
            raise RuntimeError(f"No numerically eligible global fit in {context}")
        best = min(good, key=lambda row: row["objective"])
        global_eta = np.array([best[f"value_{j}"] for j in range(8)])
    pieces["fits"].append(pd.DataFrame(global_rows))
    scaling = information_scale(
        evaluator.evaluate(np.zeros(8), information=True)["information"],
        cfg["information_scale_minimum"],
        cfg["information_scale_maximum"],
    )
    client = ClientLikelihood(data, positions, dt, q, r, coordinates)
    mixture = ClientMixture(client)
    rng = np.random.default_rng(seed + 2_700_000)
    directions = rng.normal(size=(cfg["restart_count"], 8))
    old, old_initials, _ = (
        archived_mixture(seed, scale) if context == "full" else (None, None, None)
    )
    for restart, direction in enumerate(directions):
        initial = project_split(global_eta, direction, cfg["split_log_amplitude"], coordinates)
        pieces["initials"].append(
            pd.DataFrame({"restart": restart, "index": np.arange(17), "value": initial})
        )
        if context == "full" and restart < 3:
            before = (
                old_initials[old_initials.restart == restart].sort_values("index").value.to_numpy()
            )
            if not np.allclose(before, initial, rtol=0, atol=1e-12):
                raise RuntimeError("Reused restart initial split differs")
            archived = old[old.restart == restart].iloc[0]
            fitted = np.array([archived[f"value_{j}"] for j in range(17)])
            value, gradient = mixture.value_gradient(fitted)
            if abs(value - archived.objective) > 1e-10:
                raise RuntimeError("Reused mixture objective differs")
            audit = mixture_audit(fitted, gradient, coordinates, cfg)
            success, calls, invalid = bool(archived.success), 0, 0
            reused = True
        else:
            fitted, result, phases, calls, invalid, audit = optimize_mixture(
                initial, mixture, coordinates, cfg, scaling
            )
            success, reused = bool(result.success), False
            value = mixture.value(fitted)
            pieces["phases"].append(pd.DataFrame(phases).assign(model="K2", restart=restart))
        pieces["fits"].append(
            pd.DataFrame(
                [
                    dict(
                        model="K2",
                        restart=restart,
                        objective=value,
                        success=success,
                        reused=reused,
                        evaluations=calls,
                        invalid_evaluations=invalid,
                        **scalar_audit(audit),
                        **{f"value_{j}": float(x) for j, x in enumerate(fitted)},
                    )
                ]
            )
        )
        weights = mixture.evaluate(fitted)["memberships"]
        pieces["memberships"].append(
            pd.DataFrame(
                {
                    "restart": restart,
                    "client_position": client.clients,
                    "weight_0": weights[:, 0],
                    "weight_1": weights[:, 1],
                    "selected_component": np.argmax(weights, axis=1),
                }
            )
        )
        print(
            f"10H-L seed={seed} Q={scale:g} {context} restart={restart} L={value:.7f} "
            f"KKT={audit['kkt_residual']:.2g} reused={reused}",
            flush=True,
        )
    frames = {
        name: pd.concat(rows, ignore_index=True).assign(context=context)
        for name, rows in pieces.items()
    }
    outputs = {}
    for name, frame in frames.items():
        path = OUT / f"{cache}_{name}.csv"
        frame.to_csv(path, index=False)
        outputs[path.name] = sha(path)
    manifest = {
        "source_sha256": sources,
        "training_array_sha256": dataset_hash(data),
        "output_sha256": outputs,
        "context": context,
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
    }
    completion.write_text(json.dumps(manifest, indent=2) + "\n")
    return frames


def context_models(frames, cfg):
    fits = frames["fits"]
    good = fits[(fits.model == "K1") & fits.apply(lambda row: numerical_eligible(row, cfg), axis=1)]
    selected = good.loc[good.objective.idxmin()]
    global_eta = np.array([selected[f"value_{j}"] for j in range(8)])
    models = {
        int(row.restart): np.array([row[f"value_{j}"] for j in range(17)])
        for _, row in fits[fits.model == "K2"].iterrows()
    }
    return global_eta, models


def model_parameters(global_eta, values, coordinates):
    return [coordinates.effective(eta) for eta in [global_eta, values[:8], values[8:16]]]


def calibration_rows(clients, scores, context, restart):
    return [
        {
            "context": context,
            "restart": restart,
            "client_position": int(client),
            "candidate": candidate,
            "prefix_score": float(score),
        }
        for client, values in zip(clients, scores)
        for candidate, score in enumerate(values)
    ]


def map_choices(data, positions, parameters, values, cfg, dt, q, r):
    clients, prefix = first_record_prefix(data, positions, cfg["membership_prefix_samples"])
    _, weights = calibrate_memberships(
        prefix, clients, parameters[1:], [values[16], 1 - values[16]], dt, q, r
    )
    return 1 + np.argmax(weights, axis=1)


def evaluate_seed(seed, scale):
    cfg, inherited = load_configuration()
    if seed not in cfg["development_seeds"] or scale not in cfg["process_noise_scales"]:
        raise ValueError("Only declared development jobs may execute")
    started = time.perf_counter()
    sources = source_hashes(seed, scale)
    data, split = measured_datasets(seed, cfg, inherited)
    hashes = {name: dataset_hash(value) for name, value in data.items()}
    global_archive = archived_baseline(seed, scale, hashes)
    dt = inherited["10h"]["sample_time"]
    q = np.diag(np.asarray(inherited["10ha"]["base_process_noise_diagonal"]) * scale)
    r = measurement_covariance(inherited["10ha"])
    coordinates = make_coordinates(cfg, inherited["10hf"])
    positions = split.loc[split.split == "train", "client_position"].to_numpy()
    test_positions = split.loc[split.split == "heldout", "client_position"].to_numpy()
    clients = np.unique(positions)
    lookup = {client: rank % cfg["inner_client_fold_count"] for rank, client in enumerate(clients)}
    folds = np.array([lookup[client] for client in positions])
    frames = {name: [] for name in TABLES}
    contexts, inner = {}, []
    for fold in range(cfg["inner_client_fold_count"]):
        context = f"fold{fold}"
        mask = folds == fold
        frames_context = fit_context(
            seed,
            scale,
            context,
            subset(data["train"], ~mask),
            positions[~mask],
            global_archive,
            cfg,
            dt,
            q,
            r,
            coordinates,
            sources,
        )
        contexts[context] = frames_context
        global_eta, models = context_models(frames_context, cfg)
        validation = subset(data["train"], mask)
        for restart, values in models.items():
            parameters = model_parameters(global_eta, values, coordinates)
            val_clients, scores = prefix_scores(
                validation,
                positions[mask],
                parameters,
                dt,
                q,
                r,
                cfg["membership_prefix_samples"],
                cfg["calibration_horizons_samples"],
                cfg["calibration_burn_in_samples"],
            )
            _, predictions, _, _ = candidate_forecasts(
                validation,
                positions[mask],
                parameters,
                dt,
                q,
                r,
                cfg["forecast_horizons_samples"],
                cfg["forecast_burn_in_samples"],
            )
            frames["calibration"].append(
                pd.DataFrame(calibration_rows(val_clients, scores, context, restart))
            )
            reference = {
                (row["client_position"], row["horizon_samples"]): row["sensor_normalized_mse"]
                for row in predictions
                if row["candidate"] == 0
            }
            for margin in cfg["fallback_minimum_prefix_gain_candidates"]:
                choices = forecast_choices(scores, margin)
                deployed = deploy_rows(predictions, val_clients, choices)
                for row in deployed:
                    if (
                        row["client_position"] >= 0
                        and row["horizon_samples"] in cfg["selection_horizons_samples"]
                    ):
                        baseline = reference[row["client_position"], row["horizon_samples"]]
                        inner.append(
                            {
                                "fold": fold,
                                "restart": restart,
                                "margin": margin,
                                "client_position": row["client_position"],
                                "horizon_samples": row["horizon_samples"],
                                "candidate": row["candidate"],
                                "global_error": baseline,
                                "candidate_error": row["sensor_normalized_mse"],
                                "selection_score": float(
                                    np.log(
                                        max(row["sensor_normalized_mse"], 1e-12)
                                        / max(baseline, 1e-12)
                                    )
                                ),
                            }
                        )
    contexts["full"] = fit_context(
        seed,
        scale,
        "full",
        data["train"],
        positions,
        global_archive,
        cfg,
        dt,
        q,
        r,
        coordinates,
        sources,
    )
    for context, pieces in contexts.items():
        for name in FIT_TABLES:
            frames[name].append(pieces[name])
        global_eta, models = context_models(pieces, cfg)
        for restart, values in models.items():
            for candidate, z in enumerate(model_parameters(global_eta, values, coordinates)):
                frames["parameters"].append(
                    pd.DataFrame(
                        {
                            "context": context,
                            "restart": restart,
                            "candidate": candidate,
                            "parameter": PARAMETER_NAMES,
                            "fitted": np.exp(z),
                            "coupling_log_residual": physical_coupling_diagnostic(z)[
                                "log_coupling_residual"
                            ],
                        }
                    )
                )
    eligibility = {}
    for restart in range(cfg["restart_count"]):
        eligibility[restart] = all(
            numerical_eligible(
                pieces["fits"][
                    (pieces["fits"].model == "K2") & (pieces["fits"].restart == restart)
                ].iloc[0],
                cfg,
            )
            for pieces in contexts.values()
        )
    selected, candidates = select_inner_candidate(
        inner, eligibility, cfg["inner_client_fold_count"], cfg["selection_tie_tolerance"]
    )
    global_eta, models = context_models(contexts["full"], cfg)
    full_fits = contexts["full"]["fits"]
    eligible_full = full_fits[
        (full_fits.model == "K2")
        & full_fits.apply(lambda row: numerical_eligible(row, cfg), axis=1)
    ]
    likelihood_winner = int(eligible_full.loc[eligible_full.objective.idxmin()].restart)
    _, _, old_runs = archived_mixture(seed, scale)
    old_winner = int(old_runs.loc[old_runs.component_count == 2, "best_restart"].iloc[0])
    selected_restart = selected["restart"] if selected is not None else likelihood_winner
    margin = selected["margin"] if selected is not None else 0.0
    frames["selection"].append(
        pd.DataFrame(
            [
                {
                    "selected_restart": selected_restart,
                    "selected_margin": margin,
                    "inner_score": selected["score"] if selected is not None else np.nan,
                    "eligible_strategies": sum(eligibility.values()),
                    "validation_selection_available": selected is not None,
                    "likelihood_restart": likelihood_winner,
                    "archived_restart": old_winner,
                }
            ]
        )
    )
    frames["inner_scores"].append(pd.DataFrame(inner))
    frames["cv_candidates"].append(pd.DataFrame(candidates, columns=["restart", "margin", "score"]))
    outer = {}
    for restart, values in models.items():
        parameters = model_parameters(global_eta, values, coordinates)
        test_clients, scores = prefix_scores(
            data["heldout"],
            test_positions,
            parameters,
            dt,
            q,
            r,
            cfg["membership_prefix_samples"],
            cfg["calibration_horizons_samples"],
            cfg["calibration_burn_in_samples"],
        )
        _, predictions, residuals, radii = candidate_forecasts(
            data["heldout"],
            test_positions,
            parameters,
            dt,
            q,
            r,
            cfg["forecast_horizons_samples"],
            cfg["forecast_burn_in_samples"],
        )
        choices = map_choices(data["heldout"], test_positions, parameters, values, cfg, dt, q, r)
        outer[restart] = (test_clients, scores, predictions, residuals, radii, choices)
        frames["calibration"].append(
            pd.DataFrame(calibration_rows(test_clients, scores, "outer", restart))
        )
        frames["restart_forecasts"].append(
            pd.DataFrame(deploy_rows(predictions, test_clients, choices)).assign(restart=restart)
        )
    n = len(test_clients)
    methods = {
        "global": (selected_restart, np.zeros(n, dtype=int)),
        "likelihood_MAP": (likelihood_winner, outer[likelihood_winner][-1]),
        "validated_MAP": (selected_restart, outer[selected_restart][-1]),
        "validated_group": (
            selected_restart,
            forecast_choices(outer[selected_restart][1], fallback=False),
        ),
        "validated_fallback": (
            selected_restart,
            forecast_choices(outer[selected_restart][1], margin),
        ),
        "archived_MAP": (old_winner, outer[old_winner][-1]),
    }
    if selected is None:
        methods["validated_fallback"] = (selected_restart, np.zeros(n, dtype=int))
    for method, (restart, choices) in methods.items():
        clients, scores, predictions, residuals, radii, _ = outer[restart]
        inverse = np.searchsorted(clients, test_positions)
        deployed = deploy_rows(predictions, clients, choices)
        frames["forecasts"].append(pd.DataFrame(deployed).assign(method=method, restart=restart))
        frames["memberships"].append(
            pd.DataFrame(
                {
                    "context": "outer",
                    "restart": restart,
                    "method": method,
                    "client_position": clients,
                    "selected_component": choices - 1,
                    "global_fallback": choices == 0,
                }
            )
        )
        selected_residuals = residuals[choices[inverse], np.arange(len(inverse))]
        stats, correlations = residual_stats(
            selected_residuals,
            data["heldout"].commands[:, cfg["forecast_burn_in_samples"] :],
            cfg["residual_maximum_lag"],
        )
        frames["residuals"].append(
            pd.DataFrame([stats]).assign(
                method=method, maximum_plant_spectral_radius=float(radii[np.unique(choices)].max())
            )
        )
        frames["correlations"].append(pd.DataFrame(correlations).assign(method=method))
    clients, predictions, _, _ = candidate_forecasts(
        data["heldout"],
        test_positions,
        [coordinates.nominal],
        dt,
        q,
        r,
        cfg["forecast_horizons_samples"],
        cfg["forecast_burn_in_samples"],
    )
    frames["forecasts"].append(
        pd.DataFrame(deploy_rows(predictions, clients, np.zeros(len(clients), dtype=int))).assign(
            method="nominal", restart=-1
        )
    )
    frames["splits"].append(
        split.assign(inner_fold=[lookup.get(client, -1) for client in split.client_position])
    )
    outputs = {}
    for name, pieces in frames.items():
        path = OUT / f"{job_prefix(seed, scale)}_{name}.csv"
        pd.concat(pieces, ignore_index=True).assign(seed=seed, process_noise_scale=scale).to_csv(
            path, index=False
        )
        outputs[path.name] = sha(path)
    manifest = {
        "seed": seed,
        "process_noise_scale": scale,
        "source_sha256": sources,
        "measured_array_sha256": hashes,
        "output_sha256": outputs,
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "elapsed_seconds": time.perf_counter() - started,
        "confirmation_run": False,
    }
    (OUT / f"{job_prefix(seed, scale)}_complete.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(f"10H-L selected seed={seed} Q={scale:g} {selected}", flush=True)
    return seed, scale


def summarize():
    cfg, _ = load_configuration()
    pieces = {name: [] for name in TABLES}
    manifests = []
    for seed in cfg["development_seeds"]:
        for scale in cfg["process_noise_scales"]:
            manifest = json.loads((OUT / f"{job_prefix(seed, scale)}_complete.json").read_text())
            if manifest["source_sha256"] != source_hashes(seed, scale):
                raise RuntimeError("Execution source changed")
            for name in TABLES:
                path = OUT / f"{job_prefix(seed, scale)}_{name}.csv"
                if sha(path) != manifest["output_sha256"][path.name]:
                    raise RuntimeError("Execution output changed")
                pieces[name].append(pd.read_csv(path))
            manifests.append(manifest)
    for name, frames in pieces.items():
        pd.concat(frames, ignore_index=True).to_csv(OUT / f"{PREFIX}_{name}.csv", index=False)
    (OUT / f"{PREFIX}_execution_manifest.json").write_text(json.dumps(manifests, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--q-scale", type=float)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    cfg, _ = load_configuration()
    if args.summarize_only:
        summarize()
        return
    seeds = cfg["development_seeds"] if args.seed is None else [args.seed]
    scales = cfg["process_noise_scales"] if args.q_scale is None else [args.q_scale]
    if any(seed not in cfg["development_seeds"] for seed in seeds) or any(
        scale not in cfg["process_noise_scales"] for scale in scales
    ):
        parser.error("Only declared development jobs may execute")
    jobs = [
        (seed, scale)
        for seed in seeds
        for scale in scales
        if not (args.resume and (OUT / f"{job_prefix(seed, scale)}_complete.json").exists())
    ]
    failures = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(evaluate_seed, *job): job for job in jobs}
        for future in as_completed(futures):
            error = future.exception()
            if error is None:
                print(f"Completed 10H-L {future.result()}", flush=True)
            else:
                failures.append(futures[future])
                print(f"Failed 10H-L {futures[future]}: {error!r}", flush=True)
    if failures:
        raise RuntimeError(f"Incomplete jobs {failures}; context caches preserve completed fits")
    if args.seed is None and args.q_scale is None:
        summarize()


if __name__ == "__main__":
    main()
