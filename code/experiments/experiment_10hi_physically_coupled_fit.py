"""10H-I: physical eight-coordinate fit and output-only covariance selection.

Part A freezes the historical Q only to isolate model structure. Part B selects
Q with client-blocked inner validation; its fitting path never reads that value.
No confirmation option exists. All outer clients are scored after selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from experiment_10hf_output_identifiability import (
    measurement_covariance,
    nominal_effective_parameters,
)
from experiment_10hh_single_model_repair import dataset_hash, measured_datasets

from federated_lpv.innovation_likelihood import (
    PARAMETER_NAMES,
    C,
    InnovationLikelihood,
    MeasuredDataset,
    physical_coupling_diagnostic,
    structured_matrices,
)
from federated_lpv.physical_coupled_fit import (
    PHYSICAL_NAMES,
    CoupledLikelihood,
    PhysicalCoordinates,
    constraint_audit,
    optimize_coupled,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "code/config/experiment_10hi.json"
OUT = ROOT / "results/tables"
PREFIX = "experiment_10hi"
TABLES = [
    "runs",
    "restarts",
    "parameters",
    "physical_parameters",
    "profiles",
    "curvature",
    "phases",
    "gradient_checks",
    "splits",
    "covariance_candidates",
    "covariance_restarts",
    "covariance_selection",
]


def load_configuration():
    cfg = json.loads(CONFIG.read_text())
    if set(cfg["development_seeds"]) & set(cfg["reserved_confirmation_seeds"]):
        raise ValueError("Development and confirmation overlap")
    inherited = {
        name: json.loads((ROOT / f"code/config/experiment_{name}.json").read_text())
        for name in ["10h", "10ha", "10hf"]
    }
    return cfg, inherited


def make_coordinates(cfg, hf):
    nominal = np.log(nominal_effective_parameters(hf))
    return PhysicalCoordinates(
        nominal,
        nominal + np.log(cfg["parameter_lower_multiplier"]),
        nominal + np.log(cfg["parameter_upper_multiplier"]),
    )


def make_initials(seed, cfg, coordinates):
    rng = np.random.default_rng(seed + 2_100_000)
    return [
        np.zeros(8),
        *[
            coordinates.feasible_initial(rng.normal(0, cfg["restart_log_standard_deviation"], 8))
            for _ in range(cfg["restart_count"] - 1)
        ],
    ]


def subset(data, mask):
    return MeasuredDataset(data.speeds[mask], data.commands[mask], data.measurements[mask])


def inner_client_folds(training_data, training_positions, folds=2):
    """Client identities are array positions, not labels or simulator attributes."""
    positions = np.asarray(training_positions)
    if len(positions) != len(training_data.speeds):
        raise ValueError("Every measured record must have its client position")
    unique = np.unique(positions)
    assignment = {client: rank % folds for rank, client in enumerate(unique)}
    fold_ids = np.array([assignment[client] for client in positions])
    for fold in range(folds):
        validation = fold_ids == fold
        if not validation.any() or validation.all():
            raise ValueError("Inner folds must contain both fitting and validation records")
        yield fold, subset(training_data, ~validation), subset(training_data, validation), fold_ids


def scalar_audit(audit):
    return {
        key: audit[key]
        for key in [
            "kkt_residual",
            "constraint_violation",
            "complementarity_residual",
            "active_coefficient_bounds",
        ]
    }


def fit_restarts(seed, part, method, evaluator, coordinates, cfg, initials):
    estimates, objectives, rows, phases = [], [], [], []
    for restart, initial in enumerate(initials):
        eta, result, trace, calls, invalid, audit = optimize_coupled(
            initial, evaluator, coordinates, cfg, method
        )
        value = evaluator.value(eta)
        estimates.append(eta)
        objectives.append(value)
        rows.append(
            dict(
                seed=seed,
                part=part,
                method=method,
                restart=restart,
                objective=value,
                success=bool(result.success),
                iterations=int(result.nit),
                evaluations=calls,
                invalid_evaluations=invalid,
                message=str(result.message),
                **scalar_audit(audit),
            )
        )
        phases.extend(
            dict(seed=seed, part=part, method=method, restart=restart, **p) for p in trace
        )
        print(
            f"10H-I seed={seed} part={part} method={method} restart={restart} "
            f"L={value:.6f} KKT={audit['kkt_residual']:.2g} success={result.success}",
            flush=True,
        )
    best = int(np.argmin(objectives))  # Training likelihood only.
    physical = np.exp(np.asarray(estimates))
    cv = np.std(physical, axis=0) / np.mean(physical, axis=0)
    stats = {
        "best_restart": best,
        "all_restart_success": all(row["success"] for row in rows),
        "maximum_restart_kkt_residual": max(row["kkt_residual"] for row in rows),
        "maximum_restart_constraint_violation": max(row["constraint_violation"] for row in rows),
        "restart_objective_spread_pct": 100 * (max(objectives) - min(objectives)) / min(objectives)
        if min(objectives) > 1e-12
        else max(objectives) - min(objectives),
        "maximum_restart_parameter_cv": float(cv.max()),
        "invalid_evaluations": sum(row["invalid_evaluations"] for row in rows),
    }
    return estimates[best], stats, pd.DataFrame(rows), pd.DataFrame(phases), cv


def gradient_audit(evaluator, cfg):
    rows = []
    step = cfg["gradient_check_log_step"]
    for point, eta in [("nominal", np.zeros(8)), ("perturbed", np.linspace(-0.08, 0.10, 8))]:
        analytic = evaluator.value_gradient(eta)[1]
        numerical = np.array(
            [
                (
                    evaluator.value(eta + np.eye(8)[j] * step)
                    - evaluator.value(eta - np.eye(8)[j] * step)
                )
                / (2 * step)
                for j in range(8)
            ]
        )
        error = np.linalg.norm(analytic - numerical) / max(np.linalg.norm(numerical), 1e-8)
        rows.extend(
            {
                "point": point,
                "parameter": name,
                "analytic": analytic[j],
                "central_difference": numerical[j],
                "norm_relative_error": error,
            }
            for j, name in enumerate(PHYSICAL_NAMES)
        )
    return pd.DataFrame(rows)


def curvature_audit(evaluator, fitted, coordinates, cfg):
    step = cfg["hessian_log_step"]
    hessian = np.column_stack(
        [
            (
                evaluator.value_gradient(fitted + np.eye(8)[j] * step)[1]
                - evaluator.value_gradient(fitted - np.eye(8)[j] * step)[1]
            )
            / (2 * step)
            for j in range(8)
        ]
    )
    asymmetry = np.linalg.norm(hessian - hessian.T) / max(np.linalg.norm(hessian), 1e-12)
    hessian = (hessian + hessian.T) / 2
    audit = constraint_audit(
        fitted, evaluator.value_gradient(fitted)[1], coordinates, cfg["active_bound_tolerance"]
    )
    basis = audit["face_basis"]
    full = np.linalg.eigvalsh(hessian)
    face = np.linalg.eigvalsh(basis.T @ hessian @ basis)
    info = evaluator.evaluate(fitted, information=True)["information"]
    information_eigen = np.linalg.eigvalsh(info)
    rows = [
        {"space": space, "eigenvalue_index": j, "eigenvalue": float(value)}
        for space, values in [
            ("physical_full", full),
            ("active_face", face),
            ("physical_information", information_eigen),
        ]
        for j, value in enumerate(values)
    ]
    return {
        "minimum_full_hessian_eigenvalue": float(full[0]),
        "minimum_face_hessian_eigenvalue": float(face[0]) if len(face) else np.nan,
        "face_dimension": int(basis.shape[1]),
        "hessian_asymmetry": float(asymmetry),
        "physical_information_rank": int(np.sum(information_eigen > information_eigen[-1] * 1e-6)),
        "physical_information_condition": float(information_eigen[-1] / information_eigen[0]),
    }, pd.DataFrame(rows)


def nuisance_profiles(evaluator, fitted, coordinates, cfg):
    """Fix one physical log coordinate and reoptimize all seven others.

    An infeasible requested offset is recorded, never clipped back to the fit.
    This is stronger than the conditional slices used in historical 10H-H.
    """
    baseline = evaluator.value(fitted)
    rows, edges, two_sided = [], [], True
    for j, name in enumerate(PHYSICAL_NAMES):
        local = []
        for offset in cfg["profile_log_offsets"]:
            target = fitted[j] + offset
            # The center is the already audited fit. Reprojecting it can make
            # redundant bound/equality constraints inconsistent at roundoff.
            start = fitted if offset == 0 else coordinates.feasible_profile_start(fitted, j, target)
            if start is None:
                rows.append(
                    {
                        "parameter": name,
                        "log_offset": offset,
                        "feasible": False,
                        "success": False,
                        "objective": np.nan,
                        "increase_pct": np.nan,
                        "kkt_residual": np.nan,
                        "constraint_violation": np.nan,
                    }
                )
                continue
            if offset == 0:
                estimate = fitted
                success = True
                invalid = 0
                audit = constraint_audit(
                    estimate,
                    evaluator.value_gradient(estimate)[1],
                    coordinates,
                    cfg["active_bound_tolerance"],
                    {j: target},
                )
            else:
                estimate, result, _, _, invalid, audit = optimize_coupled(
                    start, evaluator, coordinates, cfg, "joint", {j: target}
                )
                success = bool(result.success)
            value = evaluator.value(estimate)
            row = dict(
                parameter=name,
                log_offset=offset,
                feasible=True,
                success=success,
                invalid_evaluations=invalid,
                objective=value,
                increase_pct=100 * (value - baseline) / baseline,
                actual_log_distance=float(np.linalg.norm(estimate - fitted)),
                **scalar_audit(audit),
            )
            rows.append(row)
            local.append(row)
        for sign in [-1, 1]:
            side = [row for row in local if sign * row["log_offset"] > 0]
            if side:
                edge = max(side, key=lambda row: abs(row["log_offset"]))
                edges.append(edge["increase_pct"])
            else:
                two_sided = False
    frame = pd.DataFrame(rows)
    feasible = frame[frame.feasible]
    return {
        "minimum_profile_edge_increase_pct": min(edges) if edges else np.nan,
        "all_profiles_two_sided": two_sided,
        "all_feasible_profiles_success": bool(feasible.success.all()),
        "maximum_profile_kkt_residual": float(feasible.kkt_residual.max()),
        "maximum_profile_constraint_violation": float(feasible.constraint_violation.max()),
    }, frame


def choose_covariance(training_data, training_positions, seed, coordinates, cfg, dt, base_q, r):
    """No outer data or inherited oracle-selection artifact is accepted here."""
    candidates, restarts = [], []
    initials = make_initials(seed + 10000, cfg, coordinates)
    for scale in cfg["covariance_grid"]:
        for fold, fitting, validation, _ in inner_client_folds(
            training_data, training_positions, cfg["inner_client_folds"]
        ):
            q = np.diag(base_q * scale)
            train = CoupledLikelihood(InnovationLikelihood(fitting, dt, q, r), coordinates)
            test = CoupledLikelihood(InnovationLikelihood(validation, dt, q, r), coordinates)
            fitted, stats, restart_frame, _, _ = fit_restarts(
                seed,
                f"inner_Q{scale:g}_fold{fold}",
                cfg["covariance_selection_method"],
                train,
                coordinates,
                cfg,
                initials,
            )
            score = test.evaluate(fitted, diagnostics=True)
            candidates.append(
                dict(
                    seed=seed,
                    fold=fold,
                    process_noise_scale=scale,
                    validation_sensor_mse=score["sensor_normalized_mse"],
                    validation_objective=score["objective"],
                    fitting_samples=int(fitting.measurements.size // 3),
                    validation_samples=int(validation.measurements.size // 3),
                    fitting_data_sha256=dataset_hash(fitting),
                    validation_data_sha256=dataset_hash(validation),
                    **stats,
                )
            )
            restarts.append(restart_frame.assign(fold=fold, process_noise_scale=scale))
    frame = pd.DataFrame(candidates)
    scores = frame.groupby("process_noise_scale").validation_sensor_mse.mean()
    selected = float(scores.idxmin())  # Declared grid order breaks an exact tie.
    stationary = bool(
        (frame.maximum_restart_kkt_residual <= cfg["maximum_kkt_residual"]).all()
        and (
            frame.maximum_restart_constraint_violation <= cfg["maximum_constraint_violation"]
        ).all()
        and frame.all_restart_success.all()
        and (frame.invalid_evaluations == 0).all()
    )
    selection = {
        "seed": seed,
        "selected_process_noise_scale": selected,
        "selection_metric": cfg["covariance_selection_metric"],
        "selected_inner_sensor_mse": float(scores.loc[selected]),
        "all_cv_candidates_stationary": stationary,
        "selected_grid_edge": selected
        in [min(cfg["covariance_grid"]), max(cfg["covariance_grid"])],
        "outer_heldout_used_for_selection": False,
        "inherited_oracle_scale_used_for_selection": False,
    }
    return selected, frame, pd.concat(restarts, ignore_index=True), pd.DataFrame([selection])


def improvement(reference, value):
    return 100 * (reference - value) / reference


def evaluate_seed(seed, part):
    cfg, inherited = load_configuration()
    if seed not in cfg["development_seeds"]:
        raise ValueError("Only opened development seeds may be executed")
    data, split = measured_datasets(seed, cfg, inherited)
    h, ha, hf = (inherited[name] for name in ["10h", "10ha", "10hf"])
    coordinates = make_coordinates(cfg, hf)
    dt, base_q, r = (
        h["sample_time"],
        np.asarray(ha["base_process_noise_diagonal"]),
        measurement_covariance(ha),
    )
    frames = {name: [] for name in TABLES}
    if part == "A":
        noise_scale = json.loads((OUT / "experiment_10ha_frozen_selection.json").read_text())[
            "process_noise_scale"
        ]
        methods = cfg["methods"]
        archive = pd.read_csv(OUT / "experiment_10hh_runs.csv")
        archived = archive[(archive.seed == seed) & (archive.method == "staged")].iloc[0]
        if (
            dataset_hash(data["train"]) != archived.train_data_sha256
            or dataset_hash(data["heldout"]) != archived.heldout_data_sha256
        ):
            raise RuntimeError("Paired measured-array hashes differ from 10H-H")
    else:
        train_positions = split[split.split == "train"].client_position.to_numpy()
        noise_scale, candidate, cv_restarts, selection = choose_covariance(
            data["train"], train_positions, seed, coordinates, cfg, dt, base_q, r
        )
        frames["covariance_candidates"].append(candidate)
        frames["covariance_restarts"].append(cv_restarts)
        frames["covariance_selection"].append(selection)
        methods = [cfg["covariance_selection_method"]]
    q = np.diag(base_q * noise_scale)
    train = CoupledLikelihood(InnovationLikelihood(data["train"], dt, q, r), coordinates)
    test = CoupledLikelihood(InnovationLikelihood(data["heldout"], dt, q, r), coordinates)
    nominal = train.evaluate(np.zeros(8), diagnostics=True)
    nominal_test = test.evaluate(np.zeros(8), diagnostics=True)
    checks = gradient_audit(train, cfg).assign(seed=seed, part=part)
    frames["gradient_checks"].append(checks)
    initials = make_initials(seed, cfg, coordinates)
    for method in methods:
        started = time.perf_counter()
        fitted, stats, restart_frame, phases, cv = fit_restarts(
            seed, part, method, train, coordinates, cfg, initials
        )
        # Outer scoring happens only after model/restart/Q selection is complete.
        final = train.evaluate(fitted, gradient=True, diagnostics=True)
        heldout = test.evaluate(fitted, diagnostics=True)
        curvature, eigenvalues = curvature_audit(train, fitted, coordinates, cfg)
        profile_stats, profiles = nuisance_profiles(train, fitted, coordinates, cfg)
        effective = coordinates.effective(fitted)
        audit = constraint_audit(
            fitted, final["gradient"], coordinates, cfg["active_bound_tolerance"]
        )
        diagnostic = physical_coupling_diagnostic(effective)
        row = dict(
            seed=seed,
            part=part,
            method=method,
            process_noise_scale=noise_scale,
            train_objective_nominal=nominal["objective"],
            train_objective_fitted=final["objective"],
            heldout_objective_nominal=nominal_test["objective"],
            heldout_objective_fitted=heldout["objective"],
            heldout_objective_improvement_pct=improvement(
                nominal_test["objective"], heldout["objective"]
            ),
            heldout_sensor_mse_nominal=nominal_test["sensor_normalized_mse"],
            heldout_sensor_mse_fitted=heldout["sensor_normalized_mse"],
            heldout_sensor_mse_improvement_pct=improvement(
                nominal_test["sensor_normalized_mse"], heldout["sensor_normalized_mse"]
            ),
            heldout_yaw_rmse_deg_s=heldout["yaw_rmse_deg_s"],
            heldout_acceleration_rmse_mps2=heldout["acceleration_rmse_mps2"],
            heldout_steering_rmse_deg=heldout["steering_rmse_deg"],
            coupling_log_residual=diagnostic["log_coupling_residual"],
            gradient_check_relative_error=float(checks.norm_relative_error.max()),
            elapsed_seconds=time.perf_counter() - started,
            train_data_sha256=dataset_hash(data["train"]),
            heldout_data_sha256=dataset_hash(data["heldout"]),
            **stats,
            **curvature,
            **profile_stats,
            **scalar_audit(audit),
        )
        if part == "A":
            row.update(
                archived_nine_train_objective=archived.train_objective_fitted,
                archived_nine_heldout_objective=archived.heldout_objective_fitted,
                archived_nine_heldout_sensor_mse=archived.heldout_sensor_mse_fitted,
                heldout_sensor_mse_change_vs_nine_pct=100
                * (heldout["sensor_normalized_mse"] - archived.heldout_sensor_mse_fitted)
                / archived.heldout_sensor_mse_fitted,
            )
        frames["runs"].append(pd.DataFrame([row]))
        frames["restarts"].append(restart_frame)
        frames["phases"].append(phases)
        frames["profiles"].append(profiles.assign(seed=seed, part=part, method=method))
        frames["curvature"].append(eigenvalues.assign(seed=seed, part=part, method=method))
        frames["parameters"].append(
            pd.DataFrame(
                [
                    {
                        "seed": seed,
                        "part": part,
                        "method": method,
                        "parameter": name,
                        "nominal": np.exp(coordinates.nominal[j]),
                        "fitted": np.exp(effective[j]),
                        "at_lower": bool(audit["at_lower"][j]),
                        "at_upper": bool(audit["at_upper"][j]),
                    }
                    for j, name in enumerate(PARAMETER_NAMES)
                ]
            )
        )
        frames["physical_parameters"].append(
            pd.DataFrame(
                [
                    {
                        "seed": seed,
                        "part": part,
                        "method": method,
                        "parameter": name,
                        "nominal": np.exp(coordinates.nominal_physical_logs[j]),
                        "fitted": np.exp(coordinates.nominal_physical_logs[j] + fitted[j]),
                        "restart_cv": cv[j],
                    }
                    for j, name in enumerate(PHYSICAL_NAMES)
                ]
            )
        )
    split = split.assign(part=part, inner_fold=-1)
    train_indices = split.split == "train"
    positions = split.loc[train_indices, "client_position"].to_numpy()
    _, _, _, ids = next(inner_client_folds(data["train"], positions, cfg["inner_client_folds"]))
    split.loc[train_indices, "inner_fold"] = ids
    frames["splits"].append(split)
    for name, items in frames.items():
        if items:
            pd.concat(items, ignore_index=True).to_csv(
                OUT / f"{PREFIX}_seed{seed}_{part}_{name}.csv", index=False
            )
    return seed, part


def homogeneous_diagnostics():
    cfg, inherited = load_configuration()
    h, ha, hf = (inherited[name] for name in ["10h", "10ha", "10hf"])
    coordinates = make_coordinates(cfg, hf)
    template, _ = measured_datasets(cfg["development_seeds"][0], cfg, inherited)
    template = template["train"]
    # Public planted fixture; never passed to the estimator or its starts.
    planted = np.array([0.08, -0.07, 0.04, 0.12, -0.09, 0.06, -0.04, 0.10])
    planted_z = coordinates.effective(planted)
    rng = np.random.default_rng(cfg["homogeneous_diagnostic_seed"])
    r = measurement_covariance(ha)
    rows, params, restarts = [], [], []
    for scenario in ["noiseless", "sensor_noise_and_quiet_bias"]:
        y = np.zeros_like(template.measurements)
        for index, speed in enumerate(template.speeds):
            a, b = structured_matrices(planted_z, speed, h["sample_time"])
            x = np.zeros(5)
            for sample, command in enumerate(template.commands[index]):
                x = a @ x + b * command
                y[index, sample] = C @ x
            if scenario != "noiseless":
                y[index] += rng.multivariate_normal(np.zeros(3), r, size=y.shape[1])
                quiet = rng.multivariate_normal(np.zeros(3), r, size=hf["quiet_bias_samples"])
                y[index] -= quiet.mean(axis=0)
        data = MeasuredDataset(template.speeds, template.commands, y)
        q = (
            np.zeros((5, 5))
            if scenario == "noiseless"
            else np.diag(np.asarray(ha["base_process_noise_diagonal"]) * 0.01)
        )
        evaluator = CoupledLikelihood(
            InnovationLikelihood(data, h["sample_time"], q, r), coordinates
        )
        fitted, stats, restart, _, _ = fit_restarts(
            cfg["homogeneous_diagnostic_seed"],
            scenario,
            "staged",
            evaluator,
            coordinates,
            cfg,
            make_initials(cfg["homogeneous_diagnostic_seed"], cfg, coordinates),
        )
        curvature, _ = curvature_audit(evaluator, fitted, coordinates, cfg)
        audit = constraint_audit(
            fitted, evaluator.value_gradient(fitted)[1], coordinates, cfg["active_bound_tolerance"]
        )
        rows.append(
            dict(
                scenario=scenario,
                fitted_objective=evaluator.value(fitted),
                maximum_retrospective_parameter_relative_error=float(
                    np.max(np.abs(np.exp(fitted - planted) - 1))
                ),
                coupling_log_residual=physical_coupling_diagnostic(coordinates.effective(fitted))[
                    "log_coupling_residual"
                ],
                **stats,
                **curvature,
                **scalar_audit(audit),
            )
        )
        params.extend(
            {
                "scenario": scenario,
                "parameter": name,
                "planted": np.exp(coordinates.nominal_physical_logs[j] + planted[j]),
                "fitted": np.exp(coordinates.nominal_physical_logs[j] + fitted[j]),
            }
            for j, name in enumerate(PHYSICAL_NAMES)
        )
        restarts.append(restart)
    pd.DataFrame(rows).to_csv(OUT / f"{PREFIX}_homogeneous.csv", index=False)
    pd.DataFrame(params).to_csv(OUT / f"{PREFIX}_homogeneous_parameters.csv", index=False)
    pd.concat(restarts).to_csv(OUT / f"{PREFIX}_homogeneous_restarts.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


def gates_for_runs(local, cfg):
    gates = {
        "physical_coupling": bool(
            (local.coupling_log_residual.abs() <= cfg["maximum_coupling_log_residual"]).all()
        ),
        "analytic_gradients": bool(
            (local.gradient_check_relative_error <= cfg["maximum_gradient_relative_error"]).all()
        ),
        "optimizer_success": bool(local.all_restart_success.all()),
        "stationarity": bool(
            (local.maximum_restart_kkt_residual <= cfg["maximum_kkt_residual"]).all()
        ),
        "feasibility": bool(
            (
                local.maximum_restart_constraint_violation <= cfg["maximum_constraint_violation"]
            ).all()
        ),
        "valid_likelihoods": bool((local.invalid_evaluations == 0).all()),
        "restart_objective": bool(
            (
                local.restart_objective_spread_pct <= cfg["maximum_restart_objective_spread_pct"]
            ).all()
        ),
        "restart_parameters": bool(
            (local.maximum_restart_parameter_cv <= cfg["maximum_restart_parameter_cv"]).all()
        ),
        "heldout_likelihood": bool(
            (
                local.heldout_objective_improvement_pct
                >= cfg["minimum_heldout_objective_improvement_pct"]
            ).all()
        ),
        "heldout_fixed_weight_prediction": bool(
            (
                local.heldout_sensor_mse_improvement_pct
                > cfg["minimum_heldout_sensor_mse_improvement_pct"]
            ).all()
        ),
    }
    gates["numerical_prediction_gate"] = all(gates.values())
    gates.update(
        active_face_curvature=bool(
            (local.minimum_face_hessian_eigenvalue >= cfg["minimum_face_hessian_eigenvalue"]).all()
        ),
        profile_optimizers=bool(
            local.all_feasible_profiles_success.all()
            and (local.maximum_profile_kkt_residual <= cfg["maximum_kkt_residual"]).all()
            and (
                local.maximum_profile_constraint_violation <= cfg["maximum_constraint_violation"]
            ).all()
        ),
        informative_two_sided_profiles=bool(
            local.all_profiles_two_sided.all()
            and (
                local.minimum_profile_edge_increase_pct >= cfg["minimum_profile_edge_increase_pct"]
            ).all()
        ),
        interior_bounds=bool(
            (local.active_coefficient_bounds <= cfg["maximum_active_coefficient_bounds"]).all()
        ),
    )
    gates["physical_readiness_gate"] = all(gates.values())
    return gates


def summarize():
    cfg, _ = load_configuration()
    frames = {}
    for name in TABLES:
        paths = [
            OUT / f"{PREFIX}_seed{seed}_{part}_{name}.csv"
            for seed in cfg["development_seeds"]
            for part in ["A", "B"]
            if part == "B"
            or name not in ["covariance_candidates", "covariance_restarts", "covariance_selection"]
        ]
        if not all(path.exists() for path in paths):
            raise RuntimeError(f"Missing predeclared outputs for {name}")
        frames[name] = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
        frames[name].to_csv(OUT / f"{PREFIX}_{name}.csv", index=False)
    runs = frames["runs"]
    summary, conclusions = [], {}
    for (part, method), local in runs.groupby(["part", "method"]):
        aggregate = {
            "mean_heldout_objective_improvement_pct": float(
                local.heldout_objective_improvement_pct.mean()
            ),
            "minimum_heldout_objective_improvement_pct": float(
                local.heldout_objective_improvement_pct.min()
            ),
            "mean_heldout_sensor_mse_improvement_pct": float(
                local.heldout_sensor_mse_improvement_pct.mean()
            ),
            "minimum_heldout_sensor_mse_improvement_pct": float(
                local.heldout_sensor_mse_improvement_pct.min()
            ),
            "maximum_kkt_residual": float(local.maximum_restart_kkt_residual.max()),
            "maximum_constraint_violation": float(local.maximum_restart_constraint_violation.max()),
            "maximum_coupling_log_residual": float(local.coupling_log_residual.abs().max()),
            "maximum_restart_parameter_cv": float(local.maximum_restart_parameter_cv.max()),
            "maximum_restart_objective_spread_pct": float(local.restart_objective_spread_pct.max()),
            "minimum_face_hessian_eigenvalue": float(local.minimum_face_hessian_eigenvalue.min()),
            "minimum_profile_edge_increase_pct": float(
                local.minimum_profile_edge_increase_pct.min()
            ),
            "maximum_profile_kkt_residual": float(local.maximum_profile_kkt_residual.max()),
            "fleets_with_active_bounds": int((local.active_coefficient_bounds > 0).sum()),
        }
        gates = gates_for_runs(local, cfg)
        if part == "B":
            selected = frames["covariance_selection"]
            gates["inner_cv_stationarity"] = bool(selected.all_cv_candidates_stationary.all())
            gates["covariance_grid_interior"] = bool((~selected.selected_grid_edge).all())
            gates["output_only_covariance_pipeline_gate"] = bool(
                gates["numerical_prediction_gate"]
                and gates["inner_cv_stationarity"]
                and gates["covariance_grid_interior"]
            )
            gates["physical_readiness_gate"] = bool(
                gates["physical_readiness_gate"] and gates["output_only_covariance_pipeline_gate"]
            )
        conclusions[f"{part}_{method}"] = {"aggregate": aggregate, "gates": gates}
        summary.append(dict(part=part, method=method, **aggregate, **gates))
    pd.DataFrame(summary).to_csv(OUT / f"{PREFIX}_summary.csv", index=False)
    paths = [
        CONFIG,
        Path(__file__),
        ROOT / "code/src/federated_lpv/physical_coupled_fit.py",
        ROOT / "code/src/federated_lpv/innovation_likelihood.py",
        ROOT / "code/experiments/experiment_10hh_single_model_repair.py",
        ROOT / "code/experiments/experiment_10hf_output_identifiability.py",
        ROOT / "code/experiments/experiment_10h_higher_order_lpv_gate.py",
        ROOT / "code/experiments/experiment_10ha_tire_force_observer.py",
        *[ROOT / f"code/config/experiment_{name}.json" for name in ["10h", "10ha", "10hf"]],
        OUT / "experiment_10ha_frozen_selection.json",
    ]
    output = {
        "methods": conclusions,
        "confirmation_run": False,
        "reserved_confirmation_seeds": cfg["reserved_confirmation_seeds"],
        "part_A": "Paired physical-structure comparison with historical oracle-selected Q frozen; identical measured-array hashes to 10H-H.",
        "part_B": "Q scale chosen solely by two client-blocked inner training folds, fixed-R prediction scores, and a predeclared six-point grid; outer clients scored after selection. Public R and base Q shape held fixed.",
        "limitations": "Development evidence only. Stationary Gaussian working likelihood, heterogeneous pooling and unmodeled quiet-bias uncertainty remain. Active-face curvature is not a proof of global uniqueness. Physical ratios do not identify absolute mass/inertia/stiffness. No latent-state or closed-loop-control gain is claimed.",
        "provenance_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths
        },
    }
    (OUT / f"{PREFIX}_conclusions.json").write_text(json.dumps(output, indent=2) + "\n")
    plot(frames)
    print(json.dumps(output["methods"], indent=2), flush=True)


def plot(frames):
    runs = frames["runs"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for part, method, color in [
        ("A", "joint", "#0072B2"),
        ("A", "staged", "#D55E00"),
        ("B", "staged", "#009E73"),
    ]:
        local = runs[(runs.part == part) & (runs.method == method)]
        label = f"{part}: {method}"
        axes[0].plot(
            local.seed.astype(str),
            local.heldout_sensor_mse_improvement_pct,
            "o-",
            color=color,
            label=label,
        )
        axes[1].plot(
            local.seed.astype(str), local.active_coefficient_bounds, "o-", color=color, label=label
        )
    archived = pd.read_csv(OUT / "experiment_10hh_runs.csv")
    archived = archived[archived.method == "staged"]
    axes[0].plot(
        archived.seed.astype(str),
        archived.heldout_sensor_mse_improvement_pct,
        "x--",
        color="#777777",
        label="10H-H nine coordinates",
    )
    cv = frames["covariance_candidates"]
    for seed, local in cv.groupby("seed"):
        scores = local.groupby("process_noise_scale").validation_sensor_mse.mean()
        axes[2].semilogx(scores.index, scores, "o-", label=str(seed))
    axes[0].set(
        title="Held-out fixed-weight prediction",
        ylabel="Improvement over nominal (%)",
        xlabel="Opened fleet",
    )
    axes[1].set(title="Physical coefficient bounds", ylabel="Active bounds", xlabel="Opened fleet")
    axes[2].set(
        title="Training-only covariance selection",
        ylabel="Inner validation sensor MSE",
        xlabel="Process-noise scale",
    )
    for ax in axes:
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(ROOT / "results/figures/experiment_10hi_physically_coupled_fit.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--part", choices=["A", "B", "all"], default="all")
    parser.add_argument("--homogeneous-only", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    cfg, _ = load_configuration()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.homogeneous_only:
        homogeneous_diagnostics()
        return
    if args.summarize_only:
        summarize()
        return
    seeds = [args.seed] if args.seed is not None else cfg["development_seeds"]
    if any(seed not in cfg["development_seeds"] for seed in seeds):
        parser.error("Only opened development seeds are permitted; confirmation is sealed")
    parts = ["A", "B"] if args.part == "all" else [args.part]
    jobs = []
    for part in parts:
        required = [name for name in TABLES if part == "B" or not name.startswith("covariance_")]
        for seed in seeds:
            complete = all(
                (OUT / f"{PREFIX}_seed{seed}_{part}_{name}.csv").exists() for name in required
            )
            if not args.resume or not complete:
                jobs.append((seed, part))
    failures = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(evaluate_seed, seed, part): (seed, part) for seed, part in jobs}
        for future in as_completed(futures):
            error = future.exception()
            if error is None:
                print(f"Completed 10H-I {future.result()}", flush=True)
            else:
                failures.append(futures[future])
                print(f"Failed 10H-I {futures[future]}: {error!r}; other jobs continue", flush=True)
    if failures:
        raise RuntimeError(f"Incomplete jobs {failures}; use --resume after addressing the error")
    if args.seed is None and args.part == "all":
        summarize()


if __name__ == "__main__":
    main()
