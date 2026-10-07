"""Independent provenance/selection audit and reporting for experiment 10H-L."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from experiment_10hf_output_identifiability import measurement_covariance
from experiment_10hh_single_model_repair import dataset_hash, measured_datasets
from experiment_10hk_joint_client_mixture import partition_ari
from experiment_10hl_validated_model_selection import (
    FIT_TABLES,
    OUT,
    PREFIX,
    ROOT,
    TABLES,
    job_prefix,
    load_configuration,
    sha,
    source_hashes,
)

from federated_lpv.innovation_likelihood import (
    PARAMETER_NAMES,
    C,
    MeasuredDataset,
    structured_matrices,
)
from federated_lpv.output_validation import causal_forecasts, output_error_metrics
from federated_lpv.physical_coupled_fit import PHYSICAL_NAMES

METHODS = (
    "global",
    "archived_MAP",
    "likelihood_MAP",
    "validated_MAP",
    "validated_group",
    "validated_fallback",
    "nominal",
)
LABELS = {
    "global": "Global",
    "archived_MAP": "Archived MAP",
    "likelihood_MAP": "6-start MAP",
    "validated_MAP": "CV fit + MAP",
    "validated_group": "CV fit + forecast choice",
    "validated_fallback": "CV fit + fallback",
    "nominal": "Nominal",
}


def audit_execution():
    cfg, inherited = load_configuration()
    errors = {"prefix_score": 0.0, "forecast": 0.0, "simulation_suffix": 0.0}
    pieces = {name: [] for name in TABLES}
    manifests = []
    context_count = 0
    dt = inherited["10h"]["sample_time"]
    r = measurement_covariance(inherited["10ha"])
    for seed in cfg["development_seeds"]:
        for scale in cfg["process_noise_scales"]:
            prefix = job_prefix(seed, scale)
            manifest = json.loads((OUT / f"{prefix}_complete.json").read_text())
            manifests.append(manifest)
            assert manifest["source_sha256"] == source_hashes(seed, scale)
            assert not manifest["confirmation_run"]
            for relative, expected in manifest["source_sha256"].items():
                committed = subprocess.check_output(
                    ["git", "show", f"{manifest['revision']}:{relative}"], cwd=ROOT
                )
                assert hashlib.sha256(committed).hexdigest() == expected
            for context in ("fold0", "fold1", "full"):
                cached = json.loads((OUT / f"{prefix}_{context}_complete.json").read_text())
                assert cached["source_sha256"] == manifest["source_sha256"]
                for name in FIT_TABLES:
                    path = OUT / f"{prefix}_{context}_{name}.csv"
                    assert sha(path) == cached["output_sha256"][path.name]
                context_count += 1
            frames = {}
            for name in TABLES:
                path = OUT / f"{prefix}_{name}.csv"
                assert sha(path) == manifest["output_sha256"][path.name]
                frames[name] = pd.read_csv(path)
                pieces[name].append(frames[name])
            data, split = measured_datasets(seed, cfg, inherited)
            assert {name: dataset_hash(value) for name, value in data.items()} == manifest[
                "measured_array_sha256"
            ]
            train_clients = set(split.loc[split.split == "train", "client_position"])
            train_positions = split.loc[split.split == "train", "client_position"].to_numpy()
            association = {
                client: rank % 2 for rank, client in enumerate(np.unique(train_positions))
            }
            record_folds = np.array([association[client] for client in train_positions])
            for context in ("fold0", "fold1", "full"):
                mask = (
                    np.ones(len(train_positions), dtype=bool)
                    if context == "full"
                    else record_folds != int(context[-1])
                )
                records = MeasuredDataset(
                    data["train"].speeds[mask],
                    data["train"].commands[mask],
                    data["train"].measurements[mask],
                )
                cached = json.loads((OUT / f"{prefix}_{context}_complete.json").read_text())
                assert cached["training_array_sha256"] == dataset_hash(records)
            outer_clients = set(split.loc[split.split == "heldout", "client_position"])
            assert train_clients.isdisjoint(outer_clients)
            inner = frames["inner_scores"]
            assert set(inner.client_position) == train_clients
            memberships = frames["memberships"]
            for fold in (0, 1):
                fitting = set(
                    memberships.loc[memberships.context == f"fold{fold}", "client_position"]
                )
                validation = set(inner.loc[inner.fold == fold, "client_position"])
                assert fitting.isdisjoint(validation) and fitting | validation == train_clients
            fits = frames["fits"]
            eligibility = []
            for restart in range(cfg["restart_count"]):
                rows = fits[(fits.model == "K2") & (fits.restart == restart)]
                assert set(rows.context) == {"fold0", "fold1", "full"}
                eligibility.append(
                    bool(
                        rows.success.all()
                        and (rows.kkt_residual <= cfg["maximum_kkt_residual"]).all()
                        and (rows.constraint_violation <= cfg["maximum_constraint_violation"]).all()
                    )
                )
            scores = inner.groupby(["restart", "margin"]).selection_score.mean().reset_index()
            scores = scores[
                scores.restart.map(lambda value, eligibility=eligibility: eligibility[int(value)])
            ]
            selection = frames["selection"].iloc[0]
            assert selection.eligible_strategies == sum(eligibility)
            if len(scores):
                ties = scores[
                    scores.selection_score
                    <= scores.selection_score.min() + cfg["selection_tie_tolerance"]
                ]
                winner = ties.sort_values(["margin", "restart"], ascending=[False, True]).iloc[0]
                assert selection.selected_restart == winner.restart
                assert selection.selected_margin == winner.margin
                assert abs(selection.inner_score - winner.selection_score) < 1e-12
            else:
                assert not selection.validation_selection_available
            q = np.diag(np.asarray(inherited["10ha"]["base_process_noise_diagonal"]) * scale)
            parameters = frames["parameters"]
            parameters = parameters[
                (parameters.context == "full") & (parameters.restart == selection.selected_restart)
            ]
            z = [
                np.log(
                    parameters[parameters.candidate == candidate]
                    .set_index("parameter")
                    .loc[list(PARAMETER_NAMES)]
                    .fitted.to_numpy()
                )
                for candidate in (0, 1, 2)
            ]
            positions = split.loc[split.split == "heldout", "client_position"].to_numpy()
            clients, first = np.unique(positions, return_index=True)
            decisions = []
            calibration = frames["calibration"]
            calibration = calibration[
                (calibration.context == "outer")
                & (calibration.restart == selection.selected_restart)
            ]
            for client, record in zip(clients, first):
                count = cfg["membership_prefix_samples"]
                prefix_data = MeasuredDataset(
                    data["heldout"].speeds[record : record + 1],
                    data["heldout"].commands[record : record + 1, :count],
                    data["heldout"].measurements[record : record + 1, :count],
                )
                candidate_scores = []
                for candidate, model in enumerate(z):
                    predictions = causal_forecasts(
                        prefix_data,
                        model,
                        dt,
                        q,
                        r,
                        cfg["calibration_horizons_samples"],
                        cfg["calibration_burn_in_samples"],
                    )
                    score = np.mean(
                        [
                            row["sensor_normalized_mse"]
                            for row in predictions
                            if row["mode"] == "forecast"
                        ]
                    )
                    saved = (
                        calibration[
                            (calibration.client_position == client)
                            & (calibration.candidate == candidate)
                        ]
                        .iloc[0]
                        .prefix_score
                    )
                    errors["prefix_score"] = max(errors["prefix_score"], abs(score - saved))
                    candidate_scores.append(score)
                group = 1 + int(np.argmin(candidate_scores[1:]))
                decisions.append(
                    group
                    if candidate_scores[group]
                    < (1 - selection.selected_margin) * candidate_scores[0]
                    else 0
                )
            saved = (
                memberships[
                    (memberships.context == "outer") & (memberships.method == "validated_fallback")
                ]
                .set_index("client_position")
                .loc[clients]
            )
            if selection.validation_selection_available:
                assert np.array_equal(np.array(decisions), saved.selected_component.to_numpy() + 1)
            else:
                assert saved.global_fallback.all()
                decisions = [0] * len(clients)
            forecasts = frames["forecasts"]
            primary = forecasts[forecasts.method == "validated_fallback"]
            for candidate in (0, 1, 2):
                selected_clients = clients[np.array(decisions) == candidate]
                mask = np.isin(positions, selected_clients)
                if not mask.any():
                    continue
                measured = MeasuredDataset(
                    data["heldout"].speeds[mask],
                    data["heldout"].commands[mask],
                    data["heldout"].measurements[mask],
                )
                predicted = causal_forecasts(
                    measured,
                    z[candidate],
                    dt,
                    q,
                    r,
                    cfg["forecast_horizons_samples"],
                    cfg["forecast_burn_in_samples"],
                )
                for expected in predicted[:-1]:
                    scored = primary[
                        primary.client_position.isin(selected_clients)
                        & (primary.horizon_samples == expected["horizon_samples"])
                    ]
                    actual = np.average(
                        scored.sensor_normalized_mse, weights=scored.scored_output_samples
                    )
                    errors["forecast"] = max(
                        errors["forecast"], abs(actual - expected["sensor_normalized_mse"])
                    )
                simulation = np.empty_like(measured.measurements)
                for record, speed in enumerate(measured.speeds):
                    a, b = structured_matrices(z[candidate], speed, dt)
                    state = np.zeros(5)
                    for time, command in enumerate(measured.commands[record]):
                        state = a @ state + b * command
                        simulation[record, time] = measured.measurements[record, time] - C @ state
                expected = output_error_metrics(
                    simulation[:, cfg["forecast_burn_in_samples"] :], r
                )["sensor_normalized_mse"]
                scored = primary[
                    primary.client_position.isin(selected_clients) & (primary.horizon_samples == 0)
                ]
                actual = np.average(
                    scored.sensor_normalized_mse, weights=scored.scored_output_samples
                )
                errors["simulation_suffix"] = max(
                    errors["simulation_suffix"], abs(actual - expected)
                )
            assert (
                forecasts.loc[forecasts["mode"] == "forecast", "origins_per_record"] == 51
            ).all()
    assert max(errors.values()) <= 1e-7
    for name, frames in pieces.items():
        expected = pd.concat(frames, ignore_index=True).to_csv(index=False).encode()
        assert sha(OUT / f"{PREFIX}_{name}.csv") == hashlib.sha256(expected).hexdigest()
    assert json.loads((OUT / f"{PREFIX}_execution_manifest.json").read_text()) == manifests
    for name in ("10hi", "10hj", "10hk"):
        for artifact in ("conclusions", "validation"):
            path = OUT / f"experiment_{name}_{artifact}.json"
            assert path.read_bytes() == subprocess.check_output(
                ["git", "show", f"f27acae:{path.relative_to(ROOT)}"], cwd=ROOT
            )
    result = {
        "execution_jobs": 10,
        "context_caches": context_count,
        "sources_match_committed_revision": True,
        "output_hashes_verified": True,
        "aggregate_tables_reproduce_verified_jobs": True,
        "measured_hashes_verified": True,
        "whole_client_fold_isolation_verified": True,
        "inner_only_selection_reproduced": True,
        "first_record_prefix_choices_independently_reproduced": True,
        "independent_maximum_errors": errors,
        "historical_gate_artifacts_unchanged": True,
        "confirmation_run": False,
        "audit_source_sha256": sha(Path(__file__)),
    }
    (OUT / f"{PREFIX}_result_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def report_results():
    cfg, _ = load_configuration()
    forecasts = pd.read_csv(OUT / f"{PREFIX}_forecasts.csv")
    aggregate = forecasts[forecasts.client_position == -1]
    summary = (
        aggregate.groupby(["process_noise_scale", "method", "mode", "horizon_samples"])
        .sensor_normalized_mse.agg(["mean", "min", "max"])
        .reset_index()
    )
    summary.to_csv(OUT / f"{PREFIX}_forecast_summary.csv", index=False)
    per_client = forecasts[forecasts.client_position >= 0]
    index = ["seed", "process_noise_scale", "client_position", "mode", "horizon_samples"]
    pivot = per_client.pivot(
        index=index, columns="method", values="sensor_normalized_mse"
    ).reset_index()
    paired, client_summary = [], []
    for reference in ("global", "likelihood_MAP", "archived_MAP"):
        for method in METHODS:
            comparison = pivot[index].copy()
            comparison["method"], comparison["reference"] = method, reference
            comparison["error_ratio"] = pivot[method] / pivot[reference]
            comparison["gain_pct"] = 100 * (1 - comparison.error_ratio)
            paired.append(comparison)
            for key, group in comparison.groupby(
                ["process_noise_scale", "mode", "horizon_samples"]
            ):
                client_summary.append(
                    {
                        "process_noise_scale": key[0],
                        "mode": key[1],
                        "horizon_samples": key[2],
                        "method": method,
                        "reference": reference,
                        "improved_clients": int((group.error_ratio < 1).sum()),
                        "degraded_clients": int((group.error_ratio > 1).sum()),
                        "unchanged_clients": int((group.error_ratio == 1).sum()),
                        "median_gain_pct": float(group.gain_pct.median()),
                        "worst_error_ratio": float(group.error_ratio.max()),
                        "mean_log_error_ratio": float(np.log(group.error_ratio).mean()),
                    }
                )
    pd.concat(paired, ignore_index=True).to_csv(OUT / f"{PREFIX}_paired_clients.csv", index=False)
    pd.DataFrame(client_summary).to_csv(OUT / f"{PREFIX}_client_summary.csv", index=False)
    tails = (
        per_client.groupby(["process_noise_scale", "method", "mode", "horizon_samples"])
        .sensor_normalized_mse.agg(
            median="median", percentile90=lambda values: values.quantile(0.9), maximum="max"
        )
        .reset_index()
    )
    tails.to_csv(OUT / f"{PREFIX}_absolute_client_tail.csv", index=False)
    fleet = []
    for key, group in aggregate.groupby(["process_noise_scale", "mode", "horizon_samples"]):
        table = group.pivot(index="seed", columns="method", values="sensor_normalized_mse")
        for reference in ("global", "likelihood_MAP", "archived_MAP"):
            for method in METHODS:
                gain = 100 * (1 - table[method] / table[reference])
                fleet.append(
                    {
                        "process_noise_scale": key[0],
                        "mode": key[1],
                        "horizon_samples": key[2],
                        "method": method,
                        "reference": reference,
                        "mean_gain_pct": float(gain.mean()),
                        "minimum_gain_pct": float(gain.min()),
                        "maximum_gain_pct": float(gain.max()),
                        "improved_fleets": int((gain > 0).sum()),
                    }
                )
    pd.DataFrame(fleet).to_csv(OUT / f"{PREFIX}_paired_fleet_summary.csv", index=False)
    membership = pd.read_csv(OUT / f"{PREFIX}_memberships.csv")
    outer = membership[membership.context == "outer"]
    outer.groupby(["process_noise_scale", "method"]).global_fallback.sum().reset_index().to_csv(
        OUT / f"{PREFIX}_fallback_counts.csv", index=False
    )
    repeated = pd.read_csv(OUT / f"{PREFIX}_restart_forecasts.csv")
    repeated = repeated[(repeated.client_position >= 0) & (repeated.horizon_samples == 50)]
    variability = (
        repeated.groupby(["seed", "process_noise_scale", "client_position"])
        .sensor_normalized_mse.agg(["min", "max", "mean", "std"])
        .reset_index()
    )
    variability["maximum_minimum_error_ratio"] = variability["max"] / variability["min"]
    variability.to_csv(OUT / f"{PREFIX}_restart_prediction_variability.csv", index=False)
    selected = pd.read_csv(OUT / f"{PREFIX}_selection.csv")
    fits = pd.read_csv(OUT / f"{PREFIX}_fits.csv")
    repeatability = []
    parameter_spread = []
    for _, row in selected.iterrows():
        local = membership[
            (membership.context == "full")
            & (membership.seed == row.seed)
            & (membership.process_noise_scale == row.process_noise_scale)
        ]
        chosen = (
            local[local.restart == row.selected_restart]
            .sort_values("client_position")
            .selected_component.to_numpy()
            .astype(int)
        )
        full_fits = fits[
            (fits.context == "full")
            & (fits.model == "K2")
            & (fits.seed == row.seed)
            & (fits.process_noise_scale == row.process_noise_scale)
        ].sort_values("restart")
        target = (
            full_fits[full_fits.restart == row.selected_restart]
            .iloc[0][[f"value_{index}" for index in range(16)]]
            .to_numpy(dtype=float)
            .reshape(2, 8)
        )
        aligned = []
        for _, fit in full_fits.iterrows():
            plants = (
                fit[[f"value_{index}" for index in range(16)]].to_numpy(dtype=float).reshape(2, 8)
            )
            swap = np.sum((plants[::-1] - target) ** 2) < np.sum((plants - target) ** 2)
            aligned.append(plants[::-1] if swap else plants)
        aligned = np.exp(np.asarray(aligned))
        cv = aligned.std(axis=0) / aligned.mean(axis=0)
        for component in range(2):
            for index, name in enumerate(PHYSICAL_NAMES):
                parameter_spread.append(
                    {
                        "seed": int(row.seed),
                        "process_noise_scale": row.process_noise_scale,
                        "component": component,
                        "physical_coordinate": name,
                        "minimum_nominal_multiplier": float(aligned[:, component, index].min()),
                        "maximum_nominal_multiplier": float(aligned[:, component, index].max()),
                        "coefficient_of_variation": float(cv[component, index]),
                    }
                )
        repeatability.append(
            {
                "seed": int(row.seed),
                "process_noise_scale": row.process_noise_scale,
                "maximum_aligned_physical_coordinate_cv": float(cv.max()),
                "full_objective_spread_pct": float(
                    100 * np.ptp(full_fits.objective) / abs(full_fits.objective.min())
                ),
                "selected_active_coefficient_bounds": int(
                    full_fits[full_fits.restart == row.selected_restart]
                    .iloc[0]
                    .active_coefficient_bounds
                ),
                "minimum_restart_partition_ari": min(
                    partition_ari(
                        chosen,
                        local[local.restart == restart]
                        .sort_values("client_position")
                        .selected_component.to_numpy()
                        .astype(int),
                    )
                    for restart in range(cfg["restart_count"])
                ),
            }
        )
    pd.DataFrame(repeatability).to_csv(OUT / f"{PREFIX}_repeatability.csv", index=False)
    pd.DataFrame(parameter_spread).to_csv(
        OUT / f"{PREFIX}_aligned_parameter_spread.csv", index=False
    )
    inner = pd.read_csv(OUT / f"{PREFIX}_inner_scores.csv")
    fold_winners = []
    for key, records in inner.groupby(["seed", "process_noise_scale", "fold"]):
        scores = records.groupby(["restart", "margin"]).selection_score.mean().reset_index()
        ties = scores[
            scores.selection_score <= scores.selection_score.min() + cfg["selection_tie_tolerance"]
        ]
        winner = ties.sort_values(["margin", "restart"], ascending=[False, True]).iloc[0]
        fold_winners.append(
            {
                "seed": int(key[0]),
                "process_noise_scale": key[1],
                "fold": int(key[2]),
                "restart": int(winner.restart),
                "margin": float(winner.margin),
                "selection_score": float(winner.selection_score),
            }
        )
    pd.DataFrame(fold_winners).to_csv(OUT / f"{PREFIX}_descriptive_fold_winners.csv", index=False)
    residuals = pd.read_csv(OUT / f"{PREFIX}_residuals.csv")
    residuals.groupby(["process_noise_scale", "method"]).mean(numeric_only=True).drop(
        columns="seed"
    ).reset_index().to_csv(OUT / f"{PREFIX}_residual_summary.csv", index=False)
    numerical = {
        "fit_records": len(fits),
        "new_fits": int((~fits.reused).sum()),
        "reused_fits": int(fits.reused.sum()),
        "successful_fits": int(fits.success.sum()),
        "stationary_feasible_fits": int(
            (
                (fits.kkt_residual <= cfg["maximum_kkt_residual"])
                & (fits.constraint_violation <= cfg["maximum_constraint_violation"])
            ).sum()
        ),
        "maximum_kkt_residual": float(fits.kkt_residual.max()),
        "maximum_constraint_violation": float(fits.constraint_violation.max()),
        "invalid_fitting_evaluations": int(fits.invalid_evaluations.sum()),
        "maximum_coupling_log_residual": float(
            pd.read_csv(OUT / f"{PREFIX}_parameters.csv").coupling_log_residual.abs().max()
        ),
        "all_jobs_have_validated_selection": bool(selected.validation_selection_available.all()),
        "selected_margins": selected.selected_margin.to_list(),
        "primary_0p5s_comparison": [
            record
            for record in fleet
            if record["horizon_samples"] == 50 and record["method"] == "validated_fallback"
        ],
        "primary_0p5s_client_outcomes": [
            record
            for record in client_summary
            if record["horizon_samples"] == 50 and record["method"] == "validated_fallback"
        ],
        "selection_interpretation": "Primary improves six-start likelihood MAP in all five Q=0.01 fleets, but worsens it in all five Q=1 fleets. Global-relative losses decline, with no per-client guarantee. The initialization-index transfer procedure is not validated as a reliable improvement over likelihood selection.",
        "confirmation_run": False,
        "reserved_confirmation_seeds": cfg["reserved_confirmation_seeds"],
        "limitations": "Opened development, only two inner folds, strategy transfer can reach different full-fit basins. Fixed covariance remains uncalibrated; bounds remain heuristic. Prefix fallback is empirical with no per-client guarantee; no state, controller or FL communication claim.",
    }
    (OUT / f"{PREFIX}_conclusions.json").write_text(
        json.dumps(numerical, indent=2, default=lambda value: value.item()) + "\n"
    )
    plot_results(aggregate, pivot)
    print(summary[(summary.horizon_samples == 50)].to_string(index=False))
    print(json.dumps(numerical, indent=2, default=lambda value: value.item()))


def plot_results(aggregate, clients):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    colors = {
        "global": "black",
        "archived_MAP": "tab:gray",
        "likelihood_MAP": "tab:blue",
        "validated_MAP": "tab:orange",
        "validated_group": "tab:purple",
        "validated_fallback": "tab:green",
    }
    for row, scale in enumerate((0.01, 1.0)):
        local = aggregate[
            (aggregate.process_noise_scale == scale) & (aggregate["mode"] == "forecast")
        ]
        for method, color in colors.items():
            curve = (
                local[local.method == method]
                .groupby("horizon_samples")
                .sensor_normalized_mse.mean()
            )
            axes[row, 0].plot(curve.index * 0.01, curve, "o-", label=LABELS[method], color=color)
        axes[row, 0].set(
            title=f"Q scale {scale:g}: identical future windows",
            yscale="log",
            xlabel="Forecast horizon (s)",
            ylabel="Fixed-R normalized error",
        )
        axes[row, 0].legend(fontsize=7)
        points = clients[(clients.process_noise_scale == scale) & (clients.horizon_samples == 50)]
        for seed in sorted(points.seed.unique()):
            fleet = points[points.seed == seed]
            axes[row, 1].scatter(
                fleet.likelihood_MAP, fleet.validated_fallback, s=23, label=str(seed)
            )
        limits = [
            min(points.likelihood_MAP.min(), points.validated_fallback.min()) * 0.8,
            max(points.likelihood_MAP.max(), points.validated_fallback.max()) * 1.2,
        ]
        axes[row, 1].plot(limits, limits, "k--", linewidth=0.7)
        axes[row, 1].set(
            xscale="log",
            yscale="log",
            xlim=limits,
            ylim=limits,
            title="Client 0.5-second error: primary vs likelihood",
            xlabel="Six-start likelihood-MAP error",
            ylabel="Validated fallback error",
        )
        axes[row, 1].legend(fontsize=7, title="Fleet seed", title_fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT / "results/figures/experiment_10hl_validated_selection.pdf")
    plt.close(fig)


if __name__ == "__main__":
    audit_execution()
    report_results()
