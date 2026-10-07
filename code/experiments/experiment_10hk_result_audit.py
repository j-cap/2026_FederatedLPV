"""Independent provenance, client-calibration and forecast audit after 10H-K."""

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
from experiment_10hi_physically_coupled_fit import make_coordinates
from experiment_10hk_joint_client_mixture import (
    OUT,
    PREFIX,
    ROOT,
    TABLES,
    job_prefix,
    load_configuration,
    sha,
    source_hashes,
)
from scipy.special import logsumexp

from federated_lpv.client_mixture import ClientLikelihood, ClientMixture
from federated_lpv.innovation_likelihood import (
    PARAMETER_NAMES,
    InnovationLikelihood,
    MeasuredDataset,
)
from federated_lpv.output_validation import causal_forecasts


def audit_results():
    cfg, inherited = load_configuration()
    maximum_objective_error = maximum_membership_error = maximum_forecast_error = 0.0
    manifest_count = 0
    refinements = []
    dt = inherited["10h"]["sample_time"]
    r = measurement_covariance(inherited["10ha"])
    for seed in cfg["development_seeds"]:
        for scale in cfg["process_noise_scales"]:
            prefix = job_prefix(seed, scale)
            manifest = json.loads((OUT / f"{prefix}_complete.json").read_text())
            assert manifest["source_sha256"] == source_hashes(seed, scale)
            assert not manifest["confirmation_run"]
            for path, expected in manifest["source_sha256"].items():
                original = subprocess.check_output(
                    ["git", "show", f"{manifest['revision']}:{path}"], cwd=ROOT
                )
                assert hashlib.sha256(original).hexdigest() == expected
            frames = {}
            for name in TABLES:
                path = OUT / f"{prefix}_{name}.csv"
                assert sha(path) == manifest["output_sha256"][path.name]
                frames[name] = pd.read_csv(path)
            data, split = measured_datasets(seed, cfg, inherited)
            assert {name: dataset_hash(value) for name, value in data.items()} == manifest[
                "measured_array_sha256"
            ]
            q = np.diag(np.asarray(inherited["10ha"]["base_process_noise_diagonal"]) * scale)
            if (
                frames["gradient_checks"].norm_relative_error.max()
                > cfg["maximum_gradient_relative_error"]
            ):
                train_positions = split.loc[split.split == "train", "client_position"].to_numpy()
                evaluator = ClientMixture(
                    ClientLikelihood(
                        data["train"],
                        train_positions,
                        dt,
                        q,
                        r,
                        make_coordinates(cfg, inherited["10hf"]),
                    )
                )
                initial_rows = frames["initials"]
                point = (
                    initial_rows[initial_rows.restart == 0].sort_values("index").value.to_numpy()
                )
                analytic = evaluator.value_gradient(point)[1]
                rows = []
                for step in (1e-5, 5e-6, 2.5e-6):
                    numeric = np.array(
                        [
                            (
                                evaluator.value(point + np.eye(17)[j] * step)
                                - evaluator.value(point - np.eye(17)[j] * step)
                            )
                            / (2 * step)
                            for j in range(17)
                        ]
                    )
                    rows.append(
                        {
                            "step": step,
                            "relative_error": float(
                                np.linalg.norm(analytic - numeric) / np.linalg.norm(numeric)
                            ),
                        }
                    )
                refinements.append(
                    {
                        "seed": seed,
                        "process_noise_scale": scale,
                        "restart": 0,
                        "point": "unchanged predeclared first split",
                        "original_declared_check_passed": False,
                        "refinement": rows,
                        "fits_or_criteria_changed": False,
                    }
                )
            positions = split.loc[split.split == "heldout", "client_position"].to_numpy()
            clients, first = np.unique(positions, return_index=True)
            runs = frames["runs"]
            selected = runs[runs.component_count == 2].iloc[0]
            restarts = frames["restarts"]
            if not selected.collapsed_reference_selected:
                best = restarts.loc[restarts.objective.idxmin()]
                assert selected.best_restart == best.restart
                maximum_objective_error = max(
                    maximum_objective_error, abs(selected.train_objective - best.objective)
                )
            parameters = frames["parameters"]
            models = []
            for group in (0, 1):
                rows = parameters[(parameters.model == "K2") & (parameters.component == group)]
                models.append(
                    np.log(rows.set_index("parameter").loc[list(PARAMETER_NAMES)].fitted.to_numpy())
                )
            mixing = np.array([selected.mixing_weight_0, 1 - selected.mixing_weight_0])
            costs = []
            for record in first:
                count = cfg["membership_prefix_samples"]
                prefix_data = MeasuredDataset(
                    data["heldout"].speeds[record : record + 1],
                    data["heldout"].commands[record : record + 1, :count],
                    data["heldout"].measurements[record : record + 1, :count],
                )
                costs.append(
                    [
                        InnovationLikelihood(prefix_data, dt, q, r).value(z) * 3 * count / 2
                        for z in models
                    ]
                )
            scores = np.log(mixing)[None] - np.array(costs)
            weights = np.exp(scores - logsumexp(scores, axis=1)[:, None])
            saved = frames["memberships"]
            saved = saved[saved.split == "heldout_prefix"].set_index("client_position").loc[clients]
            maximum_membership_error = max(
                maximum_membership_error,
                float(np.max(np.abs(weights - saved[["weight_0", "weight_1"]].to_numpy()))),
            )
            selected_groups = np.argmax(weights, axis=1)
            full = frames["forecasts"]
            for group in (0, 1):
                mask = np.isin(positions, clients[selected_groups == group])
                if not mask.any():
                    continue
                subset = MeasuredDataset(
                    data["heldout"].speeds[mask],
                    data["heldout"].commands[mask],
                    data["heldout"].measurements[mask],
                )
                independent = causal_forecasts(
                    subset,
                    models[group],
                    dt,
                    q,
                    r,
                    cfg["forecast_horizons_samples"],
                    cfg["forecast_burn_in_samples"],
                )
                for expected in independent[:-1]:
                    per_client = full[
                        (full.model == "K2")
                        & (full.rule == "map")
                        & (full.horizon_samples == expected["horizon_samples"])
                        & full.client_position.isin(clients[selected_groups == group])
                    ]
                    actual = np.average(
                        per_client.sensor_normalized_mse, weights=per_client.scored_output_samples
                    )
                    maximum_forecast_error = max(
                        maximum_forecast_error, abs(actual - expected["sensor_normalized_mse"])
                    )
            assert np.all(full.loc[full["mode"] == "forecast", "origins_per_record"] == 101)
            manifest_count += 1
    for name in ("10hi", "10hj"):
        for artifact in ("conclusions", "validation"):
            path = OUT / f"experiment_{name}_{artifact}.json"
            original = subprocess.check_output(
                ["git", "show", f"c7abcdf:{path.relative_to(ROOT)}"], cwd=ROOT
            )
            assert path.read_bytes() == original
    for artifact in ("conclusions", "bound_diagnostic_conclusions"):
        path = OUT / f"experiment_10hh_{artifact}.json"
        original = subprocess.check_output(
            ["git", "show", f"c7abcdf:{path.relative_to(ROOT)}"], cwd=ROOT
        )
        assert path.read_bytes() == original
    assert maximum_objective_error <= 1e-12
    assert maximum_membership_error <= 1e-9
    assert maximum_forecast_error <= 1e-7
    forecasts = pd.read_csv(OUT / f"{PREFIX}_forecasts.csv")
    per_client = forecasts[(forecasts.client_position >= 0) & (forecasts.rule == "map")]
    paired = per_client.pivot(
        index=["seed", "process_noise_scale", "client_position", "mode", "horizon_samples"],
        columns="model",
        values="sensor_normalized_mse",
    ).reset_index()
    paired["gain_pct"] = 100 * (paired.K1 - paired.K2) / paired.K1
    paired.to_csv(OUT / f"{PREFIX}_paired_clients.csv", index=False)
    paired.groupby(["process_noise_scale", "mode", "horizon_samples"]).gain_pct.agg(
        ["mean", "median", "min", "max", lambda a: (a > 0).sum()]
    ).rename(columns={"<lambda_0>": "improved_clients_of_50"}).reset_index().to_csv(
        OUT / f"{PREFIX}_client_summary.csv", index=False
    )
    (OUT / f"{PREFIX}_gradient_refinement.json").write_text(
        json.dumps(refinements, indent=2) + "\n"
    )
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for axis, scale in zip(axes, cfg["process_noise_scales"]):
        local = paired[(paired.process_noise_scale == scale) & (paired.horizon_samples == 50)]
        for seed in cfg["development_seeds"]:
            fleet = local[local.seed == seed]
            axis.scatter(fleet.K1, fleet.K2, s=22, label=str(seed))
        limits = [
            min(local.K1.min(), local.K2.min()) * 0.8,
            max(local.K1.max(), local.K2.max()) * 1.2,
        ]
        axis.plot(limits, limits, color="black", linestyle="--", linewidth=0.8)
        axis.set(
            xscale="log",
            yscale="log",
            xlim=limits,
            ylim=limits,
            title=f"Q scale {scale:g}: 50 held-out clients",
            xlabel="K=1 fixed-R error",
            ylabel="K=2 prefix-MAP fixed-R error",
        )
        axis.legend(fontsize=7, title="Fleet seed", title_fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT / "results/figures/experiment_10hk_client_forecasts.pdf")
    plt.close(fig)
    result = {
        "execution_manifests": manifest_count,
        "sources_match_committed_revision": True,
        "output_hashes_verified": True,
        "measured_hashes_verified": True,
        "training_only_restart_selection_verified": True,
        "independent_first_record_membership_maximum_error": maximum_membership_error,
        "independent_MAP_forecast_maximum_error": maximum_forecast_error,
        "maximum_selected_objective_error": maximum_objective_error,
        "historical_gate_artifacts_unchanged": True,
        "confirmation_run": False,
        "original_gradient_check_failures": len(refinements),
        "finer_gradient_checks_passed": all(
            row["refinement"][-1]["relative_error"] <= cfg["maximum_gradient_relative_error"]
            for row in refinements
        ),
        "fitting_or_protocol_changed": False,
        "audit_source_sha256": sha(Path(__file__)),
    }
    (OUT / f"{PREFIX}_result_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    audit_results()
