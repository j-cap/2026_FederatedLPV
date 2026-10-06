"""Post-hoc innovation calibration audit; does not refit or change 10H-I gates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from experiment_10hf_output_identifiability import measurement_covariance
from experiment_10hh_single_model_repair import measured_datasets
from experiment_10hi_physically_coupled_fit import (
    OUT,
    PREFIX,
    ROOT,
    load_configuration,
    make_coordinates,
)

from federated_lpv.innovation_likelihood import PARAMETER_NAMES, C, steady_filter


def whitened_diagnostics(data, z, dt, q, r):
    filters = {speed: steady_filter(z, speed, dt, q, r) for speed in np.unique(data.speeds)}
    whitened = np.empty_like(data.measurements)
    for index, speed in enumerate(data.speeds):
        f = filters[speed]
        factor = np.linalg.cholesky(f["s"])
        posterior = np.zeros(5)
        for time, (command, measured) in enumerate(
            zip(data.commands[index], data.measurements[index])
        ):
            predicted = f["a"] @ posterior + f["b"] * command
            innovation = measured - C @ predicted
            whitened[index, time] = np.linalg.solve(factor, innovation)
            posterior = predicted + f["k"] @ innovation
    flat = whitened.reshape(-1, 3)
    mean = flat.mean(axis=0)
    covariance = np.cov(flat, rowvar=False, bias=True)
    previous = (whitened[:, :-1] - mean).reshape(-1, 3)
    following = (whitened[:, 1:] - mean).reshape(-1, 3)
    lag = np.sum(previous * following, axis=0) / np.sqrt(
        np.sum(previous**2, axis=0) * np.sum(following**2, axis=0)
    )
    return {
        "normalized_innovation_squared_mean": float(np.mean(flat**2)),
        "whitened_covariance_minimum_eigenvalue": float(np.linalg.eigvalsh(covariance)[0]),
        "whitened_covariance_maximum_eigenvalue": float(np.linalg.eigvalsh(covariance)[-1]),
        "maximum_absolute_lag_one_correlation": float(np.max(np.abs(lag))),
        "yaw_lag_one_correlation": float(lag[0]),
        "acceleration_lag_one_correlation": float(lag[1]),
        "steering_lag_one_correlation": float(lag[2]),
        "maximum_absolute_whitened_mean": float(np.max(np.abs(mean))),
    }


def main():
    cfg, inherited = load_configuration()
    coordinates = make_coordinates(cfg, inherited["10hf"])
    runs = pd.read_csv(OUT / f"{PREFIX}_runs.csv")
    parameters = pd.read_csv(OUT / f"{PREFIX}_parameters.csv")
    rows = []
    for seed in cfg["development_seeds"]:
        data, _ = measured_datasets(seed, cfg, inherited)
        for part in ["A", "B"]:
            local = parameters[
                (parameters.seed == seed)
                & (parameters.part == part)
                & (parameters.method == "staged")
            ]
            fitted = np.log(
                local.set_index("parameter").loc[list(PARAMETER_NAMES), "fitted"].to_numpy()
            )
            scale = float(
                runs[
                    (runs.seed == seed) & (runs.part == part) & (runs.method == "staged")
                ].process_noise_scale.iloc[0]
            )
            q = np.diag(np.asarray(inherited["10ha"]["base_process_noise_diagonal"]) * scale)
            r = measurement_covariance(inherited["10ha"])
            for model, z in [("nominal", coordinates.nominal), ("fitted", fitted)]:
                diagnostic = whitened_diagnostics(
                    data["heldout"], z, inherited["10h"]["sample_time"], q, r
                )
                rows.append(
                    dict(seed=seed, part=part, model=model, process_noise_scale=scale, **diagnostic)
                )
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / f"{PREFIX}_innovation_calibration.csv", index=False)
    paths = [
        Path(__file__),
        OUT / f"{PREFIX}_runs.csv",
        OUT / f"{PREFIX}_parameters.csv",
        ROOT / "code/src/federated_lpv/innovation_likelihood.py",
    ]
    output = {
        "posthoc_diagnostic": True,
        "fitting_selection_and_gates_unchanged": True,
        "interpretation": "Ideal correctly specified Gaussian innovations have normalized NIS one, identity covariance, and zero serial correlation. These diagnostics assess the working-model approximation; Q was selected for fixed-R prediction, not innovation calibration. No latent state or client truth is used.",
        "provenance_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths
        },
    }
    (OUT / f"{PREFIX}_innovation_calibration_conclusions.json").write_text(
        json.dumps(output, indent=2) + "\n"
    )
    print(
        frame[frame.model == "fitted"]
        .groupby("part")
        .agg(
            normalized_NIS_mean=("normalized_innovation_squared_mean", "mean"),
            maximum_absolute_lag_one_correlation=("maximum_absolute_lag_one_correlation", "max"),
        )
        .to_string(),
        flush=True,
    )


if __name__ == "__main__":
    main()
