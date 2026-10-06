"""Post-fit reporting of all predeclared 10H-J models; no model selection."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/tables"
PREFIX = "experiment_10hj"


def main():
    runs = pd.read_csv(OUT / f"{PREFIX}_runs.csv")
    forecasts = pd.read_csv(OUT / f"{PREFIX}_forecasts.csv")
    residuals = pd.read_csv(OUT / f"{PREFIX}_residuals.csv")
    physical = pd.read_csv(OUT / f"{PREFIX}_physical_parameters.csv")
    paired = []
    panels = {}
    for scale, local in runs.groupby("process_noise_scale"):
        original = local[local.box == "original"].set_index("seed")
        for box in ("wider", "widest"):
            other = local[local.box == box].set_index("seed")
            for seed in original.index:
                paired.append(
                    {
                        "seed": seed,
                        "process_noise_scale": scale,
                        "box": box,
                        "heldout_full_one_step_reduction_pct": 100
                        * (
                            original.loc[seed, "heldout_sensor_mse"]
                            - other.loc[seed, "heldout_sensor_mse"]
                        )
                        / original.loc[seed, "heldout_sensor_mse"],
                        "training_objective_reduction_pct": 100
                        * (
                            original.loc[seed, "train_objective"]
                            - other.loc[seed, "train_objective"]
                        )
                        / original.loc[seed, "train_objective"],
                    }
                )
        widest = local[local.box == "widest"]
        res = residuals[
            (residuals.process_noise_scale == scale)
            & (residuals.box == "widest")
            & (residuals.model == "fitted")
        ]
        phys = physical[(physical.process_noise_scale == scale) & (physical.box == "widest")]
        panels[str(scale)] = {
            "widest_active_bound_fleets": int((widest.active_coefficient_bounds > 0).sum()),
            "widest_all_profiles_two_sided": bool(widest.all_profiles_two_sided.all()),
            "minimum_local_profile_edge_increase_pct": float(
                widest.minimum_profile_edge_increase_pct.min()
            ),
            "mean_normalized_NIS": float(res.normalized_innovation_squared_mean.mean()),
            "maximum_within_record_autocorrelation_lags_1_20": float(
                res.maximum_absolute_within_record_autocorrelation.max()
            ),
            "maximum_within_record_past_input_correlation_lags_1_20": float(
                res.maximum_absolute_within_record_past_input.max()
            ),
            "maximum_fitted_plant_spectral_radius": float(res.maximum_plant_spectral_radius.max()),
            "physical_parameter_ranges": {
                name: {"minimum": float(values.fitted.min()), "maximum": float(values.fitted.max())}
                for name, values in phys.groupby("parameter")
            },
        }
    paired = pd.DataFrame(paired)
    paired.to_csv(OUT / f"{PREFIX}_paired_comparison.csv", index=False)
    forecast_pairs = []
    for (scale, seed, mode, horizon), local in forecasts.groupby(
        ["process_noise_scale", "seed", "mode", "horizon_samples"]
    ):
        original = float(
            local[(local.box == "original") & (local.model == "fitted")].sensor_normalized_mse.iloc[
                0
            ]
        )
        nominal = float(
            local[
                (local.box == "original") & (local.model == "nominal")
            ].sensor_normalized_mse.iloc[0]
        )
        for box in ("original", "wider", "widest"):
            value = float(
                local[(local.box == box) & (local.model == "fitted")].sensor_normalized_mse.iloc[0]
            )
            forecast_pairs.append(
                {
                    "seed": seed,
                    "process_noise_scale": scale,
                    "box": box,
                    "mode": mode,
                    "horizon_samples": horizon,
                    "fitted_sensor_mse": value,
                    "original_sensor_mse": original,
                    "nominal_sensor_mse": nominal,
                    "reduction_vs_original_pct": 100 * (original - value) / original,
                    "reduction_vs_nominal_pct": 100 * (nominal - value) / nominal,
                }
            )
    forecast_pairs = pd.DataFrame(forecast_pairs)
    forecast_pairs.to_csv(OUT / f"{PREFIX}_paired_forecasts.csv", index=False)
    comparison = (
        forecast_pairs.groupby(["process_noise_scale", "box", "mode", "horizon_samples"])
        .agg(
            mean_reduction_vs_original_pct=("reduction_vs_original_pct", "mean"),
            minimum_reduction_vs_original_pct=("reduction_vs_original_pct", "min"),
            maximum_reduction_vs_original_pct=("reduction_vs_original_pct", "max"),
            mean_reduction_vs_nominal_pct=("reduction_vs_nominal_pct", "mean"),
            fleets_worse_than_original=(
                "reduction_vs_original_pct",
                lambda x: int((x < -1e-5).sum()),
            ),
            fleets_worse_than_nominal=(
                "reduction_vs_nominal_pct",
                lambda x: int((x < -1e-5).sum()),
            ),
        )
        .reset_index()
    )
    comparison.to_csv(OUT / f"{PREFIX}_forecast_comparison_summary.csv", index=False)
    sources = [
        Path(__file__),
        *[
            OUT / f"{PREFIX}_{name}.csv"
            for name in ("runs", "forecasts", "residuals", "physical_parameters")
        ],
    ]
    report = {
        "post_fit_reporting_only": True,
        "no_model_bound_covariance_or_gate_selected": True,
        "panels": panels,
        "provenance_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources
        },
    }
    (OUT / f"{PREFIX}_result_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        paired.groupby(["process_noise_scale", "box"])
        .heldout_full_one_step_reduction_pct.agg(["mean", "min", "max"])
        .to_string()
    )
    print(comparison[comparison.box == "widest"].to_string(index=False))
    print(json.dumps(panels, indent=2))


if __name__ == "__main__":
    main()
