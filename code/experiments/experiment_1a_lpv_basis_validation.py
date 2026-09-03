"""Experiment 1A: compare LPV bases on exact frozen-speed matrices."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from federated_lpv import discrete_bicycle_matrices, fit_lpv_matrix_model, sample_fleet


ROOT = Path(__file__).resolve().parents[2]
FIGURE = ROOT / "results" / "figures" / "experiment_1a_basis_errors.pdf"
DETAIL_TABLE = ROOT / "results" / "tables" / "experiment_1a_client_basis_metrics.csv"
SUMMARY_TABLE = ROOT / "results" / "tables" / "experiment_1a_basis_summary.csv"
SUMMARY_JSON = ROOT / "results" / "tables" / "experiment_1a_summary.json"
SAMPLE_TIME = 0.01
FIT_SPEEDS = np.asarray([12.0, 16.0, 20.0, 24.0, 28.0])
INTERPOLATION_SPEEDS = np.asarray([14.0, 18.0, 22.0, 26.0])
DENSE_SPEEDS = np.linspace(10.0, 30.0, 201)
INTERIOR_SPEEDS = DENSE_SPEEDS[(DENSE_SPEEDS >= FIT_SPEEDS.min()) & (DENSE_SPEEDS <= FIT_SPEEDS.max())]
EDGE_SPEEDS = DENSE_SPEEDS[(DENSE_SPEEDS < FIT_SPEEDS.min()) | (DENSE_SPEEDS > FIT_SPEEDS.max())]
BASES = ("linear", "reciprocal", "mixed")
SEEDS = tuple(range(1, 11))


def true_matrices(client, speeds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pairs = [discrete_bicycle_matrices(float(speed), client.parameters, SAMPLE_TIME) for speed in speeds]
    return np.asarray([pair[0] for pair in pairs]), np.asarray([pair[1] for pair in pairs])


def relative_errors(truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    numerator = np.linalg.norm(prediction - truth, axis=(1, 2))
    denominator = np.linalg.norm(truth, axis=(1, 2))
    return numerator / denominator


def evaluate_client(seed: int, client, basis: str) -> tuple[dict[str, float | str | int], dict[str, np.ndarray]]:
    fit_a, fit_b = true_matrices(client, FIT_SPEEDS)
    model_a, condition = fit_lpv_matrix_model(FIT_SPEEDS, fit_a, basis)
    model_b, condition_b = fit_lpv_matrix_model(FIT_SPEEDS, fit_b, basis)
    if not np.isclose(condition, condition_b):
        raise RuntimeError("A and B fits unexpectedly use different design matrices")

    regions = {
        "anchors": FIT_SPEEDS,
        "unseen_interpolation": INTERPOLATION_SPEEDS,
        "dense_interior": INTERIOR_SPEEDS,
        "edge_extrapolation": EDGE_SPEEDS,
        "full_envelope": DENSE_SPEEDS,
    }
    errors: dict[str, np.ndarray] = {}
    row: dict[str, float | str | int] = {
        "seed": seed,
        "client_id": client.client_id,
        "family": client.family,
        "basis": basis,
        "terms": model_a.coefficients.shape[0],
        "standardized_design_condition": condition,
        "max_abs_coefficient_a": float(np.max(np.abs(model_a.coefficients))),
        "max_abs_coefficient_b": float(np.max(np.abs(model_b.coefficients))),
    }
    for region, speeds in regions.items():
        truth_a, truth_b = true_matrices(client, speeds)
        error_a = relative_errors(truth_a, model_a.predict(speeds))
        error_b = relative_errors(truth_b, model_b.predict(speeds))
        combined = np.sqrt(0.5 * (error_a**2 + error_b**2))
        row[f"{region}_mean_relative_a"] = float(np.mean(error_a))
        row[f"{region}_max_relative_a"] = float(np.max(error_a))
        row[f"{region}_mean_relative_b"] = float(np.mean(error_b))
        row[f"{region}_max_relative_b"] = float(np.max(error_b))
        row[f"{region}_mean_combined"] = float(np.mean(combined))
        row[f"{region}_max_combined"] = float(np.max(combined))
        if region == "full_envelope":
            errors = {"speed": speeds, "a": error_a, "b": error_b, "combined": combined}
    return row, errors


def aggregate(rows: list[dict[str, float | str | int]]) -> list[dict[str, float | str | int]]:
    summary = []
    for basis in BASES:
        selected = [row for row in rows if row["basis"] == basis]
        item: dict[str, float | str | int] = {
            "basis": basis,
            "terms": selected[0]["terms"],
            "clients": len(selected),
            "standardized_design_condition": selected[0]["standardized_design_condition"],
        }
        for region in ("anchors", "unseen_interpolation", "dense_interior", "edge_extrapolation", "full_envelope"):
            for matrix in ("a", "b", "combined"):
                mean_key = f"{region}_mean_relative_{matrix}" if matrix != "combined" else f"{region}_mean_combined"
                max_key = f"{region}_max_relative_{matrix}" if matrix != "combined" else f"{region}_max_combined"
                item[f"{region}_fleet_mean_{matrix}"] = float(np.mean([row[mean_key] for row in selected]))
                item[f"{region}_worst_client_{matrix}"] = float(np.max([row[max_key] for row in selected]))
        summary.append(item)
    return summary


def make_figure(development_errors: dict[tuple[str, str], dict[str, np.ndarray]], summary: list[dict[str, object]]) -> None:
    # Representative nominal client from seed 1 plus population-level summary.
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.0), constrained_layout=True)
    for basis in BASES:
        errors = development_errors[("nominal_00", basis)]
        axes[0].semilogy(errors["speed"], errors["a"], label=basis)
        axes[1].semilogy(errors["speed"], errors["b"], label=basis)
    axes[0].set(title=r"Representative $A_d$ error", ylabel="relative Frobenius error")
    axes[1].set(title=r"Representative $B_d$ error", ylabel="relative Frobenius error")
    for axis in axes[:2]:
        for speed in FIT_SPEEDS:
            axis.axvline(speed, color="0.8", linewidth=0.5)
        axis.set_xlabel("speed [m/s]")
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)

    positions = np.arange(len(BASES))
    interpolation = [100 * row["unseen_interpolation_worst_client_combined"] for row in summary]
    edge = [100 * row["edge_extrapolation_worst_client_combined"] for row in summary]
    width = 0.36
    axes[2].bar(positions - width / 2, interpolation, width, label="interpolation")
    axes[2].bar(positions + width / 2, edge, width, label="edge extrapolation")
    axes[2].set(
        xticks=positions,
        xticklabels=BASES,
        ylabel="worst-client combined error [%]",
        title="All 300 clients",
    )
    axes[2].set_yscale("log")
    axes[2].grid(axis="y", alpha=0.25)
    axes[2].legend(frameon=False)
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, bbox_inches="tight")
    plt.close(fig)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rows = []
    development_errors = {}
    for seed in SEEDS:
        for client in sample_fleet(seed):
            for basis in BASES:
                row, errors = evaluate_client(seed, client, basis)
                rows.append(row)
                if seed == 1:
                    development_errors[(client.client_id, basis)] = errors
    summary = aggregate(rows)
    compact_summary = {
        row["basis"]: {
            "terms": row["terms"],
            "standardized_design_condition": row["standardized_design_condition"],
            "interpolation_fleet_mean_combined": row["unseen_interpolation_fleet_mean_combined"],
            "interpolation_worst_client_combined": row["unseen_interpolation_worst_client_combined"],
            "edge_fleet_mean_combined": row["edge_extrapolation_fleet_mean_combined"],
            "edge_worst_client_combined": row["edge_extrapolation_worst_client_combined"],
            "full_envelope_worst_client_combined": row["full_envelope_worst_client_combined"],
        }
        for row in summary
    }
    write_csv(DETAIL_TABLE, rows)
    write_csv(SUMMARY_TABLE, summary)
    SUMMARY_JSON.write_text(json.dumps(compact_summary, indent=2) + "\n", encoding="utf-8")
    make_figure(development_errors, summary)
    print(json.dumps(compact_summary, indent=2))
    print("Wrote Experiment 1A tables and figure")


if __name__ == "__main__":
    main()
