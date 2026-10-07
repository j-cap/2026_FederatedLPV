"""Calibration-only candidate selection and frozen client-level deployment."""

from __future__ import annotations

import numpy as np

from federated_lpv.innovation_likelihood import C, MeasuredDataset
from federated_lpv.output_validation import filter_records, output_error_metrics

RMSE_KEYS = ("yaw_rmse_deg_s", "acceleration_rmse_mps2", "steering_rmse_deg")


def first_record_prefix(data, positions, samples):
    if len(positions) != len(data.speeds) or not 0 < samples <= data.commands.shape[1]:
        raise ValueError("Client positions and a valid prefix length are required")
    clients, first = np.unique(positions, return_index=True)
    return clients, MeasuredDataset(
        data.speeds[first], data.commands[first, :samples], data.measurements[first, :samples]
    )


def candidate_forecasts(data, positions, parameters, dt, q, r, horizons, burn_in):
    """Errors of each independent plant/KF on identical causal forecast origins."""
    if not horizons or any(h < 1 or int(h) != h for h in horizons):
        raise ValueError("Positive integer forecast horizons are required")
    if burn_in < 0 or int(burn_in) != burn_in or max(horizons) + burn_in > data.commands.shape[1]:
        raise ValueError("Invalid forecast origins")
    clients, inverse = np.unique(positions, return_inverse=True)
    if len(positions) != len(data.speeds):
        raise ValueError("Every record requires a client association")
    origins = np.arange(burn_in, data.commands.shape[1] - max(horizons) + 1)
    rows, residuals, radii = [], [], []
    for candidate, z in enumerate(parameters):
        history, whitened, a, b = filter_records(data, z, dt, q, r)
        residuals.append(whitened[:, burn_in:])
        radii.append(float(np.max(np.abs(np.linalg.eigvals(a)))))
        state = history[:, origins].copy()
        errors = {}
        for horizon in range(1, max(horizons) + 1):
            target = origins + horizon - 1
            state = (
                np.einsum("nij,noj->noi", a, state) + b[:, None] * data.commands[:, target, None]
            )
            if horizon in horizons:
                errors[horizon] = data.measurements[:, target] - state @ C.T
        state = np.zeros((len(data.speeds), 5))
        simulation = np.empty_like(data.measurements)
        for time in range(data.commands.shape[1]):
            state = np.einsum("nij,nj->ni", a, state) + b * data.commands[:, time, None]
            simulation[:, time] = data.measurements[:, time] - state @ C.T
        errors[0] = simulation[:, burn_in:]
        for horizon, error in errors.items():
            meta = {
                "candidate": candidate,
                "horizon_samples": horizon,
                "mode": "forecast" if horizon else "zero_start_simulation_suffix",
                "origins_per_record": len(origins) if horizon else 1,
            }
            for index, client in enumerate(clients):
                rows.append(
                    dict(
                        **meta,
                        client_position=int(client),
                        **output_error_metrics(error[inverse == index], r),
                    )
                )
    return clients, rows, np.array(residuals), np.array(radii)


def prefix_scores(data, positions, parameters, dt, q, r, samples, horizons, burn_in):
    """This API slices the first record before any filter/score is computed."""
    clients, prefix = first_record_prefix(data, positions, samples)
    _, rows, _, _ = candidate_forecasts(prefix, clients, parameters, dt, q, r, horizons, burn_in)
    scores = np.zeros((len(clients), len(parameters)))
    client_index = {client: index for index, client in enumerate(clients)}
    for row in rows:
        if row["horizon_samples"]:
            scores[client_index[row["client_position"]], row["candidate"]] += row[
                "sensor_normalized_mse"
            ] / len(horizons)
    return clients, scores


def forecast_choices(scores, margin=0.0, fallback=True):
    scores = np.asarray(scores)
    if scores.ndim != 2 or scores.shape[1] != 3 or not np.isfinite(scores).all():
        raise ValueError("Expected finite global/two-group prefix scores")
    if not 0 <= margin < 1:
        raise ValueError("Fallback margin must be in [0,1)")
    group = 1 + np.argmin(scores[:, 1:], axis=1)
    if not fallback:
        return group
    improvement = scores[np.arange(len(scores)), group] < (1 - margin) * scores[:, 0]
    return np.where(improvement, group, 0)


def deploy_rows(candidate_rows, clients, choices):
    """Choose one model per client, then aggregate squared errors by sample count."""
    if len(clients) != len(choices):
        raise ValueError("Every client needs a frozen model choice")
    mapping = dict(zip(clients, choices))
    chosen = [
        row.copy() for row in candidate_rows if row["candidate"] == mapping[row["client_position"]]
    ]
    keys = sorted({(row["mode"], row["horizon_samples"]) for row in chosen})
    aggregate = []
    for mode, horizon in keys:
        rows = [row for row in chosen if row["mode"] == mode and row["horizon_samples"] == horizon]
        if len(rows) != len(clients):
            raise ValueError("Candidate rows must cover every client and horizon once")
        counts = np.array([row["scored_output_samples"] for row in rows])
        result = {
            "mode": mode,
            "horizon_samples": horizon,
            "client_position": -1,
            "candidate": -1,
            "origins_per_record": rows[0]["origins_per_record"],
            "scored_output_samples": int(counts.sum()),
            "sensor_normalized_mse": float(
                np.average([row["sensor_normalized_mse"] for row in rows], weights=counts)
            ),
        }
        result.update(
            {
                key: float(np.sqrt(np.average([row[key] ** 2 for row in rows], weights=counts)))
                for key in RMSE_KEYS
            }
        )
        aggregate.append(result)
    return [*chosen, *aggregate]


def select_inner_candidate(records, eligible, folds, tolerance=1e-12):
    """Ranks strategies using inner scores only; outer metrics are not accepted."""
    keys = sorted({(int(row["restart"]), float(row["margin"])) for row in records})
    candidates = []
    for restart, margin in keys:
        rows = [row for row in records if row["restart"] == restart and row["margin"] == margin]
        if eligible[restart] and {row["fold"] for row in rows} == set(range(folds)):
            candidates.append(
                {
                    "restart": restart,
                    "margin": margin,
                    "score": float(np.mean([row["selection_score"] for row in rows])),
                }
            )
    if not candidates:
        return None, candidates
    best_score = min(row["score"] for row in candidates)
    ties = [row for row in candidates if row["score"] <= best_score + tolerance]
    return min(ties, key=lambda row: (-row["margin"], row["restart"])), candidates
