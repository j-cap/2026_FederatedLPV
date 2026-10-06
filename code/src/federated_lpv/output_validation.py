"""Causal output-only forecasts and record-aware residual diagnostics.

Future commands are known exogenous inputs. Future measured outputs never
enter a forecast. Correlations are descriptive, not IID hypothesis tests.
"""

from __future__ import annotations

import numpy as np

from federated_lpv.innovation_likelihood import C, steady_filter


def filter_records(data, parameters, dt, q, r):
    filters = [steady_filter(parameters, speed, dt, q, r) for speed in data.speeds]
    a, b, k = (np.array([f[key] for f in filters]) for key in ("a", "b", "k"))
    factors = np.array([np.linalg.cholesky(f["s"]) for f in filters])
    n, t = data.commands.shape
    history = np.zeros((n, t + 1, 5))
    innovations = np.empty_like(data.measurements)
    for time in range(t):
        predicted = np.einsum("nij,nj->ni", a, history[:, time]) + b * data.commands[:, time, None]
        error = data.measurements[:, time] - predicted @ C.T
        innovations[:, time] = np.linalg.solve(factors, error[..., None])[..., 0]
        history[:, time + 1] = predicted + np.einsum("nij,nj->ni", k, error)
    return history, innovations, a, b


def output_error_metrics(errors, r):
    mse = np.mean(errors**2, axis=tuple(range(errors.ndim - 1)))
    count = errors.size // 3
    flat = errors.reshape(-1, 3)
    return {
        "sensor_normalized_mse": float(
            np.einsum("ni,ij,nj->", flat, np.linalg.inv(r), flat) / (3 * count)
        ),
        "yaw_rmse_deg_s": float(np.rad2deg(np.sqrt(mse[0]))),
        "acceleration_rmse_mps2": float(np.sqrt(mse[1])),
        "steering_rmse_deg": float(np.rad2deg(np.sqrt(mse[2]))),
        "scored_output_samples": count,
    }


def causal_forecasts(data, parameters, dt, q, r, horizons, burn_in=20):
    """Forecast y[origin+h-1] using outputs strictly before origin.

    Every horizon uses origins burn_in ... T-max(horizons), inclusive. The
    zero-initialized simulation is scored separately over all T samples.
    """
    if not horizons or any(int(h) != h or h < 1 for h in horizons):
        raise ValueError("Positive integer forecast horizons are required")
    if burn_in < 0 or int(burn_in) != burn_in:
        raise ValueError("Nonnegative integer burn-in is required")
    maximum = int(max(horizons))
    if maximum + burn_in > data.commands.shape[1]:
        raise ValueError("No common causal forecast origins remain")
    history, _, a, b = filter_records(data, parameters, dt, q, r)
    origins = np.arange(burn_in, data.commands.shape[1] - maximum + 1)
    state = history[:, origins].copy()
    rows = []
    for horizon in range(1, maximum + 1):
        target = origins + horizon - 1
        state = np.einsum("nij,noj->noi", a, state) + b[:, None] * data.commands[:, target, None]
        if horizon in horizons:
            error = data.measurements[:, target] - state @ C.T
            rows.append(
                dict(
                    mode="forecast",
                    horizon_samples=horizon,
                    horizon_seconds=horizon * dt,
                    origins_per_record=len(origins),
                    **output_error_metrics(error, r),
                )
            )
    state = np.zeros((len(data.speeds), 5))
    errors = np.empty_like(data.measurements)
    for time in range(data.commands.shape[1]):
        state = np.einsum("nij,nj->ni", a, state) + b * data.commands[:, time, None]
        errors[:, time] = data.measurements[:, time] - state @ C.T
    rows.append(
        dict(
            mode="zero_start_simulation",
            horizon_samples=0,
            horizon_seconds=np.nan,
            origins_per_record=1,
            **output_error_metrics(errors, r),
        )
    )
    return rows


def residual_correlations(innovations, commands, maximum_lag=20):
    """Positive lag k pairs e[t] with e[t-k] or command[t-k]."""
    if maximum_lag < 1 or maximum_lag >= innovations.shape[1]:
        raise ValueError("Correlation lags must fit within each record")
    rows = []
    for centering in ("pooled", "within_record"):
        axes = (0, 1) if centering == "pooled" else (1,)
        error = innovations - innovations.mean(axis=axes, keepdims=True)
        inputs = commands - commands.mean(axis=axes, keepdims=True)
        for lag in range(1, maximum_lag + 1):
            following = error[:, lag:]
            for kind, previous in (
                ("autocorrelation", error[:, :-lag]),
                ("past_input", np.repeat(inputs[:, :-lag, None], 3, axis=2)),
            ):
                denominator = np.sqrt(
                    np.sum(following**2, axis=(0, 1)) * np.sum(previous**2, axis=(0, 1))
                )
                correlation = np.divide(
                    np.sum(following * previous, axis=(0, 1)),
                    denominator,
                    out=np.full(3, np.nan),
                    where=denominator > 0,
                )
                for channel, value in zip(("yaw", "acceleration", "steering"), correlation):
                    rows.append(
                        {
                            "centering": centering,
                            "kind": kind,
                            "lag": lag,
                            "output": channel,
                            "correlation": float(value),
                        }
                    )
    return rows


def residual_diagnostics(data, parameters, dt, q, r, maximum_lag=20):
    _, innovations, a, _ = filter_records(data, parameters, dt, q, r)
    flat = innovations.reshape(-1, 3)
    mean = flat.mean(axis=0)
    eigenvalues = np.linalg.eigvalsh(np.cov(flat, rowvar=False, bias=True))
    correlations = residual_correlations(innovations, data.commands, maximum_lag)
    stats = {
        "normalized_innovation_squared_mean": float(np.mean(flat**2)),
        "whitened_covariance_minimum_eigenvalue": float(eigenvalues[0]),
        "whitened_covariance_maximum_eigenvalue": float(eigenvalues[-1]),
        "maximum_absolute_whitened_mean": float(np.max(np.abs(mean))),
        "maximum_plant_spectral_radius": float(np.max(np.abs(np.linalg.eigvals(a)))),
    }
    for centering in ("pooled", "within_record"):
        for kind in ("autocorrelation", "past_input"):
            values = [
                abs(row["correlation"])
                for row in correlations
                if row["centering"] == centering and row["kind"] == kind
            ]
            stats[f"maximum_absolute_{centering}_{kind}"] = float(np.nanmax(values))
    return stats, correlations
