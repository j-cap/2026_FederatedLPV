"""Measured-output multihorizon loss with total KF/plant sensitivities."""

from __future__ import annotations

import numpy as np

from federated_lpv.innovation_likelihood import C, steady_filter
from federated_lpv.physical_coupled_fit import LOG_MAP


class MultihorizonLoss:
    def __init__(self, data, coordinates, dt, q, r, horizons, burn_in, weights=None):
        self.data, self.coordinates, self.dt, self.q, self.r = data, coordinates, dt, q, r
        self.horizons = tuple(horizons)
        if not horizons or any(h < 1 or int(h) != h for h in horizons):
            raise ValueError("Positive integer horizons required")
        self.origins = np.arange(burn_in, data.commands.shape[1] - max(horizons) + 1)
        if burn_in < 0 or not len(self.origins):
            raise ValueError("No causal forecast origins")
        self.weights = np.ones(len(data.speeds)) if weights is None else np.asarray(weights)
        if (
            self.weights.shape != data.speeds.shape
            or np.any(self.weights < 0)
            or not np.isfinite(self.weights).all()
            or self.weights.sum() <= 0
        ):
            raise ValueError("Finite nonnegative record weights with positive sum required")
        self.w = np.linalg.inv(r)

    def value_gradient(self, eta):
        speeds, inverse = np.unique(self.data.speeds, return_inverse=True)
        filters = [
            steady_filter(
                self.coordinates.effective(eta), v, self.dt, self.q, self.r, derivatives=True
            )
            for v in speeds
        ]
        a, b, k = (np.asarray([f[key] for f in filters])[inverse] for key in ["a", "b", "k"])
        da, db, dk = (
            np.einsum("np...,pj->nj...", np.asarray([f[key] for f in filters])[inverse], LOG_MAP)
            for key in ["da", "db", "dk"]
        )
        n, t = self.data.commands.shape
        history = np.zeros((n, t + 1, 5))
        sensitivity = np.zeros((n, t + 1, 5, 8))
        for time in range(t):
            x, dx, u = history[:, time], sensitivity[:, time], self.data.commands[:, time]
            predicted = np.einsum("nij,nj->ni", a, x) + b * u[:, None]
            dp = (
                np.einsum("nij,njp->nip", a, dx)
                + np.einsum("npij,nj->nip", da, x)
                + db.transpose(0, 2, 1) * u[:, None, None]
            )
            error = self.data.measurements[:, time] - predicted @ C.T
            de = -np.einsum("ij,njp->nip", C, dp)
            history[:, time + 1] = predicted + np.einsum("nij,nj->ni", k, error)
            sensitivity[:, time + 1] = (
                dp + np.einsum("nij,njp->nip", k, de) + np.einsum("npij,nj->nip", dk, error)
            )
        x, dx = history[:, self.origins].copy(), sensitivity[:, self.origins].copy()
        value, gradient = 0.0, np.zeros(8)
        for h in range(1, max(self.horizons) + 1):
            target = self.origins + h - 1
            u = self.data.commands[:, target]
            dx = (
                np.einsum("nij,nojp->noip", a, dx)
                + np.einsum("npij,noj->noip", da, x)
                + db.transpose(0, 2, 1)[:, None] * u[:, :, None, None]
            )
            x = np.einsum("nij,noj->noi", a, x) + b[:, None] * u[:, :, None]
            if h in self.horizons:
                e = self.data.measurements[:, target] - x @ C.T
                de = -np.einsum("ij,nojp->noip", C, dx)
                alpha = e @ self.w
                value += np.einsum("n,noi,noi->", self.weights, e, alpha)
                gradient += 2 * np.einsum("n,noi,noip->p", self.weights, alpha, de)
        divisor = 3 * self.weights.sum() * len(self.origins) * len(self.horizons)
        return float(value / divisor), gradient / divisor
