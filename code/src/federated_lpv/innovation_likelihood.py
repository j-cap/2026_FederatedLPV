"""Measured-output five-state LPV working likelihood and exact sensitivities.

The learner accepts measured arrays and public design quantities only. The
Gaussian covariance is a frozen-Q steady-state working model, not an assertion
that heterogeneous simulated trajectories follow one Gaussian dynamical model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm, expm_frechet, solve_discrete_are, solve_discrete_lyapunov


PARAMETER_NAMES = (
    "yaw_front", "yaw_rear", "front_beta", "front_yaw", "rear_beta",
    "rear_yaw", "front_decay", "rear_decay", "actuator_rate",
)
C = np.array([[0., 1., 0., 0., 0.], [0., 0., 1., 1., 0.], [0., 0., 0., 0., 1.]])


@dataclass(frozen=True)
class MeasuredDataset:
    speeds: np.ndarray
    commands: np.ndarray
    measurements: np.ndarray

    def __post_init__(self):
        n, t = self.commands.shape
        if self.speeds.shape != (n,) or self.measurements.shape != (n, t, 3):
            raise ValueError("Expected speeds (N,), commands (N,T), outputs (N,T,3)")
        if n == 0 or t == 0 or np.any(self.speeds <= 0):
            raise ValueError("Nonempty measured records and positive speed are required")
        if not all(np.isfinite(x).all() for x in (
            self.speeds, self.commands, self.measurements
        )):
            raise ValueError("Measured records must be finite")


def structured_matrices(log_parameters, speed, dt, derivatives=False):
    """Exact ZOH discretization, with Frechet derivatives in log coordinates."""
    v = np.exp(log_parameters)
    h = np.zeros((6, 6))
    h[0, 1] = -1.
    h[0, 2:4] = 1. / speed
    entries = (
        ((1, 2, 1.),), ((1, 3, -1.),),
        ((2, 0, -speed), (2, 4, speed)), ((2, 1, -1.),),
        ((3, 0, -speed),), ((3, 1, 1.),),
        ((2, 2, -speed),), ((3, 3, -speed),),
        ((4, 4, -1.), (4, 5, 1.)),
    )
    directions = np.zeros((9, 6, 6))
    for j, locations in enumerate(entries):
        for row, column, factor in locations:
            h[row, column] += factor * v[j]
            directions[j, row, column] = factor * v[j] * dt
    hd = h * dt
    discrete = expm(hd)
    a, b = discrete[:5, :5], discrete[:5, 5]
    if not derivatives:
        return a, b
    differential = np.array([
        expm_frechet(hd, direction, compute_expm=False)
        for direction in directions
    ])
    return a, b, differential[:, :5, :5], differential[:, :5, 5]


def steady_filter(log_parameters, speed, dt, q, r, derivatives=False):
    """DARE returns PRIOR P; K=P C' (C P C'+R)^-1, without extra prediction."""
    matrices = structured_matrices(log_parameters, speed, dt, derivatives)
    a, b = matrices[:2]
    p = solve_discrete_are(a.T, C.T, q, r)
    p = (p + p.T) / 2
    s = C @ p @ C.T + r
    inv_s = np.linalg.inv(s)
    k = p @ C.T @ inv_s
    m = np.eye(5) - k @ C
    posterior = m @ p
    result = dict(a=a, b=b, p=p, s=s, inv_s=inv_s, k=k,
                  logdet=float(np.linalg.slogdet(s)[1]))
    if not derivatives:
        return result
    da, db = matrices[2:]
    # Differentiate P=A U A'+Q, U=P-P C' S^-1 C P.
    # dU=(I-KC) dP (I-KC)' with fixed C,Q,R.
    f = a @ m
    dp = np.array([
        solve_discrete_lyapunov(f, d @ posterior @ a.T + a @ posterior @ d.T)
        for d in da
    ])
    dp = (dp + dp.transpose(0, 2, 1)) / 2
    ds = np.array([C @ d @ C.T for d in dp])
    dk = np.array([(d @ C.T - k @ ds[j]) @ inv_s for j, d in enumerate(dp)])
    result.update(da=da, db=db, dp=dp, ds=ds, dk=dk)
    return result


class InnovationLikelihood:
    """Mean sensor-normalized Gaussian NLL, with analytic total derivatives.

    L = mean_t [ e_t' S^-1 e_t + logdet(S)-logdet(R) ] / 3.
    The subtracted logdet(R) is a parameter-independent reference. A stationary
    prior covariance and zero posterior mean initialize each designed quiet-start
    record. Derivatives include P, K, S, the innovation recursion, and logdet(S).
    """

    def __init__(self, data, dt, q, r):
        self.data, self.dt = data, dt
        self.q, self.r = np.asarray(q), np.asarray(r)
        self.r_inverse = np.linalg.inv(self.r)
        self.r_logdet = float(np.linalg.slogdet(self.r)[1])

    def evaluate(self, parameters, gradient=False, information=False, diagnostics=False):
        sensitivities = gradient or information
        speeds, inverse = np.unique(self.data.speeds, return_inverse=True)
        filters = [steady_filter(parameters, v, self.dt, self.q, self.r, sensitivities)
                   for v in speeds]
        keys = ["a", "b", "inv_s", "k"] + (
            ["da", "db", "ds", "dk"] if sensitivities else []
        )
        arrays = {key: np.array([f[key] for f in filters])[inverse] for key in keys}
        a, b, w, k = (arrays[key] for key in ("a", "b", "inv_s", "k"))
        n, t = self.data.commands.shape
        x = np.zeros((n, 5))
        dx = np.zeros((n, 5, 9))
        grad = np.zeros(9)
        info = np.zeros((9, 9))
        logdet = np.array([f["logdet"] - self.r_logdet for f in filters])[inverse]
        total = t * logdet.sum()
        squared = np.zeros(3)
        sensor_squared = 0.
        if sensitivities:
            da, db, ds, dk = (arrays[key] for key in ("da", "db", "ds", "dk"))
            grad = t * np.einsum("nij,npji->p", w, ds)
            if information:
                wds = np.einsum("nij,npjk->npik", w, ds)
                info = t * .5 * np.einsum("npij,nqji->pq", wds, wds)
        for time in range(t):
            u = self.data.commands[:, time]
            predicted = np.einsum("nij,nj->ni", a, x) + b * u[:, None]
            e = self.data.measurements[:, time] - predicted @ C.T
            alpha = np.einsum("nij,nj->ni", w, e)
            total += np.einsum("ni,ni->", e, alpha)
            if sensitivities:
                dpredicted = (
                    np.einsum("nij,njp->nip", a, dx)
                    + np.einsum("npij,nj->nip", da, x)
                    + db.transpose(0, 2, 1) * u[:, None, None]
                )
                de = -np.einsum("ij,njp->nip", C, dpredicted)
                grad += (2 * np.einsum("nip,ni->p", de, alpha)
                         - np.einsum("ni,npij,nj->p", alpha, ds, alpha))
                if information:
                    info += np.einsum("nip,nij,njq->pq", de, w, de)
                dx = (dpredicted + np.einsum("npij,nj->nip", dk, e)
                      + np.einsum("nij,njp->nip", k, de))
            x = predicted + np.einsum("nij,nj->ni", k, e)
            if diagnostics:
                squared += np.square(e).sum(axis=0)
                sensor_squared += np.einsum("ni,ij,nj->", e, self.r_inverse, e)
        count = n * t * 3
        result = {"objective": float(total / count)}
        if sensitivities:
            result["gradient"] = grad / count
        if information:
            result["information"] = (info + info.T) / (2 * count)
        if diagnostics:
            result.update(sensor_normalized_mse=float(sensor_squared / count),
                          yaw_rmse_deg_s=float(np.rad2deg(np.sqrt(squared[0] / (n*t)))),
                          acceleration_rmse_mps2=float(np.sqrt(squared[1] / (n*t))),
                          steering_rmse_deg=float(np.rad2deg(np.sqrt(squared[2] / (n*t)))))
        return result

    def value_gradient(self, parameters):
        result = self.evaluate(parameters, gradient=True)
        return result["objective"], result["gradient"]

    def value(self, parameters):
        return self.evaluate(parameters)["objective"]


def projected_gradient(parameters, gradient, lower, upper, tolerance=1e-7):
    g = gradient.copy()
    g[(parameters <= lower + tolerance) & (g > 0)] = 0
    g[(parameters >= upper - tolerance) & (g < 0)] = 0
    return g


def information_scale(matrix, minimum=.2, maximum=5.):
    raw = 1 / np.sqrt(np.maximum(np.diag(matrix), 1e-12))
    return np.clip(raw / np.median(raw), minimum, maximum)


def physical_coupling_diagnostic(log_parameters):
    """Check an exact bicycle relation using fitted coefficients alone.

    yaw_front * front_beta / front_yaw and
    yaw_rear * rear_beta / rear_yaw must represent the same m/I_z.
    Neither mass, inertia nor geometry needs to be known to check this relation.
    The nine-independent-coordinate model does not enforce it during fitting.
    """
    z = np.asarray(log_parameters)
    front = float(np.exp(z[0]+z[2]-z[3]))
    rear = float(np.exp(z[1]+z[4]-z[5]))
    return dict(inferred_front_inertial_ratio=front, inferred_rear_inertial_ratio=rear,
                relative_coupling_discrepancy=abs(front-rear)/((front+rear)/2),
                log_coupling_residual=float(z[0]+z[2]-z[3]-z[1]-z[4]+z[5]))
