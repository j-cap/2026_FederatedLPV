"""Client-level working mixtures of coupled plants using measured arrays only."""

from __future__ import annotations

import numpy as np
from scipy.linalg import block_diag
from scipy.optimize import LinearConstraint, minimize
from scipy.special import logsumexp

from federated_lpv.innovation_likelihood import C, MeasuredDataset, steady_filter
from federated_lpv.output_validation import filter_records, output_error_metrics
from federated_lpv.physical_coupled_fit import LOG_MAP, constraint_audit


class ClientLikelihood:
    """Unnormalized Gaussian working NLL and total gradient for each client.

    Summation includes all a client's records before computing mixture weights.
    The omitted logdet(R) reference and Gaussian constant are model independent.
    """

    def __init__(self, data, positions, dt, q, r, coordinates):
        self.data, self.dt, self.q, self.r = data, dt, q, r
        self.coordinates = coordinates
        if np.asarray(positions).shape != data.speeds.shape:
            raise ValueError("A client association is required for every record")
        self.clients, self.client_inverse = np.unique(positions, return_inverse=True)
        self.speeds, self.speed_inverse = np.unique(data.speeds, return_inverse=True)
        self.sample_count = data.commands.size * 3

    def evaluate(self, eta, gradient=False):
        filters = [
            steady_filter(self.coordinates.effective(eta), v, self.dt, self.q, self.r, gradient)
            for v in self.speeds
        ]
        keys = ["a", "b", "k", "inv_s"] + (["da", "db", "ds", "dk"] if gradient else [])
        arrays = {key: np.array([f[key] for f in filters])[self.speed_inverse] for key in keys}
        a, b, k, w = (arrays[key] for key in ("a", "b", "k", "inv_s"))
        n, t = self.data.commands.shape
        x = np.zeros((n, 5))
        logdet = np.array([f["logdet"] for f in filters])[self.speed_inverse]
        costs = t * (logdet - np.linalg.slogdet(self.r)[1])
        if gradient:
            da, db, ds, dk = (arrays[key] for key in ("da", "db", "ds", "dk"))
            dx = np.zeros((n, 5, 9))
            gradients = t * np.einsum("nij,npji->np", w, ds)
        for time in range(t):
            u = self.data.commands[:, time]
            predicted = np.einsum("nij,nj->ni", a, x) + b * u[:, None]
            e = self.data.measurements[:, time] - predicted @ C.T
            alpha = np.einsum("nij,nj->ni", w, e)
            costs += np.einsum("ni,ni->n", e, alpha)
            if gradient:
                dpredicted = (
                    np.einsum("nij,njp->nip", a, dx)
                    + np.einsum("npij,nj->nip", da, x)
                    + db.transpose(0, 2, 1) * u[:, None, None]
                )
                de = -np.einsum("ij,njp->nip", C, dpredicted)
                gradients += 2 * np.einsum("nip,ni->np", de, alpha)
                gradients -= np.einsum("ni,npij,nj->np", alpha, ds, alpha)
                dx = (
                    dpredicted + np.einsum("npij,nj->nip", dk, e) + np.einsum("nij,njp->nip", k, de)
                )
            x = predicted + np.einsum("nij,nj->ni", k, e)
        result = np.zeros(len(self.clients))
        np.add.at(result, self.client_inverse, costs / 2)
        if not gradient:
            return result
        derivative = np.zeros((len(self.clients), 8))
        np.add.at(derivative, self.client_inverse, gradients @ LOG_MAP / 2)
        return result, derivative


def working_memberships(costs, mixing):
    mixing = np.asarray(mixing)
    if np.any(mixing <= 0) or not np.isclose(mixing.sum(), 1):
        raise ValueError("Positive normalized mixing weights are required")
    scores = np.log(mixing)[None] - costs
    normalizers = logsumexp(scores, axis=1)
    return np.exp(scores - normalizers[:, None]), normalizers


class ClientMixture:
    def __init__(self, evaluator):
        self.evaluator = evaluator

    def evaluate(self, values, gradient=False):
        plants = np.asarray(values[:16]).reshape(2, 8)
        mixing = np.array([values[16], 1 - values[16]])
        evaluated = [self.evaluator.evaluate(eta, gradient) for eta in plants]
        costs = np.column_stack([item[0] if gradient else item for item in evaluated])
        weights, normalizers = working_memberships(costs, mixing)
        scale = 2 / self.evaluator.sample_count
        result = {
            "objective": float(-scale * normalizers.sum()),
            "memberships": weights,
            "costs": costs,
            "mixing": mixing,
        }
        if gradient:
            derivative = [
                scale * np.einsum("n,np->p", weights[:, g], evaluated[g][1]) for g in range(2)
            ]
            dpi = -scale * np.sum(weights[:, 0] / mixing[0] - weights[:, 1] / mixing[1])
            result["gradient"] = np.r_[derivative[0], derivative[1], dpi]
        return result

    def value_gradient(self, values):
        result = self.evaluate(values, gradient=True)
        return result["objective"], result["gradient"]

    def value(self, values):
        return self.evaluate(values)["objective"]


def project_split(global_eta, direction, amplitude, coordinates):
    """Nearest feasible positive/negative perturbations, retaining coupling."""
    direction = direction / np.max(np.abs(direction))
    rows = []
    for sign in (-1, 1):
        target = global_eta + sign * amplitude * direction
        result = minimize(
            lambda x, target=target: (0.5 * np.sum((x - target) ** 2), x - target),
            global_eta,
            jac=True,
            method="SLSQP",
            constraints=[
                LinearConstraint(LOG_MAP, coordinates.relative_lower, coordinates.relative_upper)
            ],
            options={"ftol": 1e-12, "maxiter": 100},
        )
        if not result.success:
            raise RuntimeError(f"Split projection failed: {result.message}")
        rows.append(result.x)
    return np.r_[rows[0], rows[1], 0.5]


def mixture_audit(values, gradient, coordinates, cfg):
    audits = [
        constraint_audit(
            values[g * 8 : g * 8 + 8],
            gradient[g * 8 : g * 8 + 8],
            coordinates,
            cfg["active_bound_tolerance"],
        )
        for g in range(2)
    ]
    lower, upper = cfg["minimum_mixing_weight"], 1 - cfg["minimum_mixing_weight"]
    at_lower = values[16] <= lower + cfg["active_bound_tolerance"]
    at_upper = values[16] >= upper - cfg["active_bound_tolerance"]
    pi_gradient = gradient[16]
    if (at_lower and pi_gradient >= 0) or (at_upper and pi_gradient <= 0):
        pi_gradient = 0.0
    violation = max(0.0, lower - values[16], values[16] - upper)
    return {
        "kkt_residual": max(abs(float(pi_gradient)), *(a["kkt_residual"] for a in audits)),
        "constraint_violation": max(violation, *(a["constraint_violation"] for a in audits)),
        "active_coefficient_bounds": sum(a["active_coefficient_bounds"] for a in audits),
        "mixing_weight_at_bound": bool(at_lower or at_upper),
        "face_basis": block_diag(
            audits[0]["face_basis"],
            audits[1]["face_basis"],
            np.empty((1, 0)) if at_lower or at_upper else np.ones((1, 1)),
        ),
    }


def optimize_mixture(initial, evaluator, coordinates, cfg, plant_scale):
    scale = np.r_[plant_scale, plant_scale, 1.0]
    values = initial / scale
    matrix = block_diag(LOG_MAP, LOG_MAP, np.ones((1, 1))) * scale
    lower = np.r_[
        coordinates.relative_lower, coordinates.relative_lower, cfg["minimum_mixing_weight"]
    ]
    upper = np.r_[
        coordinates.relative_upper, coordinates.relative_upper, 1 - cfg["minimum_mixing_weight"]
    ]
    calls, invalid, phases = 0, 0, []
    stages = [np.r_[s, np.asarray(s) + 8] for s in cfg["stages"]]
    for phase, indices in [
        *[(f"stage_{i + 1}", s) for i, s in enumerate(stages)],
        ("joint", np.arange(17)),
    ]:
        active = np.asarray(indices, dtype=int)
        base = values.copy()
        shift = matrix @ base - matrix[:, active] @ base[active]
        movable = np.linalg.norm(matrix[:, active], axis=1) > 0

        def objective(local, base=base, active=active):
            nonlocal calls, invalid
            calls += 1
            proposal = base.copy()
            proposal[active] = local
            try:
                value, derivative = evaluator.value_gradient(proposal * scale)
                if not np.isfinite(value) or not np.isfinite(derivative).all():
                    raise FloatingPointError("Nonfinite mixture likelihood")
                return value, (derivative * scale)[active]
            except (np.linalg.LinAlgError, FloatingPointError):
                invalid += 1
                return 1e12, np.zeros(len(active))

        result = minimize(
            objective,
            base[active],
            jac=True,
            method="SLSQP",
            constraints=[
                LinearConstraint(
                    matrix[movable][:, active], (lower - shift)[movable], (upper - shift)[movable]
                )
            ],
            options={
                "ftol": cfg["optimizer_ftol"],
                "maxiter": cfg["optimizer_max_iterations"]
                if phase == "joint"
                else cfg["stage_max_iterations"],
            },
        )
        values[active] = result.x
        phases.append(
            {
                "phase": phase,
                "objective": float(result.fun),
                "iterations": int(result.nit),
                "success": bool(result.success),
                "message": str(result.message),
            }
        )
    fitted = values * scale
    _, gradient = evaluator.value_gradient(fitted)
    audit = mixture_audit(fitted, gradient, coordinates, cfg)
    return fitted, result, phases, calls, invalid, audit


def calibrate_memberships(data, positions, parameters, mixing, dt, q, r):
    """Only the first record per client enters this already-prefix-sliced API."""
    clients, first = np.unique(positions, return_index=True)
    prefix = MeasuredDataset(data.speeds[first], data.commands[first], data.measurements[first])
    costs = []
    for z in parameters:
        _, innovations, _, _ = filter_records(prefix, z, dt, q, r)
        logdet = np.array([steady_filter(z, v, dt, q, r)["logdet"] for v in prefix.speeds])
        costs.append(
            (
                np.sum(innovations**2, axis=(1, 2))
                + prefix.commands.shape[1] * (logdet - np.linalg.slogdet(r)[1])
            )
            / 2
        )
    weights, _ = working_memberships(np.column_stack(costs), mixing)
    return clients, weights


def mixture_forecasts(data, positions, parameters, client_weights, dt, q, r, horizons, burn_in):
    """Separately filter/propagate models; never interpolate their matrices/gains."""
    clients, inverse = np.unique(positions, return_inverse=True)
    if client_weights.shape != (len(clients), len(parameters)):
        raise ValueError("Frozen weights must match clients and components")
    if burn_in < 0 or max(horizons) + burn_in > data.commands.shape[1]:
        raise ValueError("Invalid common forecast origins")
    weights = client_weights[inverse]
    selected = np.argmax(weights, axis=1)
    origins = np.arange(burn_in, data.commands.shape[1] - max(horizons) + 1)
    predictions, simulations, whitened, radii = [], [], [], []
    for z in parameters:
        history, innovations, a, b = filter_records(data, z, dt, q, r)
        whitened.append(innovations)
        radii.append(float(np.max(np.abs(np.linalg.eigvals(a)))))
        state = history[:, origins].copy()
        by_horizon = {}
        for horizon in range(1, max(horizons) + 1):
            target = origins + horizon - 1
            state = (
                np.einsum("nij,noj->noi", a, state) + b[:, None] * data.commands[:, target, None]
            )
            if horizon in horizons:
                by_horizon[horizon] = state @ C.T
        predictions.append(by_horizon)
        state = np.zeros((len(data.speeds), 5))
        output = np.empty_like(data.measurements)
        for time in range(data.commands.shape[1]):
            state = np.einsum("nij,nj->ni", a, state) + b * data.commands[:, time, None]
            output[:, time] = state @ C.T
        simulations.append(output[:, burn_in:])
    rows = []
    rules = ("map", "ensemble") if len(parameters) > 1 else ("map",)
    for horizon in [*horizons, 0]:
        outputs = np.array(simulations if horizon == 0 else [p[horizon] for p in predictions])
        targets = (
            data.measurements[:, burn_in:]
            if horizon == 0
            else data.measurements[:, origins + horizon - 1]
        )
        for rule in rules:
            output = (
                outputs[selected, np.arange(len(selected))]
                if rule == "map"
                else np.einsum("ng,gntj->ntj", weights, outputs)
            )
            errors = targets - output
            meta = {
                "rule": rule,
                "horizon_samples": horizon,
                "mode": "forecast" if horizon else "zero_start_simulation_suffix",
                "origins_per_record": len(origins) if horizon else 1,
            }
            rows.append(dict(**meta, client_position=-1, **output_error_metrics(errors, r)))
            rows.extend(
                dict(
                    **meta,
                    client_position=int(client),
                    **output_error_metrics(errors[inverse == index], r),
                )
                for index, client in enumerate(clients)
            )
    selected_residuals = np.array(whitened)[selected, np.arange(len(selected)), burn_in:]
    return rows, selected_residuals, max(radii)
