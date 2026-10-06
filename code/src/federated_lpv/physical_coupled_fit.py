"""Eight positive bicycle coordinates with the original nine coefficient bounds.

Inputs are measured-output likelihoods, public nominal coefficients, and bounds.
No client physical parameter, latent trajectory, or fleet label is accepted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import null_space
from scipy.optimize import LinearConstraint, linprog, minimize, nnls

from federated_lpv.innovation_likelihood import information_scale

PHYSICAL_NAMES = (
    "inertial_ratio",
    "front_length",
    "rear_length",
    "front_force_gain",
    "rear_force_gain",
    "front_decay",
    "rear_decay",
    "actuator_rate",
)
LOG_MAP = np.array(
    [
        [1, 1, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
    ],
    dtype=float,
)
COUPLING_ROW = np.array([1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 0.0, 0.0, 0.0])


@dataclass(frozen=True)
class PhysicalCoordinates:
    nominal: np.ndarray
    lower: np.ndarray
    upper: np.ndarray

    def __post_init__(self):
        if any(np.asarray(v).shape != (9,) for v in (self.nominal, self.lower, self.upper)):
            raise ValueError("Expected nine log coefficients and bounds")
        if abs(COUPLING_ROW @ self.nominal) > 1e-10:
            raise ValueError("The public nominal must satisfy the physical identity")
        if np.any(self.lower > self.nominal) or np.any(self.upper < self.nominal):
            raise ValueError("The public nominal must be feasible")

    def effective(self, eta):
        return self.nominal + LOG_MAP @ np.asarray(eta)

    @property
    def relative_lower(self):
        return self.lower - self.nominal

    @property
    def relative_upper(self):
        return self.upper - self.nominal

    @property
    def nominal_physical_logs(self):
        z = self.nominal
        return np.array(
            [z[0] + z[2] - z[3], z[3] - z[2], z[5] - z[4], z[2], z[4], z[6], z[7], z[8]]
        )

    def feasible_initial(self, eta):
        """Shrink a nominal-centered ray; coefficient-wise clipping breaks coupling."""
        delta = LOG_MAP @ np.asarray(eta)
        factors = [1.0]
        for value, low, high in zip(delta, self.relative_lower, self.relative_upper):
            if value > 0:
                factors.append(high / value)
            elif value < 0:
                factors.append(low / value)
        return np.asarray(eta) * max(0.0, min(factors))

    def feasible_profile_start(self, eta, index, target):
        """Project onto the polytope with an exact fixed physical coordinate."""
        equality = np.eye(8)[index : index + 1]
        feasible = linprog(
            np.zeros(8),
            A_ub=np.vstack([LOG_MAP, -LOG_MAP]),
            b_ub=np.r_[self.relative_upper, -self.relative_lower],
            A_eq=equality,
            b_eq=[target],
            bounds=[(None, None)] * 8,
            method="highs",
        )
        if feasible.status == 2:
            return None
        if not feasible.success:
            raise RuntimeError(f"Profile feasibility LP failed: {feasible.message}")
        result = minimize(
            lambda x: (0.5 * np.sum((x - eta) ** 2), x - eta),
            feasible.x,
            jac=True,
            method="SLSQP",
            constraints=[
                LinearConstraint(LOG_MAP, self.relative_lower, self.relative_upper),
                LinearConstraint(equality, [target], [target]),
            ],
            options={"ftol": 1e-12, "maxiter": 100},
        )
        if not result.success:
            raise RuntimeError(f"Profile projection failed: {result.message}")
        return result.x


class CoupledLikelihood:
    def __init__(self, evaluator, coordinates):
        self.evaluator, self.coordinates = evaluator, coordinates

    def evaluate(self, eta, gradient=False, information=False, diagnostics=False):
        result = self.evaluator.evaluate(
            self.coordinates.effective(eta),
            gradient=gradient,
            information=information,
            diagnostics=diagnostics,
        )
        if gradient or information:
            result["gradient"] = LOG_MAP.T @ result["gradient"]
        if information:
            result["information"] = LOG_MAP.T @ result["information"] @ LOG_MAP
        return result

    def value_gradient(self, eta):
        result = self.evaluate(eta, gradient=True)
        return result["objective"], result["gradient"]

    def value(self, eta):
        return self.evaluate(eta)["objective"]


def constraint_audit(eta, gradient, coordinates, tolerance=1e-5, fixed=None):
    """Independent KKT residual using nonnegative outward-normal multipliers.

    Unlike a box projected gradient, this accounts for coupled linear bounds.
    Equality-coordinate multipliers in nuisance profiles are unrestricted.
    """
    delta = LOG_MAP @ eta
    low = delta - coordinates.relative_lower
    high = coordinates.relative_upper - delta
    at_lower, at_upper = low <= tolerance, high <= tolerance
    normals = np.vstack([-LOG_MAP[at_lower], LOG_MAP[at_upper]])
    equalities = np.eye(8)[list(fixed)] if fixed else np.empty((0, 8))
    tangent = null_space(equalities) if len(equalities) else np.eye(8)
    projected = tangent.T @ gradient
    if len(normals):
        multipliers = nnls((normals @ tangent).T, -projected, maxiter=1000)[0]
        residual = projected + (normals @ tangent).T @ multipliers
    else:
        multipliers = np.empty(0)
        residual = projected
    active_slacks = np.r_[low[at_lower], high[at_upper]]
    face = (
        null_space(np.vstack([normals, equalities]))
        if (len(normals) + len(equalities))
        else np.eye(8)
    )
    violation = max(0.0, float(-low.min()), float(-high.min()))
    if fixed:
        violation = max(violation, max(abs(eta[j] - value) for j, value in fixed.items()))
    return {
        "kkt_residual": float(np.max(np.abs(residual), initial=0.0)),
        "constraint_violation": violation,
        "complementarity_residual": float(np.max(np.abs(multipliers * active_slacks), initial=0.0)),
        "active_coefficient_bounds": int(at_lower.sum() + at_upper.sum()),
        "at_lower": at_lower,
        "at_upper": at_upper,
        "multipliers": multipliers,
        "face_basis": face,
    }


def optimize_coupled(initial, evaluator, coordinates, cfg, method="staged", fixed=None):
    """Staged or joint SLSQP with analytic gradients and exactly inherited bounds."""
    scale = information_scale(
        evaluator.evaluate(np.zeros(8), information=True)["information"],
        cfg["information_scale_minimum"],
        cfg["information_scale_maximum"],
    )
    values = np.asarray(initial).copy() / scale
    for index, target in (fixed or {}).items():
        values[index] = target / scale[index]
    matrix = LOG_MAP * scale
    phases, calls, invalid = [], 0, 0
    stages = cfg["stages"] if method == "staged" else []
    for phase, indices in [
        *[(f"stage_{i + 1}", stage) for i, stage in enumerate(stages)],
        ("joint_polish", list(range(8))),
    ]:
        active = np.array([j for j in indices if j not in (fixed or {})], dtype=int)
        if not len(active):
            continue
        base = values.copy()
        shift = matrix @ base - matrix[:, active] @ base[active]
        movable = np.linalg.norm(matrix[:, active], axis=1) > 0
        constraint = LinearConstraint(
            matrix[movable][:, active],
            (coordinates.relative_lower - shift)[movable],
            (coordinates.relative_upper - shift)[movable],
        )

        def objective(local, base=base, active=active):
            nonlocal calls, invalid
            calls += 1
            proposal = base.copy()
            proposal[active] = local
            try:
                value, gradient = evaluator.value_gradient(scale * proposal)
                if not np.isfinite(value) or not np.isfinite(gradient).all():
                    raise FloatingPointError("Nonfinite working likelihood")
                return value, (gradient * scale)[active]
            except (np.linalg.LinAlgError, FloatingPointError):
                invalid += 1
                return 1e12, np.zeros(len(active))

        result = minimize(
            objective,
            base[active],
            jac=True,
            method="SLSQP",
            constraints=[constraint],
            options={
                "ftol": cfg["optimizer_ftol"],
                "maxiter": (
                    cfg["optimizer_max_iterations"]
                    if phase == "joint_polish"
                    else cfg["stage_max_iterations"]
                ),
            },
        )
        values[active] = result.x
        phases.append(
            {
                "phase": phase,
                "success": bool(result.success),
                "iterations": int(result.nit),
                "objective": float(result.fun),
                "message": str(result.message),
            }
        )
    eta = scale * values
    final = evaluator.evaluate(eta, gradient=True)
    audit = constraint_audit(
        eta, final["gradient"], coordinates, cfg["active_bound_tolerance"], fixed
    )
    return eta, result, phases, calls, invalid, audit
