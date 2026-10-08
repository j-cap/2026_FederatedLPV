"""Truth-boundary diagnostics for frozen output-feedback experiments.

These calculations are simulation interventions and scorers, not learners.
The rollout adapter must keep corrected controller input out of KF internals.
"""

from dataclasses import dataclass
from itertools import product

import numpy as np

STATE_NAMES = (
    "beta", "yaw_rate", "front_force_per_mass", "rear_force_per_mass", "applied_steering",
)
CORRECTION_INDICES = {
    "none": (),
    "beta": (0,),
    "forces": (2, 3),
    "beta_forces": (0, 2, 3),
    "all": (0, 1, 2, 3, 4),
}


@dataclass(frozen=True)
class FeedbackCase:
    name: str
    controller: str
    state_source: str
    correction: str


def diagnostic_cases(config):
    """Return the locked factorial and the learned-Both partial corrections."""
    cases = [
        FeedbackCase(
            f"{controller}_with_{state_source}", controller, state_source,
            "all" if state_source == "true_state" else "none",
        )
        for controller, state_source in product(
            config["controller_designs"], config["state_sources"],
        )
    ]
    cases.extend(
        FeedbackCase(
            f"learned_with_learned_kf_correct_{correction}",
            "learned", "learned_kf", correction,
        )
        for correction in config["learned_both_corrections"]
    )
    if len({case.name for case in cases}) != len(cases):
        raise ValueError("Diagnostic case names must be unique")
    if any(case.correction not in CORRECTION_INDICES for case in cases):
        raise ValueError("Unknown truth correction")
    return tuple(cases)


def _state_pair(estimate, truth):
    estimate = np.asarray(estimate, dtype=float)
    truth = np.asarray(truth, dtype=float)
    if estimate.shape != truth.shape or estimate.ndim not in (1, 2):
        raise ValueError("Estimate and truth must have matching state/trajectory shapes")
    if estimate.shape[-1] != 5 or estimate.size == 0:
        raise ValueError("Expected five physical state coordinates and nonempty data")
    if not (np.isfinite(estimate).all() and np.isfinite(truth).all()):
        raise ValueError("Nonfinite trajectories must be recorded as failures, not scored")
    return estimate, truth


def corrected_controller_state(estimate, truth, correction):
    """Return a separate controller input; never mutate either state array."""
    estimate, truth = _state_pair(estimate, truth)
    if correction not in CORRECTION_INDICES:
        raise ValueError(f"Unknown truth correction: {correction}")
    result = estimate.copy()
    indices = CORRECTION_INDICES[correction]
    if indices:
        result[..., indices] = truth[..., indices]
    return result


def command_error_components(estimate, truth, state_feedback_gain):
    """Signed contributions to u(estimate)-u(truth), before clipping.

Hold reference and integrator state fixed. Pass only the five plant-state
columns of K; an integrator column is deliberately rejected. Gains may be
constant or already interpolated along the matching trajectory.
"""
    estimate, truth = _state_pair(estimate, truth)
    gain = np.asarray(state_feedback_gain, dtype=float)
    if gain.shape not in ((5,), estimate.shape):
        raise ValueError("Use five state gains, constant or matching the trajectory")
    if not np.isfinite(gain).all():
        raise ValueError("Nonfinite controller gain")
    return -gain * (estimate - truth)


def command_error_summary(contributions, start=0):
    """Summarize command errors without dropping their cross terms.

Inputs and RMS results use steering radians. The uncentered second-moment
matrix uses radians squared and retains both bias and cancellation.
"""
    values = np.asarray(contributions, dtype=float)
    if values.ndim != 2 or values.shape[1] != 5:
        raise ValueError("Expected a trajectory of five command contributions")
    if not isinstance(start, (int, np.integer)) or not 0 <= start < len(values):
        raise ValueError("The scoring window must contain at least one sample")
    values = values[start:]
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite contributions must be recorded as failures")
    moments = values.T @ values / len(values)
    joint_force = values[:, 2] + values[:, 3]
    total = values.sum(axis=1)
    return {
        "samples": len(values),
        "component_rms_rad": dict(zip(STATE_NAMES, np.sqrt(np.diag(moments)).tolist())),
        "joint_force_rms_rad": float(np.sqrt(np.mean(joint_force**2))),
        "total_rms_rad": float(np.sqrt(np.mean(total**2))),
        "mean_command_error_rad": float(total.mean()),
        "second_moment_rad2": moments.tolist(),
    }
