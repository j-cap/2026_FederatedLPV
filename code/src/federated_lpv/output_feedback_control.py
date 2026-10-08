"""Causal scheduled KF/LQI rollout and fixed-input replay, no learner inputs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_discrete_are

from federated_lpv.feedback_diagnostics import CORRECTION_INDICES, corrected_controller_state
from federated_lpv.innovation_likelihood import C


@dataclass(frozen=True)
class Schedule:
    grid: np.ndarray
    a: np.ndarray
    b: np.ndarray
    observer: np.ndarray
    feedback: np.ndarray
    prefilter: np.ndarray

    def at(self, speeds):
        return {
            name: schedule_values(getattr(self, name), self.grid, speeds)
            for name in ("a", "b", "observer", "feedback", "prefilter")
        }


def schedule_values(values, grid, speeds):
    grid, speeds = np.asarray(grid), np.asarray(speeds)
    if np.any(speeds < grid[0] - 1e-10) or np.any(speeds > grid[-1] + 1e-10):
        raise ValueError("Scheduling speed outside design grid")
    index = np.clip(np.searchsorted(grid, speeds, side="right") - 1, 0, len(grid) - 2)
    weight = (speeds - grid[index]) / (grid[index + 1] - grid[index])
    shape = weight.shape + (1,) * (np.asarray(values).ndim - 1)
    weight = weight.reshape(shape)
    return (1 - weight) * values[index] + weight * values[index + 1]


def prior_kalman_gain(a, q, r):
    p = solve_discrete_are(a.T, C.T, q, r)
    return np.linalg.solve(C @ p @ C.T + r, C @ p).T


def lqi_design(a, b, dt, weights):
    b = np.asarray(b).reshape(5, 1)
    cy = C[0:1]
    aa = np.block([[a, np.zeros((5, 1))], [dt * cy, np.ones((1, 1))]])
    bb = np.vstack([b, np.zeros((1, 1))])
    q = np.diag(
        [
            weights["q_beta"],
            weights["q_yaw"],
            weights["q_force"],
            weights["q_force"],
            weights["q_actuator"],
            weights["q_integral"],
        ]
    )
    r = np.array([[weights["r"]]])
    p = solve_discrete_are(aa, bb, q, r)
    gain = np.linalg.solve(r + bb.T @ p @ bb, bb.T @ p @ aa).ravel()
    equilibrium = np.linalg.solve(
        np.block([[np.eye(5) - a, -b], [cy, np.zeros((1, 1))]]), np.r_[np.zeros(5), 1.0]
    )
    prefilter = float(equilibrium[5] + gain[:5] @ equilibrium[:5])
    return gain, prefilter


def design_schedule(matrices, grid, q, r, weights, dt):
    a = np.asarray([m[0] for m in matrices])
    b = np.asarray([m[1].reshape(5) for m in matrices])
    gains = np.asarray([prior_kalman_gain(item, q, r) for item in a])
    controller = [lqi_design(ai, bi, dt, weights) for ai, bi in zip(a, b)]
    return Schedule(
        np.asarray(grid),
        a,
        b,
        gains,
        np.asarray([item[0] for item in controller]),
        np.asarray([item[1] for item in controller]),
    )


def rollout(
    true_matrices,
    true_grid,
    observer,
    controller,
    scenario,
    corrected_noise,
    dt=0.01,
    limit=0.2617993877991494,
    correction="none",
):
    """u[k] from current estimate; predict using clipped u[k], correct y[k+1]."""
    speed, reference = scenario["speed"], scenario["reference"]
    n = len(speed)
    if np.asarray(corrected_noise).shape != (n, 3):
        raise ValueError("One corrected sensor-noise vector per state sample required")
    true_a = schedule_values(np.asarray([m[0] for m in true_matrices]), true_grid, speed[:-1])
    true_b = schedule_values(
        np.asarray([m[1].reshape(5) for m in true_matrices]), true_grid, speed[:-1]
    )
    obs, control = observer.at(speed), controller.at(speed)
    x, estimate, supplied = (np.zeros((n, 5)) for _ in range(3))
    measured = np.zeros((n, 3))
    eta = np.zeros(n)
    raw, command = np.zeros(n - 1), np.zeros(n - 1)
    measured[0] = corrected_noise[0]
    estimate[0] = obs["observer"][0] @ measured[0]
    for k in range(n - 1):
        supplied[k] = corrected_controller_state(estimate[k], x[k], correction)
        gain = control["feedback"][k]
        raw[k] = -gain[:5] @ supplied[k] - gain[5] * eta[k] + control["prefilter"][k] * reference[k]
        command[k] = np.clip(raw[k], -limit, limit)
        increment = dt * (measured[k, 0] - reference[k])
        worsens_saturation = abs(raw[k]) > limit and raw[k] * (-gain[5] * increment) > 0
        eta[k + 1] = eta[k] if worsens_saturation else eta[k] + increment
        x[k + 1] = true_a[k] @ x[k] + true_b[k] * command[k]
        measured[k + 1] = C @ x[k + 1] + corrected_noise[k + 1]
        predicted = obs["a"][k] @ estimate[k] + obs["b"][k] * command[k]
        estimate[k + 1] = predicted + obs["observer"][k] @ (measured[k + 1] - C @ predicted)
    supplied[-1] = corrected_controller_state(estimate[-1], x[-1], correction)
    return {
        "truth": x,
        "estimate": estimate,
        "supplied": supplied,
        "measurement": measured,
        "integral": eta,
        "raw_command": raw,
        "command": command,
    }


def replay_observer(record, observer, speeds):
    """The replay accepts commands/measurements only; truth is outside its API."""
    command, measured = record["command"], record["measurement"]
    if len(measured) != len(command) + 1 or len(speeds) != len(measured):
        raise ValueError("Replay requires causal state/input timing")
    scheduled = observer.at(speeds)
    estimate = np.zeros((len(measured), 5))
    estimate[0] = scheduled["observer"][0] @ measured[0]
    for k, u in enumerate(command):
        predicted = scheduled["a"][k] @ estimate[k] + scheduled["b"][k] * u
        estimate[k + 1] = predicted + scheduled["observer"][k] @ (measured[k + 1] - C @ predicted)
    return estimate


def feedback_radius(
    true_a,
    true_b,
    observer_a,
    observer_b,
    observer_gain,
    gain,
    dt,
    full_state=False,
    correction="none",
):
    """Frozen unsaturated eigenvalue diagnostic, not a scheduled stability proof."""
    bt = np.asarray(true_b).reshape(5, 1)
    kx, ki = gain[:5][None], gain[5]
    if full_state:
        matrix = np.block([[true_a - bt @ kx, -bt * ki], [dt * C[0:1], np.ones((1, 1))]])
    else:
        lc = observer_gain @ C
        m = np.eye(5) - lc
        d = m @ np.asarray(observer_b).reshape(5, 1) + lc @ bt
        selector = np.diag([float(j in CORRECTION_INDICES[correction]) for j in range(5)])
        truth_gain, estimate_gain = kx @ selector, kx @ (np.eye(5) - selector)
        matrix = np.block(
            [
                [true_a - bt @ truth_gain, -bt @ estimate_gain, -bt * ki],
                [lc @ true_a - d @ truth_gain, m @ observer_a - d @ estimate_gain, -d * ki],
                [dt * C[0:1], np.zeros((1, 5)), np.ones((1, 1))],
            ]
        )
    return float(np.max(np.abs(np.linalg.eigvals(matrix))))


def metrics(record, scenario, start, limit, beta_limit):
    truth, estimate = record["truth"], record["estimate"]
    error = estimate[start:] - truth[start:]
    tracking = np.rad2deg(truth[:, 1] - scenario["reference"])
    command = record["command"]
    return {
        "tracking_rmse_deg_s": float(np.sqrt(np.mean(tracking**2))),
        "tracking_peak_deg_s": float(np.max(np.abs(tracking))),
        "beta_rmse_deg": float(np.rad2deg(np.sqrt(np.mean(error[:, 0] ** 2)))),
        "yaw_rmse_deg_s": float(np.rad2deg(np.sqrt(np.mean(error[:, 1] ** 2)))),
        "front_force_rmse": float(np.sqrt(np.mean(error[:, 2] ** 2))),
        "rear_force_rmse": float(np.sqrt(np.mean(error[:, 3] ** 2))),
        "applied_steer_rmse_deg": float(np.rad2deg(np.sqrt(np.mean(error[:, 4] ** 2)))),
        "command_rms_deg": float(np.rad2deg(np.sqrt(np.mean(command**2)))),
        "command_rate_rms_deg_s": float(
            np.rad2deg(np.sqrt(np.mean(np.diff(command) ** 2)))
            / (scenario["time"][1] - scenario["time"][0])
        ),
        "saturation_samples": int(np.count_nonzero(abs(record["raw_command"]) > limit)),
        "beta_violation_samples": int(np.count_nonzero(abs(truth[:, 0]) > beta_limit)),
        "steering_violation_samples": int(np.count_nonzero(abs(truth[:, 4]) > limit)),
        "finite": bool(all(np.isfinite(v).all() for v in record.values())),
    }
