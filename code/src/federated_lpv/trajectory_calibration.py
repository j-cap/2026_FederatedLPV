"""Continuous fixed-speed calibration; fitting functions accept observations only."""
from __future__ import annotations

import hashlib
import json
import numpy as np

from .vehicle import nonlinear_bicycle_rhs


def array_hash(value):
    a = np.asarray(value, dtype="<f8", order="C")
    return hashlib.sha256(json.dumps(list(a.shape)).encode() + a.tobytes()).hexdigest()


def rk4(parameters, x, u, speed, dt):
    def f(state):
        return nonlinear_bicycle_rhs(state, u, speed, parameters)
    k1 = f(x)
    k2 = f(x + dt*k1/2)
    k3 = f(x + dt*k2/2)
    k4 = f(x + dt*k3)
    return x + dt*(k1 + 2*k2 + 2*k3 + k4)/6


def window_indices(cfg):
    dt = cfg["sample_time"]
    onset = round((cfg["settle_s"] + cfg["hold_s"])/dt)
    quiet = np.arange(onset-round(cfg["center_window_s"]/dt), onset)
    fitting = np.arange(onset, onset+round(cfg["excitation_s"]/dt))
    return quiet, fitting


def waveform(cfg, amplitude_deg, speed, seed, validation=False):
    """Deterministic bounded waveform; no state/reference matrix is consulted."""
    dt, duration = cfg["sample_time"], cfg["excitation_s"]
    t = np.arange(round(duration/dt))*dt
    if validation:
        f0, f1 = cfg["validation_chirp_hz"]
        signal = np.sin(2*np.pi*(f0*t + (f1-f0)*t*t/(2*duration)))
    else:
        phases = np.random.default_rng(seed).uniform(0, 2*np.pi, len(cfg["frequencies_hz"]))
        signal = np.sin(2*np.pi*t[:, None]*np.asarray(cfg["frequencies_hz"])
                        + phases).sum(axis=1)
    edge = cfg["taper_s"]
    taper = np.sin(0.5*np.pi*np.minimum(1, t/edge))**2
    taper *= np.sin(0.5*np.pi*np.minimum(1, (duration-dt-t)/edge))**2
    signal *= taper
    signal /= max(np.max(np.abs(signal)), 1e-15)
    amplitude = min(cfg["max_excitation_deg"], amplitude_deg*(20/speed)**2)
    return np.deg2rad(amplitude)*signal


def generate_trajectory(parameters, speed, amplitude_deg, seed, cfg, validation=False):
    """One initial state, then continuous dynamics. Controller uses noisy yaw only.

    True parameters are used exclusively in the nonlinear simulator. No analytic
    equilibrium or reference matrix is used to design the input or center data.
    """
    dt = cfg["sample_time"]
    quiet, fitting = window_indices(cfg)
    n = int(fitting[-1]+1)
    rng = np.random.default_rng(np.random.SeedSequence([seed, int(validation), 11]))
    state_noise = rng.normal(size=(n+1, 2))*np.deg2rad(cfg["state_noise_deg"])
    input_noise = rng.normal(size=n)*np.deg2rad(cfg["input_noise_deg"])
    bridge_noise = rng.normal(size=(n, 2))*np.deg2rad(cfg["state_noise_deg"])
    wave = waveform(cfg, amplitude_deg, speed, seed, validation)
    x, u = np.zeros((n+1, 2)), np.zeros(n)
    ay, limited = np.zeros(n), np.zeros(n)
    settle = round(cfg["settle_s"]/dt)
    avg = round(cfg["center_window_s"]/dt)
    integral, base = 0., 0.
    limit, rate = np.deg2rad([cfg["steering_limit_deg"], cfg["steering_rate_deg_s"]])
    for k in range(n):
        if k < settle:
            fraction = min(1., k*dt/cfg["turn_ramp_s"])
            ramp = 0.5*(1-np.cos(np.pi*fraction))
            curvature = cfg["curvature_per_m"]*ramp
            error = speed*curvature - (x[k, 1]+state_noise[k, 1])
            raw = (np.arctan(cfg["nominal_wheelbase_m"]*curvature)
                   + cfg["yaw_kp"]*error + cfg["yaw_ki"]*integral)
        else:
            if k == settle:
                base = float(u[settle-avg:settle].mean())
            raw = base + (wave[k-fitting[0]] if k >= fitting[0] else 0.)
        previous = u[k-1] if k else 0.
        u[k] = np.clip(np.clip(raw, previous-rate*dt, previous+rate*dt), -limit, limit)
        limited[k] = abs(u[k]-raw) > 1e-12
        if k < settle and (not limited[k] or error*(raw-u[k]) <= 0):
            integral += dt*error
        dx = nonlinear_bicycle_rhs(x[k], u[k], speed, parameters)
        ay[k] = speed*(dx[0]+x[k, 1])
        x[k+1] = rk4(parameters, x[k], u[k], speed, dt)
        if not np.isfinite(x[k+1]).all():
            raise FloatingPointError("Nonfinite physical trajectory; no fit is admissible")
    psi = np.r_[0., np.cumsum(dt*(x[:-1, 1]+x[1:, 1])/2)]
    # v is longitudinal speed, not the magnitude of planar velocity.
    lateral = speed*np.tan(x[:, 0])
    velocity = np.column_stack([speed*np.cos(psi)-lateral*np.sin(psi),
                                speed*np.sin(psi)+lateral*np.cos(psi)])
    xy = np.vstack([np.zeros(2), np.cumsum(dt*(velocity[:-1]+velocity[1:])/2,axis=0)])
    return dict(time=np.arange(n+1)*dt, state=x, steering=u, ay=ay,
                observed_state=x+state_noise, observed_steering=u+input_noise,
                bridge_noise=bridge_noise, limited=limited, position=xy, heading=psi)


def observations(trace, cfg, mode):
    """No parameters or true equilibrium are arguments to this preprocessing."""
    quiet, fitting = window_indices(cfg)
    if mode == "sensor":
        state, steering = trace["observed_state"], trace["observed_steering"]
        targets = state[1:]
    else:
        state, steering = trace["state"], trace["steering"]
        targets = state[1:] + (trace["bridge_noise"] if mode == "bridge" else 0.)
    center = np.r_[state[quiet].mean(axis=0), steering[quiet].mean()]
    records = np.column_stack([state[fitting], steering[fitting], targets[fitting]])
    return records, center


def fit_records(records, center, cfg):
    z = np.asarray(records[:, :3])-np.asarray(center)
    y = np.asarray(records[:, 3:])-np.asarray(center[:2])
    scales = np.deg2rad(cfg["regressor_scales_deg"])
    singular = np.linalg.svd(z/scales, compute_uv=False)/np.sqrt(len(z))
    coefficient, _, rank, _ = np.linalg.lstsq(z, y, rcond=None)
    finite = np.isfinite(coefficient).all()
    target = np.r_[coefficient[:2].T.ravel(), coefficient[2:].T.ravel()]
    return target, dict(rank=int(rank), records=len(z), fit_valid=bool(rank == 3 and finite),
        scaled_condition=float(singular[0]/singular[-1]) if singular[-1] > 0 else None,
        scaled_sigma_min=float(singular[-1]),
        residual_rmse=np.sqrt(np.mean((y-z@coefficient)**2, axis=0)).tolist())


def prediction_diagnostics(target, trace, center, cfg):
    """Validation against clean simulator response, after freezing the fitted A/B.

    Each 1 s window starts once at its actual state; no state correction is made
    within that window. Predictions and persistence use the same initial state.
    """
    _, ix = window_indices(cfg)
    a, b = np.asarray(target[:4]).reshape(2, 2), np.asarray(target[4:])
    x, u = trace["state"]-center[:2], trace["steering"]-center[2]
    scales = np.deg2rad(cfg["regressor_scales_deg"][:2])
    one = x[ix]@a.T + u[ix, None]*b - x[ix+1]
    steps = round(cfg["prediction_horizon_s"]/cfg["sample_time"])
    errors, persistence = [], []
    diverged = False
    for start in ix[::steps]:
        estimate = x[start].copy()
        for k in range(int(start), min(int(start)+steps, int(ix[-1])+1)):
            estimate = a@estimate+b*u[k]
            if not np.isfinite(estimate).all() or np.linalg.norm(estimate/scales) > 1e9:
                diverged = True
                break
            errors.append((estimate-x[k+1])/scales)
            persistence.append((x[start]-x[k+1])/scales)
        if diverged:
            break
    return dict(one_step_nrmse=float(np.sqrt(np.mean((one/scales)**2))),
        rollout_diverged=diverged,
        rollout_nrmse=None if diverged else float(np.sqrt(np.mean(np.square(errors)))),
        rollout_state_rmse=None if diverged else
            (np.sqrt(np.mean(np.square(errors), axis=0))*scales).tolist(),
        persistence_nrmse=None if diverged else float(np.sqrt(np.mean(np.square(persistence)))))


def physical_diagnostics(trace, speed, cfg):
    quiet, fitting = window_indices(cfg)
    dt, x, u = cfg["sample_time"], trace["state"], trace["steering"]
    rate = np.diff(np.r_[0., u])/dt
    scale = np.deg2rad(cfg["regressor_scales_deg"][:2])
    slope = np.polyfit(np.arange(len(quiet))*dt, x[quiet]/scale, 1)[0]
    yaw_error = abs(float(x[quiet, 1].mean())-speed*cfg["curvature_per_m"])
    result = dict(max_beta_deg=float(np.rad2deg(np.abs(x[:, 0]).max())),
        max_ay=float(np.abs(trace["ay"]).max()),
        max_jerk=float(np.abs(np.diff(trace["ay"])/dt).max()),
        max_steering_deg=float(np.rad2deg(np.abs(u).max())),
        max_rate_deg_s=float(np.rad2deg(np.abs(rate).max())),
        excitation_limited_steps=int(trace["limited"][fitting].sum()),
        quiet_yaw_error=yaw_error, quiet_drift=float(np.max(np.abs(slope))),
        physical_state_continuous=True, initialization_count=1,
        total_duration_s=float(trace["time"][-1]),
        identification_duration_s=cfg["excitation_s"])
    result["physical_pass"] = bool(result["max_beta_deg"] <= cfg["max_beta_deg"]
        and result["max_ay"] <= cfg["max_ay"]
        and result["max_steering_deg"] <= cfg["steering_limit_deg"]+1e-8
        and result["max_rate_deg_s"] <= cfg["steering_rate_deg_s"]+1e-8
        and result["excitation_limited_steps"] == 0)
    result["settling_pass"] = bool(yaw_error <= max(cfg["settling_abs_yaw"],
        cfg["settling_relative_yaw"]*speed*cfg["curvature_per_m"])
        and result["quiet_drift"] <= cfg["max_quiet_drift"])
    return result
