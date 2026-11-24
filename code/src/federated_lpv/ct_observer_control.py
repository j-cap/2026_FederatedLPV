"""Yaw-only output feedback with a physical-coordinate discrete Kalman observer.

The online feedback function accepts yaw, reference and internal memory only.
True states and recipient parameters are confined to simulation and diagnostics.
OE affine offsets and fitted initial states never enter this module.
"""
from __future__ import annotations

import hashlib
import numpy as np
from scipy.linalg import solve_discrete_are

from .physical_ct_oe import discrete_matrices
from .oe_lqi_validation import interpolate, rk4
from .vehicle import nonlinear_bicycle_rhs

C = np.array([[0., 1.]])


def observer_model(theta, speed, settings):
    """Physical beta/r coordinates at straight running, exact ZOH; no OE gauge."""
    return discrete_matrices(theta, speed, settings['geometry_m'], 0.,
                             settings['sample_time'])


def kalman_design(a, b, settings, samples):
    """Finite-time gains from P0=0 plus steady-state diagnostics.

    Q is a common design allowance for model discrepancy, not injected plant
    noise or an estimated covariance. R is the simulated yaw sensor variance.
    """
    a = np.asarray(a, float).reshape(2, 2)
    b = np.asarray(b, float).reshape(2)
    q = np.diag(np.deg2rad(settings['process_step_std_deg'])**2)
    r = float(np.deg2rad(settings['yaw_noise_std_deg_s'])**2)
    if samples < 1 or r <= 0 or np.any(np.diag(q) <= 0):
        raise ValueError('Positive samples and noise design variances required')
    prior = solve_discrete_are(a.T, C.T, q, np.array([[r]]))
    steady = prior[:, 1] / (prior[1, 1] + r)
    rho = float(max(abs(np.linalg.eigvals(a @ (np.eye(2)-steady[:, None]@C)))))
    if not np.isfinite(prior).all() or rho >= 1:
        raise ValueError('No stabilizing observer Riccati solution')
    # Conditioning uses fixed physical scales, not method-specific magnitudes.
    scales = np.deg2rad(settings['observability_scales_deg'])
    observability = np.vstack([C, C@a])*scales[None, :]/scales[1]
    singular = np.linalg.svd(observability, compute_uv=False)
    rank = int(np.linalg.matrix_rank(observability))
    gains = np.zeros((samples, 2)); variances = np.zeros(samples)
    covariance = np.zeros((2, 2))  # known zero physical initial state
    for k in range(samples):
        variances[k] = covariance[1, 1] + r
        gains[k] = covariance[:, 1]/variances[k]
        j = np.eye(2)-gains[k, :, None]@C
        posterior = j@covariance@j.T + r*np.outer(gains[k], gains[k])
        covariance = a@posterior@a.T + q
        covariance = (covariance+covariance.T)/2
    diagnostics = dict(a=a.tolist(), b=b.tolist(), q=q.tolist(), r=r,
        initial_covariance=np.zeros((2, 2)).tolist(), steady_gain=steady.tolist(),
        error_rho=rho, observability_rank=rank,
        observability_singular_values=singular.tolist(),
        observability_condition=float(singular[0]/singular[-1]) if rank == 2 else None)
    return dict(a=a, b=b, gains=gains, innovation_variances=variances,
                diagnostics=diagnostics)


def feedback_step(prior, integral, previous_input, measured_yaw, reference,
                  gain, prefilter, observer_gain, a, b, cfg):
    """Correct -> control/limit -> predict using applied input; no true state."""
    innovation = float(measured_yaw-prior[1])
    posterior = prior + observer_gain*innovation
    requested = float(-gain[:2]@posterior-gain[2]*integral+prefilter*reference)
    dt = cfg['sample_time']
    limit = np.deg2rad(cfg['steering_limit_deg'])
    step = np.deg2rad(cfg['steering_rate_limit_deg_s'])*dt
    rate_command = float(np.clip(requested, previous_input-step, previous_input+step))
    applied = float(np.clip(rate_command, -limit, limit))
    limited = abs(applied-requested) > 1e-12
    # Measured yaw, not latent sideslip or simulated yaw, drives integral action.
    candidate = float(np.clip(integral+dt*(measured_yaw-reference),
                              -cfg['integral_limit'], cfg['integral_limit']))
    next_integral = candidate if (not limited or
        (-gain[2]*(candidate-integral))*(requested-applied) <= 0) else integral
    return dict(posterior=posterior, next_prior=a@posterior+b*applied,
        integral=next_integral, applied=applied, requested=requested,
        innovation=innovation, limited=limited,
        rate_limited=abs(rate_command-requested) > 1e-12,
        angle_limited=abs(applied-rate_command) > 1e-12)


def origin_closed_loop(ap, bp, ao, bo, gain, observer_gain, dt):
    """5-state deterministic Jacobian: [true x, predicted x, integral].

    Uses the steady observer gain, measured-yaw integration and inactive limits.
    This is a local frozen-speed diagnostic, not a nonlinear stability proof.
    """
    bp = np.asarray(bp).reshape(2, 1); bo = np.asarray(bo).reshape(2, 1)
    lc = np.asarray(observer_gain).reshape(2, 1)@C
    j = np.eye(2)-lc; ks = np.asarray(gain[:2]).reshape(1, 2)
    return np.block([
        [ap-bp@ks@lc, -bp@ks@j, -bp*gain[2]],
        [(ao-bo@ks)@lc, (ao-bo@ks)@j, -bo*gain[2]],
        [dt*C, np.zeros((1, 2)), np.ones((1, 1))]])


def trace_hash(arrays):
    """Canonical little-endian numeric payload, independent of NPZ timestamps."""
    digest = hashlib.sha256()
    for key, value in sorted(arrays.items()):
        a = np.ascontiguousarray(value, dtype='<f8')
        digest.update(key.encode('utf-8'))
        digest.update(str(a.shape).encode('ascii'))
        digest.update(a.tobytes())
    return digest.hexdigest()


def yaw_noise(seed, client_index, speed, samples, settings):
    sequence = np.random.SeedSequence([settings['noise_namespace'], seed,
                                      client_index, int(round(1000*speed))])
    return np.random.default_rng(sequence).normal(0.,
        np.deg2rad(settings['yaw_noise_std_deg_s']), samples)


def simulate_observed(parameters, schedule, profile, observer, noise, cfg):
    """No full-state feedback; complete finite violations stay in all metrics."""
    time, speed, reference = [np.asarray(profile[k], float)
                              for k in ['time', 'speed', 'reference']]
    n = len(time)-1; dt = cfg['sample_time']
    if len(speed) != n+1 or len(reference) != n+1 or np.shape(noise) != (n+1,):
        raise ValueError('Inconsistent trace lengths')
    np.testing.assert_allclose(np.diff(time), dt, atol=1e-12, rtol=0)
    if not np.isfinite(np.r_[time, speed, reference, noise]).all():
        raise ValueError('Nonfinite profile/noise')
    if not np.all(speed == speed[0]):
        raise ValueError('This diagnostic is fixed speed only')
    gains, prefilters = interpolate(schedule, cfg['speed_grid'], speed[:-1])
    if observer['gains'].shape != (n+1, 2):
        raise ValueError('Observer horizon mismatch')
    state = np.zeros((n+1, 2)); estimate = np.zeros_like(state)
    measured = np.zeros(n+1); innovation = np.zeros(n+1)
    steering = np.zeros(n); requested = np.zeros(n); ay = np.zeros(n)
    integral = np.zeros(n+1); limited = np.zeros(n)
    rate_limited = np.zeros(n); angle_limited = np.zeros(n)
    prior = np.zeros(2); done = 0; reason = None
    for k in range(n):
        try:
            with np.errstate(over='raise', invalid='raise', divide='raise'):
                measured[k] = state[k, 1]+noise[k]
                feedback = feedback_step(prior, integral[k], steering[k-1] if k else 0.,
                    measured[k], reference[k], gains[k], prefilters[k],
                    observer['gains'][k], observer['a'], observer['b'], cfg)
                command = feedback['applied']
                next_state = rk4(state[k], command, speed[k], parameters, dt)
                if (not np.isfinite(np.r_[next_state, feedback['next_prior']]).all() or
                    np.max(abs(np.r_[next_state, feedback['next_prior']])) > cfg['numerical_abort_abs_state']):
                    raise FloatingPointError('Nonfinite/excessive state or observer state')
                estimate[k] = feedback['posterior']; innovation[k] = feedback['innovation']
                prior = feedback['next_prior']; integral[k+1] = feedback['integral']
                steering[k] = command; requested[k] = feedback['requested']
                for name, array in [('limited', limited), ('rate_limited', rate_limited),
                                    ('angle_limited', angle_limited)]:
                    array[k] = feedback[name]
                ay[k] = speed[k]*(nonlinear_bicycle_rhs(state[k], command, speed[k], parameters)[0]+state[k, 1])
                state[k+1] = next_state; done = k+1
        except (ValueError, FloatingPointError, OverflowError, np.linalg.LinAlgError) as exc:
            reason = f'{type(exc).__name__}: {exc}'; break
    measured[done] = state[done, 1]+noise[done]
    innovation[done] = measured[done]-prior[1]
    estimate[done] = prior+observer['gains'][done]*innovation[done]
    arrays = dict(time=time[:done+1], speed=speed[:done+1], reference=reference[:done+1],
        state=state[:done+1], estimate=estimate[:done+1], measured_yaw=measured[:done+1],
        innovation=innovation[:done+1], noise=noise[:done+1], integral=integral[:done+1],
        steering=steering[:done], requested_steering=requested[:done],
        lateral_acceleration=ay[:done], limited=limited[:done],
        angle_limited=angle_limited[:done], rate_limited=rate_limited[:done])
    if reason:
        return dict(status='rollout_failed', reason=reason, metrics=None,
                    feasible=False, completed_steps=done), arrays
    rms = lambda x: float(np.sqrt(np.mean(np.square(x))))
    degrees = lambda x: float(np.rad2deg(x))
    tracking = state[:, 1]-reference
    error = estimate-state
    rate = np.diff(np.r_[0., steering])/dt
    metrics = dict(tracking_rmse_deg_s=degrees(rms(tracking)),
        tracking_peak_deg_s=degrees(max(abs(tracking))),
        beta_estimation_rmse_deg=degrees(rms(error[:, 0])),
        beta_estimation_peak_deg=degrees(max(abs(error[:, 0]))),
        yaw_estimation_rmse_deg_s=degrees(rms(error[:, 1])),
        beta_peak_deg=degrees(max(abs(state[:, 0]))),
        steering_rms_deg=degrees(rms(steering)), steering_peak_deg=degrees(max(abs(steering))),
        steering_rate_rms_deg_s=degrees(rms(rate)), steering_rate_peak_deg_s=degrees(max(abs(rate))),
        steering_total_variation_deg=degrees(np.sum(abs(np.diff(np.r_[0., steering])))),
        lateral_acceleration_peak_mps2=float(max(abs(ay))),
        limiter_fraction=float(limited.mean()), rate_limiter_fraction=float(rate_limited.mean()),
        angle_limiter_fraction=float(angle_limited.mean()),
        beta_violation_samples=int(np.sum(abs(np.rad2deg(state[:, 0])) > cfg['beta_limit_deg'])),
        innovation_rms_deg_s=degrees(rms(innovation)),
        mean_normalized_innovation_squared=float(np.mean(innovation**2/observer['innovation_variances'])))
    if not np.isfinite(list(metrics.values())).all():
        return dict(status='rollout_failed', reason='Nonfinite full metrics', metrics=None,
                    feasible=False, completed_steps=done), arrays
    feasible = bool(metrics['beta_violation_samples'] == 0 and
        metrics['steering_peak_deg'] <= cfg['steering_limit_deg']+1e-8 and
        metrics['steering_rate_peak_deg_s'] <= cfg['steering_rate_limit_deg_s']+1e-8)
    return dict(status='complete', metrics=metrics, feasible=feasible, completed_steps=done), arrays
