"""Matched LQI synthesis and nonlinear rollouts for frozen OE-derived maps.

No identification, parameter fitting, gain tuning, or model repair occurs here.
State feedback is noise-free. Simulation parameters never enter synthesis.
"""
from __future__ import annotations
import numpy as np
from .control import design_lqi_gain, augmented_tracking_matrices
from .vehicle import nonlinear_bicycle_rhs


def design_nodes(matrices, cfg):
    """Keep all node diagnostics; one failed node makes the schedule unusable."""
    nodes = []
    for speed, matrix in zip(cfg['speed_grid'], matrices, strict=True):
        node = dict(speed=float(speed), matrix=np.asarray(matrix).tolist())
        try:
            if not np.isfinite(matrix).all():
                raise ValueError('nonfinite model')
            a, b = matrix[:4].reshape(2, 2), matrix[4:].reshape(2, 1)
            gain, prefilter = design_lqi_gain(a, b, cfg['sample_time'], np.diag(cfg['q']), [[cfg['r']]])
            aa, bb = augmented_tracking_matrices(a, b, cfg['sample_time'])
            rho = float(max(abs(np.linalg.eigvals(aa-bb@gain[None, :]))))
            if not np.isfinite(np.r_[gain, prefilter, rho]).all() or rho >= 1 + 1e-9:
                raise ValueError('nonfinite or nonstabilizing Riccati solution')
            node.update(status='designed', gain=gain.tolist(), prefilter=float(prefilter), design_rho=rho)
        except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
            node.update(status='synthesis_failed', reason=f'{type(exc).__name__}: {exc}')
        nodes.append(node)
    return dict(status='designed' if all(n['status']=='designed' for n in nodes) else 'synthesis_failed', nodes=nodes)


def interpolate(schedule, grid, speeds):
    if schedule['status'] != 'designed':
        raise ValueError('Cannot interpolate a failed schedule')
    speeds = np.asarray(speeds, float)
    if np.any(speeds < grid[0]-1e-12) or np.any(speeds > grid[-1]+1e-12):
        raise ValueError('Requested speed outside frozen design grid')
    gains = np.asarray([n['gain'] for n in schedule['nodes']])
    prefilters = np.asarray([n['prefilter'] for n in schedule['nodes']])
    return (np.column_stack([np.interp(speeds, grid, gains[:,j]) for j in range(3)]),
            np.interp(speeds, grid, prefilters))


def rk4(state, steering, speed, parameters, dt):
    rhs = lambda x: nonlinear_bicycle_rhs(x, steering, speed, parameters)
    k1 = rhs(state); k2 = rhs(state + dt*k1/2)
    k3 = rhs(state + dt*k2/2); k4 = rhs(state + dt*k3)
    return state + dt*(k1+2*k2+2*k3+k4)/6


def simulate(parameters, schedule, profile, cfg):
    """Return full-duration metrics or an explicit failure and its finite prefix.

    A finite sideslip-limit violation does not stop integration or remove errors
    from RMSE. Numeric overflow/large-state protection makes full metrics null.
    """
    if schedule['status'] != 'designed':
        return dict(status='synthesis_failed', feasible=False, metrics=None, completed_steps=0), None
    time, speed, reference = [np.asarray(profile[k],float) for k in ['time','speed','reference']]
    if not (len(time)==len(speed)==len(reference)) or len(time)<2:
        raise ValueError('Invalid profile lengths')
    dt=cfg['sample_time']; np.testing.assert_allclose(np.diff(time),dt,atol=1e-12,rtol=0)
    if not np.isfinite(np.r_[time,speed,reference]).all(): raise ValueError('Nonfinite profile')
    gains, prefilters = interpolate(schedule, cfg['speed_grid'], speed[:-1])
    n=len(time)-1; state=np.zeros((n+1,2)); integral=np.zeros(n+1)
    steering=np.zeros(n); requested=np.zeros(n); ay=np.zeros(n)
    angle_limited=np.zeros(n); rate_limited=np.zeros(n); limited=np.zeros(n)
    limit=np.deg2rad(cfg['steering_limit_deg']); rate=np.deg2rad(cfg['steering_rate_limit_deg_s'])
    done=0; reason=None
    for k in range(n):
        try:
            with np.errstate(over='raise',invalid='raise',divide='raise'):
                raw=-gains[k]@np.r_[state[k],integral[k]]+prefilters[k]*reference[k]
                previous=steering[k-1] if k else 0.
                rate_command=np.clip(raw,previous-rate*dt,previous+rate*dt)
                command=float(np.clip(rate_command,-limit,limit))
                lateral=speed[k]*(nonlinear_bicycle_rhs(state[k],command,speed[k],parameters)[0]+state[k,1])
                next_state=rk4(state[k],command,speed[k],parameters,dt)
                if not np.isfinite(np.r_[raw,command,lateral,next_state]).all() or np.max(abs(next_state))>cfg['numerical_abort_abs_state']:
                    raise FloatingPointError('nonfinite or excessive state; full trajectory unavailable')
                error=state[k,1]-reference[k]
                candidate=float(np.clip(integral[k]+dt*error,-cfg['integral_limit'],cfg['integral_limit']))
                saturated=abs(command-raw)>1e-12
                integral[k+1]=candidate if not saturated or (-gains[k,2]*(candidate-integral[k]))*(raw-command)<=0 else integral[k]
                state[k+1]=next_state; steering[k]=command; requested[k]=raw; ay[k]=lateral
                limited[k]=saturated; rate_limited[k]=abs(rate_command-raw)>1e-12
                angle_limited[k]=abs(command-rate_command)>1e-12; done=k+1
        except (FloatingPointError, OverflowError, ValueError, np.linalg.LinAlgError) as exc:
            reason=f'{type(exc).__name__}: {exc}'; break
    arrays=dict(time=time[:done+1],speed=speed[:done+1],reference=reference[:done+1],
                state=state[:done+1],integral=integral[:done+1],steering=steering[:done],
                requested_steering=requested[:done],lateral_acceleration=ay[:done],
                limited=limited[:done],angle_limited=angle_limited[:done],rate_limited=rate_limited[:done])
    if reason:
        return dict(status='rollout_failed',reason=reason,feasible=False,metrics=None,completed_steps=done),arrays
    tracking=state[:,1]-reference; steering_rate=np.diff(np.r_[0.,steering])/dt
    rms=lambda x: float(np.sqrt(np.mean(np.square(x))))
    metrics=dict(tracking_rmse_deg_s=float(np.rad2deg(rms(tracking))),
        tracking_peak_deg_s=float(np.rad2deg(max(abs(tracking)))),
        beta_rms_deg=float(np.rad2deg(rms(state[:,0]))),beta_peak_deg=float(np.rad2deg(max(abs(state[:,0])))),
        steering_rms_deg=float(np.rad2deg(rms(steering))),steering_peak_deg=float(np.rad2deg(max(abs(steering)))),
        steering_rate_rms_deg_s=float(np.rad2deg(rms(steering_rate))),
        steering_rate_peak_deg_s=float(np.rad2deg(max(abs(steering_rate)))),
        steering_total_variation_deg=float(np.rad2deg(np.sum(abs(np.diff(np.r_[0.,steering]))))),
        lateral_acceleration_peak_mps2=float(max(abs(ay))),
        limiter_fraction=float(limited.mean()),angle_limiter_fraction=float(angle_limited.mean()),
        rate_limiter_fraction=float(rate_limited.mean()),
        beta_violation_samples=int(np.sum(abs(np.rad2deg(state[:,0]))>cfg['beta_limit_deg'])))
    if not np.isfinite(list(metrics.values())).all():
        return dict(status='rollout_failed',reason='nonfinite metrics',feasible=False,metrics=None,completed_steps=done),arrays
    bounds=metrics['steering_peak_deg']<=cfg['steering_limit_deg']+1e-8 and metrics['steering_rate_peak_deg_s']<=cfg['steering_rate_limit_deg_s']+1e-8
    feasible=bool(bounds and metrics['beta_violation_samples']==0)
    return dict(status='complete',feasible=feasible,metrics=metrics,completed_steps=done),arrays
