"""Matched nonlinear yaw-only OE and output-feedback control for the relaxation pilot.

Identification accepts measured yaw, applied steering and speed only. All records
start from a deliberately settled, known zero state. No affine offsets are fitted.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm, solve_discrete_are
from scipy.optimize import least_squares

from .tire_relaxation import rhs, continuous_matrices, project_nuisance, svd_summary
from .physical_ct_oe import continuous_matrices as instantaneous_matrices


def instantaneous_rhs(x, u, speed, p, geometry):
    p1, p2, p3, mu = np.asarray(p).T
    beta, yaw = np.asarray(x).T
    lf, lr = geometry
    cf = mu*9.81*lr/(lf+lr); cr = mu*9.81*lf/(lf+lr)
    qf = cf*np.tanh(p1*(u-beta-lf*yaw/speed)/cf)
    qr = cr*np.tanh(p2*(-beta+lr*yaw/speed)/cr)
    return np.stack([(qf+qr)/speed-yaw, p3*(lf*qf-lr*qr)], axis=-1)


def batch_predict(parameters, inputs, speeds, geometry, dt, max_step):
    """RK4 over equal-length records and parameter candidates in one batch.

    Returns [candidate, record, sample, state]. Complex-step differentiation is
    applied to this same discrete numerical map, including every RK4 stage.
    """
    p = np.atleast_2d(np.asarray(parameters)); u = np.atleast_2d(np.asarray(inputs))
    speeds = np.asarray(speeds, float)
    if p.shape[1] not in (4, 6) or u.shape[0] != len(speeds):
        raise ValueError('Expected four/six parameters and one input row per speed')
    if not np.isfinite(p).all() or np.any(p.real <= 0) or min(speeds) <= 0:
        raise ValueError('Positive finite parameters and speeds required')
    npar, nrec = len(p), len(u); dim = 4 if p.shape[1] == 6 else 2
    pars = np.repeat(p, nrec, axis=0); v = np.tile(speeds, npar)
    sub = int(np.ceil(dt/max_step)); h = dt/sub
    if dim == 4 and h > np.min(pars.real[:, 4:]/v[:, None])/2:
        raise ValueError('Integration step too large for relaxation time')
    f = rhs if dim == 4 else instantaneous_rhs
    x = np.zeros((npar*nrec, dim), dtype=np.result_type(p, float))
    out = np.empty((npar*nrec, u.shape[1]+1, dim), dtype=x.dtype); out[:, 0] = x
    for k in range(u.shape[1]):
        delta = np.tile(u[:, k], npar)
        for _ in range(sub):
            k1 = f(x, delta, v, pars, geometry)
            k2 = f(x+h*k1/2, delta, v, pars, geometry)
            k3 = f(x+h*k2/2, delta, v, pars, geometry)
            k4 = f(x+h*k3, delta, v, pars, geometry)
            x = x+h*(k1+2*k2+2*k3+k4)/6
        out[:, k+1] = x
    if not np.isfinite(out).all():
        raise FloatingPointError('Nonfinite model response')
    return out.reshape(npar, nrec, u.shape[1]+1, dim)


def observed_records(records):
    if not records:
        raise ValueError('No training records')
    for s in records:
        if set(s) != {'speed', 'applied_input', 'measured_yaw'}:
            raise ValueError('Estimator accepts yaw, known steering and speed only')
        u, y = np.asarray(s['applied_input']), np.asarray(s['measured_yaw'])
        if u.ndim != 1 or y.shape != (len(u)+1,) or len(u) < 10:
            raise ValueError('Expected N inputs and N+1 yaw measurements')
        if not np.isfinite(np.r_[u, y, s['speed']]).all():
            raise ValueError('Nonfinite observations')
    return (np.asarray([s['applied_input'] for s in records]),
            np.asarray([s['measured_yaw'] for s in records]),
            np.asarray([s['speed'] for s in records]))


def response_jacobian(q, base, free, u, speeds, geometry, settings):
    p = np.array(base, float); p[free] = np.exp(q)
    trials = np.tile(p, (len(free)+1, 1)).astype(complex)
    epsilon = 1e-25
    for j, index in enumerate(free):
        trials[j+1, index] *= np.exp(1j*epsilon)
    x = batch_predict(trials, u, speeds, geometry, settings['sample_time'],
                      settings['integration_max_step'])
    return x[0, :, :, 1].real, (x[1:, :, :, 1].imag/epsilon).reshape(len(free), -1).T


def fit_attempt(records, geometry, settings, start, free):
    """One bounded log-parameter least-squares start; no test data or plant truth."""
    u, y, speeds = observed_records(records)
    start = np.asarray(start, float); free = np.asarray(free, int)
    bounds = np.asarray(settings['bounds'])[:len(start)]
    if np.any(start < bounds[:, 0]) or np.any(start > bounds[:, 1]):
        raise ValueError('Start outside declared bounds')
    noise = np.deg2rad(settings['yaw_noise_std_deg_s']); cache = {}
    def evaluate(q):
        if 'q' not in cache or not np.array_equal(q, cache['q']):
            pred, jac = response_jacobian(q, start, free, u, speeds, geometry, settings)
            cache.update(q=q.copy(), r=(pred-y).ravel()/noise, j=jac/noise)
        return cache
    try:
        res = least_squares(lambda q: evaluate(q)['r'], np.log(start[free]),
            jac=lambda q: evaluate(q)['j'], bounds=np.log(bounds[free].T),
            x_scale='jac', max_nfev=settings['max_nfev'], ftol=settings['ftol'],
            xtol=settings['xtol'], gtol=settings['gtol'])
        p = start.copy(); p[free] = np.exp(res.x)
        _, singular, vh = np.linalg.svd(res.jac, full_matrices=False)
        info = svd_summary(res.jac, 1e-8)
        info['weakest_log_direction'] = vh[-1].tolist()
        if len(start) == 6 and list(free) == list(range(6)):
            projected, rank = project_nuisance(res.jac[:, 4:6], res.jac[:, :4])
            info['relaxation_after_ct_projection'] = dict(**svd_summary(projected,
                reference_scale=float(np.linalg.norm(res.jac[:, 4:6], 2))), nuisance_rank=rank)
        return dict(converged=bool(res.success), status=int(res.status), message=str(res.message),
            parameters=p.tolist(), free_indices=free.tolist(), weighted_sse=float(res.fun@res.fun),
            weighted_mse=float(np.mean(res.fun**2)), nfev=int(res.nfev), njev=int(res.njev),
            optimality=float(res.optimality), active_mask=res.active_mask.tolist(),
            near_bound=(np.minimum(res.x-np.log(bounds[free, 0]),
                np.log(bounds[free, 1])-res.x) < 1e-3).tolist(), information=info)
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
        return dict(converged=False, message=f'{type(exc).__name__}: {exc}')


def select_attempt(attempts):
    candidates = [(a['weighted_sse'], i) for i, a in enumerate(attempts) if a['converged']]
    if not candidates:
        return dict(converged=False, reason='No converged start; no fallback')
    _, index = min(candidates)
    return dict(attempts[index], selected_start=index)


def design(parameters, speed, geometry, cfg, samples):
    """Same beta/yaw/integral LQI cost; force states have zero direct penalty.

    Observer Q is a fixed engineering allowance, not injected process noise.
    Both dimensions start with zero prior covariance and known zero state.
    """
    p = np.asarray(parameters); dim = 4 if len(p) == 6 else 2; dt = cfg['sample_time']
    ac, bc = (continuous_matrices(p, speed, geometry) if dim == 4 else
              instantaneous_matrices(p, speed, geometry, 0.))
    block = np.zeros((dim+1, dim+1)); block[:dim, :dim] = ac; block[:dim, -1] = bc
    sampled = expm(dt*block); a = sampled[:dim, :dim]; b = sampled[:dim, -1:]
    c = np.zeros((1, dim)); c[0, 1] = 1.
    aa = np.block([[a, np.zeros((dim, 1))], [dt*c, np.ones((1, 1))]])
    bb = np.vstack([b, [[0.]]]); weights = np.zeros(dim+1)
    weights[:2] = cfg['state_cost']; weights[-1] = cfg['integral_cost']; r = cfg['input_cost']
    ric = solve_discrete_are(aa, bb, np.diag(weights), np.array([[r]]))
    gain = np.linalg.solve(r+bb.T@ric@bb, bb.T@ric@aa).ravel()
    eq = np.linalg.solve(np.block([[np.eye(dim)-a, -b], [c, np.zeros((1, 1))]]),
                         np.r_[np.zeros(dim), 1.])
    prefilter = float(eq[-1]+gain[:dim]@eq[:dim])
    qsd = np.r_[np.deg2rad(cfg['observer_state_step_std_deg']),
                cfg['observer_force_step_std_mps2'] if dim == 4 else []]
    q = np.diag(qsd**2); rv = np.deg2rad(cfg['yaw_noise_std_deg_s'])**2
    prior = solve_discrete_are(a.T, c.T, q, np.array([[rv]]))
    steady = prior[:, 1]/(prior[1, 1]+rv)
    rho_control = float(max(abs(np.linalg.eigvals(aa-bb@gain[None, :]))))
    rho_observer = float(max(abs(np.linalg.eigvals(a@(np.eye(dim)-steady[:, None]@c)))))
    if max(rho_control, rho_observer) >= 1:
        raise ValueError('Nonstabilizing model-based controller or observer')
    covariance = np.zeros_like(a); gains = np.zeros((samples, dim))
    for k in range(samples):
        gains[k] = covariance[:, 1]/(covariance[1, 1]+rv)
        j = np.eye(dim)-gains[k, :, None]@c
        post = j@covariance@j.T+rv*np.outer(gains[k], gains[k])
        covariance = a@post@a.T+q; covariance = (covariance+covariance.T)/2
    return dict(a=a, b=b.ravel(), gain=gain, prefilter=prefilter, observer_gains=gains,
        diagnostics=dict(control_rho=rho_control, observer_rho=rho_observer,
            gain=gain.tolist(), prefilter=prefilter, a=a.tolist(), b=b.ravel().tolist(),
            observer_q=q.tolist(), observer_r=float(rv), steady_observer_gain=steady.tolist()))



def closed_loop_matrix(plant_parameters, speed, geometry, d, cfg):
    """Offline local diagnostic for [plant x, observer prior, yaw integral].

    Uses true plant matrices only for evaluation, steady observer gain and
    inactive limits. This is not a nonlinear stability certificate.
    """
    ac, bc = continuous_matrices(plant_parameters, speed, geometry)
    block = np.zeros((5, 5)); block[:4, :4] = ac; block[:4, 4] = bc
    z = expm(cfg['sample_time']*block); ap, bp = z[:4, :4], z[:4, 4:]
    a, b = d['a'], d['b'][:, None]; n = len(b)
    cp = np.array([[0., 1., 0., 0.]])
    cm = np.zeros((1, n)); cm[0, 1] = 1.
    gain = d['gain']; ks = gain[:-1][None, :]; ki = gain[-1]
    l = np.asarray(d['diagnostics']['steady_observer_gain'])[:, None]
    j = np.eye(n)-l@cm
    return np.block([[ap-bp@ks@l@cp, -bp@ks@j, -bp*ki],
        [(a-b@ks)@l@cp, (a-b@ks)@j, -b*ki],
        [cfg['sample_time']*cp, np.zeros((1, n)), np.ones((1, 1))]])

def feedback(prior, integral, previous, yaw, reference, design_record, k, cfg):
    """Only measured yaw and controller memory enter the online feedback law."""
    d = design_record; gain = d['gain']; dt = cfg['sample_time']
    posterior = prior+d['observer_gains'][k]*(yaw-prior[1])
    requested = float(-gain[:-1]@posterior-gain[-1]*integral+d['prefilter']*reference)
    step = np.deg2rad(cfg['steering_rate_limit_deg_s'])*dt
    rate_limited = np.clip(requested, previous-step, previous+step)
    applied = float(np.clip(rate_limited, -np.deg2rad(cfg['steering_limit_deg']),
                            np.deg2rad(cfg['steering_limit_deg'])))
    limited = abs(applied-requested) > 1e-12
    candidate = np.clip(integral+dt*(yaw-reference), -cfg['integral_limit'], cfg['integral_limit'])
    integral = candidate if not limited or (-gain[-1]*(candidate-integral))*(requested-applied) <= 0 else integral
    return posterior, d['a']@posterior+d['b']*applied, float(integral), applied, requested


def control_rollout(plant, geometry, speed, reference, noise, d, cfg):
    """All methods control the same four-state nonlinear plant, using yaw only."""
    n = len(reference)-1; dt = cfg['sample_time']; dim = len(d['b'])
    state = np.zeros((n+1, 4)); estimate = np.zeros((n+1, dim))
    steering = np.zeros(n); requested = np.zeros(n); integral = np.zeros(n+1)
    prior = np.zeros(dim); done = 0; reason = None
    sub = int(np.ceil(dt/cfg['integration_max_step'])); h = dt/sub
    for k in range(n):
        try:
            with np.errstate(over='raise', invalid='raise', divide='raise'):
                estimate[k], prior, integral[k+1], steering[k], requested[k] = feedback(
                    prior, integral[k], steering[k-1] if k else 0., state[k, 1]+noise[k],
                    reference[k], d, k, cfg)
                x = state[k].copy(); u = steering[k]
                for _ in range(sub):
                    k1 = rhs(x, u, speed, plant, geometry); k2 = rhs(x+h*k1/2, u, speed, plant, geometry)
                    k3 = rhs(x+h*k2/2, u, speed, plant, geometry); k4 = rhs(x+h*k3, u, speed, plant, geometry)
                    x = x+h*(k1+2*k2+2*k3+k4)/6
                if not np.isfinite(np.r_[x, prior]).all() or max(abs(np.r_[x, prior])) > 1e6:
                    raise FloatingPointError('Nonfinite or excessive state')
                state[k+1] = x; done = k+1
        except (FloatingPointError, ValueError, OverflowError) as exc:
            reason = str(exc); break
    estimate[done] = prior+d['observer_gains'][done]*(state[done, 1]+noise[done]-prior[1])
    arrays = dict(time=np.arange(done+1)*dt, state=state[:done+1], estimate=estimate[:done+1],
        reference=reference[:done+1], noise=noise[:done+1], steering=steering[:done],
        requested=requested[:done], integral=integral[:done+1])
    if reason:
        return dict(status='rollout_failed', reason=reason, metrics=None, feasible=False), arrays
    rms = lambda x: float(np.sqrt(np.mean(x*x)))
    deg = lambda x: float(np.rad2deg(x))
    ay = state[:, 2]+state[:, 3]; rate = np.diff(np.r_[0., steering])/dt
    metrics = dict(tracking_rmse_deg_s=deg(rms(state[:, 1]-reference)),
        tracking_peak_deg_s=deg(max(abs(state[:, 1]-reference))),
        beta_estimation_rmse_deg=deg(rms(estimate[:, 0]-state[:, 0])),
        yaw_estimation_rmse_deg_s=deg(rms(estimate[:, 1]-state[:, 1])),
        beta_peak_deg=deg(max(abs(state[:, 0]))), ay_peak_mps2=float(max(abs(ay))),
        steering_rms_deg=deg(rms(steering)), steering_peak_deg=deg(max(abs(steering))),
        steering_rate_peak_deg_s=deg(max(abs(rate))),
        limiter_fraction=float(np.mean(abs(steering-requested) > 1e-12)))
    feasible = metrics['beta_peak_deg'] <= cfg['beta_limit_deg'] and metrics['ay_peak_mps2'] <= cfg['ay_limit_mps2']
    return dict(status='complete', metrics=metrics, feasible=bool(feasible)), arrays
