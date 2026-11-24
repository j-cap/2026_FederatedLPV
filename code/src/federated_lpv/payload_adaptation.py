"""Scalar loading identification from yaw and applied steering only.

Loading is centered at the original CG. Geometry, tire stiffness and friction
stay fixed; added yaw inertia is delta_mass * payload_radius_of_gyration**2.
No simulator parameters, actual loading amount or future samples enter fitting.
"""
from __future__ import annotations
from dataclasses import replace
import numpy as np
from scipy.optimize import minimize_scalar


def load_vehicle(vehicle, fraction, radius):
    if not np.isfinite([fraction, radius]).all() or fraction < 0 or radius <= 0:
        raise ValueError('Nonnegative loading and positive payload radius required')
    added = fraction*vehicle.mass
    return replace(vehicle, mass=vehicle.mass+added,
                   yaw_inertia=vehicle.yaw_inertia+added*radius**2)


def loaded_parameters(previous, fraction, radius):
    """Accept one fraction or a vector; return (...,4) physical CT ratios."""
    p1, p2, p3, mu = np.asarray(previous, float)
    fraction = np.asarray(fraction, float)
    if not np.isfinite(previous).all() or min(previous) <= 0:
        raise ValueError('Four positive finite previous parameters required')
    if not np.isfinite(fraction).all() or np.any(fraction < 0) or not np.isfinite(radius) or radius <= 0:
        raise ValueError('Invalid loading/radius')
    return np.stack([p1/(1+fraction), p2/(1+fraction),
        p3*(1+fraction)/(1+fraction*p3*radius**2), np.full_like(fraction, mu)], axis=-1)


def yaw_prediction(parameters, applied_input, speed, geometry, dt):
    """Vectorized candidate models, nonlinear tanh bicycle, zero physical x0.

    Express forces per unit mass: absolute vehicle mass is not identifiable
    from these ratios and is not required to simulate the lateral response.
    Steering is held through each RK4 step, as in the independent plant code.
    """
    theta = np.atleast_2d(np.asarray(parameters, float))
    u = np.asarray(applied_input, float)
    if theta.shape[1] != 4 or not np.isfinite(theta).all() or np.any(theta <= 0):
        raise ValueError('Invalid candidate parameters')
    if u.ndim != 1 or not np.isfinite(u).all() or speed <= 0 or dt <= 0 or min(geometry) <= 0:
        raise ValueError('Invalid input, speed, geometry or sampling')
    p1, p2, p3, mu = theta.T; lf, lr = geometry
    front = mu*9.81*lr/(lf+lr); rear = mu*9.81*lf/(lf+lr)
    def rhs(x, delta):
        beta, yaw = x[:, 0], x[:, 1]
        ff = front*np.tanh(p1*(delta-beta-lf*yaw/speed)/front)
        fr = rear*np.tanh(p2*(-beta+lr*yaw/speed)/rear)
        return np.column_stack([(ff+fr)/speed-yaw, p3*(lf*ff-lr*fr)])
    state = np.zeros((len(theta), 2)); output = np.zeros((len(theta), len(u)+1))
    for k, delta in enumerate(u):
        k1 = rhs(state, delta); k2 = rhs(state+dt*k1/2, delta)
        k3 = rhs(state+dt*k2/2, delta); k4 = rhs(state+dt*k3, delta)
        state += dt*(k1+2*k2+2*k3+k4)/6
        output[:, k+1] = state[:, 1]
    return output


def fitting_segment(trace, seconds, dt):
    """Copy only the causal prefix and the channels permitted for identification."""
    n = int(round(seconds/dt))
    if not np.isclose(n*dt, seconds) or len(trace['steering']) < n:
        raise ValueError('Prefix unavailable or not aligned with sampling')
    return dict(measured_yaw=np.asarray(trace['measured_yaw'][:n+1]).copy(),
                applied_input=np.asarray(trace['steering'][:n]).copy(),
                speed=float(trace['speed'][0]))


def fit_payload(previous, segment, settings):
    if set(segment) != {'measured_yaw','applied_input','speed'}:
        raise ValueError('Estimator accepts yaw, applied input and speed only')
    y = np.asarray(segment['measured_yaw'], float); u = np.asarray(segment['applied_input'], float)
    if y.ndim != 1 or u.ndim != 1 or len(y) != len(u)+1 or len(u) < 1 or not np.isfinite(np.r_[y,u]).all():
        raise ValueError('Invalid fitting samples')
    grid = np.linspace(*settings['bounds'], settings['grid_points'])
    scale = np.deg2rad(settings['yaw_noise_std_deg_s'])
    def prediction(fraction):
        theta = loaded_parameters(previous, fraction, settings['payload_radius_m'])
        return yaw_prediction(theta, u, segment['speed'], settings['geometry_m'], settings['sample_time'])
    predicted = prediction(grid)
    losses = np.mean(((predicted-y[None,:])/scale)**2, axis=1)
    if not np.isfinite(losses).all():
        return dict(converged=False, reason='Nonfinite grid loss')
    if np.ptp(losses) <= 1e-12*(1+float(max(losses))):
        return dict(converged=False, reason='No resolved loading information in this prefix',
                    grid=grid.tolist(), grid_loss=losses.tolist())
    candidates = [dict(fraction=float(grid[i]), weighted_mse=float(losses[i]),
                       converged=True, kind='boundary') for i in [0,len(grid)-1]]
    attempts = []
    def objective(value):
        return float(np.mean(((prediction(value)[0]-y)/scale)**2))
    # Refine every sampled local minimum, rather than assuming unimodality.
    for j in range(1, len(grid)-1):
        if losses[j] <= losses[j-1] and losses[j] <= losses[j+1]:
            result = minimize_scalar(objective, bounds=(grid[j-1],grid[j+1]), method='bounded',
                options=dict(xatol=settings['xatol'], maxiter=settings['maxiter']))
            item = dict(fraction=float(result.x), weighted_mse=float(result.fun) if np.isfinite(result.fun) else None,
                        converged=bool(result.success and np.isfinite(result.fun)),
                        evaluations=int(result.nfev), kind='refined', message=str(result.message))
            attempts.append(item)
            if item['converged']:
                candidates.append(item)
    best = min(candidates, key=lambda c:c['weighted_mse'])
    # A failed interior search cannot be disguised as a successful boundary fit.
    if any(not a['converged'] for a in attempts) and best['weighted_mse'] > float(min(losses))+1e-10:
        return dict(converged=False, reason='Interior minimizer failed', attempts=attempts,
                    grid=grid.tolist(), grid_loss=losses.tolist())
    lam = best['fraction']; h = settings['sensitivity_step']
    low, high = max(grid[0],lam-h), min(grid[-1],lam+h)
    derivative = (prediction(high)[0]-prediction(low)[0])/(high-low)
    return dict(converged=True, fraction=lam, weighted_mse=best['weighted_mse'],
        parameters=loaded_parameters(previous, lam, settings['payload_radius_m']).tolist(),
        boundary=bool(min(lam-grid[0],grid[-1]-lam) < 10*settings['xatol']),
        grid=grid.tolist(), grid_loss=losses.tolist(), attempts=attempts,
        yaw_sensitivity_norm=float(np.linalg.norm(derivative/scale)),
        previous_model_weighted_mse=float(losses[0]),
        note='Sensitivity is descriptive, not a calibrated parameter-confidence interval.')


def acquisition_profile(settings):
    dt = settings['sample_time']; time = np.arange(round(settings['duration_s']/dt)+1)*dt
    reference = np.zeros_like(time)
    for start in settings['curve_starts_s']:
        phase = (time-start)/settings['curve_duration_s']
        active = (phase > 0)&(phase < 1)
        reference[active] = (settings['lateral_demand_mps2']/settings['speed_mps'])*np.sin(2*np.pi*phase[active])**3
    return dict(time=time, speed=np.full_like(time,settings['speed_mps']), reference=reference)
