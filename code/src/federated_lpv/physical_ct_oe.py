"""Four-parameter, fixed-curvature continuous-time bicycle OE.

Only measured states, known applied inputs, speeds, geometry and settings enter
fit_physical_oe. Plant parameters and evaluation trajectories are not accepted.
Offsets and initial states are eliminated by weighted linear least squares for
each trial dynamics (variable projection), not fixed at sample means or trim.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm, expm_frechet
from scipy.optimize import least_squares

from .structured_oe import propagate


def continuous_matrices(parameters, speed, geometry, curvature, gravity=9.81):
    """Analytical tanh-tire Jacobian at steady cornering; x=[beta,r].

    p1=Cf/m, p2=Cr/m, p3=m/Iz. Static axle loads and small-angle slip
    definitions imply the same slope factor at both axles. Requires |v²k|<mu*g.
    Complex parameters are supported internally for exact chain derivatives.
    """
    p1, p2, p3, mu = np.asarray(parameters)
    lf, lr = geometry
    if speed <= 0 or min(geometry) <= 0 or gravity <= 0:
        raise ValueError('Positive speed, geometry and gravity required')
    if not np.iscomplexobj(parameters):
        if not np.isfinite(parameters).all() or min(parameters) <= 0:
            raise ValueError('Four finite positive parameters required')
        if abs(curvature) * speed**2 >= mu * gravity:
            raise ValueError('No finite steady-cornering equilibrium at this friction')
    s = 1 - (curvature * speed**2 / (mu * gravity))**2
    k = lr*p2 - lf*p1
    q = lf**2*p1 + lr**2*p2
    a = np.array([[-s*(p1+p2)/speed, -1+s*k/speed**2],
                  [s*p3*k, -s*p3*q/speed]])
    b = s*np.array([p1/speed, lf*p1*p3])
    return a, b


def discrete_matrices(parameters, speed, geometry, curvature, dt, derivatives=False):
    """Exact ZOH and, optionally, derivatives w.r.t. log physical parameters."""
    if dt <= 0:
        raise ValueError('Positive sample time required')
    parameters = np.asarray(parameters, dtype=float)
    a, b = continuous_matrices(parameters, speed, geometry, curvature)
    block = np.zeros((3, 3)); block[:2, :2] = a; block[:2, 2] = b
    z = expm(dt*block)
    if not derivatives:
        return z[:2, :2], z[:2, 2]
    dz = []
    for j in range(4):
        # Complex step differentiates the explicit rational CT expressions only.
        # The matrix exponential derivative uses the Frechet derivative.
        trial = parameters.astype(complex); trial[j] += 1j*1e-25*parameters[j]
        aj, bj = continuous_matrices(trial, speed, geometry, curvature)
        direction = np.zeros((3, 3))
        direction[:2, :2] = aj.imag/1e-25; direction[:2, 2] = bj.imag/1e-25
        dz.append(expm_frechet(dt*block, dt*direction, compute_expm=False))
    dz = np.asarray(dz)
    return z[:2, :2], z[:2, 2], dz[:, :2, :2], dz[:, :2, 2]


def projected_response(log_parameters, segment, settings, jacobian=True):
    """Residual and exact variable-projection Jacobian for one trajectory.

    y/u are centered for numerical conditioning only. Four nuisance quantities
    (two constant discrete offsets and two initial states) remain completely free.
    """
    y, u = segment['measured_state'], segment['applied_input']
    scale = np.deg2rad(settings['state_noise_deg'])
    shift = y.mean(axis=0); ushift = u.mean()
    ys = (y-shift)/scale; us = u-ushift
    parameters = np.exp(log_parameters)
    values = discrete_matrices(parameters, segment['speed'], settings['geometry_m'],
                               settings['curvature_per_m'], settings['sample_time'], jacobian)
    a, b = values[:2]
    az = a*scale[None, :]/scale[:, None]; bz = b/scale
    n = len(u)
    forcing = np.zeros((n, 2, 5))
    forcing[:, :, 0] = us[:, None]*bz
    forcing[:, :, 1:3] = np.eye(2)
    initial = np.zeros((2, 5)); initial[:, 3:5] = np.eye(2)
    basis = propagate(az, forcing, initial)
    base = basis[:, :, 0].reshape(-1)
    h = basis[:, :, 1:].reshape(-1, 4)
    hu, singular, vh = np.linalg.svd(h, full_matrices=False)
    if singular[-1] <= np.finfo(float).eps*max(h.shape)*singular[0]:
        raise np.linalg.LinAlgError('Rank-deficient offset/initial-state design')
    pinv = (vh.T/singular)@hu.T
    nuisance = pinv@(ys.ravel()-base)
    residual = base + h@nuisance - ys.ravel()
    record = dict(speed=float(segment['speed']), target=np.r_[a.ravel(), b].tolist(),
                  offset=(scale*nuisance[:2]+shift-a@shift-b*ushift).tolist(),
                  initial_state=(scale*nuisance[2:]+shift).tolist(),
                  nuisance_condition=float(singular[0]/singular[-1]),
                  weighted_mse=float(np.mean(residual**2)))
    if not jacobian:
        return residual, record
    da, db = values[2:]
    dz = da*scale[None, None, :]/scale[None, :, None]
    dbz = db/scale
    derivative_forcing = np.einsum('pij,njk->nikp', dz, basis[:-1])
    derivative_forcing[:, :, 0, :] += us[:, None, None]*dbz.T[None, :, :]
    derivative_basis = propagate(az, derivative_forcing, np.zeros((2, 5, 4)))
    dbase = derivative_basis[:, :, 0, :].reshape(-1, 4)
    dh = derivative_basis[:, :, 1:, :].reshape(-1, 4, 4)
    fixed_nuisance = dbase + np.einsum('nkp,k->np', dh, nuisance)
    # Full derivative, including the residual-dependent term. Dropping the
    # latter gives only an approximate variable-projection Jacobian.
    jac = fixed_nuisance - hu@(hu.T@fixed_nuisance) - pinv.T@np.einsum('nkp,n->kp', dh, residual)
    return residual, jac, record


def validated_segments(segments):
    out = []
    for segment in segments:
        y = np.asarray(segment['measured_state'], dtype=float)
        u = np.asarray(segment['applied_input'], dtype=float)
        speed = float(segment['speed'])
        if y.shape != (len(u)+1, 2) or u.ndim != 1 or len(u) < 10:
            raise ValueError('Expected consecutive states (N+1,2) and inputs (N,)')
        if not np.isfinite(y).all() or not np.isfinite(u).all() or not np.isfinite(speed):
            raise ValueError('Nonfinite training data')
        out.append(dict(measured_state=y, applied_input=u, speed=speed))
    if not out or len({s['speed'] for s in out}) != len(out):
        raise ValueError('Expected one training segment per distinct speed')
    return out


def fit_physical_oe(segments, settings):
    """Joint four-parameter OE over observed speeds, no validation-based selection."""
    segments = validated_segments(segments)
    lower, upper = np.asarray(settings['bounds'], dtype=float).T
    scales = np.asarray(settings['state_noise_deg'])
    if (lower.shape != (4,) or np.any(lower <= 0) or np.any(upper <= lower)
            or scales.shape != (2,) or np.any(scales <= 0)):
        raise ValueError('Invalid parameter bounds or noise scales')
    if lower[3]*9.81 <= abs(settings['curvature_per_m'])*settings['domain_max_speed_mps']**2:
        raise ValueError('Friction lower bound must permit cornering across the target domain')
    cache = {}

    def evaluate(q):
        if 'q' not in cache or not np.array_equal(q, cache['q']):
            parts = [projected_response(q, s, settings) for s in segments]
            residual = np.concatenate([v[0] for v in parts])
            jac = np.vstack([v[1] for v in parts])
            if not np.isfinite(residual).all() or not np.isfinite(jac).all():
                raise FloatingPointError('Nonfinite projected OE response')
            cache.update(q=q.copy(), residual=residual, jacobian=jac,
                         trajectories=[v[2] for v in parts])
        return cache

    attempts, candidates = [], []
    for index, start in enumerate(settings['starts']):
        if np.any(np.asarray(start) <= lower) or np.any(np.asarray(start) >= upper):
            raise ValueError('Initialization must lie strictly within parameter bounds')
        try:
            result = least_squares(lambda q: evaluate(q)['residual'], np.log(start),
                jac=lambda q: evaluate(q)['jacobian'], bounds=(np.log(lower), np.log(upper)),
                x_scale='jac', max_nfev=settings['max_nfev'], ftol=settings['ftol'],
                xtol=settings['xtol'], gtol=settings['gtol'])
            valid = bool(result.success and np.isfinite(result.fun).all())
            attempt = dict(start=index, converged=valid, status=int(result.status),
                message=str(result.message), nfev=int(result.nfev),
                weighted_mse=float(np.mean(result.fun**2)), parameters=np.exp(result.x).tolist(),
                optimality=float(result.optimality), active_mask=result.active_mask.tolist())
            attempts.append(attempt)
            if valid:
                candidates.append((attempt['weighted_mse'], index, result.x.copy()))
        except (FloatingPointError, ValueError, np.linalg.LinAlgError) as exc:
            attempts.append(dict(start=index, converged=False, reason=f'{type(exc).__name__}: {exc}'))
    out = dict(converged=bool(candidates), attempts=attempts, parameter_names=['Cf/m', 'Cr/m', 'm/Iz', 'mu'])
    if not candidates:
        return out
    mse, index, q = min(candidates, key=lambda item: (item[0], item[1]))
    chosen = evaluate(q); j = chosen['jacobian']
    _, singular, vh = np.linalg.svd(j, full_matrices=False)
    rank = int(np.linalg.matrix_rank(j))
    # A descriptive, local linearized information diagnostic, not calibrated CIs:
    # closed-loop measurement/input dependence and model mismatch remain present.
    columns = np.linalg.norm(j, axis=0)
    information = dict(log_parameter_singular_values=singular.tolist(), rank=rank,
        condition=float(singular[0]/singular[-1]) if rank == 4 else None,
        weakest_log_parameter_direction=vh[-1].tolist(), column_norms=columns.tolist(),
        cosine_similarity=((j.T@j)/np.maximum(columns[:, None]*columns[None, :], 1e-300)).tolist())
    # Physical sensitivity after allowing offsets and initial states to adapt:
    per_speed = []
    for s in segments:
        _, js, _ = projected_response(q, s, settings)
        sv = np.linalg.svd(js, compute_uv=False)
        per_speed.append(dict(speed=s['speed'], singular_values=sv.tolist(),
                              relative_rank=int(np.sum(sv > sv[0]*1e-8))))
    info_bounds = np.minimum(q-np.log(lower), np.log(upper)-q)
    out.update(parameters=np.exp(q).tolist(), selected_start=index, weighted_mse=mse,
        trajectories=chosen['trajectories'], information=information,
        single_speed_information=per_speed, near_bound=(info_bounds < 1e-3).tolist(),
        scalar_observations=sum(s['measured_state'].size for s in segments),
        free_dynamic_parameters=4, free_nuisance_parameters=4*len(segments))
    return out
