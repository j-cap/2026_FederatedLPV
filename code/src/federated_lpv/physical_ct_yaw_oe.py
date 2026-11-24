"""Yaw-only physical CT OE; no sideslip observations enter this module.

The latent initial sideslip is fixed to zero as an affine-model gauge, not as
knowledge of the physical initial state. Free two-component offsets absorb any
constant hidden-state shift. This spans exactly the yaw responses of the old
four-nuisance affine model, using three (generically independent) columns.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import least_squares
from .physical_ct_oe import discrete_matrices
from .structured_oe import propagate


def validated_segments(segments):
    out = []
    for segment in segments:
        if set(segment) != {'speed', 'measured_yaw', 'applied_input'}:
            raise ValueError('Yaw estimator accepts only speed, measured_yaw and applied_input')
        y = np.asarray(segment['measured_yaw'], dtype=float)
        u = np.asarray(segment['applied_input'], dtype=float)
        v = float(segment['speed'])
        if u.ndim != 1 or y.shape != (len(u)+1,) or len(u) < 10:
            raise ValueError('Expected yaw (N+1,) and known applied input (N,)')
        if not np.isfinite(np.r_[y, u, v]).all() or v <= 0:
            raise ValueError('Nonfinite data or invalid speed')
        out.append(dict(speed=v, measured_yaw=y, applied_input=u))
    if not out or len({s['speed'] for s in out}) != len(out):
        raise ValueError('One segment per distinct speed required')
    return out


def projected_response(log_parameters, segment, settings, jacobian=True):
    """Exact variable projection on yaw observations with a hidden-state gauge.

    With x'=x-[d,0], initial beta can be set to zero and the affine offset
    becomes c' = c+(A-I)[d,0]. Yaw is unchanged for every sample. Therefore this
    removes a redundant nuisance direction without adding physical information.
    SVD handles rank loss (e.g. a dynamically unobservable hidden state).
    The projection derivative applies on each constant-rank region.
    """
    y, u = segment['measured_yaw'], segment['applied_input']
    scale = np.deg2rad(settings['state_noise_deg'])  # fixed coordinate scales only
    shift = np.array([0., y.mean()]); ushift = u.mean()
    ys = (y-shift[1])/scale[1]; us = u-ushift
    values = discrete_matrices(np.exp(log_parameters), segment['speed'], settings['geometry_m'],
        settings['curvature_per_m'], settings['sample_time'], jacobian)
    a, b = values[:2]
    az = a*scale[None, :]/scale[:, None]; bz = b/scale
    forcing = np.zeros((len(u), 2, 4))
    forcing[:, :, 0] = us[:, None]*bz
    forcing[:, :, 1:3] = np.eye(2)
    initial = np.zeros((2, 4)); initial[1, 3] = 1.
    basis = propagate(az, forcing, initial)
    base = basis[:, 1, 0]; h = basis[:, 1, 1:]
    hu, singular, vh = np.linalg.svd(h, full_matrices=False)
    keep = singular > settings.get('nuisance_rcond', 1e-12)*singular[0]
    rank = int(keep.sum()); ur = hu[:, keep]
    pinv = (vh[keep].T/singular[keep])@ur.T
    nuisance = pinv@(ys-base)
    residual = base+h@nuisance-ys
    record = dict(speed=float(segment['speed']), target=np.r_[a.ravel(), b].tolist(),
        offset=(scale*nuisance[:2]+shift-a@shift-b*ushift).tolist(),
        initial_state=[0., float(scale[1]*nuisance[2]+shift[1])],
        initial_beta_is_gauge=True, nuisance_rank=rank,
        nuisance_singular_values=singular.tolist(),
        nuisance_condition=float(singular[0]/singular[keep][-1]),
        weighted_mse=float(np.mean(residual**2)))
    if not jacobian:
        return residual, record
    da, db = values[2:]
    dz = da*scale[None, None, :]/scale[None, :, None]
    dbz = db/scale
    forcing_d = np.einsum('pij,njk->nikp', dz, basis[:-1])
    forcing_d[:, :, 0, :] += us[:, None, None]*dbz.T[None, :, :]
    basis_d = propagate(az, forcing_d, np.zeros((2, 4, 4)))
    dbase = basis_d[:, 1, 0, :]; dh = basis_d[:, 1, 1:, :]
    fixed = dbase+np.einsum('nkp,k->np', dh, nuisance)
    jac = fixed-ur@(ur.T@fixed)-pinv.T@np.einsum('nkp,n->kp', dh, residual)
    return residual, jac, record


def validation_prediction(parameters, segment, settings, warmup_transitions):
    """Freeze dynamics; estimate only nuisance on a disjoint prefix then free-run.

    No measurement after the prefix enters predictions or centering. Both the
    full-state-trained and yaw-trained dynamics use this identical yaw-only rule.
    """
    s = validated_segments([segment])[0]
    n = int(warmup_transitions)
    if n < 10 or n >= len(s['applied_input']):
        raise ValueError('Warmup must leave a nonempty held-out suffix')
    prefix = dict(speed=s['speed'], measured_yaw=s['measured_yaw'][:n+1],
                  applied_input=s['applied_input'][:n])
    _, nuisance = projected_response(np.log(parameters), prefix, settings, False)
    target = np.array(nuisance['target']); a = target[:4].reshape(2,2); b = target[4:]
    pred = propagate(a, s['applied_input'][:,None]*b+np.array(nuisance['offset']),
                     np.array(nuisance['initial_state']))[:,1]
    return pred, nuisance


def fit_yaw_oe(segments, settings):
    """Same four parameters, starts, bounds and optimizer as full-state OE; yaw loss only."""
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
        scalar_observations=sum(s['measured_yaw'].size for s in segments),
        free_dynamic_parameters=4, free_nuisance_parameters=sum(t['nuisance_rank'] for t in chosen['trajectories']),
        nuisance_columns_per_segment=3, hidden_initial_beta_is_gauge=True)
    return out
