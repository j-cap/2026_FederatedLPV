"""Fixed-coordinate, two-state affine output-error identification.

The estimator receives measured states and known inputs only. Simulator states,
parameters, validation data and reference matrices are deliberately not arguments.
"""
from __future__ import annotations
import warnings
import numpy as np
from scipy.optimize import least_squares
from scipy.signal import lfilter


def propagate(a, forcing, initial):
    """Exact two-state recursion, using Cayley-Hamilton and compiled filtering.

    forcing has shape (N,2,...) and initial (2,...). Returns N+1 states.
    """
    a=np.asarray(a); forcing=np.asarray(forcing); initial=np.asarray(initial)
    tr=np.trace(a); det=np.linalg.det(a)
    q=np.empty((len(forcing)+1,)+initial.shape)
    q[0]=initial
    if len(forcing):
        q[1]=np.einsum('ij,j...->i...',a-tr*np.eye(2),initial)+forcing[0]
    if len(forcing)>1:
        q[2:]=forcing[1:]+np.einsum('ij,nj...->ni...',a-tr*np.eye(2),forcing[:-1])
    return lfilter([1.], [1.,-tr,det],q,axis=0)


def response_and_jacobian(p,u):
    a=p[:4].reshape(2,2); b=p[4:6]; c=p[6:8]; x0=p[8:10]
    x=propagate(a,u[:,None]*b+c,x0)
    f=np.zeros((len(u),2,10))
    f[:,0,0:2]=x[:-1]; f[:,1,2:4]=x[:-1]
    f[:,0,4]=u; f[:,1,5]=u
    f[:,0,6]=1.; f[:,1,7]=1.
    initial=np.zeros((2,10)); initial[:,8:10]=np.eye(2)
    jac=propagate(a,f,initial)
    return x,jac


def linear_start(a,y,u):
    """For a fixed A, find B, offset and initial state from measured data."""
    p=np.r_[a.ravel(),np.zeros(6)]
    _,jac=response_and_jacobian(p,u)
    theta=np.linalg.lstsq(jac[:,:,4:].reshape(-1,6),y.ravel(),rcond=None)[0]
    return np.r_[a.ravel(),theta]


def affine_ols(y,u):
    coeff=np.linalg.lstsq(np.column_stack([y[:-1],u,np.ones(len(u))]),y[1:],rcond=None)[0]
    return np.r_[coeff[:2].T.ravel(),coeff[2],coeff[3],y[0]]


def fit_oe(measured_state,applied_input,settings):
    y=np.asarray(measured_state,float); u=np.asarray(applied_input,float)
    if y.shape!=(len(u)+1,2) or len(u)<10 or not np.isfinite(y).all() or not np.isfinite(u).all():
        raise ValueError('Expected finite consecutive states (N+1,2) and input (N,)')
    # Whiten in physical measurement units. Centering is numerical only: a free
    # affine offset is fitted, so the empirical mean is not an equilibrium prior.
    scales=np.deg2rad(settings['state_noise_deg']); uscale=np.deg2rad(settings['input_scale_deg'])
    shift=y.mean(axis=0); ushift=u.mean()
    ys=(y-shift)/scales; us=(u-ushift)/uscale
    ols=affine_ols(ys,us); a0=ols[:4].reshape(2,2)
    radius=max(abs(np.linalg.eigvals(a0)))
    if radius>settings['initial_radius_max']: a0=a0*(settings['initial_radius_max']/radius)
    starts=[('affine_ols',a0)]+[(f'diagonal_{v}',np.eye(2)*v) for v in settings['diagonal_starts']]
    attempts=[]; candidates=[]; converged_parameters={}
    for name,a in starts:
        p0=linear_start(a,ys,us)
        cache={}
        def evaluate(p):
            if 'p' not in cache or not np.array_equal(cache['p'],p):
                with np.errstate(over='ignore',invalid='ignore'), warnings.catch_warnings():
                    warnings.simplefilter('ignore',RuntimeWarning)
                    x,j=response_and_jacobian(p,us)
                if not np.isfinite(x).all() or not np.isfinite(j).all() or np.max(np.abs(x))>1e100:
                    # Numerical barrier only, not a stability constraint. Such
                    # trial steps cannot be selected as successful estimates.
                    r=np.full(ys.size,1e100*(1+np.linalg.norm(p)))
                    jac=np.tile(1e100*p/max(np.linalg.norm(p),1e-30),(ys.size,1))
                else:
                    r=(x-ys).ravel(); jac=j.reshape(-1,10)
                cache.update(p=p.copy(),r=r,j=jac)
            return cache['r'],cache['j']
        try:
            result=least_squares(lambda p:evaluate(p)[0],p0,jac=lambda p:evaluate(p)[1],
                method='trf',x_scale='jac',max_nfev=settings['max_nfev'],
                ftol=settings['ftol'],xtol=settings['xtol'],gtol=settings['gtol'])
            mse=float(np.mean(result.fun**2))
            valid=bool(result.success and np.isfinite(mse) and mse<1e90 and np.isfinite(result.x).all())
            attempts.append(dict(start=name,converged=valid,status=int(result.status),
                message=str(result.message),nfev=int(result.nfev),weighted_mse=mse if np.isfinite(mse) else None,
                optimality=float(result.optimality) if np.isfinite(result.optimality) else None))
            if valid:
                candidates.append((mse,result.x,result.jac,name))
                converged_parameters[name]=result.x.copy()
        except (ValueError,FloatingPointError,np.linalg.LinAlgError) as exc:
            attempts.append(dict(start=name,converged=False,message=str(exc)))
    def physical(p):
        a=p[:4].reshape(2,2)*scales[:,None]/scales[None,:]
        b=p[4:6]*scales/uscale
        c=p[6:8]*scales+shift-a@shift-b*ushift
        initial=p[8:10]*scales+shift
        return dict(target=np.r_[a.ravel(),b].tolist(),offset=c.tolist(),initial_state=initial.tolist(),
                    spectral_radius=float(max(abs(np.linalg.eigvals(a)))))
    out=dict(attempts=attempts,converged=bool(candidates),affine_ols=physical(ols),
             state_noise_deg=settings['state_noise_deg'])
    for attempt in attempts:
        if attempt['start'] in converged_parameters:
            attempt.update(physical(converged_parameters[attempt['start']]))
    if candidates:
        mse,p,jac,name=min(candidates,key=lambda z:z[0])
        singular=np.linalg.svd(jac,compute_uv=False)
        out.update(physical(p),weighted_mse=mse,selected_start=name,
                   jacobian_rank=int(np.linalg.matrix_rank(jac)),
                   jacobian_condition=float(singular[0]/singular[-1]) if singular[-1]>0 else None)
    return out
