"""One fixed nominal LTI/LQI controller and a nominal OE initialization.

Plant parameters enter simulation only. Controller synthesis has no recipient or
speed argument; a single model, feedback gain and prefilter are reused everywhere.
"""
from __future__ import annotations
import copy
import warnings
import numpy as np
from scipy.optimize import least_squares
from .vehicle import VehicleParameters,discrete_bicycle_matrices,nonlinear_bicycle_rhs
from .control import design_lqi_gain
from .trajectory_calibration import rk4,window_indices,waveform
from .structured_oe import fit_oe,response_and_jacobian


def design_fixed_controller(settings,dt):
    a,b=discrete_bicycle_matrices(settings['design_speed_mps'],VehicleParameters(**settings['parameters']),dt)
    gain,prefilter=design_lqi_gain(a,b,dt,np.diag(settings['q']),[[settings['r']]])
    return dict(a=a.tolist(),b=b.ravel().tolist(),gain=gain.tolist(),prefilter=float(prefilter),
                design_speed_mps=settings['design_speed_mps'])


def reference_signal(speed,seed,cfg,protocol,validation=False):
    _,ix=window_indices(cfg);t=np.arange(ix[-1]+1)*cfg['sample_time']
    ramp=.5*(1-np.cos(np.pi*np.minimum(1,t/cfg['turn_ramp_s'])))
    ref=speed*cfg['curvature_per_m']*ramp
    # Reuse the prescribed taper/frequencies/phases, normalized to unit peak.
    shape=waveform(cfg,1.,20.,seed,validation)
    shape=shape/max(abs(shape))
    ref[ix]+=protocol['lateral_acceleration_reference_amplitude']/speed*shape
    return ref


def generate_fixed_lqi(parameters,speed,seed,cfg,protocol,controller,validation=False):
    dt=cfg['sample_time'];_,ix=window_indices(cfg);n=int(ix[-1]+1)
    rng=np.random.default_rng(np.random.SeedSequence([seed,int(validation),11]))
    noise=rng.normal(size=(n+1,2))*np.deg2rad(cfg['state_noise_deg'])
    input_noise=rng.normal(size=n)*np.deg2rad(cfg['input_noise_deg'])
    bridge=rng.normal(size=(n,2))*np.deg2rad(cfg['state_noise_deg'])
    ref=reference_signal(speed,seed,cfg,protocol,validation)
    gain=np.array(controller['gain']);prefilter=controller['prefilter']
    x=np.zeros((n+1,2));u=np.zeros(n);raw=np.zeros(n);ay=np.zeros(n);limited=np.zeros(n)
    xi=0.;integral=np.zeros(n+1)
    limit,rate=np.deg2rad([cfg['steering_limit_deg'],cfg['steering_rate_deg_s']])
    for k in range(n):
        measured=x[k]+noise[k]
        raw[k]=-gain[:2]@measured-gain[2]*xi+prefilter*ref[k]
        previous=u[k-1] if k else 0.
        u[k]=np.clip(np.clip(raw[k],previous-rate*dt,previous+rate*dt),-limit,limit)
        limited[k]=abs(u[k]-raw[k])>1e-12
        candidate=np.clip(xi+dt*(measured[1]-ref[k]),-protocol['integral_limit'],protocol['integral_limit'])
        if not limited[k] or (-gain[2]*(candidate-xi))*(raw[k]-u[k])<=0:xi=float(candidate)
        integral[k+1]=xi
        dx=nonlinear_bicycle_rhs(x[k],u[k],speed,parameters)
        ay[k]=speed*(dx[0]+x[k,1])
        x[k+1]=rk4(parameters,x[k],u[k],speed,dt)
        if not np.isfinite(x[k+1]).all():raise FloatingPointError('Nonfinite fixed-LQI trajectory')
    heading=np.r_[0.,np.cumsum(dt*(x[:-1,1]+x[1:,1])/2)]
    lateral=speed*np.tan(x[:,0])
    vel=np.column_stack([speed*np.cos(heading)-lateral*np.sin(heading),speed*np.sin(heading)+lateral*np.cos(heading)])
    xy=np.vstack([np.zeros(2),np.cumsum(dt*(vel[:-1]+vel[1:])/2,axis=0)])
    return dict(time=np.arange(n+1)*dt,state=x,steering=u,observed_state=x+noise,
        observed_steering=u+input_noise,measurement_noise=noise,bridge_noise=bridge,
        ay=ay,limited=limited,position=xy,heading=heading,yaw_reference=ref,
        steering_requested=raw,integral=integral)


def fit_nominal_start(y,u,settings,controller):
    """One additional start of the unchanged affine OE objective and tolerances."""
    scale=np.deg2rad(settings['state_noise_deg']);uscale=np.deg2rad(settings['input_scale_deg'])
    shift=y.mean(axis=0);ushift=u.mean();ys=(y-shift)/scale;us=(u-ushift)/uscale
    a=np.array(controller['a']);b=np.array(controller['b'])
    az=a*scale[None,:]/scale[:,None];bz=b*uscale/scale
    # Same nominal physical A/B at every speed. Empirical mean initializes the
    # offset; the first measured sample initializes x0, both subsequently free.
    p0=np.r_[az.ravel(),bz,np.zeros(2),ys[0]];cache={}
    def evaluate(p):
        if 'p' not in cache or not np.array_equal(cache['p'],p):
            with np.errstate(over='ignore',invalid='ignore'),warnings.catch_warnings():
                warnings.simplefilter('ignore',RuntimeWarning);x,j=response_and_jacobian(p,us)
            if not np.isfinite(x).all() or not np.isfinite(j).all() or np.max(abs(x))>1e100:
                r=np.full(ys.size,1e100*(1+np.linalg.norm(p)))
                j=np.tile(1e100*p/max(np.linalg.norm(p),1e-30),(ys.size,1))
            else:r=(x-ys).ravel();j=j.reshape(-1,10)
            cache.update(p=p.copy(),r=r,j=j)
        return cache['r'],cache['j']
    attempt=dict(start='fixed_nominal_20mps',initial_target=np.r_[a.ravel(),b].tolist())
    try:
        result=least_squares(lambda p:evaluate(p)[0],p0,jac=lambda p:evaluate(p)[1],method='trf',x_scale='jac',
            max_nfev=settings['max_nfev'],ftol=settings['ftol'],xtol=settings['xtol'],gtol=settings['gtol'])
        mse=float(np.mean(result.fun**2))
        valid=bool(result.success and np.isfinite(mse) and mse<1e90 and np.isfinite(result.x).all())
        attempt.update(converged=valid,status=int(result.status),message=str(result.message),nfev=int(result.nfev),
            weighted_mse=mse if np.isfinite(mse) else None,
            optimality=float(result.optimality) if np.isfinite(result.optimality) else None)
        if valid:
            p=result.x;a=p[:4].reshape(2,2)*scale[:,None]/scale[None,:];b=p[4:6]*scale/uscale
            singular=np.linalg.svd(result.jac,compute_uv=False)
            attempt.update(target=np.r_[a.ravel(),b].tolist(),offset=(p[6:8]*scale+shift-a@shift-b*ushift).tolist(),
                initial_state=(p[8:10]*scale+shift).tolist(),spectral_radius=float(max(abs(np.linalg.eigvals(a)))),
                jacobian_rank=int(np.linalg.matrix_rank(result.jac)),
                jacobian_condition=float(singular[0]/singular[-1]) if singular[-1]>0 else None)
    except (ValueError,FloatingPointError,np.linalg.LinAlgError) as exc:attempt.update(converged=False,message=str(exc))
    return attempt


def add_nominal_start(y,u,settings,controller,base=None):
    """Reuse frozen three-start results when supplied; never choose by truth error."""
    base=fit_oe(y,u,settings) if base is None else copy.deepcopy(base)
    extra=fit_nominal_start(y,u,settings,controller)
    attempts=base['attempts']+[extra];good=[a for a in attempts if a['converged']]
    combined=dict(attempts=attempts,converged=bool(good))
    if good:
        best=min(good,key=lambda a:a['weighted_mse'])
        combined.update({k:best[k] for k in ['target','offset','initial_state','spectral_radius','weighted_mse']})
        combined['selected_start']=best['start']
        combined['jacobian_rank']=best.get('jacobian_rank',base.get('jacobian_rank'))
        combined['jacobian_condition']=best.get('jacobian_condition',base.get('jacobian_condition'))
    return base,combined
