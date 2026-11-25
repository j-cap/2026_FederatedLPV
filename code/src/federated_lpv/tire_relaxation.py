"""Four-state tanh bicycle with axle-force relaxation, at fixed positive speed.

State: [beta, r, q_f, q_r], q_j=F_yj/m (m/s²).
Parameters: [C_f/m, C_r/m, m/I_z, mu, sigma_f, sigma_r].
This module does not identify absolute mass, inertia or tire force scale.
"""
from __future__ import annotations
import numpy as np
from scipy.linalg import expm


def validate(parameters, speed, geometry):
    p = np.atleast_2d(np.asarray(parameters))
    if p.shape[1] != 6 or not np.isfinite(p).all() or np.any(p.real <= 0):
        raise ValueError('Six finite positive physical ratios/relaxation parameters required')
    if not np.isfinite(speed) or speed <= 0 or len(geometry) != 2 or not np.isfinite(geometry).all() or min(geometry) <= 0:
        raise ValueError('Positive speed and two positive geometry lengths required')
    return p


def rhs(state, steering, speed, parameters, geometry):
    p1,p2,p3,mu,sf,sr = np.asarray(parameters).T
    beta,r,qf,qr = np.asarray(state).T
    lf,lr = geometry
    capf=mu*9.81*lr/(lf+lr);capr=mu*9.81*lf/(lf+lr)
    targetf=capf*np.tanh(p1*(steering-beta-lf*r/speed)/capf)
    targetr=capr*np.tanh(p2*(-beta+lr*r/speed)/capr)
    return np.stack([(qf+qr)/speed-r,p3*(lf*qf-lr*qr),
        speed/sf*(targetf-qf),speed/sr*(targetr-qr)],axis=-1)


def continuous_matrices(parameters, speed, geometry):
    p=validate(parameters,speed,geometry)[0]
    p1,p2,p3,_,sf,sr=p;lf,lr=geometry;af=speed/sf;ar=speed/sr
    a=np.array([[0,-1,1/speed,1/speed],[0,0,p3*lf,-p3*lr],
        [-af*p1,-af*p1*lf/speed,-af,0],[-ar*p2,ar*p2*lr/speed,0,-ar]])
    b=np.array([0,0,af*p1,0],dtype=a.dtype)
    return a,b


def sampled_matrices(parameters,speed,geometry,dt):
    a,b=continuous_matrices(parameters,speed,geometry)
    block=np.zeros((5,5),dtype=a.dtype);block[:4,:4]=a;block[:4,4]=b
    z=expm(block*dt)
    return z[:4,:4],z[:4,4]


def simulate(parameters, applied_input, speed, geometry, dt=.01, max_step=.001, initial=None):
    """Vectorized nonlinear RK4, ZOH applied steering, fixed internal step.

    Complex parameters/initial states are supported for analytic sensitivities.
    The complete prediction includes the initial output sample. No hidden state
    measurements or future observations are accepted by the model.
    """
    p=validate(parameters,speed,geometry);u=np.asarray(applied_input,float)
    if u.ndim != 1 or not np.isfinite(u).all() or not np.isfinite([dt,max_step]).all() or min(dt,max_step)<=0:
        raise ValueError('Finite 1D applied input and positive time steps required')
    dtype=np.result_type(p,initial if initial is not None else 0.,float)
    x=np.zeros((len(p),4),dtype=dtype) if initial is None else np.broadcast_to(np.asarray(initial,dtype=dtype),(len(p),4)).copy()
    if not np.isfinite(x).all():raise ValueError('Invalid initial state')
    sub=int(np.ceil(dt/max_step));h=dt/sub
    if h > float(np.min(p.real[:,4:]))/speed/2:
        raise ValueError('Internal integration step exceeds half the shortest relaxation time')
    out=np.empty((len(p),len(u)+1,4),dtype=dtype);out[:,0]=x
    for k,delta in enumerate(u):
        for _ in range(sub):
            k1=rhs(x,delta,speed,p,geometry);k2=rhs(x+h*k1/2,delta,speed,p,geometry)
            k3=rhs(x+h*k2/2,delta,speed,p,geometry);k4=rhs(x+h*k3,delta,speed,p,geometry)
            x=x+h*(k1+2*k2+2*k3+k4)/6
        out[:,k+1]=x
    if not np.isfinite(out).all():raise FloatingPointError('Nonfinite relaxed trajectory')
    return out


def trajectory_sensitivities(parameters,u,speed,geometry,dt,max_step,state_scales,noise_sd):
    """Whitened yaw sensitivities: six log parameters, four scaled initial states."""
    p=np.asarray(parameters,float);epsilon=1e-25
    candidates=np.tile(p,(11,1)).astype(complex);initial=np.zeros((11,4),complex)
    for j in range(6):candidates[1+j,j]*=np.exp(1j*epsilon)
    for j in range(4):initial[7+j,j]=1j*epsilon*state_scales[j]
    prediction=simulate(candidates,u,speed,geometry,dt,max_step,initial)
    return prediction[0].real,(prediction[1:,:,1].imag/epsilon/noise_sd).T


def svd_summary(matrix,rcond=1e-9,reference_scale=0.):
    s=np.linalg.svd(matrix,compute_uv=False);n=matrix.shape[1]
    threshold=rcond*max(float(s[0]) if len(s) else 0.,reference_scale)
    rank=int(np.sum(s>threshold)) if len(s) else 0
    return dict(rank_threshold=float(threshold),columns=n,rank=rank,singular_values=s.tolist(),
        condition=float(s[0]/s[-1]) if rank==n and s[-1]>0 else None,
        smallest_singular=float(s[-1]) if len(s)==n else 0.)


def project_nuisance(target,nuisance,rcond=1e-9):
    if nuisance.shape[1]==0:return target.copy(),0
    u,s,_=np.linalg.svd(nuisance,full_matrices=False)
    rank=int(np.sum(s>rcond*s[0])) if len(s) and s[0]>0 else 0
    basis=u[:,:rank]
    return target-basis@(basis.T@target),rank


def distinguishability(jac,rcond=1e-9):
    """Local information after allowing specified nuisance variations, no prior."""
    records={}
    reference=float(np.linalg.norm(jac[:,4:6],ord=2))
    for name,indices in [('conditional',[]),('unknown_initial',list(range(6,10))),
        ('unknown_ct',list(range(4))),('unknown_ct_and_initial',list(range(4))+list(range(6,10)))]:
        s,nrank=project_nuisance(jac[:,4:6],jac[:,indices],rcond)
        norms=np.linalg.norm(s,axis=0)
        cosine=float(np.dot(s[:,0],s[:,1])/np.prod(norms)) if min(norms)>0 else None
        records[name]=dict(**svd_summary(s,rcond,reference),nuisance_rank=nrank,
            column_cosine=cosine,weak_direction_delta_sse_for_log_step_0p1=float((.1*np.linalg.svd(s,compute_uv=False)[-1])**2))
    return records


def observability(parameters,speed,geometry,dt,horizon,state_scales,noise_sd,rcond=1e-9):
    ad,_=sampled_matrices(parameters,speed,geometry,dt)
    c=np.array([0,1,0,0])/noise_sd;power=np.eye(4);rows=[]
    for _ in range(round(horizon/dt)+1):
        rows.append((c@power)*state_scales);power=power@ad
    return svd_summary(np.asarray(rows),rcond)
