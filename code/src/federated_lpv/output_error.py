"""Positive-ratio output-error identification with varying-speed kinematics."""
import numpy as np
from scipy.linalg import solve_banded
from scipy.optimize import least_squares


def transitions(parameters, v0, v1, dt=0.01):
    """RK4 affine maps in [vy,r], with speed linear inside each sample."""
    f,r,h=parameters; dtype=np.result_type(parameters,float)
    def matrix(v):
        a=np.empty((len(v),2,2),dtype=dtype)
        a[:,0,0]=-(f+r)/v; a[:,0,1]=(1.6*r-1.2*f)/v-v
        a[:,1,0]=h*(1.6*r-1.2*f)/v; a[:,1,1]=-h*(1.2**2*f+1.6**2*r)/v
        return a
    a0=matrix(v0);am=matrix((v0+v1)/2);a1=matrix(v1)
    b=np.broadcast_to(np.array([f,h*f*1.2],dtype=dtype),(len(v0),2))
    eye=np.eye(2)[None]
    k1=a0;g1=b
    k2=am@(eye+dt*k1/2);g2=np.einsum('nij,nj->ni',am,dt*g1/2)+b
    k3=am@(eye+dt*k2/2);g3=np.einsum('nij,nj->ni',am,dt*g2/2)+b
    k4=a1@(eye+dt*k3);g4=np.einsum('nij,nj->ni',a1,dt*g3)+b
    return eye+dt*(k1+2*k2+2*k3+k4)/6,dt*(g1+2*g2+2*g3+g4)/6


def prepare(records):
    """Records start at known zero state; measurements are beta proxy and r."""
    lengths=[len(r['input']) for r in records]
    return dict(v0=np.concatenate([r['speed'][:-1] for r in records]),
        v1=np.concatenate([r['speed'][1:] for r in records]),
        u=np.concatenate([r['input'] for r in records]),
        y=np.concatenate([r['state'][1:] for r in records]),
        starts=np.r_[0,np.cumsum(lengths)[:-1]])


def predict(parameters,data,dt=0.01):
    """Solve the block recurrence in compiled banded linear algebra."""
    f,g=transitions(parameters,data['v0'],data['v1'],dt)
    f[data['starts']]=0 # independent zero-initial-state records
    n=len(f);bands=np.zeros((4,2*n),dtype=f.dtype);bands[0]=1
    k=np.arange(1,n)
    for a in range(2):
        for b in range(2):
            columns=2*(k-1)+b
            bands[2+a-b,columns]=-f[k,a,b]
    state=solve_banded((3,0),bands,(g*data['u'][:,None]).ravel(),check_finite=False).reshape(n,2)
    state[:,0]/=data['v1']
    return state


def fit(records, prior=None, strength=0.0, prior_scale=0.05):
    """Output error plus a log-ratio prior; strength weights mean data loss.

    A zero strength exactly preserves the unregularized 7A objective.
    Positive strength minimizes mean squared standardized output error plus
    strength * sum(((log(p)-log(prior))/prior_scale)**2).
    """
    if strength < 0 or prior_scale <= 0:
        raise ValueError('strength must be nonnegative and prior_scale positive')
    if strength and (prior is None or np.shape(prior)!=(3,) or np.any(np.asarray(prior)<=0)):
        raise ValueError('positive strength requires three positive prior ratios')
    data=prepare(records)
    scales=np.deg2rad([.05,.1])
    def residual(logp):
        output=((predict(np.exp(logp),data)-data['y'])/scales).ravel()
        if not strength:return output
        penalty=np.sqrt(output.size*strength)*(logp-np.log(prior))/prior_scale
        return np.r_[output,penalty]
    solutions=[least_squares(residual,np.log(p),jac='cs',bounds=(np.log([1,1,.05]),np.log([300,300,3])),
        ftol=1e-9,xtol=1e-9,gtol=1e-9,max_nfev=150) for p in ([40,40,.5],[80,60,1])]
    result=min(solutions,key=lambda s:s.cost);p=np.exp(result.x)
    norms=np.linalg.norm(result.jac,axis=0)
    singular=np.linalg.svd(result.jac/np.maximum(norms,1e-30),compute_uv=False)
    return p,dict(success=bool(result.success),both_starts_success=all(s.success for s in solutions),
        cost=float(result.cost),nfev=result.nfev,condition=float(singular[0]/singular[-1]),
        bound_hit=bool(np.any(result.active_mask)),
        multistart_difference=float(np.max(abs(np.exp(solutions[0].x)-np.exp(solutions[1].x))/p)))
