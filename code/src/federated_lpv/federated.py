"""Federated optimization of the structured output-error objective."""
import numpy as np
from scipy.optimize import minimize
from .output_error import prepare, predict

SCALES=np.deg2rad([.05,.1])
BOUNDS=np.log([[1,300],[1,300],[.05,3]])


def local_value_gradient(logp,records):
    """Return local loss and gradient; records never enter server aggregation."""
    data=prepare(records)
    def residual(value): return ((predict(np.exp(value),data)-data['y'])/SCALES).ravel()
    r=residual(logp);h=1e-20;j=np.column_stack([np.imag(residual(logp.astype(complex)+1j*h*np.eye(3)[k]))/h for k in range(3)])
    return np.r_[.5*np.dot(r,r),j.T@r]


def secure_sum(messages,round_index,seed=8101):
    """Pairwise-mask emulator: server receives only a mask-cancelled sum.

    This validates numerical equivalence and protocol accounting; it is not a
    cryptographic implementation or a dropout-tolerant secure-aggregation proof.
    """
    masked=np.array(messages,dtype=float,copy=True);rng=np.random.default_rng(seed+round_index)
    for i in range(len(masked)):
        for j in range(i+1,len(masked)):
            mask=rng.normal(size=masked.shape[1]);masked[i]+=mask;masked[j]-=mask
    return masked.sum(axis=0)


def fit_federated(client_records,secure=False,mask_seed=8101):
    """Minimize the sum of client output-error objectives from four-number messages."""
    evaluations=0;messages=0;clients=len(client_records)
    def evaluate(logp):
        nonlocal evaluations,messages
        local=[local_value_gradient(logp,r) for r in client_records]
        total=secure_sum(local,evaluations,mask_seed) if secure else np.sum(local,axis=0)
        evaluations+=1;messages+=clients
        return total[0],total[1:]
    solutions=[]
    for start in ([40,40,.5],[80,60,1]):
        result=minimize(evaluate,np.log(start),method='L-BFGS-B',jac=True,bounds=BOUNDS,
            options={'ftol':1e-12,'gtol':1e-8,'maxiter':150,'maxls':30})
        solutions.append(result)
    result=min(solutions,key=lambda s:s.fun);p=np.exp(result.x)
    return p,dict(success=bool(result.success),both_starts_success=all(s.success for s in solutions),
        objective=float(result.fun),iterations=int(result.nit),evaluations=evaluations,
        client_messages=messages,uplink_floats=4*messages,downlink_floats=3*clients*evaluations,
        bound_hit=bool(np.any(np.isclose(result.x,BOUNDS[:,0],atol=1e-7)|np.isclose(result.x,BOUNDS[:,1],atol=1e-7))),
        multistart_difference=float(np.max(abs(np.exp(solutions[0].x)-np.exp(solutions[1].x))/p)))
