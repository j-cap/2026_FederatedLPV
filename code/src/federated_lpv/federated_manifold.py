"""Secure-aggregation-compatible EM and manifold sufficient statistics."""
import numpy as np
from scipy.special import logsumexp


def _psd(matrix,floor):
    values,vectors=np.linalg.eigh((matrix+matrix.T)/2)
    return (vectors*np.maximum(values,floor))@vectors.T


def responsibilities(x,covariances,mix,means,intrinsic):
    n,k=len(x),len(mix);logp=np.empty((n,k))
    for g in range(k):
        total=covariances+intrinsic[g];inverse=np.linalg.inv(total);delta=x-means[g];_,logdet=np.linalg.slogdet(total);quad=np.einsum('ni,nij,nj->n',delta,inverse,delta);logp[:,g]=np.log(max(mix[g],1e-12))-.5*(x.shape[1]*np.log(2*np.pi)+logdet+quad)
    normalizer=logsumexp(logp,axis=1)
    return np.exp(logp-normalizer[:,None]),normalizer


def initialize(x,k,floor):
    mean=x.mean(0);cov=np.cov(x.T,bias=True)+floor*np.eye(x.shape[1]);values,vectors=np.linalg.eigh(cov);axis=vectors[:,-1];spread=np.sqrt(values[-1]);offsets=np.linspace(-1.5,1.5,k);means=mean+offsets[:,None]*spread*axis;return np.ones(k)/k,means,np.repeat(cov[None],k,axis=0)


def client_statistics(x,covariances,responsibility,means,intrinsic):
    """Return additive per-client messages; no individual message is exposed downstream."""
    n,k=responsibility.shape;d=x.shape[1];mass=responsibility.sum(0);precision=np.zeros((k,d,d));rhs=np.zeros((k,d));scatter=np.zeros((k,d,d))
    for g in range(k):
        inv=np.linalg.inv(covariances+intrinsic[g]);precision[g]=np.einsum('n,nij->ij',responsibility[:,g],inv);rhs[g]=np.einsum('n,nij,nj->i',responsibility[:,g],inv,x);delta=x-means[g];scatter[g]=np.einsum('n,ni,nj->ij',responsibility[:,g],delta,delta)-np.einsum('n,nij->ij',responsibility[:,g],covariances)
    return mass,precision,rhs,scatter


def fit_federated_mixture(x,covariances,k,rounds,participation,seed,damping=.35,floor=0.0025):
    x=np.asarray(x);covariances=np.asarray(covariances);n,d=x.shape;rng=np.random.default_rng(seed);size=max(k*2,int(np.ceil(participation*n)));initial_indices=np.sort(rng.choice(n,min(size,n),replace=False));mix,means,intrinsic=initialize(x[initial_indices],k,floor**2);messages=len(initial_indices);seen=set(initial_indices.tolist())
    for round_index in range(rounds):
        size=max(k*2,int(np.ceil(participation*n)));indices=np.sort(rng.choice(n,min(size,n),replace=False));seen.update(indices.tolist());r,ll=responsibilities(x[indices],covariances[indices],mix,means,intrinsic);mass,precision,rhs,scatter=client_statistics(x[indices],covariances[indices],r,means,intrinsic);new_means=np.array([np.linalg.solve(precision[g]+1e-12*np.eye(d),rhs[g]) for g in range(k)]);new_intrinsic=np.array([_psd(scatter[g]/max(mass[g],1e-12),floor**2) for g in range(k)]);new_mix=mass/mass.sum();mix=(1-damping)*mix+damping*new_mix;means=(1-damping)*means+damping*new_means;intrinsic=(1-damping)*intrinsic+damping*new_intrinsic;messages+=len(indices)
    r,ll=responsibilities(x,covariances,mix,means,intrinsic);parameters=(k-1)+k*d+k*d*(d+1)//2;bic=float(-2*ll.sum()+parameters*np.log(n));return dict(k=k,mix=mix,means=means,intrinsic=intrinsic,responsibility=r,bic=bic,messages=messages,coverage=len(seen)/n,rounds=rounds)


def select_federated_mixture(x,covariances,candidates,rounds,participation,seed,damping=.35,floor=.0025,minimum_size=10):
    models=[]
    for k in candidates:
        model=fit_federated_mixture(x,covariances,k,rounds,participation,seed+101*k,damping,floor);model['minimum_size']=float((model['responsibility'].sum(0)).min());model['admissible']=model['minimum_size']>=minimum_size;models.append(model)
    admissible=[m for m in models if m['admissible']];return min(admissible or models,key=lambda m:m['bic']),models


def aggregate_group_moments(log_parameters,responsibility):
    """Aggregate count, first and second moments for hard local assignments."""
    labels=np.argmax(responsibility,axis=1);moments=[]
    for g in range(responsibility.shape[1]):
        z=log_parameters[labels==g];moments.append((len(z),z.sum(0),np.einsum('ni,nj->ij',z,z)))
    return moments
