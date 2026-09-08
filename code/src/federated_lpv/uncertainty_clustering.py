"""Uncertainty estimates and heteroscedastic mixtures for LPV ratios."""
import numpy as np
from scipy.special import logsumexp

from .output_error import prepare, predict


def log_parameter_covariance(records, parameters, output_noise):
    """Gauss--Newton covariance of log-ratios for known output noise."""
    data=prepare(records);logp=np.log(np.asarray(parameters,float));scales=np.deg2rad(np.asarray(output_noise,float));h=1e-5
    def residual(q):return ((predict(np.exp(q),data)-data['y'])/scales).ravel()
    jac=np.column_stack([(residual(logp+h*np.eye(3)[j])-residual(logp-h*np.eye(3)[j]))/(2*h) for j in range(3)])
    information=jac.T@jac
    values,vectors=np.linalg.eigh(information);floor=max(values.max()*1e-10,1e-10)
    covariance=(vectors*(1/np.maximum(values,floor)))@vectors.T
    return covariance,float(values.max()/max(values.min(),floor))


def normalized_covariance(log_covariance,bounds):
    """Map a log-parameter covariance into normalized log-box coordinates."""
    width=np.log(np.asarray(bounds,float)[:,1])-np.log(np.asarray(bounds,float)[:,0]);scale=np.diag(1/width)
    return scale@np.asarray(log_covariance)@scale


def _log_gaussian(x,mean,covariance):
    sign,logdet=np.linalg.slogdet(covariance)
    if sign<=0:return -np.inf
    delta=x-mean
    return -.5*(len(x)*np.log(2*np.pi)+logdet+delta@np.linalg.solve(covariance,delta))


def fit_mixture(x,covariances,k,seed=0,restarts=10,max_iter=100,intrinsic_floor=0.0025):
    """Fit z_i ~ sum_k pi_k N(mu_k, C_i + S_k) by generalized EM."""
    from sklearn.cluster import KMeans
    x=np.asarray(x,float);covariances=np.asarray(covariances,float);n,d=x.shape;best=None
    for restart in range(restarts):
        labels=KMeans(k,n_init=1,random_state=seed+37*restart+k).fit_predict(x)
        means=np.array([x[labels==j].mean(0) for j in range(k)]);mix=np.bincount(labels,minlength=k)/n
        intrinsic=np.array([np.cov(x[labels==j].T,bias=True) if np.sum(labels==j)>1 else np.eye(d)*intrinsic_floor**2 for j in range(k)])
        intrinsic=np.array([_project_psd(s,intrinsic_floor**2) for s in intrinsic])
        previous=-np.inf
        for _ in range(max_iter):
            logprob=np.empty((n,k))
            inverses=[]
            for j in range(k):
                total=covariances+intrinsic[j];inverse=np.linalg.inv(total);inverses.append(inverse);delta=x-means[j];_,logdet=np.linalg.slogdet(total);quadratic=np.einsum('ni,nij,nj->n',delta,inverse,delta);logprob[:,j]=np.log(max(mix[j],1e-12))-.5*(d*np.log(2*np.pi)+logdet+quadratic)
            normalizer=logsumexp(logprob,axis=1);responsibility=np.exp(logprob-normalizer[:,None]);likelihood=float(normalizer.sum())
            mix=responsibility.mean(0)
            for j in range(k):
                precision=np.einsum('n,nij->ij',responsibility[:,j],inverses[j]);rhs=np.einsum('n,nij,nj->i',responsibility[:,j],inverses[j],x)
                means[j]=np.linalg.solve(precision,rhs)
                delta=x-means[j];scatter=(np.einsum('n,ni,nj->ij',responsibility[:,j],delta,delta)-np.einsum('n,nij->ij',responsibility[:,j],covariances))/max(responsibility[:,j].sum(),1e-12)
                intrinsic[j]=_project_psd(scatter,intrinsic_floor**2)
            if abs(likelihood-previous)<=1e-7*(1+abs(likelihood)):break
            previous=likelihood
        parameters=(k-1)+k*d+k*d*(d+1)//2;bic=-2*likelihood+parameters*np.log(n)
        candidate=dict(k=k,means=means.copy(),mix=mix.copy(),intrinsic=intrinsic.copy(),responsibility=responsibility.copy(),log_likelihood=likelihood,bic=float(bic))
        if best is None or candidate['bic']<best['bic']:best=candidate
    return best


def _project_psd(matrix,floor):
    matrix=(matrix+matrix.T)/2;values,vectors=np.linalg.eigh(matrix)
    return (vectors*np.maximum(values,floor))@vectors.T


def predict_membership(x,covariances,model):
    """Return posterior compatibility probabilities for unseen clients."""
    x=np.asarray(x,float);covariances=np.asarray(covariances,float);logprob=np.empty((len(x),model['k']))
    for j in range(model['k']):
        total=covariances+model['intrinsic'][j];inverse=np.linalg.inv(total);delta=x-model['means'][j];_,logdet=np.linalg.slogdet(total);quadratic=np.einsum('ni,nij,nj->n',delta,inverse,delta);logprob[:,j]=np.log(max(model['mix'][j],1e-12))-.5*(x.shape[1]*np.log(2*np.pi)+logdet+quadratic)
    return np.exp(logprob-logsumexp(logprob,axis=1)[:,None])
