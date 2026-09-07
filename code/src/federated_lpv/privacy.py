"""One-shot client-level analytic Gaussian aggregation in bounded log space."""
import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr

LOG_BOUNDS=np.log(np.array([[1.,300.],[1.,300.],[.05,3.]]))
CENTER=LOG_BOUNDS.mean(axis=1);HALF_RANGE=np.diff(LOG_BOUNDS,axis=1).ravel()/2


def analytic_delta(epsilon,mu):
    """Exact hockey-stick divergence of unit Gaussians separated by mu."""
    return float(ndtr(mu/2-epsilon/mu)-np.exp(epsilon)*ndtr(-mu/2-epsilon/mu))


def analytic_gaussian_sigma(epsilon,delta,sensitivity):
    if epsilon<=0 or not 0<delta<1 or sensitivity<=0:raise ValueError('invalid privacy arguments')
    root=brentq(lambda mu:analytic_delta(epsilon,mu)-delta,1e-12,1e3)
    return sensitivity/root


def normalize_log_parameters(parameters):
    z=(np.log(np.asarray(parameters))-CENTER)/HALF_RANGE
    return np.clip(z,-1,1)


def denormalize_log_parameters(value):
    return np.exp(np.clip(value,-1,1)*HALF_RANGE+CENTER)


def private_mean(parameters,epsilon,delta,rng,clip_norm=np.sqrt(3)):
    """Replacement-adjacent client DP mean; returns estimate and audit data."""
    z=normalize_log_parameters(parameters);norms=np.linalg.norm(z,axis=1);scale=np.minimum(1,clip_norm/np.maximum(norms,1e-30));clipped=z*scale[:,None]
    mean=clipped.mean(axis=0);sensitivity=2*clip_norm/len(z)
    if np.isinf(epsilon):sigma=0.;noise=np.zeros(3)
    else:sigma=analytic_gaussian_sigma(epsilon,delta,sensitivity);noise=rng.normal(0,sigma,3)
    return denormalize_log_parameters(mean+noise),dict(sigma=sigma,sensitivity=sensitivity,clip_norm=clip_norm,clipped_clients=int(np.sum(scale<1)),noise_norm=float(np.linalg.norm(noise)))
