import numpy as np
from federated_lpv.uncertainty_clustering import fit_mixture,predict_membership,normalized_covariance


def test_normalized_covariance_preserves_symmetry_and_positivity():
    covariance=normalized_covariance(np.diag([.04,.01,.09]),[[40,70],[35,75],[.45,.8]])
    assert np.allclose(covariance,covariance.T)
    assert np.linalg.eigvalsh(covariance).min()>0


def test_heteroscedastic_mixture_separates_clear_groups():
    rng=np.random.default_rng(3);x=np.r_[rng.normal(-1,.05,(30,2)),rng.normal(1,.05,(30,2))];cov=np.repeat((np.eye(2)*.0025)[None],60,axis=0)
    model=fit_mixture(x,cov,2,seed=4,restarts=3)
    posterior=predict_membership(x,cov,model)
    assert np.mean(np.max(posterior,axis=1))>.99
    assert len(np.unique(np.argmax(posterior,axis=1)))==2
