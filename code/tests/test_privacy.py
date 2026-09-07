import numpy as np
from federated_lpv.privacy import analytic_delta,analytic_gaussian_sigma,private_mean,shaped_private_mean


def test_analytic_gaussian_calibration():
    for epsilon in (.5,1,4,8):
        sigma=analytic_gaussian_sigma(epsilon,1e-5,.2)
        np.testing.assert_allclose(analytic_delta(epsilon,.2/sigma),1e-5,rtol=1e-8)


def test_nonprivate_mean_is_deterministic_and_bounded():
    p=np.array([[50,55,.6],[52,54,.62]])
    a,d=private_mean(p,np.inf,1e-5,np.random.default_rng(1));b,_=private_mean(p,np.inf,1e-5,np.random.default_rng(2))
    np.testing.assert_array_equal(a,b);assert np.all(a>0);assert d['sigma']==0


def test_custom_public_bounds_are_respected():
    bounds=np.array([[40,70],[35,75],[.45,.8]])
    p,_=private_mean([[1,300,.05],[300,1,3]],np.inf,1e-5,np.random.default_rng(1),bounds=bounds)
    assert np.all(p>=bounds[:,0]) and np.all(p<=bounds[:,1])


def test_shaped_private_mean_matches_nonprivate_mean_and_geometry():
    bounds=np.array([[40.,70.],[35.,75.],[.45,.8]])
    parameters=np.array([[50.,45.,.55],[60.,65.,.7]])
    weights=np.array([1.6,1.8,.34])
    shaped,diag=shaped_private_mean(parameters,np.inf,1e-5,np.random.default_rng(2),weights,bounds)
    plain,_=private_mean(parameters,np.inf,1e-5,np.random.default_rng(2),bounds=bounds)
    assert np.allclose(shaped,plain)
    assert np.isclose(diag['clip_norm'],np.linalg.norm(weights))
    assert diag['clipped_clients']==0


def test_shaped_private_mean_uses_expected_replacement_sensitivity():
    parameters=np.tile([50.,50.,.6],(20,1));weights=np.array([2.,1.,.5])
    _,diag=shaped_private_mean(parameters,2.,1e-5,np.random.default_rng(3),weights)
    assert np.isclose(diag['sensitivity'],2*np.linalg.norm(weights)/len(parameters))
