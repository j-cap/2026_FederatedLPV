import numpy as np
from federated_lpv.privacy import analytic_delta,analytic_gaussian_sigma,private_mean


def test_analytic_gaussian_calibration():
    for epsilon in (.5,1,4,8):
        sigma=analytic_gaussian_sigma(epsilon,1e-5,.2)
        np.testing.assert_allclose(analytic_delta(epsilon,.2/sigma),1e-5,rtol=1e-8)


def test_nonprivate_mean_is_deterministic_and_bounded():
    p=np.array([[50,55,.6],[52,54,.62]])
    a,d=private_mean(p,np.inf,1e-5,np.random.default_rng(1));b,_=private_mean(p,np.inf,1e-5,np.random.default_rng(2))
    np.testing.assert_array_equal(a,b);assert np.all(a>0);assert d['sigma']==0
