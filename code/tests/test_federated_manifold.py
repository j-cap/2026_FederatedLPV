import numpy as np
from federated_lpv.federated_manifold import responsibilities,aggregate_group_moments,fit_federated_mixture_cohorts


def test_responsibilities_normalize():
    x=np.array([[0.,0.],[1.,1.]]);c=np.repeat((.1*np.eye(2))[None],2,axis=0);mix=np.array([.5,.5]);means=x.copy();s=np.repeat((.2*np.eye(2))[None],2,axis=0);r,_=responsibilities(x,c,mix,means,s)
    assert np.allclose(r.sum(1),1)


def test_group_moments_reconstruct_hard_group_mean():
    z=np.array([[1.,2.],[3.,4.],[9.,8.]]);r=np.array([[1,0],[1,0],[0,1]]);moments=aggregate_group_moments(z,r)
    assert np.allclose(moments[0][1]/moments[0][0],[2,3])


def test_external_cohorts_control_message_count_and_coverage():
    x=np.array([[-1.,0.],[-.8,.1],[.8,-.1],[1.,0.]])
    cov=np.repeat((.01*np.eye(2))[None],4,axis=0)
    model=fit_federated_mixture_cohorts(x,cov,1,[np.array([0,1]),np.array([1,2])])
    assert model['messages']==4
    assert model['coverage']==.75
