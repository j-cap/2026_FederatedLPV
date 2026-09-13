import numpy as np
from experiment_9f_personalized_manifold import learn_basis,reconstruct


def test_full_rank_reconstructs_unseen_vector():
    rng=np.random.default_rng(2);z=rng.normal(size=(20,3));w=np.diag([1.,3.,7.]);center,basis,_=learn_basis(z,w,3);target=rng.normal(size=3)
    assert np.allclose(reconstruct(target,center,basis,w),target)


def test_rank_zero_returns_shared_center():
    z=np.array([[0.,1.,2.],[2.,3.,4.]])
    center,basis,_=learn_basis(z,np.eye(3),0)
    assert basis.shape==(3,0)
    assert np.allclose(reconstruct(np.ones(3),center,basis,np.eye(3)),center)
