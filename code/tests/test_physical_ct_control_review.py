"""Check the new diagnostic against the actual nonlinear RK4 implementation."""
from pathlib import Path
import sys
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'code/src'),str(ROOT/'code/experiments')]
from analyze_physical_ct_control_pilot import rk4_origin_matrices
from federated_lpv.oe_lqi_validation import rk4
from federated_lpv.vehicle import family_centers
from federated_lpv.control import augmented_tracking_matrices


@pytest.mark.parametrize('speed',[13.,21.,29.])
@pytest.mark.parametrize('family',['nominal','heavy','handling'])
def test_origin_jacobian_matches_nonlinear_plant_and_integrator(speed,family):
    plant=family_centers()[family];dt=.01;eps=1e-7
    a,b=rk4_origin_matrices(plant,speed,dt)
    # Include integral feedback with a negative gain as in the diagnosed group.
    gain=np.array([-23.,3.3,-.63])
    def step(z):
        u=-gain@z
        return np.r_[rk4(z[:2],u,speed,plant,dt),z[2]+dt*z[1]]
    eye=np.eye(3)*eps
    numerical=np.column_stack([(step(d)-step(-d))/(2*eps) for d in eye])
    aa,bb=augmented_tracking_matrices(a,b,dt)
    np.testing.assert_allclose(aa-bb@gain[None,:],numerical,rtol=2e-8,atol=2e-9)
