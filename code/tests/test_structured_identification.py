"""Check that the reduced parameters preserve the validated plant convention."""
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from experiment_4f_structured_identification import matrices, step
from federated_lpv.vehicle import family_centers, continuous_bicycle_matrices, discrete_bicycle_matrices

class StructuredTests(unittest.TestCase):
    def test_reduced_matrices_match_all_family_physics(self):
        for p in family_centers().values():
            ratios=[p.front_stiffness/p.mass,p.rear_stiffness/p.mass,p.mass/p.yaw_inertia]
            for v in (12.,20.,28.):
                a,b=matrices(ratios,v); expected_a,expected_b=continuous_bicycle_matrices(v,p)
                np.testing.assert_allclose(a,expected_a,atol=1e-12)
                np.testing.assert_allclose(b,expected_b[:,0],atol=1e-12)

    def test_rk4_identification_step_matches_zoh_within_tolerance(self):
        for p in family_centers().values():
            ratios=[p.front_stiffness/p.mass,p.rear_stiffness/p.mass,p.mass/p.yaw_inertia]
            for v in (12.,20.,28.):
                x=np.array([[.01,.1]]); u=np.array([.02])
                a,b=discrete_bicycle_matrices(v,p,.01)
                expected=a@x[0]+b[:,0]*u[0]
                np.testing.assert_allclose(step(ratios,x,u,np.array([v]))[0],expected,atol=1e-6,rtol=0)
