import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from experiment_5a_lti_complexity import candidates,controller
from experiment_4f_structured_identification import StructuredModel
from experiment_4d_nonlinear_controllers import redesign_controllers

class ComplexityTests(unittest.TestCase):
    def test_five_endpoint_anchors_reproduce_existing_m5(self):
        params={f:np.array([50.,55.,.6]) for f in ('nominal','heavy','handling')}
        proposed=controller(StructuredModel('family','lpv',params),np.linspace(12,28,5))
        old=redesign_controllers({'M5':StructuredModel('family','grid',params)})['M5']
        for v in (12.,14.,19.,22.,28.):
            np.testing.assert_allclose(proposed.evaluate('heavy',v)[0],old.evaluate('heavy',v)[0])
            self.assertAlmostEqual(proposed.evaluate('heavy',v)[1],old.evaluate('heavy',v)[1])

    def test_anchor_candidates_stay_in_support_and_have_requested_count(self):
        for n in (1,2,3,5,8,10):
            for name,grid in candidates(n,np.linspace(12,28,101)):
                self.assertEqual(len(grid),n)
                self.assertTrue(np.all((grid>=12)&(grid<=28)))
                self.assertTrue(np.all(np.diff(grid)>0))
