import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
import experiment_4c_nonlinear_oracle_models as base
from experiment_4d_nonlinear_controllers import trajectory_metrics
from experiment_4g_independent_validation import simulate_fleet
from federated_lpv import sample_fleet,fit_oracle_architectures,design_oracle_controllers

class IndependentValidationTests(unittest.TestCase):
    def test_vectorized_fleet_matches_reference_simulator_and_metrics(self):
        clients=sample_fleet(11,1)
        models=fit_oracle_architectures(clients,base.FIT_SPEEDS,base.DT)
        controller=design_oracle_controllers(models,base.DT,base.Q,base.R,base.CONTROL_SPEEDS,base.FIT_SPEEDS)['M4']
        time=np.arange(0,1.01,base.DT);speed=18+2*np.sin(time);ref=.04*np.sin(3*time)
        x,u,metrics=simulate_fleet(clients,controller,speed,ref)
        for j,c in enumerate(clients):
            expected,inputs=base.simulate_nonlinear(c,controller,speed,ref)
            np.testing.assert_allclose(x[:,j],expected,atol=1e-12)
            np.testing.assert_allclose(u[:,j],inputs,atol=1e-12)
            values=trajectory_metrics(c,expected,inputs,speed,ref)
            for name,key in [('tracking','tracking_rmse_deg_s'),('peak_acceleration','peak_lateral_accel_mps2'),('peak_steering_rate','peak_steering_rate_deg_s')]:
                self.assertAlmostEqual(metrics[name][j],values[key],places=10)
