"""Paired-intervention checks; run directly with Python."""
import copy
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/"code/src"),str(ROOT/"code/experiments")]
from paper_known_steering_diagnostic import known_observations, verify_arrays, compare_cell
from federated_lpv.trajectory_calibration import array_hash, observations, fit_records, prediction_diagnostics, window_indices
import paper_trajectory_calibration as parent


class KnownSteeringTests(unittest.TestCase):
    def setUp(self):
        self.cfg=parent.load(parent.CONFIG)
        quiet,indices=window_indices(self.cfg)
        rng=np.random.default_rng(41)
        n=int(indices[-1]+1)
        a=np.array([[.97,.01],[-.01,.94]])
        b=np.array([.06,.11])
        steering=.01*np.sin(np.arange(n)*.11)+.006*np.sin(np.arange(n)*.37)
        state=np.zeros((n+1,2))
        for k in range(n): state[k+1]=a@state[k]+b*steering[k]
        self.truth=np.r_[a.ravel(),b]
        self.trace=dict(state=state,steering=steering,observed_state=state+rng.normal(0,1e-4,state.shape),
                        observed_steering=steering+rng.normal(0,1e-4,n))

    def fixture(self):
        records,center=observations(self.trace,self.cfg,"sensor")
        target,diag=fit_records(records,center,self.cfg)
        frozen=dict(target=target.tolist(),center=center.tolist(),validation_center=center.tolist(),
            record_sha256=array_hash(records),**prediction_diagnostics(target,self.trace,center,self.cfg))
        cell=dict(client="test",speed=20.,cases={"TrajectoryAllSensor":frozen},
                  training=dict(physical_pass=True,settling_pass=True),validation=dict(physical_pass=True,settling_pass=True))
        return cell

    def test_only_steering_and_its_center_change(self):
        before={k:array_hash(v) for k,v in self.trace.items()}
        old,oldcenter=observations(self.trace,self.cfg,"sensor")
        new,newcenter=known_observations(self.trace,self.cfg)
        quiet,ix=window_indices(self.cfg)
        np.testing.assert_array_equal(old[:,[0,1,3,4]],new[:,[0,1,3,4]])
        np.testing.assert_array_equal(oldcenter[:2],newcenter[:2])
        np.testing.assert_array_equal(new[:,2],self.trace["steering"][ix])
        self.assertEqual(newcenter[2],self.trace["steering"][quiet].mean())
        self.assertTrue(np.any(old[:,2]!=new[:,2]))
        self.assertEqual(before,{k:array_hash(v) for k,v in self.trace.items()})

    def test_adjacent_state_observations_remain_identical(self):
        records,_=known_observations(self.trace,self.cfg)
        np.testing.assert_array_equal(records[:-1,3:],records[1:,:2])

    def test_applied_input_not_unconstrained_request(self):
        self.trace["steering_request"]=np.full_like(self.trace["steering"],1.)
        records,_=known_observations(self.trace,self.cfg)
        _,ix=window_indices(self.cfg)
        np.testing.assert_array_equal(records[:,2],self.trace["steering"][ix])
        self.assertFalse(np.array_equal(records[:,2],self.trace["steering_request"][ix]))

    def test_no_input_noise_gives_exactly_equal_fits(self):
        self.trace["observed_steering"]=self.trace["steering"].copy()
        cell=self.fixture()
        result=compare_cell(self.trace,self.trace,cell,self.truth,self.cfg,dict(baseline_atol=1e-10,baseline_rtol=1e-9))
        np.testing.assert_array_equal(result["OriginalSensor"]["target"],result["KnownSteering"]["target"])
        self.assertEqual(result["matrix_change_pp"],0.)
        self.assertEqual(result["OriginalSensor"]["rollout_nrmse"],result["KnownSteering"]["rollout_nrmse"])

    def test_common_validation_center_used_for_both_models(self):
        cell=self.fixture()
        result=compare_cell(self.trace,self.trace,cell,self.truth,self.cfg,dict(baseline_atol=1e-10,baseline_rtol=1e-9))
        _,center=known_observations(self.trace,self.cfg)
        for method in ["OriginalSensor","KnownSteering"]:
            expected=prediction_diagnostics(np.array(result[method]["target"]),self.trace,center,self.cfg)
            self.assertEqual(expected["rollout_nrmse"],result[method]["rollout_nrmse"])

    def test_source_array_corruption_rejected(self):
        arrays={"train_state":self.trace["state"].copy()}
        cell=dict(client="test",speed=20.,array_sha256={k:array_hash(v) for k,v in arrays.items()})
        verify_arrays(arrays,cell)
        arrays["train_state"][0,0]+=1e-9
        with self.assertRaises(ValueError): verify_arrays(arrays,cell)

    def test_changed_frozen_fit_rejected(self):
        cell=self.fixture();cell["cases"]["TrajectoryAllSensor"]["target"][0]+=.01
        with self.assertRaises(AssertionError):
            compare_cell(self.trace,self.trace,cell,self.truth,self.cfg,dict(baseline_atol=1e-10,baseline_rtol=1e-9))


if __name__=="__main__": unittest.main(verbosity=2)
