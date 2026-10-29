"""Run directly with Python; no additional test framework is required."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/"code/src"),str(ROOT/"code/experiments")]
import numpy as np
import pandas as pd
from federated_lpv.vehicle import family_centers
from federated_lpv.trajectory_calibration import (
    array_hash, fit_records, generate_trajectory, observations,
    prediction_diagnostics, rk4, waveform, window_indices,
)
from paper_trajectory_calibration import pilot_decision, read_checkpoint, save_checkpoint


class CalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = json.loads((ROOT/"code/config/paper_trajectory_calibration.json").read_text())
        cls.parameters = family_centers()["nominal"]
        cls.trace = generate_trajectory(cls.parameters,20.,.3,178,cls.cfg)

    def test_continuity_including_phase_boundaries(self):
        t = self.trace
        settle = round(self.cfg["settle_s"]/self.cfg["sample_time"])
        _,fitting = window_indices(self.cfg)
        points = [0,399,400,settle-1,settle,int(fitting[0])-1,int(fitting[0]),
                  len(t["steering"])-1,*range(0,len(t["steering"]),73)]
        for k in points:
            np.testing.assert_array_equal(t["state"][k+1],rk4(self.parameters,t["state"][k],t["steering"][k],20.,.01))

    def test_sensor_record_next_state_is_next_regressor(self):
        records,center = observations(self.trace,self.cfg,"sensor")
        np.testing.assert_array_equal(records[:-1,3:],records[1:,:2])
        quiet,_ = window_indices(self.cfg)
        np.testing.assert_array_equal(center[:2],self.trace["observed_state"][quiet].mean(axis=0))

    def test_sensor_preprocessing_needs_no_hidden_truth(self):
        records,center = observations(self.trace,self.cfg,"sensor")
        measured = {k:self.trace[k] for k in ["observed_state","observed_steering"]}
        observed,empirical = observations(measured,self.cfg,"sensor")
        np.testing.assert_array_equal(observed,records)
        np.testing.assert_array_equal(empirical,center)

    def test_steering_limits_and_smooth_window(self):
        u = self.trace["steering"]
        self.assertLessEqual(np.abs(u).max(),np.deg2rad(self.cfg["steering_limit_deg"]))
        self.assertLessEqual(np.abs(np.diff(np.r_[0.,u])/.01).max(),np.deg2rad(self.cfg["steering_rate_deg_s"])+1e-12)
        signal = waveform(self.cfg,.3,20.,178)
        self.assertAlmostEqual(signal[0],0.)
        self.assertAlmostEqual(signal[-1],0.)
        self.assertLessEqual(np.abs(signal).max(),np.deg2rad(.3)+1e-15)

    def test_regression_recovers_known_linear_map(self):
        rng = np.random.default_rng(1)
        z = rng.normal(size=(80,3))*.01
        a = np.array([[.98,.02],[-.01,.95]])
        b = np.array([.03,.12])
        center = np.array([.03,.1,.02])
        y = z[:,:2]@a.T+z[:,2,None]*b
        target,diag = fit_records(np.column_stack([z+center,y+center[:2]]),center,self.cfg)
        np.testing.assert_allclose(target,np.r_[a.ravel(),b],atol=1e-14)
        self.assertEqual(diag["rank"],3)

    def test_rank_failure_not_silently_accepted(self):
        target,diag = fit_records(np.zeros((36,5)),np.zeros(3),self.cfg)
        self.assertFalse(diag["fit_valid"])
        self.assertEqual(diag["rank"],0)
        self.assertIsNone(diag["scaled_condition"])

    def test_free_run_alignment_on_known_system(self):
        cfg = {**self.cfg,"settle_s":0.,"hold_s":0.,"excitation_s":2.}
        a = np.array([[.98,.02],[-.01,.95]])
        b = np.array([.03,.12])
        u = np.sin(np.arange(200)*.07)*.01
        x = np.zeros((201,2))
        for k in range(200):
            x[k+1] = a@x[k]+b*u[k]
        diag = prediction_diagnostics(np.r_[a.ravel(),b],dict(state=x,steering=u),np.zeros(3),cfg)
        self.assertLess(diag["one_step_nrmse"],1e-13)
        self.assertLess(diag["rollout_nrmse"],1e-13)
        self.assertGreater(diag["persistence_nrmse"],0.)

    def test_deterministic_raw_replay(self):
        replay = generate_trajectory(self.parameters,20.,.3,178,self.cfg)
        for key in replay:
            self.assertEqual(array_hash(replay[key]),array_hash(self.trace[key]))

    def test_checkpoint_corruption_and_stale_provenance(self):
        provenance = {"source_sha256":{"test":"one"}}
        value = dict(stage="pilot",seed=1,profile="test",status="execution_complete",
                     provenance=provenance,clients=[],client_count=0,cells=[])
        with tempfile.TemporaryDirectory() as tmp:
            path = save_checkpoint(value,Path(tmp))
            self.assertEqual(read_checkpoint(path,provenance),value)
            with self.assertRaises(ValueError):
                read_checkpoint(path,{"source_sha256":{"test":"two"}})
            corrupt = json.loads(path.read_text())
            corrupt["payload"]["seed"] = 2
            path.write_text(json.dumps(corrupt))
            with self.assertRaises(ValueError):
                read_checkpoint(path,provenance)

    def test_selection_blocks_failed_profiles_and_uses_smallest_passing_input(self):
        rows = []
        for profile in self.cfg["profiles"]:
            for case in ["TrajectoryAllClean","TrajectoryAllSensor"]:
                rows.append(dict(profile=profile,case=case,rank=3,scaled_condition=10.,
                    fit_valid=True,matrix_error=.01,rollout_nrmse=.1,persistence_nrmse=.5,
                    physical_pass=True,settling_pass=True))
        frame = pd.DataFrame(rows)
        self.assertEqual(pilot_decision(frame,self.cfg)[1],"gentle")
        frame.loc[frame.profile == "gentle","physical_pass"] = False
        self.assertEqual(pilot_decision(frame,self.cfg)[1],"standard")
        frame["persistence_nrmse"] = .01
        self.assertIsNone(pilot_decision(frame,self.cfg)[1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
