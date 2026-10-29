"""Checks for one controller across the speed envelope and unbiased comparisons."""
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
import unittest
import hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'code/src'),str(ROOT/'code/experiments')]
import paper_fixed_lti_identification as runner
from federated_lpv.fixed_lti_identification import design_fixed_controller,generate_fixed_lqi,reference_signal,add_nominal_start
from federated_lpv.structured_oe import propagate,fit_oe
from federated_lpv.vehicle import family_centers
from federated_lpv.trajectory_calibration import window_indices,array_hash,physical_diagnostics


class FixedLTITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol,cls.settings,cls.cfg,cls.lock=runner.preflight()
        cls.controller=cls.lock['controller']

    def test_one_frozen_model_gain_and_prefilter(self):
        designed=design_fixed_controller(self.protocol['controller'],self.cfg['sample_time'])
        for key in ['a','b','gain','prefilter']:
            np.testing.assert_allclose(designed[key],self.controller[key],atol=1e-10)
        self.assertEqual(self.controller['design_speed_mps'],20.)
        self.assertEqual(self.protocol['controller']['q'],[25.,50.,1000.])
        self.assertEqual(self.protocol['controller']['r'],.15)

    def test_reference_scales_lateral_acceleration_not_controller(self):
        _,ix=window_indices(self.cfg)
        wave=[]
        for speed in [10.,20.,30.]:
            ref=reference_signal(speed,99,self.cfg,self.protocol)
            wave.append((ref[ix]-speed*self.cfg['curvature_per_m'])*speed)
            self.assertEqual(ref[0],0.)
        for w in wave:
            np.testing.assert_allclose(w,wave[0],atol=1e-14)
            self.assertAlmostEqual(max(abs(w)),.3)

    def test_zero_start_measured_feedback_and_applied_input_limits(self):
        t=generate_fixed_lqi(family_centers()['nominal'],10.,99,self.cfg,self.protocol,self.controller)
        np.testing.assert_array_equal(t['state'][0],[0.,0.])
        gain=np.array(self.controller['gain'])
        expected=-(t['observed_state'][:-1]@gain[:2])-gain[2]*t['integral'][:-1]+self.controller['prefilter']*t['yaw_reference']
        np.testing.assert_allclose(t['steering_requested'],expected,atol=1e-14)
        self.assertLessEqual(max(abs(np.rad2deg(t['steering']))),12.+1e-8)
        self.assertLessEqual(max(abs(np.rad2deg(np.diff(np.r_[0.,t['steering']]))/.01)),30.+1e-8)
        self.assertTrue(np.any(t['limited']))
        self.assertFalse(physical_diagnostics(t,10.,self.cfg)['physical_pass'])
        # A limit activation remains visible, never hidden by passing bounds.
        self.assertGreater(runner.tracking(t,self.cfg)['limiter_fraction'],0.)

    def test_noise_seed_matches_original_and_repeatability(self):
        a=generate_fixed_lqi(family_centers()['nominal'],20.,71,self.cfg,self.protocol,self.controller)
        b=generate_fixed_lqi(family_centers()['nominal'],20.,71,self.cfg,self.protocol,self.controller)
        for key in a:self.assertEqual(array_hash(a[key]),array_hash(b[key]))
        rng=np.random.default_rng(np.random.SeedSequence([71,0,11]))
        expected=rng.normal(size=a['state'].shape)*np.deg2rad(self.cfg['state_noise_deg'])
        np.testing.assert_array_equal(a['measurement_noise'],expected)
        y,u=runner.inputs(a,self.cfg);_,ix=window_indices(self.cfg)
        np.testing.assert_array_equal(u,a['steering'][ix])
        np.testing.assert_array_equal(y,a['observed_state'][ix[0]:ix[-1]+2])

    def test_nominal_start_reuses_baseline_and_selects_training_loss(self):
        a=np.array([[.94,.015],[-.02,.97]]);b=np.array([.04,.12]);c=np.array([.0001,-.0002])
        u=.01*np.random.default_rng(40).normal(size=700)
        y=propagate(a,u[:,None]*b+c,np.array([.003,-.004]))
        base=fit_oe(y,u,self.settings['estimator']);before=runner.parent.canonical(base)
        with patch('federated_lpv.fixed_lti_identification.fit_oe',side_effect=AssertionError('baseline must not refit')):
            _,combined=add_nominal_start(y,u,self.settings['estimator'],self.controller,base)
        self.assertEqual(before,runner.parent.canonical(base));self.assertEqual(len(combined['attempts']),4)
        expected=np.r_[np.array(self.controller['a']).ravel(),self.controller['b']]
        np.testing.assert_array_equal(combined['attempts'][-1]['initial_target'],expected)
        self.assertLessEqual(combined['weighted_mse'],base['weighted_mse'])
        np.testing.assert_allclose(combined['target'],np.r_[a.ravel(),b],atol=1e-5)

    def test_exact_new_array_restoration_and_corruption_rejection(self):
        seed=987;speed=20.;acq=74;params=family_centers()['nominal']
        traces=[generate_fixed_lqi(params,speed,acq,self.cfg,self.protocol,self.controller,v) for v in [False,True]]
        arrays=runner.parent.pack_traces(*traces)
        row=dict(status='simulated',client_index=0,speed=speed,parameters=runner.asdict(params),acquisition_seed=acq,
                 array_sha256={k:array_hash(v) for k,v in arrays.items()})
        with tempfile.TemporaryDirectory() as folder,patch.object(runner,'DATA',Path(folder)),patch.object(runner,'read_result',return_value=dict(cells=[row])):
            self.assertEqual(runner.restore_new_job(seed)['restored'],1)
            self.assertEqual(runner.restore_new_job(seed)['restored'],0)
            path=runner.raw_path(seed,0,speed)
            arrays['train_state'][0,0]+=.01;runner.parent.write_raw(path,arrays)
            with self.assertRaises(AssertionError):runner.restore_new_job(seed)

    def test_analysis_includes_failures_and_keeps_confirmation_closed(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as folder,patch.object(runner,'OUT',Path(folder)),patch.object(runner,'REPORT',Path(folder)/'report.html'),patch.object(runner.parent,'tracked_unchanged',return_value=True):
            for seed in self.protocol['seeds']:
                cells=[]
                for i in range(83):
                    models={m:dict(fit_valid=(i!=0),accuracy_pass=(i!=0),matrix_error=.01 if i else None,
                        rollout_nrmse=.1 if i else None,affine_rollout_nrmse=.1 if i else None) for m in runner.METHODS}
                    cells.append(dict(client=f'test{i}',speed=10+2*(i%11),status='simulated',models=models,
                        physical_pass=False,settling_pass=True,tracking_validation=dict(yaw_rmse_deg_s=.2,limiter_fraction=.05),
                        closed_optimizer=dict(attempts=[dict(converged=True)],selected_start='affine_ols')))
                payload=dict(status='complete',seed=seed,provenance=self.lock,cells=cells,previews={})
                runner.parent.save_json(runner.path_for(seed),dict(payload=payload,payload_sha256=hashlib.sha256(runner.parent.canonical(payload)).hexdigest()))
            runner.analyze()
            result=runner.parent.load(runner.OUT/f'{runner.PREFIX}_analysis.json')
            self.assertEqual(result['summaries'][0]['total'],249)
            self.assertEqual(result['summaries'][0]['valid'],246)
            self.assertFalse(result['identification_gate_pass'])
            self.assertFalse(result['lpv_control_benefit_tested'])
            self.assertFalse(result['confirmation_unlocked'])
            self.assertTrue(runner.REPORT.exists())

    def test_single_cell_worker_saves_all_conditions_and_raw_hashes(self):
        # Technical nominal fixture, not a sampled development client.
        seed=987;speed=20.;params=family_centers()['nominal']
        traces=[generate_fixed_lqi(params,speed,82,self.cfg,self.protocol,self.controller,v) for v in [False,True]]
        y,u=runner.inputs(traces[0],self.cfg)
        base=fit_oe(y,u,self.settings['estimator'])
        reference=np.r_[np.array(self.controller['a']).ravel(),self.controller['b']]
        oldcell=dict(client='fixture',speed=speed,reference=reference.tolist(),optimizer=base,
            training_measurements_sha256=array_hash(y),training_input_sha256=array_hash(u),
            models=dict(StructuredOE=runner.score(base,traces[1],reference,self.cfg)),
            physical_pass=False,settling_pass=True)
        source=dict(cells=[dict(client='fixture',client_index=0,speed=speed)],
            clients=[dict(parameters=runner.asdict(params))])
        with tempfile.TemporaryDirectory() as folder,patch.object(runner,'OUT',Path(folder)),patch.object(runner,'DATA',Path(folder)/'raw'),patch.object(runner.oe.known,'source_result',return_value=source),patch.object(runner.oe.known,'load_traces',return_value=traces),patch.object(runner.oe,'read_result',return_value=dict(cells=[oldcell])):
            result=runner.run_job(seed)
            self.assertEqual(result['cells'],1)
            saved=runner.parent.load(runner.path_for(seed))['payload']['cells'][0]
            self.assertEqual(set(saved['models']),set(runner.METHODS))
            self.assertEqual(len(saved['closed_optimizer']['attempts']),4)
            with np.load(runner.raw_path(seed,0,speed)) as raw:
                for key,sha in saved['array_sha256'].items():self.assertEqual(array_hash(raw[key]),sha)


if __name__=='__main__':unittest.main(verbosity=2)
