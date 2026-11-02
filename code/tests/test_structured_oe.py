"""Estimator and paired-protocol checks. Run directly with Python."""
from pathlib import Path
import sys
import unittest
import tempfile
from unittest.mock import patch
import hashlib
import numpy as np
from scipy.optimize._numdiff import approx_derivative
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'code/src'),str(ROOT/'code/experiments')]
from federated_lpv.structured_oe import propagate,response_and_jacobian,fit_oe
import paper_structured_oe_diagnostic as runner
from federated_lpv.trajectory_calibration import window_indices,fit_records,prediction_diagnostics


class OETests(unittest.TestCase):
    def setUp(self):
        self.settings=runner.parent.load(runner.CONFIG)
        self.a=np.array([[.93,.025],[-.03,.97]])
        self.b=np.array([.06,.14]);self.c=np.array([.0002,-.0001])
        self.x0=np.array([.002,-.003]);rng=np.random.default_rng(75)
        self.u=.02*rng.normal(size=600)
        self.x=propagate(self.a,self.u[:,None]*self.b+self.c,self.x0)

    def test_filter_recursion_matches_direct_simulation(self):
        expected=[self.x0]
        for u in self.u:expected.append(self.a@expected[-1]+self.b*u+self.c)
        np.testing.assert_allclose(self.x,expected,atol=1e-13,rtol=1e-11)

    def test_analytic_jacobian(self):
        p=np.r_[self.a.ravel(),self.b,self.c,self.x0]
        x,j=response_and_jacobian(p,self.u[:100])
        numeric=approx_derivative(lambda z:response_and_jacobian(z,self.u[:100])[0].ravel(),p,method='3-point')
        np.testing.assert_allclose(j.reshape(-1,10),numeric,atol=1e-6,rtol=2e-5)

    def test_clean_recovery_offset_and_initial_state(self):
        fit=fit_oe(self.x,self.u,self.settings['estimator'])
        self.assertTrue(fit['converged']);self.assertEqual(fit['jacobian_rank'],10)
        np.testing.assert_allclose(fit['target'],np.r_[self.a.ravel(),self.b],atol=2e-6)
        np.testing.assert_allclose(fit['offset'],self.c,atol=1e-7)
        np.testing.assert_allclose(fit['initial_state'],self.x0,atol=1e-6)

    def test_observation_translation_changes_only_offset_and_initial_state(self):
        fit=fit_oe(self.x+np.array([.1,-.2]),self.u+.07,self.settings['estimator'])
        expected=self.c+np.array([.1,-.2])-self.a@np.array([.1,-.2])-self.b*.07
        np.testing.assert_allclose(fit['target'],np.r_[self.a.ravel(),self.b],atol=2e-6)
        np.testing.assert_allclose(fit['offset'],expected,atol=1e-6)

    def test_measurements_not_mutated_and_no_truth_arguments(self):
        y=self.x+np.random.default_rng(4).normal(0,1e-4,self.x.shape)
        before=y.copy();u=self.u.copy()
        result=fit_oe(y,u,self.settings['estimator'])
        np.testing.assert_array_equal(y,before);np.testing.assert_array_equal(u,self.u)
        self.assertTrue(result['converged'])
        self.assertEqual(len(result['attempts']),3)
        self.assertEqual(result['weighted_mse'],min(v['weighted_mse'] for v in result['attempts'] if v['converged']))

    def test_nonconvergence_does_not_silently_fall_back_to_ols(self):
        opts=dict(self.settings['estimator'],max_nfev=1)
        y=self.x+np.random.default_rng(5).normal(0,1e-4,self.x.shape)
        result=fit_oe(y,self.u,opts)
        self.assertFalse(result['converged']);self.assertNotIn('target',result)

    def test_runner_paired_baseline_and_affine_prediction(self):
        cfg=runner.parent.load(runner.parent.CONFIG)
        _,ix=window_indices(cfg);n=int(ix[-1]+1)
        u=.01*np.random.default_rng(5).normal(size=n)
        x=propagate(self.a,u[:,None]*self.b+self.c,self.x0)
        trace=dict(state=x,steering=u,observed_state=x.copy())
        records,center=runner.known.known_observations(trace,cfg)
        target,_=fit_records(records,center,cfg)
        frozen=dict(known_center=center,common_validation_center=center,
            KnownSteering=dict(target=target, matrix_error=float(np.linalg.norm(target-np.r_[self.a.ravel(),self.b])/np.linalg.norm(np.r_[self.a.ravel(),self.b])),
            **prediction_diagnostics(target,trace,center,cfg)))
        result=runner.fit_cell(trace,trace,frozen,np.r_[self.a.ravel(),self.b],cfg,self.settings)
        self.assertTrue(result['optimizer']['converged'])
        self.assertLess(result['models']['StructuredOE']['affine_rollout_nrmse'],1e-5)
        self.assertLess(result['models']['StructuredOE']['matrix_error'],1e-5)
        self.assertTrue(result['models']['KnownOLS']['fit_valid'])

    def test_analysis_keeps_failed_fits_in_denominator(self):
        settings,cfg,lock=runner.preflight()
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            out=Path(directory);report=out/'test.html'
            with patch.object(runner,'OUT',out),patch.object(runner,'REPORT',report),patch.object(runner.parent,'tracked_unchanged',return_value=True):
                for seed in settings['seeds']:
                    for profile in settings['profiles']:
                        cells=[]
                        for i in range(83):
                            models={}
                            for method in runner.METHODS:
                                valid=not (method=='StructuredOE' and i==0)
                                models[method]=dict(fit_valid=valid,accuracy_pass=valid,
                                    matrix_error=.01 if valid else None,rollout_nrmse=.1 if valid else None,
                                    affine_rollout_nrmse=.1 if valid else None)
                            cells.append(dict(client=f'fixture_{i}',speed=10+2*(i%11),models=models,
                                physical_pass=True,settling_pass=True,clean_matrix_error=.001,elapsed_s=.1))
                        payload=dict(seed=seed,profile=profile,status='complete',provenance=lock,cells=cells)
                        runner.parent.save_json(runner.path_for(seed,profile),dict(payload=payload,
                            payload_sha256=hashlib.sha256(runner.parent.canonical(payload)).hexdigest()))
                runner.analyze()
                import csv
                with (out/f'{runner.PREFIX}_summary.csv').open() as source:
                    summary=list(csv.DictReader(source))
                for row in summary:
                    if row['method']=='StructuredOE':
                        self.assertEqual(int(row['n']),249)
                        self.assertEqual(int(row['n_valid']),246)
                        self.assertEqual(int(row['passing']),246)
                self.assertEqual(report.read_text().count('<svg'),3)
                audit=runner.parent.load(out/f'{runner.PREFIX}_analysis.json')
                self.assertFalse(audit['confirmation_unlocked'])


if __name__=='__main__':unittest.main(verbosity=2)
