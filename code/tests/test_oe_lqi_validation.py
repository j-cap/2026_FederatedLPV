"""Technical fixtures only: no reserved fleet controller or rollout evaluation."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'code/src'),str(ROOT/'code/experiments')]
import paper_oe_lqi_validation as runner
from federated_lpv.oe_lqi_validation import design_nodes, interpolate, simulate
from federated_lpv.vehicle import family_centers


class OELQIValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads=threadpool_limits(limits=1); cls.threads.__enter__()
        cls.cfg=runner.load(runner.CONFIG)
        cls.schedule=runner.nominal_schedule(cls.cfg)
        cls.parameters=family_centers()['nominal']
        t=np.arange(51)*cls.cfg['sample_time']
        cls.profile=dict(time=t,speed=np.full_like(t,20.),reference=.01*np.sin(2*np.pi*t))

    @classmethod
    def tearDownClass(cls):
        cls.threads.__exit__(None,None,None)

    def test_frozen_nominal_controller_is_constant(self):
        gains,prefilters=interpolate(self.schedule,self.cfg['speed_grid'],np.array([10.,17.3,30.]))
        frozen=runner.load(ROOT/'code/config/paper_fixed_lti_identification.json')['controller_frozen']
        np.testing.assert_array_equal(gains,np.tile(frozen['gain'],(3,1)))
        np.testing.assert_array_equal(prefilters,np.full(3,frozen['prefilter']))
        with self.assertRaises(ValueError):
            interpolate(self.schedule,self.cfg['speed_grid'],np.array([31.]))

    def test_failed_node_blocks_schedule_without_fallback(self):
        matrices=np.array([n['matrix'] for n in self.schedule['nodes']]); matrices[4,4:]=0
        failed=design_nodes(matrices,self.cfg)
        self.assertEqual(failed['status'],'synthesis_failed')
        self.assertEqual(sum(n['status']=='synthesis_failed' for n in failed['nodes']),1)
        outcome,arrays=simulate(self.parameters,failed,self.profile,self.cfg)
        self.assertEqual(outcome['status'],'synthesis_failed'); self.assertIsNone(outcome['metrics']); self.assertIsNone(arrays)

    def test_known_input_feedback_and_rmse_alignment(self):
        outcome,arrays=simulate(self.parameters,self.schedule,self.profile,self.cfg)
        self.assertEqual(outcome['status'],'complete')
        np.testing.assert_array_equal(arrays['state'][0],[0.,0.])
        node=self.schedule['nodes'][0]
        expected=-arrays['state'][:-1]@np.array(node['gain'][:2])-node['gain'][2]*arrays['integral'][:-1]+node['prefilter']*self.profile['reference'][:-1]
        np.testing.assert_allclose(arrays['requested_steering'],expected,atol=1e-14)
        rmse=np.rad2deg(np.sqrt(np.mean((arrays['state'][:,1]-self.profile['reference'])**2)))
        self.assertAlmostEqual(outcome['metrics']['tracking_rmse_deg_s'],rmse)
        self.assertEqual(len(arrays['state']),len(arrays['steering'])+1)

    def test_limit_activation_and_antiwindup(self):
        profile=dict(self.profile,reference=np.ones(51)*2.)
        outcome,arrays=simulate(self.parameters,self.schedule,profile,self.cfg)
        self.assertEqual(outcome['status'],'complete')
        self.assertGreater(outcome['metrics']['limiter_fraction'],0)
        self.assertLessEqual(outcome['metrics']['steering_peak_deg'],12.+1e-8)
        self.assertLessEqual(outcome['metrics']['steering_rate_peak_deg_s'],120.+1e-8)
        self.assertEqual(arrays['integral'][1],0.)  # worsening saturation is blocked
        self.assertLessEqual(max(abs(arrays['integral'])),self.cfg['integral_limit'])

    def test_infeasible_is_not_dropped_but_numeric_failure_is_explicit(self):
        cfg=dict(self.cfg,beta_limit_deg=1e-12)
        outcome,arrays=simulate(self.parameters,self.schedule,self.profile,cfg)
        self.assertEqual(outcome['status'],'complete'); self.assertFalse(outcome['feasible'])
        self.assertIsNotNone(outcome['metrics']); self.assertEqual(outcome['completed_steps'],50)
        cfg=dict(self.cfg,numerical_abort_abs_state=1e-12)
        outcome,arrays=simulate(self.parameters,self.schedule,self.profile,cfg)
        self.assertEqual(outcome['status'],'rollout_failed'); self.assertIsNone(outcome['metrics'])
        self.assertLess(outcome['completed_steps'],50)
        self.assertEqual(len(arrays['state']),outcome['completed_steps']+1)

    def test_exact_restore_and_corruption_rejection(self):
        outcome,arrays=simulate(self.parameters,self.schedule,self.profile,self.cfg)
        client=dict(index=0,parameters=asdict(self.parameters),schedules={'NominalLTI':self.schedule},
                    rows=[dict(method='NominalLTI',scenario='moderate',outcome=outcome,arrays=runner.array_hashes(arrays))])
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder); path=folder/'checkpoint.json'
            runner.atomic_json(path,dict(client=client,label='β — UTF-8'))
            restored=runner.envelope(path)['client']
            self.assertEqual(runner.restore_client(restored,0,{'moderate':self.profile},self.cfg,folder),1)
            self.assertEqual(runner.restore_client(restored,0,{'moderate':self.profile},self.cfg,folder),0)
            rawpath=runner.raw_path(folder,0,0,'NominalLTI','moderate')
            arrays['state'][0,0]+=.001; runner.save_arrays(rawpath,arrays)
            with self.assertRaisesRegex(ValueError,'Existing raw array hash mismatch'):
                runner.restore_client(restored,0,{'moderate':self.profile},self.cfg,folder)
            saved=runner.load(path); saved['payload']['label']='tampered'
            path.write_text(__import__('json').dumps(saved),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'Payload hash mismatch'):runner.envelope(path)

    def test_group_assignment_and_local_coefficients(self):
        m={'Federated':[np.ones((7,6)).tolist(),(2*np.ones((7,6))).tolist()], 'assignments':[1,0],
           'RestrictedLocal':[np.ones((3,6)).tolist(),(3*np.ones((3,6))).tolist()]}
        np.testing.assert_array_equal(runner.coefficient(m,'Federated',0),2*np.ones((7,6)))
        np.testing.assert_array_equal(runner.coefficient(m,'RestrictedLocal',1),3*np.ones((3,6)))

    def test_interrupted_worker_resumes_without_duplicate_cases(self):
        # Mock plant/synthesis calls, exercise actual worker persistence and resume.
        seed=0; models={m:[np.zeros((3 if m=='RestrictedLocal' else 7,6)).tolist()]*30
                        for m in ['RestrictedLocal','SmoothLocal','ExactLPV']}
        models.update(Global=np.zeros((7,6)).tolist(),Federated=[np.zeros((7,6)).tolist()],assignments=[0]*30)
        source=dict(clients=[dict(client=f'fixture{i}',family='fixture',parameters=asdict(self.parameters),speeds=[20.]) for i in range(30)],
                    conditions={'ClosedOE4':dict(models=models)})
        outcome,arrays=simulate(self.parameters,self.schedule,self.profile,self.cfg)
        with tempfile.TemporaryDirectory() as folder, patch.object(runner,'source_path',return_value=Path(folder)/'source.json'), \
                patch.object(runner,'result_path',return_value=Path(folder)/'result.json'),patch.object(runner,'RAW',Path(folder)/'raw'), \
                patch.object(runner,'provenance',return_value={'fixture':True}), \
                patch.object(runner,'profiles',return_value={s:self.profile for s in runner.SCENARIOS}), \
                patch.object(runner,'nominal_schedule',return_value=self.schedule),patch.object(runner,'design_nodes',return_value=self.schedule), \
                patch.object(runner,'spectral_diagnostic',return_value={'fixture':True}),patch.object(runner,'save_arrays'),patch('builtins.print'):
            runner.atomic_json(Path(folder)/'source.json',source)
            with patch.object(runner,'simulate',side_effect=[(outcome,arrays)]*24+[RuntimeError('interruption')]):
                with self.assertRaisesRegex(RuntimeError,'interruption'):runner.run_fleet(seed)
            partial=runner.envelope(Path(folder)/'result.json'); self.assertEqual(len(partial['clients']),2)
            with patch.object(runner,'simulate',return_value=(outcome,arrays)) as sim:
                self.assertEqual(runner.run_fleet(seed)['status'],'execution_complete'); self.assertEqual(sim.call_count,28*12)
            with patch.object(runner,'simulate',side_effect=AssertionError('must not repeat')):
                self.assertEqual(runner.run_fleet(seed)['status'],'already_complete')
            saved=runner.envelope(Path(folder)/'result.json'); self.assertEqual(sum(len(c['rows']) for c in saved['clients']),360)
            saved['clients'][0]['rows'][0]=deepcopy(saved['clients'][0]['rows'][1])
            with self.assertRaisesRegex(ValueError,'Duplicate or missing'):runner.validate_checkpoint(saved,self.cfg,seed)

    def test_exclusive_lock(self):
        with tempfile.TemporaryDirectory() as folder,patch.object(runner,'RAW',Path(folder)):
            with runner.execution_lock():
                with self.assertRaisesRegex(RuntimeError,'Execution lock exists'):
                    with runner.execution_lock():pass
            self.assertFalse((Path(folder)/'execution.lock').exists())


if __name__=='__main__':unittest.main()
