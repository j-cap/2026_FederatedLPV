"""Confirmation safeguards; never sample or inspect reserved fleets."""
import copy
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'code/src'),str(ROOT/'code/experiments')]
import paper_oe_lpv_confirmation as run


class ConfirmationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.setup=run.preflight()

    def test_frozen_settings(self):
        protocol,cfg,basis,mixture,acq,*_=self.setup
        self.assertEqual(protocol['seeds'],list(range(1311,1321)))
        self.assertFalse(set(protocol['seeds']) & set(cfg['seeds']))
        self.assertEqual(cfg['smooth_strength'],1e6)
        self.assertEqual(mixture['candidate_clusters'],[1,2,3,4,5])

    def test_checkpoint_integrity_and_stale_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'cell.json';value=dict(seed=1300,provenance={'version':1},text='β → r')
            run.save(path,value)
            self.assertEqual(run.read(path,1300,{'version':1}),value)
            with self.assertRaises(ValueError):run.read(path,1300,{'version':2})
            envelope=run.parent.load(path);envelope['payload']['text']='corrupted'
            run.parent.save_json(path,envelope)
            with self.assertRaises(ValueError):run.read(path,1300,{'version':1})

    def test_duplicate_and_partial_resume(self):
        result=dict(clients=[dict(client='a',parameters={},speeds=[10.,12.])],cells=[])
        self.assertEqual(run.validate_cells(result),{(0,10.),(0,12.)})
        cell=dict(client='a',parameters={},client_index=0,speed=10.)
        result['cells']=[cell]
        self.assertEqual(run.validate_cells(result),{(0,12.)})
        result['cells'].append(cell)
        with self.assertRaises(ValueError):run.validate_cells(result)

    def test_failed_cell_blocks_entire_fleet_without_refit(self):
        protocol,cfg,basis,mixture,acq,*_=self.setup
        value=dict(seed=1300,provenance={},clients=[dict(client='fixture',parameters={},speeds=[10.])],
            cells=[dict(client='fixture',parameters={},client_index=0,speed=10.,status='numerical_failure')])
        with patch.object(run.lpv,'fit_models',side_effect=AssertionError('Must not fit subset')):
            result=run.integrate(value,cfg,basis,mixture,acq)
        self.assertEqual(result['status'],'execution_blocked')
        self.assertEqual(set(result['blocked_conditions']),set(cfg['conditions']))

    def test_intervals_use_fleets_and_require_all_ten(self):
        protocol=self.setup[0]
        rows=[]
        for i,seed in enumerate(protocol['seeds']):
            for method,value in [('Federated',.02+i*.0001),('SmoothLocal',.04),('Global',.045)]:
                rows.append(dict(seed=seed,condition='ClosedOE4',region='unseen',method=method,matrix=value))
        units=pd.DataFrame(rows)
        intervals=run.paired_intervals(units,protocol)
        self.assertEqual(len(intervals),2)
        self.assertTrue(all(c['lower_pp']>0 and c['fleets']==10 for c in intervals))
        self.assertAlmostEqual(intervals[0]['mean_difference_pp'],1.955)
        self.assertEqual(run.paired_intervals(units[units.seed!=1320],protocol),[])
        reverse=units.copy();reverse.loc[reverse.method=='Federated','matrix']=.06
        self.assertTrue(all(c['upper_pp']<0 for c in run.paired_intervals(reverse,protocol)))

    def test_concurrent_execution_rejected_and_lock_removed(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp)
            with run.execution_lock(directory):
                with self.assertRaises(RuntimeError):
                    with run.execution_lock(directory):pass
            self.assertFalse((directory/'execution.lock').exists())

    def test_run_resume_does_not_repeat_completed_cells(self):
        protocol,cfg,basis,mixture,acq,controller_protocol,settings,controller,lock=self.setup
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp)
            client=dict(client='fixture',parameters={},speeds=[10.,12.])
            first=dict(client='fixture',parameters={},client_index=0,speed=10.,status='simulated')
            initial=dict(seed=1300,provenance=lock,clients=[client],cells=[first],status='acquiring')
            path=lambda seed,kind='result',directory_unused=None: directory/f'{kind}_{seed}.json'
            run.save(path(1300,'acquisition'),initial)
            def fake_fit(seed,index,client,speed,*args):
                return dict(client='fixture',parameters={},client_index=index,speed=speed,status='simulated')
            def fake_integrate(acquired,*args):
                return dict(seed=1300,provenance=lock,status='execution_complete',blocked_conditions={})
            with patch.object(run,'path_for',side_effect=path), patch.object(run,'preflight',return_value=self.setup), patch.object(run,'fit_cell',side_effect=fake_fit) as fit, patch.object(run,'integrate',side_effect=fake_integrate):
                run.run_fleet(1300)
                self.assertEqual(fit.call_count,1)
                self.assertEqual(fit.call_args.args[3],12.)
                self.assertTrue(run.run_fleet(1300)['resumed'])
                self.assertEqual(fit.call_count,1)
                acquired=run.read(path(1300,'acquisition'),1300,lock)
                acquired['cells'][0]['status']='modified'
                run.save(path(1300,'acquisition'),acquired)
                with self.assertRaises(ValueError):run.run_fleet(1300)

    def test_report_failed_confirmation_utf8(self):
        manifest=dict(primary_intervals=[],primary_confirmation_supported=False,primary_fleets=0,failures=[{'seed':1311,'reason':'failed'}])
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'report.html'
            run.write_report([],None,manifest,path)
            text=path.read_text(encoding='utf-8')
            self.assertIn('0/10',text)
            self.assertIn('failed',text)


if __name__=='__main__':unittest.main()
