"""Coverage budgets, physical ambiguity, checkpoint failures and fair summaries."""
from pathlib import Path
import copy
import sys
import numpy as np
import pandas as pd
import pytest
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'code/src'),str(ROOT/'code/experiments')]
import paper_physical_ct_coverage as run


def config():return run.old.load(run.CONFIG)


def test_frozen_counts_and_budget_preserving_windows():
    cfg=config();specs=[p for s in cfg['seeds'] for p in run.fleet_plan(s,cfg)]
    assert len(specs)==588
    assert sum(p['reuse'] is None for p in specs)==477
    for p in specs:
        ns=[w['n'] for w in p['windows']]
        if p['kind']=='multi_20s':assert sum(ns)==2000 and max(ns)-min(ns)<=1
        if p['kind']=='single':assert ns==[2000]
        if p['kind']=='adjacent_pair':
            assert ns==[2000,2000]
            assert p['windows'][1]['speed']-p['windows'][0]['speed']==2
    two=run.plan([28.,30.],0,cfg)
    assert next(x for x in two if x['kind']=='adjacent_pair')['reuse']=='full'


def test_segments_remain_contiguous_with_terminal_measurement_and_no_mutation():
    specs=run.plan([10,12,14],0,config());spec=specs[-1]
    originals=[dict(speed=v,measured_state=np.arange(4002).reshape(2001,2)+10000*v,applied_input=np.arange(2000)+10000*v) for v in [10,12,14]]
    before=copy.deepcopy(originals);segments=run.sliced_segments(originals,spec)
    for s,w,original in zip(segments,spec['windows'],originals):
        assert s['measured_state'].shape==(w['n']+1,2)
        np.testing.assert_array_equal(s['applied_input'],original['applied_input'][w['start']:w['start']+w['n']])
        assert np.all(np.diff(s['applied_input'])==1)
        s['measured_state'][0,0]=-1
    for a,b in zip(originals,before):np.testing.assert_array_equal(a['measured_state'],b['measured_state'])


@pytest.mark.parametrize('speed',[10.,20.,30.])
def test_exact_single_speed_ambiguity_changes_unseen_models(speed):
    settings=run.old.load(run.old.CONFIG)['estimator'];theta=np.array([47.,59.,.56,1.02])
    candidates=run.equivalent_parameters(theta,speed,settings,41)
    lower,upper=np.array(settings['bounds']).T
    assert len(candidates)>10 and np.all(candidates>=lower) and np.all(candidates<=upper)
    base=run.models(theta,[speed],settings)
    for p in candidates:np.testing.assert_allclose(run.models(p,[speed],settings),base,atol=2e-12,rtol=2e-10)
    other=12 if speed!=12 else 28
    assert np.max(abs(run.models(candidates[0],[other],settings)-run.models(candidates[-1],[other],settings)))>1e-4


def test_variant_averages_do_not_overweight_three_speed_clients_and_failures_stay():
    frame=pd.DataFrame([dict(seed=1,client='a',kind='single',error=e) for e in [1.,2.,3.]]+
                       [dict(seed=1,client='b',kind='single',error=e) for e in [10.,10.]])
    clients,fleets,summary=run.aggregate(frame,['error'])
    assert summary.set_index('kind').loc['single','error']==6
    frame.loc[0,'error']=np.nan
    assert np.isnan(run.aggregate(frame,['error'])[2].set_index('kind').loc['single','error'])


def payload(control=False):
    cfg=config();spec=run.fleet_plan(1301,cfg)[0];c=run.old.sources(1301)[1]['clients'][0]
    p=dict(seed=1301,status='running',provenance={},records=[dict(spec=spec,client=c['client'])])
    if not control:p['records'][0].update(transitions=6000,fit={'converged':False})
    else:
        m=dict(steering_peak_deg=1.,steering_rate_peak_deg_s=2.,beta_violation_samples=0)
        p['records'][0]['rows']=[dict(speed=v,outcome=dict(status='complete',completed_steps=1200,feasible=True,metrics=m.copy())) for v in [13,21,29]]
    return p,cfg


def test_checkpoint_missing_duplicate_and_wrong_budget(monkeypatch):
    monkeypatch.setattr(run,'provenance',lambda *a:{})
    p,cfg=payload();run.validate(p,1301,cfg)
    bad=copy.deepcopy(p);bad['records'][0]['transitions']=2000
    with pytest.raises(ValueError,match='budget'):run.validate(bad,1301,cfg)
    bad=copy.deepcopy(p);bad['records'].append(copy.deepcopy(bad['records'][0]))
    with pytest.raises(ValueError,match='order'):run.validate(bad,1301,cfg)
    p['status']='execution_complete'
    with pytest.raises(ValueError,match='Incomplete'):run.validate(p,1301,cfg)


def test_control_failed_metrics_and_false_feasibility_rejected(monkeypatch):
    monkeypatch.setattr(run,'provenance',lambda *a:{})
    p,cfg=payload(True);run.validate(p,1301,cfg,True)
    p['records'][0]['rows'][0]['outcome']['metrics']['beta_violation_samples']=1
    with pytest.raises(ValueError,match='feasibility'):run.validate(p,1301,cfg,True)
    p['records'][0]['rows'][0]['outcome']['feasible']=False
    run.validate(p,1301,cfg,True)
    p['records'][0]['rows'][0]['outcome']['status']='identification_failed'
    with pytest.raises(ValueError,match='failed rollout'):run.validate(p,1301,cfg,True)


def test_report_handles_failed_fit_and_unicode(tmp_path,monkeypatch):
    monkeypatch.setattr(run,'REPORT',tmp_path/'report.html')
    frame=pd.DataFrame([dict(seed=1,client='a',kind=k,mu=.9,equivalence_max_increment_deviation_pct=2.,common_unseen_increment_pct=np.nan if k=='single' else 1.) for k in run.KINDS])
    clients,_,summary=run.aggregate(frame,['common_unseen_increment_pct'])
    run.write_report(frame,clients,summary,None,dict(condition_records=4,control_cases=0))
    text=run.REPORT.read_text(encoding='utf-8')
    assert '<svg' in text and 'unavailable' in text and 'μ' in text
    assert not any(line.rstrip()!=line for line in text.splitlines())
