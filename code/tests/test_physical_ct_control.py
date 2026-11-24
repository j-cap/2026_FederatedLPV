"""Failures, case completeness and reporting in the fixed control comparison."""
import copy
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'code/src'),str(ROOT/'code/experiments')]
import paper_physical_ct_control as run


def checkpoint():
    cfg=run.old.load(run.CONFIG);seed=1301
    _,source=run.old.sources(seed)
    control=run.old.load(run.old.storage.CONFIG)
    schedule=run.old.storage.nominal_schedule(control)
    metrics=dict(tracking_rmse_deg_s=.1,tracking_peak_deg_s=.2,beta_peak_deg=1.,
        steering_rms_deg=.3,steering_peak_deg=2.,steering_rate_peak_deg_s=5.,
        lateral_acceleration_peak_mps2=2.5,limiter_fraction=.0,beta_violation_samples=0)
    outcome=dict(status='complete',feasible=True,metrics=metrics,completed_steps=1200)
    client=dict(index=0,client=source['clients'][0]['client'],observed_speeds=source['clients'][0]['speeds'],
        schedules={method:copy.deepcopy(schedule) for method in cfg['methods']},
        rows=[dict(method=method,speed=v,outcome=copy.deepcopy(outcome)) for method in cfg['methods'] for v in cfg['speeds_mps']])
    payload=dict(seed=seed,status='running',provenance=run.old.provenance(seed,run.old.load(run.old.CONFIG)),
        identification_sha256=run.old.file_hash(run.old.result_path(seed)),clients=[client])
    return payload,seed,cfg,source


def test_missing_duplicate_case_rejected():
    p,seed,cfg,source=checkpoint()
    run.validate(p,seed,cfg,source)
    p['clients'][0]['rows'][-1]=copy.deepcopy(p['clients'][0]['rows'][0])
    with pytest.raises(ValueError,match='Missing, duplicate'):
        run.validate(p,seed,cfg,source)


def test_finite_infeasible_rollout_is_valid_but_false_feasibility_is_not():
    p,seed,cfg,source=checkpoint();o=p['clients'][0]['rows'][0]['outcome']
    o['metrics']['beta_violation_samples']=1;o['feasible']=False
    run.validate(p,seed,cfg,source)
    o['feasible']=True
    with pytest.raises(ValueError,match='Constraint status'):
        run.validate(p,seed,cfg,source)


def test_numeric_failure_cannot_carry_full_duration_metric():
    p,seed,cfg,source=checkpoint();o=p['clients'][0]['rows'][0]['outcome']
    o.update(status='rollout_failed',feasible=False,completed_steps=10)
    with pytest.raises(ValueError,match='Failed trajectories'):
        run.validate(p,seed,cfg,source)
    o['metrics']=None
    run.validate(p,seed,cfg,source)


def test_failed_case_invalidates_fleet_mean_and_stays_in_counts():
    cfg=run.old.load(run.CONFIG);records=[]
    names=['tracking_rmse_deg_s','tracking_peak_deg_s','beta_peak_deg','steering_rms_deg',
        'steering_rate_peak_deg_s','lateral_acceleration_peak_mps2','limiter_fraction']
    for seed in cfg['seeds']:
        for method in cfg['methods']:
            for client in ['a','b']:
                for v in cfg['speeds_mps']:
                    failed=seed==1301 and method=='Global' and client=='a' and v==13
                    records.append(dict(seed=seed,method=method,client=client,speed=v,
                        status='rollout_failed' if failed else 'complete',feasible=False,
                        **{name:np.nan if failed else 1. for name in names}))
    _,fleets,summary,_,paired=run.tables(pd.DataFrame(records),cfg)
    g=summary.set_index('method').loc['Global']
    assert g['cases']==18 and g['complete']==17 and g['rollout_failed']==1 and g['feasible']==0
    assert np.isnan(g.tracking_rmse_deg_s)
    assert summary.set_index('method').loc['Federated','tracking_rmse_deg_s']==1.
    assert not paired[(paired.seed==1301)&(paired.baseline=='Global')].full_fleet_available.item()


def test_report_accepts_missing_fleet_means(tmp_path,monkeypatch):
    cfg=run.old.load(run.CONFIG)
    names=['tracking_rmse_deg_s','tracking_peak_deg_s','beta_peak_deg','steering_rms_deg',
        'steering_rate_peak_deg_s','lateral_acceleration_peak_mps2','limiter_fraction']
    records=[dict(seed=seed,method=m,client='synthetic',speed=v,status='complete',feasible=True,
        **{name:np.nan if m=='SmoothLocal' else .1 for name in names})
        for seed in cfg['seeds'] for m in cfg['methods'] for v in cfg['speeds_mps']]
    _,fleets,summary,speed,paired=run.tables(pd.DataFrame(records),cfg)
    path=tmp_path/'report.html';monkeypatch.setattr(run,'REPORT',path)
    run.write_report(summary,fleets,speed,paired,dict(seeds=cfg['seeds']))
    text=path.read_text(encoding='utf-8')
    assert text.count('<svg')==1 and 'unavailable' in text and '120°/s' in text
