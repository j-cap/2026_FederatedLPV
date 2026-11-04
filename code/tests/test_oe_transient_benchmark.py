"""Protocol, failure accounting and independent physical/plot checks."""
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'code/experiments'))
import paper_oe_transient_benchmark as b


def test_profile_has_constant_speed_common_demand_and_quiet_endpoints():
    spec = b.run.load(b.CONFIG)
    for speed in spec['speeds_mps']:
        p = b.profile(spec, speed)
        t = p['time']; a = p['nominal_lateral_acceleration']
        assert len(t) == 1201 and np.all(p['speed'] == speed)
        assert np.all(a[(t <= 2) | (t >= 10)] == 0)
        assert max(a) == pytest.approx(2.5) and min(a) == pytest.approx(-2.5)
        assert np.sum(a) == pytest.approx(0, abs=1e-12)
        np.testing.assert_allclose(p['reference']*speed, a, atol=1e-14)
        np.testing.assert_allclose(p['curvature']*speed**2, a, atol=1e-14)
        assert abs(np.gradient(a,.01)[200]) < 1e-3
        assert abs(np.gradient(a,.01)[1000]) < 1e-3


def test_failure_accounting_keeps_infeasible_and_discloses_pair_availability():
    rows = []
    for seed in [1,2]:
        for method in ['Global','Federated']:
            for i in range(30):
                status = 'rollout_failed' if seed == 2 and method == 'Global' and i == 0 else 'complete'
                # Failed run has no full-duration metric; one very bad finite run must stay.
                value = np.nan if status != 'complete' else (100. if i == 1 else 1.)
                rows.append(dict(seed=seed,method=method,speed_mps=13.,family='fixture',status=status,
                    feasible=status == 'complete' and i != 1,**{k:value for k in b.METRICS}))
    fleet,summary,pairs = b.summaries(pd.DataFrame(rows))
    global_rows = summary[(summary.family == 'all') & (summary.method == 'Global')].iloc[0]
    assert global_rows.cases == 60 and global_rows.finite == 59 and global_rows.feasible == 57
    assert global_rows.rollout_failed == 1
    assert global_rows.tracking_rmse_deg_s > 4
    assert pairs.iloc[0].complete_paired_fleets == 1
    assert pairs.iloc[0].baseline_minus_federated_mean_deg_s == 0
    assert pairs.iloc[0].total_fleets == 2


def test_all_failed_method_remains_visible():
    rows = []
    for method in ['Global','Federated']:
        for i in range(30):
            rows.append(dict(seed=1,method=method,speed_mps=13.,family='fixture',
                status='synthesis_failed',feasible=False,**{k:np.nan for k in b.METRICS}))
    _,summary,pairs = b.summaries(pd.DataFrame(rows))
    assert (summary.finite == 0).all()
    assert (summary.synthesis_failed == 30).all()
    assert summary.tracking_rmse_deg_s.isna().all()
    assert pairs.iloc[0].complete_paired_fleets == 0
    assert np.isnan(pairs.iloc[0].baseline_minus_federated_mean_deg_s)


def fixture():
    spec=b.run.load(b.CONFIG);cfg=b.run.load(b.run.CONFIG)
    selection=dict(cases=[dict(seed=0,index=i,family=f,client=f+'_fixture',observed_speeds=[10,12])
                          for i,f in enumerate(['heavy','handling'])])
    traces={}
    for case in selection['cases']:
        for speed in spec['speeds_mps']:
            p=b.profile(spec,speed);t=p['time']
            for method in spec['display_methods']:
                traces[(case['family'],float(speed),method)]=dict(time=t,speed=p['speed'],reference=p['reference'],
                    state=np.c_[.001*np.sin(t),p['reference']+.001*np.sin(t)],steering=.03*np.sin(t[:-1]),
                    lateral_acceleration=p['nominal_lateral_acceleration'][:-1],
                    angle_limited=np.zeros(len(t)-1),rate_limited=np.zeros(len(t)-1))
    return spec,cfg,selection,traces


def test_panels_have_shared_scales_and_complete_unclipped_labels():
    spec,cfg,selection,traces=fixture()
    panels=b.make_panels(spec,cfg,traces,selection)
    assert len(panels)==19
    for signal in ['response','error','steering']:
        axes=[fig.axes[0] for name,fig in panels.items() if name.endswith(signal)]
        assert len(axes)==6 and len({ax.get_ylim() for ax in axes})==1
    for name,fig in panels.items():
        fig.canvas.draw();renderer=fig.canvas.get_renderer()
        if name!='legend':
            ax=fig.axes[0]
            assert ax.get_title()=='' and ax.get_legend() is None
            assert fig.get_figwidth()==3.48
            boxes=[ax.get_tightbbox(renderer)]
        else:
            boxes=[fig.legends[0].get_window_extent(renderer)]
        for box in boxes:
            assert box.x0 >= -1 and box.y0 >= -1
            assert box.x1 <= fig.bbox.width+1 and box.y1 <= fig.bbox.height+1
        b.previous.plt.close(fig)
    frame=b.trace_table(traces,selection)
    assert len(frame)==18*1201
    assert frame.steering_deg.isna().sum()==18
    assert frame.lateral_acceleration_mps2.isna().sum()==18


def test_replay_rejects_even_small_metric_drift(monkeypatch):
    spec,cfg,selection,traces=fixture()
    expected=dict(method='Global',speed_mps=13.,outcome={'status':'complete'},arrays={},extra_metrics={})
    original=dict(seed=0,clients=[dict(index=0,family='heavy',client='heavy_fixture')])
    result=dict(clients=[dict(rows=[expected])])
    changed=deepcopy(expected);changed['extra_metrics']['rmse']=1e-15
    monkeypatch.setattr(b,'evaluate',lambda *args:(changed,None))
    with pytest.raises(ValueError,match='Exact replay mismatch'):
        b.replay_cases(spec,cfg,[original],[result],selection)


def test_checkpoint_resume_and_provenance_guards(tmp_path, monkeypatch):
    spec=b.run.load(b.CONFIG);cfg=b.run.load(b.run.CONFIG)
    seed=spec['seeds'][0]
    original=dict(seed=seed,protocol=cfg,clients=[dict(index=i,client=f'fixture_{i}',family='fixture',
                  observed_speeds=[10,12]) for i in range(30)])
    source=tmp_path/'source.json';b.run.atomic_json(source,original)
    monkeypatch.setattr(b.run,'result_path',lambda s:source)
    calls=[]
    def failed(client,method,p,config,specification):
        calls.append((client['index'],method,float(p['speed'][0])))
        return dict(method=method,speed_mps=float(p['speed'][0]),
                    outcome=dict(status='synthesis_failed',metrics=None,feasible=False,completed_steps=0),
                    extra_metrics=None,arrays=None),None
    monkeypatch.setattr(b,'evaluate',failed)
    assert b.run_fleet(seed,tmp_path)['status']=='execution_complete'
    assert len(calls)==540
    path=b.result_path(tmp_path,seed);before=path.read_bytes()
    b.run_fleet(seed,tmp_path)
    assert len(calls)==540 and path.read_bytes()==before
    payload=b.run.envelope(path)
    changed=deepcopy(payload);changed['clients'][0]['rows'].pop()
    with pytest.raises(ValueError,match='Missing or duplicate'):
        b.validate(changed,spec,seed,original)
    changed=deepcopy(payload);changed['provenance']['protocol_sha256']='changed'
    with pytest.raises(ValueError,match='provenance mismatch'):
        b.validate(changed,spec,seed,original)
    # Resume a partial fleet from exactly its last complete client, even if all
    # existing cases are failures. Failure is an outcome, not a retry policy.
    payload['status']='running';payload['clients']=payload['clients'][:29]
    b.run.atomic_json(path,payload)
    b.run_fleet(seed,tmp_path)
    assert len(calls)==558
    assert {i for i,_,_ in calls[-18:]}=={29}
