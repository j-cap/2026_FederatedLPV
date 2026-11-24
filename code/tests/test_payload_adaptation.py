"""Physical consistency, prefix isolation, scalar recovery and checkpoint policy."""
import copy
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'code/src'),str(ROOT/'code/experiments')]
import paper_ct_payload_adaptation as run
from federated_lpv.payload_adaptation import (load_vehicle,loaded_parameters,yaw_prediction,
    fit_payload,fitting_segment,acquisition_profile)
from federated_lpv.vehicle import family_centers
from federated_lpv.oe_lqi_validation import rk4
from federated_lpv.ct_observer_control import trace_hash


def config():
    return run.base.old.load(run.CONFIG)


def ratios(p):
    return np.array([p.front_stiffness/p.mass,p.rear_stiffness/p.mass,p.mass/p.yaw_inertia,.9])


def test_loading_ratios_match_independent_mass_and_inertia_update():
    for p in family_centers().values():
        for fraction in [0,.1,.2,.3,.5]:
            loaded=load_vehicle(p,fraction,1.)
            np.testing.assert_allclose(loaded_parameters(ratios(p),fraction,1.),ratios(loaded),rtol=2e-15)
            assert loaded.front_length==p.front_length and loaded.rear_length==p.rear_length
            assert loaded.front_stiffness==p.front_stiffness and loaded.rear_stiffness==p.rear_stiffness


def test_vectorized_predictor_matches_independent_nonlinear_plant():
    base=family_centers()['heavy'];fractions=[0,.1,.3]
    u=.012*np.sin(np.arange(300)*.01*2*np.pi*.7)
    predicted=yaw_prediction(loaded_parameters(ratios(base),fractions,1.),u,21.,[1.2,1.6],.01)
    for j,fraction in enumerate(fractions):
        plant=load_vehicle(base,fraction,1.);x=np.zeros(2);yaw=[0.]
        for steering in u:
            x=rk4(x,steering,21.,plant,.01);yaw.append(x[1])
        np.testing.assert_allclose(predicted[j],yaw,atol=2e-15,rtol=2e-14)


def test_scalar_fit_recovers_loading_without_true_states_or_parameters():
    cfg=config()['fit'];theta=np.array([48.,57.,.61,.9]);value=.217
    u=.009*np.sin(np.arange(250)*.01*2*np.pi*.9)
    y=yaw_prediction(loaded_parameters(theta,value,1.),u,20.,[1.2,1.6],.01)[0]
    fit=fit_payload(theta,dict(measured_yaw=y,applied_input=u,speed=20.),cfg)
    assert fit['converged'] and abs(fit['fraction']-value)<1e-5
    with pytest.raises(ValueError,match='yaw, applied input'):
        fit_payload(theta,dict(measured_yaw=y,applied_input=u,speed=20.,true_beta=np.zeros_like(y)),cfg)


def test_no_excitation_is_not_reported_as_successful_zero_loading_estimate():
    fit=fit_payload([50.,50.,.6,.9],dict(measured_yaw=np.zeros(51),applied_input=np.zeros(50),speed=20.),config()['fit'])
    assert not fit['converged'] and 'No resolved' in fit['reason']


def test_prefix_has_no_hidden_state_or_future_sample_dependency():
    trace=dict(measured_yaw=np.arange(1001)/1000,steering=np.arange(1000)/10000,
               speed=np.full(1001,20.),state=np.zeros((1001,2)),estimate=np.zeros((1001,2)))
    old=fitting_segment(trace,4.,.01)
    trace['measured_yaw'][401:]=1e6;trace['steering'][400:]=-1e6
    trace['state'][:]=np.nan;trace['estimate'][:]=np.nan
    new=fitting_segment(trace,4.,.01)
    assert set(old)=={'measured_yaw','applied_input','speed'}
    for key in old:np.testing.assert_array_equal(old[key],new[key])
    assert len(old['measured_yaw'])==401 and len(old['applied_input'])==400


def test_road_inspired_profile_lengths_limits_and_prefix_timing():
    cfg=config()['acquisition'];p=acquisition_profile(cfg)
    assert len(p['time'])==4001 and p['time'][-1]==40.
    assert np.max(abs(p['reference']*p['speed']))<=1.5+1e-12
    for t in [0,8,10,18,20,28,30,38,40]:assert abs(p['reference'][round(t/.01)])<1e-12


def test_known_change_keeps_imperfect_previous_model_and_paired_noise(monkeypatch):
    cfg=config();previous=np.array([44.,61.,.52,1.1])
    schedule=run.schedule_for(previous,cfg);plant=load_vehicle(family_centers()['nominal'],.2,1.)
    monkeypatch.setattr(run,'material',lambda *args:(0,plant,previous,schedule))
    def fake_simulate(plant,schedule,profile,observer,noise,cc):
        return dict(status='complete',feasible=True,completed_steps=1200,metrics={}),{'noise':noise}
    monkeypatch.setattr(run,'simulate_observed',fake_simulate)
    old,_=run.evaluate('nominal_00',20,'Previous',21,cfg)
    known,_=run.evaluate('nominal_00',20,'KnownChange',21,cfg)
    np.testing.assert_array_equal(known['parameters'],loaded_parameters(previous,.2,1.))
    assert not np.allclose(known['parameters'],ratios(plant))
    assert old['noise_sha256']==known['noise_sha256']


def test_failed_case_invalidates_mean_and_no_headroom_ratio_for_nonpositive_gap(tmp_path,monkeypatch):
    cfg=config();rows=[]
    for client in ['nominal_00','nominal_01']:
        for load in cfg['loading_percent']:
            for method in run.methods(cfg):
                for speed in cfg['test_speeds_mps']:
                    rows.append(dict(client=client,loading_percent=load,method=method,speed=speed,
                        status='complete',feasible=True,true_origin_rho=.9,**{m:.1 for m in run.METRICS}))
    frame=pd.DataFrame(rows)
    frame.loc[0,'status']='rollout_failed';frame.loc[0,'feasible']=False;frame.loc[0,run.METRICS]=np.nan
    _,_,summary,pairs,sensitivity=run.tables(frame,cfg)
    assert np.isnan(summary.set_index('method').loc['Previous','tracking_rmse_deg_s'])
    assert summary.set_index('method').loc['Previous','failures']==1
    assert pairs.fraction_of_positive_gap_closed.isna().all()
    fits=pd.DataFrame([dict(client='nominal_00',loading_percent=20,prefix_s=t,estimated_loading_percent=20.) for t in cfg['prefix_seconds']])
    target=tmp_path/'review.html';monkeypatch.setattr(run,'REPORT',target)
    run.write_report(summary,pairs,fits,pd.DataFrame([dict(status='complete')]),sensitivity,[],cfg)
    text=target.read_text(encoding='utf-8');assert 'unavailable' in text and text.count('<svg')==2


def test_exact_restore_rejects_corruption_without_changing_checkpoint(tmp_path,monkeypatch):
    cfg=config();monkeypatch.setattr(run,'RAW',tmp_path)
    arrays=dict(time=np.arange(10.),state=np.zeros((10,2)))
    saved=dict(trace_sha256=trace_hash(arrays),outcome={'status':'complete'})
    monkeypatch.setattr(run,'acquire',lambda *args:(saved,arrays))
    restored=run.restore_acquisition('nominal_00',20,cfg,saved)
    assert trace_hash(restored)==saved['trace_sha256']
    run.save_arrays(run.raw_path('nominal_00',20),dict(time=np.arange(10.),state=np.ones((10,2))))
    with pytest.raises(ValueError,match='hash mismatch'):
        run.restore_acquisition('nominal_00',20,cfg,saved)


def test_resume_preserves_completed_fit_and_acquisition(tmp_path,monkeypatch):
    cfg=config();monkeypatch.setattr(run,'OUT',tmp_path/'tables');monkeypatch.setattr(run,'RAW',tmp_path/'raw')
    arrays=dict(measured_yaw=np.zeros(4001),steering=np.zeros(4000),speed=np.full(4001,20.))
    acq=dict(trace_sha256=trace_hash(arrays),noise_sha256='noise',operational_checks_passed=True,
             outcome=dict(status='complete',completed_steps=4000,feasible=True,metrics={}))
    monkeypatch.setattr(run,'acquire',lambda *args:(acq,arrays))
    monkeypatch.setattr(run,'material',lambda *args:(0,None,np.array([50.,50.,.6,.9]),None))
    calls=[]
    def fake_fit(previous,segment,settings):
        calls.append(len(segment['applied_input']))
        return dict(converged=True,fraction=.2,parameters=loaded_parameters(previous,.2,1.).tolist())
    monkeypatch.setattr(run,'fit_payload',fake_fit)
    metrics={m:.1 for m in run.METRICS};metrics.update(beta_violation_samples=0,steering_peak_deg=1.)
    def fake_evaluate(client,percent,method,speed,cfg,fit):
        return dict(method=method,speed=speed,noise_sha256=f'noise{speed}',trace_sha256='trace',
            outcome=dict(status='complete',completed_steps=1200,feasible=True,metrics=metrics)),None
    monkeypatch.setattr(run,'evaluate',fake_evaluate)
    writer=run.base.atomic_json
    def interrupted(path,payload):
        if len(payload['fits'])==2:raise PermissionError('Synthetic checkpoint interruption')
        writer(path,payload)
    monkeypatch.setattr(run.base,'atomic_json',interrupted)
    with pytest.raises(PermissionError,match='Synthetic'):run.run_case('nominal_00',20)
    assert len(run.base.old.envelope(run.path('nominal_00',20))['fits'])==1
    calls.clear();monkeypatch.setattr(run.base,'atomic_json',writer)
    def no_regeneration(*args):raise AssertionError('Acquisition must be reused')
    monkeypatch.setattr(run,'acquire',no_regeneration)
    assert run.run_case('nominal_00',20)['status']=='execution_complete'
    assert calls==[800,1600,3200,4000]
