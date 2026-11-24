from pathlib import Path
import sys
import numpy as np
import pytest
from scipy.optimize._numdiff import approx_derivative
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'code/src'),str(ROOT/'code/experiments'),str(ROOT/'code/tests')]
from test_physical_ct_oe import fixture
from federated_lpv.physical_ct_yaw_oe import projected_response,fit_yaw_oe,validated_segments,validation_prediction
from federated_lpv.physical_ct_oe import discrete_matrices
from federated_lpv.structured_oe import propagate


def yaw_segments(segments):
    return [dict(speed=s['speed'],measured_yaw=s['measured_state'][:,1].copy(),applied_input=s['applied_input'].copy()) for s in segments]


def test_yaw_gauge_spans_original_four_nuisance_model():
    theta,cfg,raw=fixture(noise=1.)
    s=yaw_segments(raw)[0];q=np.log(theta);res,jac,record=projected_response(q,s,cfg)
    a,b=discrete_matrices(theta,s['speed'],cfg['geometry_m'],.005,.01)
    forcing=np.zeros((len(s['applied_input']),2,5));forcing[:,:,0]=s['applied_input'][:,None]*b
    forcing[:,:,1:3]=np.eye(2);initial=np.zeros((2,5));initial[:,3:]=np.eye(2)
    basis=propagate(a,forcing,initial);h=basis[:,1,1:];base=basis[:,1,0]
    coef=np.linalg.lstsq(h,s['measured_yaw']-base,rcond=1e-12)[0]
    reference=(base+h@coef-s['measured_yaw'])/np.deg2rad(cfg['state_noise_deg'][1])
    np.testing.assert_allclose(res,reference,atol=1e-8,rtol=1e-7)
    assert np.linalg.matrix_rank(h)==3 and record['nuisance_rank']==3
    numerical=approx_derivative(lambda z:projected_response(z,s,cfg,False)[0],q,method='3-point',rel_step=1e-5)
    np.testing.assert_allclose(jac,numerical,atol=5e-5,rtol=3e-5)


def test_noiseless_joint_recovery_and_no_input_mutation():
    theta,cfg,raw=fixture(noise=0.,n=800,speeds=(12.,22.,28.))
    segments=yaw_segments(raw);before=[s['measured_yaw'].copy() for s in segments]
    fit=fit_yaw_oe(segments,cfg)
    assert fit['converged'] and fit['weighted_mse']<1e-12
    np.testing.assert_allclose(fit['parameters'],theta,rtol=1e-4)
    assert fit['information']['rank']==4
    assert fit['scalar_observations']==3*801 and fit['free_nuisance_parameters']==9
    for s,y in zip(segments,before):np.testing.assert_array_equal(s['measured_yaw'],y)


def test_no_sideslip_channel_or_metadata_accepted():
    _,cfg,raw=fixture(n=60)
    with pytest.raises(ValueError):validated_segments(raw)
    s=yaw_segments(raw)[0];s['true_beta']=raw[0]['measured_state'][:,0]
    with pytest.raises(ValueError):fit_yaw_oe([s],cfg)
    del s['true_beta'];s['measured_yaw']=raw[0]['measured_state']
    with pytest.raises(ValueError):fit_yaw_oe([s],cfg)


def test_validation_suffix_cannot_affect_predictions():
    theta,cfg,raw=fixture(noise=1.,n=500)
    s=yaw_segments(raw)[0];pred,_=validation_prediction(theta,s,cfg,200)
    s['measured_yaw'][201:]+=100
    changed,_=validation_prediction(theta,s,cfg,200)
    np.testing.assert_array_equal(pred,changed)


def test_noiseless_single_speed_has_physical_ambiguity():
    theta,cfg,raw=fixture(noise=0.,n=500,speeds=(20.,))
    s=yaw_segments(raw)[0];alt=theta.copy();alt[3]=1.2
    factor=lambda mu:1-(.005*20**2/(9.81*mu))**2
    alt[:2]*=factor(theta[3])/factor(alt[3])
    r,_,_=projected_response(np.log(theta),s,cfg)
    ra,_,_=projected_response(np.log(alt),s,cfg)
    np.testing.assert_allclose(r,ra,atol=1e-8)


def test_unobservable_hidden_state_uses_rank_revealing_projection():
    theta,cfg,raw=fixture(noise=0.,n=100,speeds=(20.,))
    theta[1]=cfg['geometry_m'][0]/cfg['geometry_m'][1]*theta[0]
    s=yaw_segments(raw)[0]
    residual,jac,record=projected_response(np.log(theta),s,cfg)
    assert record['nuisance_rank']==2
    assert np.isfinite(residual).all() and np.isfinite(jac).all()


def test_preprocessing_and_fit_boundary_ignore_all_beta_values(monkeypatch):
    import paper_physical_ct_yaw_only as runner
    theta,cfg,raw=fixture(noise=1.,n=2000,speeds=(20.,))
    r=raw[0];state=np.zeros((4401,2));state[2400:]=r['measured_state']
    steering=np.zeros(4400);steering[2400:]=r['applied_input']
    trace=dict(observed_state=state.copy(),state=state.copy(),steering=steering)
    monkeypatch.setattr(runner.old,'traces',lambda *args:( (trace,trace),False))
    received=[]
    def fitted(segments,settings):
        validated_segments(segments)
        received.append(segments[0]['measured_yaw'].copy())
        return dict(converged=True,parameters=theta.tolist())
    monkeypatch.setattr(runner,'fit_yaw_oe',fitted)
    params=dict(mass=1500.,yaw_inertia=2500.,front_length=1.2,rear_length=1.6,front_stiffness=80000.,rear_stiffness=80000.)
    source=dict(client='synthetic',speeds=[20.],parameters=params,reference=runner.model_grid(theta,range(10,31,2),cfg).tolist())
    acquired=dict(cells=[dict(client_index=0,speed=20.,training_measurements_sha256='synthetic')])
    full=dict(fit=dict(parameters=theta.tolist()));config=runner.old.load(runner.CONFIG)
    before=runner.evaluate(1301,0,source,acquired,full,config,False)
    trace['observed_state'][:,0]=np.nan;trace['state'][:,0]=np.nan
    after=runner.evaluate(1301,0,source,acquired,full,config,False)
    np.testing.assert_array_equal(received[0],received[1])
    assert before['validation']==after['validation']
    assert before['comparisons']==after['comparisons']


def test_analysis_preserves_failed_fits_and_renders_html(tmp_path,monkeypatch):
    import paper_physical_ct_yaw_only as runner
    cfg=runner.old.load(runner.CONFIG)
    monkeypatch.setattr(runner,'OUT',tmp_path)
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    monkeypatch.setattr(runner,'REPORT',tmp_path/'report.html')
    monkeypatch.setattr(runner,'validate',lambda *args:None)
    for seed in cfg['seeds']:
        errors={k:[.01]*11 for k in ['matrix','increment','a','b','pole_rms_absolute']}
        comparisons={m:dict(errors=errors,scaled_increment=[.02]*11,entry_absolute=[[.001]*6]*11) for m in runner.METHODS}
        failed=seed==cfg['seeds'][0]
        if failed:comparisons.pop('YawOnlyCT')
        result=dict(seed=seed,status='execution_complete',clients=[dict(index=0,client='synthetic',observed_speeds=[12.,28.],
            elapsed_s=0.,fit=dict(converged=not failed,parameters=[47.,59.,.56,1.02],information=dict(log_parameter_singular_values=[4.,3.,2.,1.])),
            comparisons=comparisons,validation=[dict(method=m,speed=v,status='complete',measured_yaw_rmse_deg_s=.03,true_yaw_rmse_deg_s=.01)
                for m in comparisons for v in [12.,28.]])])
        runner.atomic_json(runner.path(seed),result)
    runner.analyze(cfg)
    import pandas as pd
    summary=pd.read_csv(tmp_path/f'{runner.PREFIX}_summary.csv').set_index('method')
    assert np.isnan(summary.loc['YawOnlyCT','increment'])
    assert np.isfinite(summary.loc['FullStateCT','increment'])
    text=(tmp_path/'report.html').read_text(encoding='utf-8')
    assert text.count('<svg')==1 and '2/3 yaw-only fits converged' in text
