"""Model structure, singular cases, sensitivity and checkpoint safety."""
import copy
from pathlib import Path
import sys
import numpy as np
import pytest
from scipy.linalg import expm

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'code/src'),str(ROOT/'code/experiments')]
from federated_lpv.tire_relaxation import (rhs,simulate,continuous_matrices,observability,
    trajectory_sensitivities,distinguishability)
from federated_lpv.physical_ct_oe import continuous_matrices as static_matrices
import paper_tire_relaxation_gate as run

P=np.array([50.,55.,.6,.9,.4,.7]);G=[1.2,1.6]


def test_origin_linearization_matches_independent_rhs_derivative():
    a,b=continuous_matrices(P,21,G);epsilon=1e-25
    for j in range(4):
        x=np.zeros(4,complex);x[j]=1j*epsilon
        np.testing.assert_allclose(rhs(x,0,21,P,G).imag/epsilon,a[:,j],atol=1e-12)
    np.testing.assert_allclose(rhs(np.zeros(4),1j*epsilon,21,P,G).imag/epsilon,b,atol=1e-12)


def test_eliminating_quasistatic_force_states_recovers_existing_ct_model():
    a,b=continuous_matrices(P,21,G)
    reduced=a[:2,:2]-a[:2,2:]@np.linalg.solve(a[2:,2:],a[2:,:2])
    input_reduced=b[:2]-a[:2,2:]@np.linalg.solve(a[2:,2:],b[2:])
    expected_a,expected_b=static_matrices(P[:4],21,G,0)
    np.testing.assert_allclose(reduced,expected_a,atol=1e-12)
    np.testing.assert_allclose(input_reduced,expected_b,atol=1e-12)


def test_equal_time_constant_observability_determinant_and_degeneracy():
    p=P.copy();p[4:]=.5;v=21.;tau=.5/v;a,_=continuous_matrices(p,v,G);c=np.array([[0,1,0,0]])
    o=np.vstack([c@np.linalg.matrix_power(a,j) for j in range(4)])
    expected=sum(G)*p[2]**3*(G[0]*p[0]-G[1]*p[1])**2/(tau**2*v)
    np.testing.assert_allclose(np.linalg.det(o),expected,rtol=1e-12)
    p[1]=p[0]*G[0]/G[1]
    assert observability(p,v,G,.01,2.,[.01,.03,2.,2.],.001)['rank']==2
    p[1]*=1.1
    assert observability(p,v,G,.01,2.,[.01,.03,2.,2.],.001)['rank']==4


def test_normalized_rhs_matches_physical_force_equations():
    from federated_lpv.vehicle import VehicleParameters,nonlinear_tire_forces
    m=1500.;plant=VehicleParameters(m,m/P[2],*G,m*P[0],m*P[1])
    x=np.array([.01,.04,.5,.8]);delta=.02;v=21.
    ff,fr,*_=nonlinear_tire_forces(x[:2],delta,v,plant,P[3])
    expected=np.array([(m*x[2]+m*x[3])/(m*v)-x[1],
        (G[0]*m*x[2]-G[1]*m*x[3])/plant.yaw_inertia,
        v/P[4]*(ff/m-x[2]),v/P[5]*(fr/m-x[3])])
    np.testing.assert_allclose(rhs(x,delta,v,P,G),expected,rtol=1e-14)


def test_small_signal_rk4_matches_exact_linear_zoh():
    p=P;a,b=continuous_matrices(p,21,G);z=np.zeros((5,5));z[:4,:4]=a;z[:4,4]=b;z=expm(.01*z)
    u=1e-7*np.sin(np.arange(100)*.07);x=np.zeros(4);linear=[x.copy()]
    for delta in u:x=z[:4,:4]@x+z[:4,4]*delta;linear.append(x.copy())
    actual=simulate(p,u,21,G,max_step=.0005)[0]
    np.testing.assert_allclose(actual,linear,atol=2e-12,rtol=1e-6)


def test_sensitivities_match_finite_difference_for_parameters_and_initial_state():
    u=.003*np.sin(np.arange(80)*.08);scale=.0005;scales=np.array([.01,.03,2.,2.])
    _,jac=trajectory_sensitivities(P,u,21,G,.01,.001,scales,scale)
    h=1e-4
    for j in [0,3,4,5]:
        plus=P.copy();minus=P.copy();plus[j]*=np.exp(h);minus[j]*=np.exp(-h)
        y=simulate([plus,minus],u,21,G)[:,:,1]
        np.testing.assert_allclose(jac[:,j],(y[0]-y[1])/(2*h*scale),atol=2e-6,rtol=2e-5)
    init=np.zeros((2,4));init[0,2]=h*scales[2];init[1,2]=-h*scales[2]
    y=simulate([P,P],u,21,G,initial=init)[:,:,1]
    np.testing.assert_allclose(jac[:,8],(y[0]-y[1])/(2*h*scale),atol=1e-5,rtol=2e-5)


def test_zero_input_carries_no_parameter_information():
    _,jac=trajectory_sensitivities(P,np.zeros(50),21,G,.01,.001,[.01,.03,2.,2.],.001)
    assert np.count_nonzero(jac[:,:6])==0
    assert distinguishability(jac)['conditional']['rank']==0


def test_nuisance_projection_does_not_turn_roundoff_into_information():
    rng=np.random.default_rng(1);jac=rng.normal(size=(100,10));jac[:,4]=jac[:,0];jac[:,5]=jac[:,1]
    info=distinguishability(jac)
    assert info['conditional']['rank']==2
    assert info['unknown_ct']['rank']==0


def test_integration_step_and_prefix_causality():
    with pytest.raises(ValueError,match='integration step'):
        simulate([50,55,.6,.9,.01,.01],np.zeros(10),29,G,max_step=.01)
    u=.004*np.sin(np.arange(80)*.07)
    whole=simulate(P,u,21,G);prefix=simulate(P,u[:40],21,G)
    np.testing.assert_array_equal(whole[:,:41],prefix)


def test_windows_safe_checkpoint_and_hash_validation(tmp_path):
    path=tmp_path/'x.json';run.atomic_json(path,dict(note='β μ → UTF-8'))
    assert run.read_checkpoint(path)['note']=='β μ → UTF-8'
    text=path.read_text(encoding='utf-8').replace('UTF-8','corrupt');path.write_text(text,encoding='utf-8')
    with pytest.raises(ValueError,match='digest'):run.read_checkpoint(path)


def test_exact_restore_refuses_changed_arrays(tmp_path,monkeypatch):
    monkeypatch.setattr(run,'RAW',tmp_path)
    cfg=run.load_config();job=run.jobs(cfg)[0];arrays={'state':np.zeros((3,4))}
    saved={'trace_sha256':run.trace_hash(arrays)}
    monkeypatch.setattr(run,'generate',lambda *args:({},arrays))
    run.restore(job,cfg,saved)
    run.save_arrays(tmp_path/f'{run.key(job)}.npz',{'state':np.ones((3,4))})
    with pytest.raises(ValueError,match='Original preserved'):run.restore(job,cfg,saved)


def test_complete_resume_and_protocol_change_rejection(tmp_path,monkeypatch):
    monkeypatch.setattr(run,'OUT',tmp_path)
    cfg=run.preflight();job=run.jobs(cfg)[0]
    p=dict(job=list(job),provenance=run.identity(cfg),status='execution_complete',data={},sensitivity={},surfaces={'true_ct':{},'previous_ct':{}})
    run.atomic_json(run.path(job),p)
    monkeypatch.setattr(run,'generate',lambda *a:pytest.fail('Completed cases must not rerun'))
    assert run.worker(job)['status']=='already_complete'
    p['provenance']['files']['wrong']='hash';run.atomic_json(run.path(job),p)
    with pytest.raises(ValueError,match='protocol'):run.worker(job)


def test_conditional_surface_and_utf8_html_rendering(tmp_path,monkeypatch):
    import pandas as pd
    cfg=copy.deepcopy(run.load_config());cfg['grid_points']=3
    u=.004*np.sin(np.arange(50)*.1)
    x=simulate(P,u,21,G)[0]
    arrays=dict(steering=u,state=x,measured_yaw=x[:,1]+.0005*np.sin(np.arange(51)))
    grid=run.surface(P[:4],arrays,21,G,cfg)
    assert grid['objectives']['clean']['minimum_grid_sse']<1e-20
    assert grid['objectives']['clean']['best_grid_lengths_m']==[.4,.7]
    record=dict(job=[cfg['clients'][0]['client'],21,'slow'],surfaces={'true_ct':grid,'previous_ct':grid})
    rows=[dict(client='synthetic',speed=v,probe=probe,assumption=assumption,smallest_singular=1.)
        for v in cfg['speeds_mps'] for probe in cfg['inputs'] for assumption in ['conditional','unknown_ct','unknown_ct_and_initial']]
    output=tmp_path/'smoke.html';monkeypatch.setattr(run,'REPORT',output)
    run.render([record],pd.DataFrame([{'rank':4}]),pd.DataFrame(rows),cfg,run.negative_controls(cfg))
    html=output.read_text(encoding='utf-8')
    assert html.count('<svg')==3 and 'β' in html and 'confidence intervals' in html


def test_worst_grid_corner_step_halving_is_below_sensor_resolution():
    p=P.copy();p[4:]=.15
    u=np.deg2rad(.35)*np.sin(np.arange(200)*.01*2*np.pi*1.2)
    coarse=simulate(p,u,29,G,max_step=.001)[0]
    fine=simulate(p,u,29,G,max_step=.0005)[0]
    assert np.max(abs(coarse[:,1]-fine[:,1]))<np.deg2rad(.03)*1e-3
