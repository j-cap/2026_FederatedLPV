"""Population physics, fixed-group information control, and resumable orchestration."""
import copy
from pathlib import Path
import sys
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'code/src'),str(ROOT/'code/experiments')]
import paper_cfl_heterogeneity as run
from federated_lpv import cfl_population as pop


@pytest.fixture(scope='module')
def cfg():return run.preflight()


def test_population_slope_and_payload_physics(cfg):
    spec=copy.deepcopy(cfg['population'])
    s=spec['variation'];s.update(family_log_sd=0.,client_log_sd=0.,BCE_additive_sd=[0.,0.,0.],
        payload_fraction=[.2,.2],payload_position_wheelbase=[.1,.1],tire_condition=[1.,1.])
    a=spec['anchors']['Mini'];m=a['mass_kg'];lf,lr=a['geometry_m'];x=.1*(lf+lr)
    draw=pop.draw_vehicle('Mini',np.random.default_rng(1),spec)
    new_lf,new_lr=draw['geometry_m'];mass=1.2*m
    np.testing.assert_allclose(new_lf+new_lr,lf+lr)
    np.testing.assert_allclose(new_lf,lf-.2*x/1.2)
    np.testing.assert_allclose(draw['physical_draw']['inertia_kg_m2'],a['inertia_kg_m2']+m*.2/1.2*x*x)
    for i,length in enumerate([new_lr,new_lf]):
        b,c,_=a['lateral_BCE'][i]
        expected=b*c*.9*9.81*length/(lf+lr)
        np.testing.assert_allclose(draw['physical_parameters'][i],expected)
        np.testing.assert_allclose(draw['physical_draw']['stiffness_N_rad'][i],mass*expected)


def test_family_and_mixed_groups_have_matched_size_and_coverage(cfg):
    for fleet in cfg['fleets']:
        groups=run.blocks(fleet)
        assert len(groups)==7
        for prefix in ['OracleFamily','MixedGroups']:
            ids=[]
            for key,members in groups.items():
                if not key.startswith(prefix):continue
                assert len(members)==6
                assert sorted(c['training_speed'] for c in members)==[13,13,21,21,29,29]
                families=[c['family'] for c in members]
                assert len(set(families))==(1 if prefix=='OracleFamily' else 3)
                ids.extend(c['client'] for c in members)
            assert len(ids)==len(set(ids))==18
    # Nominal center depends on the independent population design, not fleet realizations.
    other=pop.generate_fleet(9999,cfg['population'],cfg['fleet_training_speeds'])
    assert other['clients']!=cfg['fleets'][0]['clients']
    np.testing.assert_allclose(pop.nominal_center(cfg['population']),cfg['nominal_parameters'])


def test_progression_requires_all_controls_and_valid_convergence(cfg):
    import pandas as pd
    values=pd.DataFrame({'LocalOE':[2.]*3,'NominalPrior':[1.]*3,
        'SharedPrior':[1.]*3,'OracleFamily':[.8]*3,'MixedGroups':[1.]*3})
    assert run.progression(values,True,True,cfg)['status']=='supports_testing_learned_groups'
    assert run.progression(values,False,True,cfg)['status']=='no_automatic_progression'
    assert run.progression(values,True,False,cfg)['status']=='no_automatic_progression'
    values['MixedGroups']=.79
    assert run.progression(values,True,True,cfg)['status']=='no_automatic_progression'


def test_writer_lock_rejects_collision_and_releases(tmp_path,monkeypatch):
    monkeypatch.setattr(run,'RAW',tmp_path/'raw')
    pth=tmp_path/'tables'/'checkpoint.json'
    with run.task_lock(pth):
        with pytest.raises(OSError):
            with run.task_lock(pth):pass
    with pytest.raises(RuntimeError):
        with run.task_lock(pth):raise RuntimeError('crash simulation')
    with run.task_lock(pth):pass


def test_checkpoint_resume_restore_and_analysis(tmp_path,monkeypatch,cfg):
    """Real data generation/evaluation and checkpointing, stub nonlinear optimizer."""
    cfg=copy.deepcopy(cfg);cfg['fleets']=cfg['fleets'][:1];cfg['seeds']=[cfg['fleets'][0]['seed']]
    cfg['duration_s']=.12;cfg['taper_s']=.03;cfg['evaluation_speeds']=[13,21,29]
    cfg['sharing']['personal_max_rounds']=2
    monkeypatch.setattr(run,'OUT',tmp_path/'tables');monkeypatch.setattr(run,'RAW',tmp_path/'raw')
    monkeypatch.setattr(run,'REPORT',tmp_path/'report.html')
    calls=[]
    def fit(self,start,center=None,weights=None,max_nfev=None):
        calls.append(1);q=np.asarray(start);w=np.zeros(6) if weights is None else np.asarray(weights)
        eta=q if center is None else np.asarray(center);penalty=float(np.sum((w*(q-eta))**2))
        return dict(available=True,optimizer_converged=True,log_parameters=q.tolist(),parameters=np.exp(q).tolist(),
            objective=1+penalty,data_sse=1.,prior_penalty=penalty,initial_objective=1+penalty,
            nfev=1,optimality=0.,near_bound=[False]*6)
    monkeypatch.setattr(run.fl.Client,'fit',fit)
    provenance=run.identity();fleet=cfg['fleets'][0];seed=fleet['seed']
    for c in fleet['clients']:run.local_task(seed,c['client'],cfg,provenance)
    for key in run.blocks(fleet):run.group_task(seed,key,cfg,provenance)
    for c in fleet['clients']:run.eval_task(seed,c['client'],cfg,provenance)
    before=len(calls);key=fleet['clients'][0]['client']
    run.local_task(seed,key,cfg,provenance);run.group_task(seed,'SharedPrior_all',cfg,provenance)
    pth=run.path(seed,'eval',key);p=run.storage.read_checkpoint(pth)
    p['cases'].pop('v13_slow');p['status']='running';run.storage.atomic_json(pth,p)
    run.eval_task(seed,key,cfg,provenance)
    assert len(calls)==before, 'Resume must not rerun completed fits'
    # Missing ignored raw arrays are rebuilt without changing committed checkpoints.
    hashes={p:run.base.canonical_hash(p) for p in run.OUT.glob('*.json')}
    target=run.RAW/str(seed)/f'{key}_training.npz';target.unlink()
    run.restore(cfg);assert target.exists()
    assert hashes=={p:run.base.canonical_hash(p) for p in hashes}
    with monkeypatch.context() as context:
        context.setattr(run,'ROOT',tmp_path)
        context.setattr(run,'identity',lambda:provenance)
        run.analyze(cfg)
    text=run.REPORT.read_text(encoding='utf-8')
    assert text.count('<svg')==2 and 'not an upper bound' in text
    # Changing frozen parent evidence must block stale evaluation reuse.
    p=run.storage.read_checkpoint(run.path(seed,'local',key));p['tampered']=True
    run.storage.atomic_json(run.path(seed,'local',key),p)
    with pytest.raises(ValueError,match='parent'):run.eval_task(seed,key,cfg,provenance)


def test_partial_group_round_uses_saved_client_updates(tmp_path,monkeypatch,cfg):
    """A crash after one client update must not recompute that optimizer call."""
    cfg=copy.deepcopy(cfg);cfg['duration_s']=.12;cfg['taper_s']=.03
    cfg['sharing']['personal_max_rounds']=1
    monkeypatch.setattr(run,'OUT',tmp_path/'tables');monkeypatch.setattr(run,'RAW',tmp_path/'raw')
    provenance=run.identity();seed=cfg['seeds'][0];key='OracleFamily_Mini'
    members=run.blocks(run.fleet_for(cfg,seed))[key]
    q=np.log(cfg['nominal_parameters']).tolist()
    def fit(self,start,center=None,weights=None,max_nfev=None):
        return dict(available=True,optimizer_converged=True,log_parameters=q,parameters=np.exp(q).tolist(),
            objective=1.,data_sse=1.,prior_penalty=0.,initial_objective=1.,nfev=1,optimality=0.,near_bound=[False]*6)
    monkeypatch.setattr(run.fl.Client,'fit',fit)
    for c in members:run.local_task(seed,c['client'],cfg,provenance)
    saved=run.save
    def interrupt(pth,p,stage):
        saved(pth,p,stage)
        if stage=='round1_client0':raise RuntimeError('simulated interruption')
    monkeypatch.setattr(run,'save',interrupt)
    with pytest.raises(RuntimeError,match='simulated'):run.group_task(seed,key,cfg,provenance)
    monkeypatch.setattr(run,'save',saved);calls=[]
    def counted(*args,**kwargs):calls.append(1);return fit(*args,**kwargs)
    monkeypatch.setattr(run.fl.Client,'fit',counted)
    run.group_task(seed,key,cfg,provenance)
    assert len(calls)==len(members)-1
