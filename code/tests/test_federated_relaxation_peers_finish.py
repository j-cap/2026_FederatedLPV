"""Only the unfinished conditions may change; exact resume and restore."""
import copy
from pathlib import Path
import sys
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code/experiments'))
import paper_federated_relaxation_peers_finish as run
base=run.base


def test_two_conditions_source_lock_and_exact_initial_state():
    cfg,protocol=run.preflight()
    for seed in protocol['seeds']:
        original,_,previous=run.sources(seed,cfg,protocol);before=copy.deepcopy(previous)
        p=run.initialize(seed,previous,cfg,protocol)
        expected=copy.deepcopy(previous['conditions']['13']['optimization']);expected['status']='running'
        assert p['optimization']==expected and previous==before
        assert len(run.affected_keys(original,cfg,seed))==42
        run.validate(p,original,previous,cfg,protocol)
        bad=copy.deepcopy(p);bad['optimization']['history'][0]['objective']+=1
        with pytest.raises(AssertionError):run.validate(bad,original,previous,cfg,protocol)
    with pytest.raises(ValueError,match='Only fleets'):run.initialize(1303,previous,cfg,protocol)
    bad=copy.deepcopy(protocol);bad['source_checkpoints'][run.peers.path(1301).relative_to(ROOT).as_posix()]='corrupt'
    with pytest.raises(ValueError,match='checkpoint changed'):run.sources(1301,cfg,bad)


def test_training_hash_failure_preserves_file(tmp_path,monkeypatch):
    cfg,protocol=run.preflight();original,_,previous=run.sources(1301,cfg,protocol);c=cfg['fleets'][0]['clients'][0]
    monkeypatch.setattr(run.peers,'RAW',tmp_path)
    f=tmp_path/'1301'/f'{run.peers.record_key(c,13)}.npz';base.storage.save_arrays(f,dict(noise=np.zeros(2)))
    before=f.read_bytes()
    with pytest.raises(ValueError,match='training hash mismatch'):run.record(original,previous,c,cfg)
    assert f.read_bytes()==before


def test_partial_resume_reuses_seven_conditions_and_restores(tmp_path,monkeypatch):
    cfg,protocol=run.preflight();cfg=copy.deepcopy(cfg);cfg['duration_s']=.15;cfg['evaluation_speeds']=[13,21]
    for fleet in cfg['fleets']:
        old=fleet['clients'];fleet['clients']=[next(c for c in old if c['training_speed']==v) for v in [13,21,29]]
    monkeypatch.setattr(base,'RAW',tmp_path/'originalraw');monkeypatch.setattr(run.peers,'RAW',tmp_path/'peerraw')
    monkeypatch.setattr(run,'RAW',tmp_path/'finishraw');monkeypatch.setattr(run,'OUT',tmp_path/'tables');monkeypatch.setattr(run,'REPORT',tmp_path/'report.html')
    q=np.log(cfg['nominal_parameters']).tolist();bundles={}
    for fleet in cfg['fleets']:
        seed=fleet['seed'];clients=fleet['clients']
        original=dict(seed=seed,data={},local={},nominal_prior={},evaluation={},selected={
            'FedPersonal':dict(available=True,log_parameters=[q]*3),'Global':dict(available=False),'FedGroups':dict(available=False)})
        previous=dict(seed=seed,records={},conditions={},evaluation={});complementary=dict(evaluation={})
        for speed in [13,21,29]:
            state=base.fl.initialize_personal([q]*3,1,0,cfg['sharing']['weights'])
            end=30 if seed in protocol['seeds'] and speed==13 else 2
            for j in range(end):
                state=base.fl.finish_personal_round(state,[dict(log_parameters=q,data_sse=1.)]*3,cfg['sharing']['weights'],dict(cfg['sharing'],personal_relative_tolerance=0,personal_max_rounds=30),True)
            state['status']='budget_exhausted' if end==30 else 'converged'
            previous['conditions'][str(speed)]=dict(status=state['status'],optimization=state)
        for i,c in enumerate(clients):
            local=dict(available=True,optimizer_converged=True,parameters=cfg['nominal_parameters'])
            original['local'][c['client']]=dict(selected=local);original['nominal_prior'][c['client']]=local
            a,meta=base.generate(c,seed,'training',c['training_speed'],cfg);original['data'][c['client']]=meta
            base.storage.save_arrays(base.RAW/str(seed)/f'{c["client"]}_training.npz',a)
            a,meta=run.peers.make_record(original,c,13,cfg);key=run.peers.record_key(c,13)
            previous['records'][key]=meta;base.storage.save_arrays(run.peers.RAW/str(seed)/f'{key}.npz',a)
            for v in cfg['evaluation_speeds']:
                for probe in ['slow','broadband']:
                    key=f'{c["client"]}_v{v}_{probe}';entry,a=base.evaluation_case(original,c,i,v,probe,cfg)
                    original['evaluation'][key]=entry
                    base.storage.save_arrays(base.RAW/str(seed)/f'{key}.npz',a)
                    complementary['evaluation'][key]=dict(metrics=entry['methods']['FedPersonal'])
                    previous['evaluation'][key]=dict(metrics=entry['methods']['FedPersonal'],source_trace_sha256=entry['trace_sha256'])
        bundles[seed]=(original,complementary,previous)
    before=copy.deepcopy(bundles);files={str(f):base.storage.file_hash(f) for folder in [base.RAW,run.peers.RAW] for f in folder.rglob('*.npz')}
    monkeypatch.setattr(run,'preflight',lambda execution=False:(cfg,protocol))
    monkeypatch.setattr(run,'sources',lambda seed,*args,**kwargs:bundles[seed])
    calls=[]
    def fit(self,start,center=None,weights=None,max_nfev=None):
        assert center is not None  # no new initial/local-only fit
        calls.append(list(start));return dict(available=True,optimizer_converged=True,log_parameters=list(start),data_sse=1.,nfev=1,optimality=0.,near_bound=[False]*6)
    monkeypatch.setattr(base.fl.Client,'fit',fit)
    save=run.save
    def interrupt(p,stage):
        save(p,stage)
        if stage=='round31_client0':raise InterruptedError('test interruption')
    monkeypatch.setattr(run,'save',interrupt)
    with pytest.raises(InterruptedError):run.worker(1301)
    assert len(calls)==1
    monkeypatch.setattr(run,'save',save)
    for seed in protocol['seeds']:run.worker(seed)
    assert len(calls)==6 and not run.path(1303).exists()
    for seed in protocol['seeds']:
        p=base.storage.read_checkpoint(run.path(seed));assert len(p['evaluation'])==4
        assert p['optimization']['round']==31 and p['optimization']['status']=='converged'
    p=base.storage.read_checkpoint(run.path(1301));p['status']='running';p['evaluation'].pop(next(iter(p['evaluation'])));base.storage.atomic_json(run.path(1301),p)
    run.worker(1301);assert len(calls)==6
    f=next((run.RAW/'1301').glob('*.npz'));f.unlink();checkpoint=base.storage.file_hash(run.path(1301))
    run.restore(cfg,protocol,[1301]);assert f.exists() and base.storage.file_hash(run.path(1301))==checkpoint
    run.analyze(cfg,protocol)
    import pandas as pd
    table=pd.read_csv(run.OUT/f'{run.PREFIX}_optimization.csv');assert table.reused.sum()==7
    html=run.REPORT.read_text(encoding='utf-8');assert html.count('<svg')==1 and '\ufffd' not in html
    assert bundles==before and files=={str(f):base.storage.file_hash(f) for folder in [base.RAW,run.peers.RAW] for f in folder.rglob('*.npz')}
