"""Continuation isolation, total budget, partial-round resume and restoration."""
import copy
from pathlib import Path
import sys
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code/experiments'))
import paper_federated_relaxation_continue as run
base=run.base


def fixed_fit(q):
    return dict(available=True,optimizer_converged=True,log_parameters=list(q),data_sse=1.,
        nfev=1,optimality=0.,near_bound=[False]*6)


def fixture_state(cfg,converged=False):
    q=np.log(cfg['nominal_parameters']).tolist()
    state=base.fl.initialize_personal([q,q],1,0,cfg['sharing']['weights'])
    for _ in range(8):
        state=base.fl.finish_personal_round(state,[base.fl.client_round_message(fixed_fit(q))]*2,
            cfg['sharing']['weights'],dict(cfg['sharing'],personal_relative_tolerance=0),True)
    state['status']='converged' if converged else 'budget_exhausted'
    return state


def test_source_locks_and_only_round_cap_changes():
    cfg,protocol=run.preflight()
    assert protocol['total_round_cap']==30
    originals=[run.load_source(s,cfg,protocol) for s in protocol['seeds']]
    assert [run.source_state(s)['status'] for s in originals]==['budget_exhausted','budget_exhausted','converged']
    for original in originals:
        before=copy.deepcopy(original);p=run.initialize(original,cfg,protocol)
        assert original==before
        expected=copy.deepcopy(run.source_state(original))
        if expected['status']=='budget_exhausted':expected['status']='running'
        assert p['optimization']==expected
        run.validate(p,original,cfg,protocol)
    bad=copy.deepcopy(protocol);name=next(iter(bad['source_checkpoints']));bad['source_checkpoints'][name]='corrupt'
    with pytest.raises(ValueError,match='checkpoint changed'):run.load_source(1301,cfg,bad)


def test_resume_partial_round_matches_uninterrupted(tmp_path):
    cfg,_=run.preflight();state=fixture_state(cfg);state['status']='running'
    p=dict(optimization=copy.deepcopy(state));direct=copy.deepcopy(p);calls=[0,0]
    class Endpoint:
        def __init__(self,index):self.index=index
        def fit(self,q,*args,**kwargs):calls[self.index]+=1;return fixed_fit(q)
    clients=[Endpoint(0),Endpoint(1)]
    saved=tmp_path/'partial.json'
    def interrupt(p,stage):
        base.storage.atomic_json(saved,p)
        if stage.endswith('client0'):raise InterruptedError('synthetic interruption')
    with pytest.raises(InterruptedError):run.continue_rounds(p,clients,cfg,dict(total_round_cap=30),interrupt)
    assert calls==[1,0]
    p=base.storage.read_checkpoint(saved)
    run.continue_rounds(p,clients,cfg,dict(total_round_cap=30),lambda *args:None)
    assert calls==[1,1] and p['optimization']['status']=='converged'
    run.continue_rounds(direct,clients,cfg,dict(total_round_cap=30),lambda *args:None)
    assert p==direct


def test_budget_counts_original_rounds_and_failure_is_saved():
    cfg,_=run.preflight();cfg=copy.deepcopy(cfg);cfg['sharing']['personal_relative_tolerance']=0
    state=fixture_state(cfg);state['status']='running';p=dict(optimization=state)
    class Endpoint:
        def fit(self,q,*args,**kwargs):return fixed_fit(q)
    run.continue_rounds(p,[Endpoint(),Endpoint()],cfg,dict(total_round_cap=30),lambda *args:None)
    assert p['optimization']['round']==30 and len(p['optimization']['history'])==30
    assert p['optimization']['status']=='budget_exhausted'
    p=dict(optimization=fixture_state(cfg));p['optimization']['status']='running';saved=[]
    class Failed:
        def fit(self,*args,**kwargs):return dict(available=False,optimizer_converged=False)
    with pytest.raises(RuntimeError,match='Local update failed'):
        run.continue_rounds(p,[Failed(),Endpoint()],cfg,dict(total_round_cap=30),lambda p,stage:saved.append(stage))
    assert saved[-1]=='local_update_failed' and p['optimization']['pending']


def test_hash_mismatch_rejected_without_writes(tmp_path,monkeypatch):
    cfg,_=run.preflight();source=run.load_source(1301,*run.preflight())
    c=base.fleet_for(cfg,1301)['clients'][0]
    monkeypatch.setattr(base,'RAW',tmp_path)
    monkeypatch.setattr(base,'generate',lambda *args:({'steering':np.zeros(2)},{}))
    with pytest.raises(ValueError,match='training trace hash mismatch'):run.training_record(source,c,cfg)
    assert not list(tmp_path.iterdir())


def test_small_workflow_preserves_sources_reuses_baselines_and_restores(tmp_path,monkeypatch):
    cfg,protocol=run.preflight();cfg=copy.deepcopy(cfg)
    cfg['duration_s']=.15;cfg['evaluation_speeds']=[13]
    for fleet in cfg['fleets']:fleet['clients']=fleet['clients'][:2]
    monkeypatch.setattr(run,'OUT',tmp_path/'continued');monkeypatch.setattr(run,'RAW',tmp_path/'newraw')
    monkeypatch.setattr(base,'RAW',tmp_path/'originalraw');monkeypatch.setattr(run,'REPORT',tmp_path/'report.html')
    sources={};original_files=[]
    for fleet in cfg['fleets']:
        seed=fleet['seed'];state=fixture_state(cfg,converged=seed==1303)
        source=dict(seed=seed,sharing_attempts={'FedPersonal':{'0':state}},
            selected={'FedPersonal':dict(available=True,selected_attempt='0',log_parameters=state['log_parameters']),
                      'Global':dict(available=False),'FedGroups':dict(available=False)},local={},nominal_prior={},data={},evaluation={})
        for i,c in enumerate(fleet['clients']):
            local=dict(available=True,optimizer_converged=True,parameters=cfg['nominal_parameters'])
            source['local'][c['client']]=dict(selected=local);source['nominal_prior'][c['client']]=local
            arrays,metadata=base.generate(c,seed,'training',c['training_speed'],cfg);source['data'][c['client']]=metadata
            file=base.RAW/str(seed)/f'{c["client"]}_training.npz';base.storage.save_arrays(file,arrays);original_files.append(file)
            for probe in ['slow','broadband']:
                key=f'{c["client"]}_v13_{probe}';entry,arrays=base.evaluation_case(source,c,i,13,probe,cfg)
                source['evaluation'][key]=entry
                file=base.RAW/str(seed)/f'{key}.npz';base.storage.save_arrays(file,arrays);original_files.append(file)
        sources[seed]=source
    original_hashes={str(p):base.storage.file_hash(p) for p in original_files};before=copy.deepcopy(sources)
    monkeypatch.setattr(run,'preflight',lambda execution=False:(cfg,protocol))
    monkeypatch.setattr(run,'load_source',lambda seed,*args,**kwargs:sources[seed])
    calls=[]
    def fit(self,q,*args,**kwargs):calls.append(1);return fixed_fit(q)
    monkeypatch.setattr(base.fl.Client,'fit',fit)
    for seed in protocol['seeds']:run.worker(seed)
    assert len(calls)==4  # two unfinished fleets, two clients, one added round
    assert sources==before
    assert original_hashes=={str(p):base.storage.file_hash(p) for p in original_files}
    p=base.storage.read_checkpoint(run.path(1303));assert all(e['reused'] for e in p['evaluation'].values())
    assert p['optimization']==run.source_state(sources[1303])
    p=base.storage.read_checkpoint(run.path(1301));p['status']='running';p['evaluation'].pop(next(iter(p['evaluation'])))
    base.storage.atomic_json(run.path(1301),p);run.worker(1301);assert len(calls)==4
    target=next((run.RAW/'1301').glob('*.npz'));target.unlink()
    run.restore(cfg,protocol,[1301]);assert target.exists()
    # Container timestamps in NPZ can differ; the restore itself checks numerical bytes.
    checkpoint_hash=base.storage.file_hash(run.path(1301));run.restore(cfg,protocol,[1301])
    assert checkpoint_hash==base.storage.file_hash(run.path(1301))
    run.analyze(cfg,protocol);html=run.REPORT.read_text(encoding='utf-8')
    assert html.count('<svg')==1 and '\ufffd' not in html
    assert original_hashes=={str(p):base.storage.file_hash(p) for p in original_files}
