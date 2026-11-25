"""Matched noise/input, target isolation, resume and unavailable-case accounting."""
import copy
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code/experiments'))
import paper_federated_relaxation_peers as run
base=run.base


def test_protocol_and_source_integrity():
    cfg,p=run.preflight()
    assert p['same_speeds']==[13,21,29] and p['total_round_cap']==30
    for seed in p['seeds']:
        source,final=run.sources(seed,cfg,p)
        assert final['optimization']['status']=='converged'
        fleet=base.fleet_for(cfg,seed)
        for target in fleet['clients']:
            peers=[c for c in fleet['clients'] if c['client']!=target['client']]
            assert len(peers)==8 and sum(c['training_speed']!=target['training_speed'] for c in peers)==6
    bad=copy.deepcopy(p);bad['source_checkpoints'][base.path(1301).relative_to(ROOT).as_posix()]='changed'
    with pytest.raises(ValueError,match='checkpoint changed'):run.sources(1301,cfg,bad)


def test_paired_record_changes_only_speed_response(monkeypatch):
    cfg,_=run.preflight();cfg=copy.deepcopy(cfg);cfg['duration_s']=.2;c=cfg['fleets'][0]['clients'][0]
    original,meta=base.generate(c,1301,'training',c['training_speed'],cfg)
    source=dict(seed=1301,data={c['client']:meta})
    monkeypatch.setattr(run,'original_arrays',lambda *args:original)
    same,record=run.make_record(source,c,c['training_speed'],cfg)
    assert record['trace_sha256']==meta['trace_sha256']
    for v in [13,21,29]:
        arrays,record=run.make_record(source,c,v,cfg)
        assert np.array_equal(arrays['noise'],original['noise'])
        assert np.array_equal(arrays['steering'],original['steering'])
        assert np.array_equal(arrays['time'],original['time'])
        np.testing.assert_allclose(arrays['measured_yaw'],arrays['state'][:,1]+original['noise'])
        if v!=c['training_speed']:assert not np.array_equal(arrays['state'],original['state'])
        assert record['operating_checks_passed']
    assert not np.shares_memory(same['state'],original['state'])


def test_bad_original_array_never_replaced(tmp_path,monkeypatch):
    cfg,protocol=run.preflight();source,_=run.sources(1301,cfg,protocol);c=cfg['fleets'][0]['clients'][0]
    monkeypatch.setattr(base,'RAW',tmp_path)
    file=tmp_path/'1301'/f'{c["client"]}_training.npz'
    base.storage.save_arrays(file,dict(noise=np.zeros(4)))
    before=file.read_bytes()
    with pytest.raises(ValueError,match='training hash mismatch'):run.original_arrays(source,c,cfg)
    assert file.read_bytes()==before


def test_incomplete_pairs_withhold_means():
    rows=[]
    for client in ['a','b']:
        for method in ['LocalOE','NominalPrior','Complementary','SameSpeed']:
            value=None if client=='b' and method=='SameSpeed' else 1.
            rows.append(dict(seed=1,training_speed=13,region='target_unseen_anchor',client=client,speed=21,probe='slow',method=method,yaw_rmse_deg_s=value))
    summary,pairs=run.summaries(pd.DataFrame(rows));s=summary[summary.method=='SameSpeed'].iloc[0]
    assert s.complete==1 and s.expected==2 and pd.isna(s.mean_yaw_rmse_deg_s)
    pair=pairs[pairs.baseline=='SameSpeed'].iloc[0]
    assert pair.paired==1 and pair.expected==2 and pd.isna(pair.reduction_pct)


def test_small_workflow_restart_target_isolation_restore_and_report(tmp_path,monkeypatch):
    cfg,protocol=run.preflight();cfg=copy.deepcopy(cfg);protocol=copy.deepcopy(protocol)
    cfg['duration_s']=.15;cfg['evaluation_speeds']=[13,21];cfg['fleets']=cfg['fleets'][:1];protocol['seeds']=[1301]
    # One physical client per original speed, each is a target once.
    original_clients=cfg['fleets'][0]['clients']
    clients=[next(c for c in original_clients if c['training_speed']==v) for v in [13,21,29]]
    cfg['fleets'][0]['clients']=clients
    monkeypatch.setattr(base,'RAW',tmp_path/'originalraw');monkeypatch.setattr(run,'OUT',tmp_path/'tables')
    monkeypatch.setattr(run,'RAW',tmp_path/'newraw');monkeypatch.setattr(run,'REPORT',tmp_path/'report.html')
    q=np.log(cfg['nominal_parameters']).tolist();selected=dict(available=True,optimizer_converged=True,
        parameters=cfg['nominal_parameters'],log_parameters=q)
    source=dict(seed=1301,local={},nominal_prior={},data={},evaluation={},selected={
        'FedPersonal':dict(available=True,log_parameters=[q]*3),'Global':dict(available=False),'FedGroups':dict(available=False)})
    complementary=dict(evaluation={},optimization=dict(status='converged',round=9,history=[dict(objective=3.)],communication_float64=0))
    for i,c in enumerate(clients):
        source['local'][c['client']]=dict(selected=copy.deepcopy(selected));source['nominal_prior'][c['client']]=copy.deepcopy(selected)
        arrays,metadata=base.generate(c,1301,'training',c['training_speed'],cfg);source['data'][c['client']]=metadata
        base.storage.save_arrays(base.RAW/'1301'/f'{c["client"]}_training.npz',arrays)
        for v in cfg['evaluation_speeds']:
            for probe in ['slow','broadband']:
                key=f'{c["client"]}_v{v}_{probe}';entry,arrays=base.evaluation_case(source,c,i,v,probe,cfg)
                source['evaluation'][key]=entry;complementary['evaluation'][key]=dict(metrics=entry['methods']['FedPersonal'])
                base.storage.save_arrays(base.RAW/'1301'/f'{key}.npz',arrays)
    original=copy.deepcopy(source)
    files={str(p):base.storage.file_hash(p) for p in base.RAW.rglob('*.npz')}
    monkeypatch.setattr(run,'preflight',lambda execution=False:(cfg,protocol))
    monkeypatch.setattr(run,'sources',lambda *args,**kwargs:(source,complementary))
    calls=[]
    def fit(self,start,center=None,weights=None,max_nfev=None):
        calls.append((float(self.speeds[0]),center is None,np.array(start).tolist()))
        # Return identical nominal solutions for deterministic orchestration.
        return dict(available=True,optimizer_converged=True,parameters=np.exp(q).tolist(),log_parameters=q,
            objective=1.,data_sse=1.,nfev=1,optimality=0.,near_bound=[False]*6)
    monkeypatch.setattr(base.fl.Client,'fit',fit)
    real_save=run.save
    def interrupt(p,stage):
        real_save(p,stage)
        if stage=='speed13_round1_client0':raise InterruptedError('after saved client update')
    monkeypatch.setattr(run,'save',interrupt)
    with pytest.raises(InterruptedError):run.worker(1301)
    saved=base.storage.read_checkpoint(run.path(1301));assert list(saved['conditions']['13']['optimization']['pending'])==['0']
    monkeypatch.setattr(run,'save',real_save);run.worker(1301)
    # Six new local records * three starts; three reused local fits; no refits on resume.
    local_calls=[x for x in calls if x[1]];assert len(local_calls)==18
    for speed in [13,21,29]:
        starts=[x[2] for x in local_calls if x[0]==speed]
        np.testing.assert_allclose(starts,np.tile(np.log(cfg['starts']),(2,1)))
    assert len([x for x in calls if not x[1]])==18  # 3 conditions, 2 rounds, 3 clients
    p=base.storage.read_checkpoint(run.path(1301));run.validate(p,cfg,protocol)
    for c in clients:
        entry=p['local'][run.record_key(c,c['training_speed'])]
        assert entry['reused'] and entry['selected']==source['local'][c['client']]['selected']
    count=len(calls);run.worker(1301);assert len(calls)==count
    # Resuming missing evaluation does not refit models.
    p['status']='running';p['evaluation'].pop(next(iter(p['evaluation'])));base.storage.atomic_json(run.path(1301),p)
    run.worker(1301);assert len(calls)==count
    target=next((run.RAW/'1301').glob('*train*.npz'));target.unlink()
    output=next((run.RAW/'1301').glob('*_slow.npz'));output.unlink()
    checkpoint=base.storage.file_hash(run.path(1301));run.restore(cfg,protocol,[1301])
    assert target.exists() and output.exists() and base.storage.file_hash(run.path(1301))==checkpoint
    run.analyze(cfg,protocol);html=run.REPORT.read_text(encoding='utf-8')
    assert html.count('<svg')==1 and '\ufffd' not in html
    assert source==original and files=={str(p):base.storage.file_hash(p) for p in base.RAW.rglob('*.npz')}
