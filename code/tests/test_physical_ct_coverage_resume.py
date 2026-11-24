"""Regression tests for cross-platform provenance and list-based model grids."""
import copy
from pathlib import Path
import sys
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'code/src'),str(ROOT/'code/experiments')]
import paper_physical_ct_coverage_resume as fix


def test_windows_provenance_validates_without_mutation():
    base=fix.base;cfg=base.old.load(base.CONFIG)
    p=base.old.envelope(base.path(1301));before=copy.deepcopy(p)
    fix.validate(p,1301,cfg)
    assert p==before
    changed=copy.deepcopy(p)
    first=next(iter(changed['provenance']['sources']))
    changed['provenance']['sources'][first]='0'*64
    with pytest.raises(ValueError,match='hash/runtime'):fix.validate(changed,1301,cfg)


def test_duplicate_portable_paths_rejected():
    p=dict(sources={'code/a':'x','code\\a':'x'},frozen={})
    with pytest.raises(ValueError,match='Duplicate'):fix.portable_provenance(p)


def test_saved_list_and_array_produce_identical_schedules():
    base=fix.base;cfg=base.old.load(base.old.storage.CONFIG)
    theta=[47.,59.,.56,1.02];settings=base.old.load(base.old.CONFIG)['estimator']
    matrices=base.models(theta,cfg['speed_grid'],settings)
    schedule=fix.design_nodes(matrices.tolist(),cfg)
    assert schedule==fix._original_design(matrices,cfg)
    assert schedule['status']=='designed'


def test_resume_loads_partial_checkpoint_and_records_adapter(monkeypatch):
    base=fix.base;cfg=base.old.load(base.CONFIG)
    monkeypatch.setattr(base,'validate',fix.validate)
    p=fix.checkpoint(1301,cfg,True)
    assert p['execution_adapter']==fix.adapter_identity()
    assert len(p['records'])>=1
    fix.validate(p,1301,cfg,True)


def locked():
    error=PermissionError('Simulated Windows file lock')
    error.winerror=5
    return error


def test_transient_replace_lock_retries_without_deleting_checkpoint(tmp_path,monkeypatch):
    target=tmp_path/'checkpoint.json';target.write_text('original')
    replace=fix.os.replace;attempts=[];sleeps=[]
    def interrupted(source,destination):
        attempts.append(source)
        assert target.read_text()=='original'
        if len(attempts)<3:raise locked()
        replace(source,destination)
    monkeypatch.setattr(fix.os,'replace',interrupted)
    monkeypatch.setattr(fix.time,'sleep',sleeps.append)
    fix.atomic_json(target,dict(label='β',records=[1,2]))
    assert len(attempts)==3 and len(set(attempts))==1
    assert sleeps==[.05,.1]
    assert fix.base.old.envelope(target)==dict(label='β',records=[1,2])
    assert not list(tmp_path.glob('*.tmp'))


def test_permanent_replace_lock_preserves_both_valid_versions(tmp_path,monkeypatch):
    target=tmp_path/'checkpoint.json'
    fix.atomic_json(target,dict(records=[1]))
    monkeypatch.setattr(fix.os,'replace',lambda *args:(_ for _ in ()).throw(locked()))
    monkeypatch.setattr(fix.time,'sleep',lambda seconds:None)
    with pytest.raises(PermissionError):fix.atomic_json(target,dict(records=[1,2]),attempts=3)
    assert fix.base.old.envelope(target)==dict(records=[1])
    pending=list(tmp_path.glob('*.tmp'))
    assert len(pending)==1 and fix.base.old.envelope(pending[0])==dict(records=[1,2])


def test_non_windows_permission_error_not_retried(tmp_path,monkeypatch):
    def fail(*args):raise PermissionError('Persistent filesystem permission error')
    monkeypatch.setattr(fix.os,'replace',fail)
    monkeypatch.setattr(fix.time,'sleep',lambda seconds:pytest.fail('Unexpected retry'))
    with pytest.raises(PermissionError):fix.atomic_json(tmp_path/'checkpoint.json',dict(records=[]))


def test_completed_legacy_and_partial_checkpoints_still_validate():
    base=fix.base;cfg=base.old.load(base.CONFIG)
    for seed in [1301,1302,1303]:
        p=base.old.envelope(base.path(seed,True))
        fix.validate(p,seed,cfg,True)
        p=copy.deepcopy(p);p['execution_adapter']=copy.deepcopy(fix.LEGACY_ADAPTER_ID)
        fix.validate(p,seed,cfg,True)
    bad=copy.deepcopy(p);bad['execution_adapter']['sha256']='0'*64
    with pytest.raises(ValueError,match='execution adapter'):fix.validate(bad,1303,cfg,True)
