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
