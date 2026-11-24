r"""Compatibility adapter for the frozen coverage run; no refitting required.

python code\experiments\paper_physical_ct_coverage_resume.py control --workers 3
Normalizes provenance path separators for comparison only, preserving all hashes.
Converts serialized grid lists to arrays at the controller-design boundary.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import json
import multiprocessing
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
import paper_physical_ct_coverage as base
import numpy as np
from threadpoolctl import threadpool_limits

FROZEN_RUNNER_SHA256='0576e0251e801e00c54ee3c9dff4fbefef4720ff0fb75975f5c3300f16822726'
_original_validate=base.validate
_original_checkpoint=base.checkpoint
_original_design=base.design_nodes


def adapter_identity():
    return dict(file=Path(__file__).relative_to(base.ROOT).as_posix(),sha256=base.old.file_hash(__file__),
        frozen_runner_sha256=FROZEN_RUNNER_SHA256,
        changes=['compare provenance with portable path separators; all hashes and runtime still checked',
                 'convert saved grid matrices to float array before unchanged controller synthesis'])


def portable_provenance(provenance):
    result=copy.deepcopy(provenance)
    for section in ['sources','frozen']:
        normalized={k.replace('\\','/'):v for k,v in result[section].items()}
        if len(normalized)!=len(result[section]):raise ValueError('Duplicate normalized provenance paths')
        result[section]=normalized
    return result


def validate(payload,seed,cfg,control=False):
    actual=portable_provenance(payload['provenance'])
    expected=base.provenance(seed,cfg,control)
    if actual!=portable_provenance(expected):raise ValueError('Coverage provenance hash/runtime mismatch')
    if 'execution_adapter' in payload and payload['execution_adapter']!=adapter_identity():raise ValueError('Changed coverage execution adapter')
    # Only a shallow top-level copy is needed; original validator never mutates it.
    copy_payload=dict(payload,provenance=expected)
    _original_validate(copy_payload,seed,cfg,control)


def design_nodes(matrices,cfg):
    values=np.asarray(matrices,dtype=float)
    if values.shape!=(len(cfg['speed_grid']),6):raise ValueError('Invalid saved controller model grid')
    return _original_design(values,cfg)


def checkpoint(seed,cfg,control=False):
    payload=_original_checkpoint(seed,cfg,control)
    if control:payload['execution_adapter']=adapter_identity()
    return payload


def install():
    if base.old.file_hash(base.__file__)!=FROZEN_RUNNER_SHA256:raise ValueError('Frozen coverage runner changed')
    base.validate=validate;base.checkpoint=checkpoint;base.design_nodes=design_nodes


def worker(seed):
    install()
    with threadpool_limits(limits=1):return base.control_fleet(seed)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['status','control','analyze'])
    parser.add_argument('--workers',type=int,default=3)
    args=parser.parse_args()
    if args.workers<1:parser.error('--workers must be positive')
    install();cfg=base.preflight()
    if args.command=='status':
        for seed in cfg['seeds']:
            for control in [False,True]:
                if base.path(seed,control).exists():
                    p=base.old.envelope(base.path(seed,control));validate(p,seed,cfg,control)
                    print(json.dumps(dict(seed=seed,stage='control' if control else 'identification',status=p['status'],saved=len(p['records']))))
    elif args.command=='analyze':
        base.analyze(cfg)
        p=base.OUT/f'{base.PREFIX}_analysis.json';audit=base.old.load(p)
        audit['analysis_adapter']=adapter_identity();base.old.acquisition.parent.save_json(p,audit)
    else:
        with base.lock(),base.old.execution_lock('pilot','identification'),ProcessPoolExecutor(max_workers=min(args.workers,3),mp_context=multiprocessing.get_context('spawn')) as pool:
            for result in as_completed([pool.submit(worker,seed) for seed in cfg['seeds']]):print(json.dumps(result.result()),flush=True)


if __name__=='__main__':main()
