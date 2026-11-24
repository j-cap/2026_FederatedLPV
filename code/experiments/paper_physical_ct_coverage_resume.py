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
import os
import tempfile
import time
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
import paper_physical_ct_coverage as base
import numpy as np
from threadpoolctl import threadpool_limits

FROZEN_RUNNER_SHA256='0576e0251e801e00c54ee3c9dff4fbefef4720ff0fb75975f5c3300f16822726'
LEGACY_ADAPTER_ID={'file': 'code/experiments/paper_physical_ct_coverage_resume.py', 'sha256': 'a84e99e24bc97f38bdcdddd6ddd98673542c0bd588a9135601460c2e67564fdc', 'frozen_runner_sha256': '0576e0251e801e00c54ee3c9dff4fbefef4720ff0fb75975f5c3300f16822726', 'changes': ['compare provenance with portable path separators; all hashes and runtime still checked', 'convert saved grid matrices to float array before unchanged controller synthesis']}
_original_validate=base.validate
_original_checkpoint=base.checkpoint
_original_design=base.design_nodes


def adapter_identity():
    return dict(file=Path(__file__).relative_to(base.ROOT).as_posix(),sha256=base.old.file_hash(__file__),
        frozen_runner_sha256=FROZEN_RUNNER_SHA256,
        changes=['compare provenance with portable path separators; all hashes and runtime still checked',
                 'convert saved grid matrices to float array before unchanged controller synthesis',
                 'unique checkpoint temporary files and bounded retry of Windows replacement locks'])


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
    if 'execution_adapter' in payload and payload['execution_adapter'] not in [LEGACY_ADAPTER_ID,adapter_identity()]:raise ValueError('Changed coverage execution adapter')
    if 'adapter_migration' in payload:
        migration=payload['adapter_migration']
        if migration['previous']!=LEGACY_ADAPTER_ID or not 0<=migration['saved_conditions']<=len(payload['records']):raise ValueError('Invalid adapter migration history')
    # Only a shallow top-level copy is needed; original validator never mutates it.
    copy_payload=dict(payload,provenance=expected)
    _original_validate(copy_payload,seed,cfg,control)


def design_nodes(matrices,cfg):
    values=np.asarray(matrices,dtype=float)
    if values.shape!=(len(cfg['speed_grid']),6):raise ValueError('Invalid saved controller model grid')
    return _original_design(values,cfg)


def checkpoint(seed,cfg,control=False):
    payload=_original_checkpoint(seed,cfg,control)
    if control:
        if payload.get('execution_adapter')==LEGACY_ADAPTER_ID:
            payload['adapter_migration']=dict(previous=LEGACY_ADAPTER_ID,saved_conditions=len(payload['records']))
        payload['execution_adapter']=adapter_identity()
    return payload


def atomic_json(path,payload,attempts=21):
    """Write once, then retry only atomic replacement on Windows lock errors.

    The existing checkpoint is never deleted. On exhausted retries the complete,
    hash-protected temporary file is kept for inspection/recovery.
    """
    if attempts<1:raise ValueError('At least one replacement attempt required')
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',newline='\n',
            prefix=path.name+'.',suffix='.tmp',dir=path.parent,delete=False) as stream:
        temp=Path(stream.name)
        json.dump(dict(payload=payload,payload_sha256=base.old.storage.digest(payload)),stream,indent=2,allow_nan=False)
        stream.write('\n');stream.flush();os.fsync(stream.fileno())
    for attempt in range(attempts):
        try:
            os.replace(temp,path)
            return
        except PermissionError as exc:
            if getattr(exc,'winerror',None) not in [5,32,33] or attempt==attempts-1:
                exc.add_note(f'Existing checkpoint preserved. Complete pending checkpoint retained at {temp}. Close processes holding the checkpoint and resume the command.')
                raise
            time.sleep(min(.05*2**min(attempt,4),.5))


def install():
    if base.old.file_hash(base.__file__)!=FROZEN_RUNNER_SHA256:raise ValueError('Frozen coverage runner changed')
    base.validate=validate;base.checkpoint=checkpoint;base.design_nodes=design_nodes
    base.old.atomic_json=atomic_json


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
