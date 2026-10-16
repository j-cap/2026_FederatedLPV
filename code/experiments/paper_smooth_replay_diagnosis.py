"""Replay frozen Smooth Local schedules; never refit or replace executed evidence."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import importlib.metadata
import json
from pathlib import Path
import platform
import tempfile
import numpy as np
from threadpoolctl import threadpool_limits, threadpool_info
import paper_smooth_local_lqi as frozen

ROOT=frozen.ROOT
OUT=ROOT/'results/tables'
PREFIX='paper_smooth_replay_diagnosis'


def dump(path, data):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    temp.replace(path)


def lock():
    result=dict(frozen.preflight()[3])
    result[Path(__file__).relative_to(ROOT).as_posix()]=frozen.digest(Path(__file__))
    for seed in range(411,421):
        p=OUT/f'{frozen.PREFIX}_seed{seed}.json'
        result[p.relative_to(ROOT).as_posix()]=frozen.digest(p)
    return result


def metadata():
    return dict(platform=platform.platform(),python=platform.python_version(),
                packages={n:importlib.metadata.version(n) for n in ['numpy','scipy','pandas']},
                threadpools=threadpool_info())


def replay(seed):
    provenance=lock(); original=frozen.read_checkpoint(seed); cfg=frozen.preflight()[1]
    clients={c.client_id:c for c in frozen.sample_fleet(seed)}
    scenarios=frozen.parent.scenarios(cfg)
    rows=[]
    with threadpool_limits(limits=1):
        for item in original['clients']:
            schedule=None if item['schedule'] is None else tuple(np.asarray(a) for a in item['schedule'])
            for rec in item['records']:
                sc=scenarios[rec['scenario']]
                result=frozen.simulate_with_arrays(clients[item['client']],schedule,sc,cfg)
                old=next(r for r in original['rows'] if r['client']==item['client'] and r['method']=='SmoothLocal' and r['scenario']==rec['scenario'])
                if result is None:
                    rows.append(dict(client=item['client'],scenario=rec['scenario'],status='replay_failed',old_feasible=old['feasible']))
                    continue
                metrics,arrays=result; hashes=frozen.array_hashes(arrays)
                rows.append(dict(client=item['client'],scenario=rec['scenario'],status='complete',
                    old_feasible=old['feasible'],feasible=metrics['feasible'],
                    old_rmse=old['tracking_rmse_deg_s'],rmse=metrics['tracking_rmse_deg_s'],
                    rmse_difference=metrics['tracking_rmse_deg_s']-old['tracking_rmse_deg_s'],
                    hash_matches={k:hashes[k]==rec['hashes'][k] for k in hashes},
                    replay_hashes=hashes,metrics=metrics))
    return dict(seed=seed,source_sha256=provenance,environment=metadata(),rows=rows,
                scenarios={k:{n:np.asarray(v).tolist() for n,v in sc.items()} for k,sc in scenarios.items()},
                plant_parameters={k:asdict(c.parameters) for k,c in clients.items()})


def worker(seed):
    path=OUT/f'{PREFIX}_seed{seed}.json'
    if path.exists():
        saved=json.loads(path.read_text())
        if saved['source_sha256']!=lock():raise ValueError('Stale replay checkpoint')
    else:dump(path,replay(seed))
    print(f'Replay {seed} saved',flush=True)


def restore(seed, destination):
    saved=json.loads((OUT/f'{PREFIX}_seed{seed}.json').read_text())
    if saved['source_sha256']!=lock():raise ValueError('Stale checkpoint')
    original=frozen.read_checkpoint(seed); cfg=frozen.preflight()[1]
    from federated_lpv.fleet import VehicleClient
    from federated_lpv.vehicle import VehicleParameters
    destination.mkdir(parents=True,exist_ok=True)
    with threadpool_limits(limits=1):
        for item in original['clients']:
            client=VehicleClient(item['client'],item['client'].rsplit('_',1)[0],VehicleParameters(**saved['plant_parameters'][item['client']]))
            schedule=None if item['schedule'] is None else tuple(np.asarray(a) for a in item['schedule'])
            for row in [r for r in saved['rows'] if r['client']==item['client'] and r['status']=='complete']:
                sc={k:np.asarray(v) for k,v in saved['scenarios'][row['scenario']].items()}
                _,arrays=frozen.simulate_with_arrays(client,schedule,sc,cfg)
                if frozen.array_hashes(arrays)!=row['replay_hashes']:raise ValueError('Replay checkpoint reconstruction differs')
                p=destination/f"seed{seed}_{item['client']}_{row['scenario']}.npz"
                np.savez_compressed(p,**arrays)
                with np.load(p) as disk:
                    if frozen.array_hashes(dict(disk))!=row['replay_hashes']:raise ValueError('Saved arrays differ')
    print(f'Restored replay checkpoint {seed} with exact hashes; original execution is unchanged')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['preflight','run','restore-data'])
    p.add_argument('--workers',type=int,default=5);p.add_argument('--seed',type=int,default=411)
    args=p.parse_args();lock()
    if args.workers<1 or args.seed not in range(411,421):p.error('Invalid workers or seed')
    if args.action=='preflight': print('Frozen inputs verified')
    elif args.action=='restore-data':restore(args.seed,ROOT/'results/data'/PREFIX)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:list(pool.map(worker,range(411,421)))


if __name__=='__main__':main()
