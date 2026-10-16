"""Controlled replay with hash-verified exported original inputs; no fitting."""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import argparse
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
import paper_smooth_replay_diagnosis as diagnosis
from federated_lpv.fleet import VehicleClient
from federated_lpv.vehicle import VehicleParameters

frozen=diagnosis.frozen
REFERENCE=frozen.OUT/'paper_smooth_replay_execution_reference.json'
PREFIX='paper_smooth_original_input_replay'


def inputs(seed):
    reference=frozen.load(REFERENCE)
    if reference['source_sha256']!=frozen.preflight()[3]:raise ValueError('Reference provenance differs')
    original=frozen.read_checkpoint(seed)
    scenarios={k:{n:np.asarray(v) for n,v in sc.items()} for k,sc in reference['scenarios'].items()}
    for item in original['clients']:
        for rec in item['records']:
            sc=scenarios[rec['scenario']]
            h=frozen.array_hashes(dict(time=sc['time'],speed=sc['speed'],reference=sc['speed']*sc['curvature']))
            if any(h[k]!=rec['hashes'][k] for k in h):raise ValueError('Export does not match original input hashes')
    clients=[VehicleClient(c.client_id,c.family,VehicleParameters(**reference['plant_parameters'][str(seed)][c.client_id])) for c in frozen.sample_fleet(seed)]
    return clients,scenarios


def lock():
    return {**diagnosis.lock(),REFERENCE.relative_to(frozen.ROOT).as_posix():frozen.digest(REFERENCE),
            Path(__file__).relative_to(frozen.ROOT).as_posix():frozen.digest(Path(__file__))}


def worker(seed):
    path=frozen.OUT/f'{PREFIX}_seed{seed}.json';provenance=lock()
    if path.exists():
        if frozen.load(path)['source_sha256']!=provenance:raise ValueError('Stale checkpoint')
        return
    clients,scenarios=inputs(seed)
    old_sampler,old_scenarios=frozen.sample_fleet,frozen.parent.scenarios
    try:
        frozen.sample_fleet=lambda _:clients
        frozen.parent.scenarios=lambda _:scenarios
        result=diagnosis.replay(seed)
    finally:
        frozen.sample_fleet, frozen.parent.scenarios=old_sampler,old_scenarios
    result['source_sha256']=provenance
    result.pop('scenarios');result.pop('plant_parameters')
    result['input_reference']=REFERENCE.relative_to(frozen.ROOT).as_posix()
    diagnosis.dump(path,result)
    print(f'Original-input replay {seed} saved',flush=True)


def restore(seed):
    saved=frozen.load(frozen.OUT/f'{PREFIX}_seed{seed}.json')
    if saved['source_sha256']!=lock():raise ValueError('Stale checkpoint')
    clients,scenarios=inputs(seed);clients={c.client_id:c for c in clients}
    original=frozen.read_checkpoint(seed);cfg=frozen.preflight()[1]
    destination=frozen.ROOT/'results/data'/PREFIX;destination.mkdir(parents=True,exist_ok=True)
    with threadpool_limits(limits=1):
        for item in original['clients']:
            schedule=None if item['schedule'] is None else tuple(np.asarray(a) for a in item['schedule'])
            for row in [r for r in saved['rows'] if r['client']==item['client'] and r['status']=='complete']:
                _,arrays=frozen.simulate_with_arrays(clients[item['client']],schedule,scenarios[row['scenario']],cfg)
                if frozen.array_hashes(arrays)!=row['replay_hashes']:raise ValueError('Restored arrays differ')
                p=destination/f"seed{seed}_{item['client']}_{row['scenario']}.npz"
                np.savez_compressed(p,**arrays)
                with np.load(p) as disk:
                    if frozen.array_hashes(dict(disk))!=row['replay_hashes']:raise ValueError('Persisted arrays differ')
    print('Original-input replay checkpoint restored and verified; historical execution retained')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['preflight','run','restore-data'])
    p.add_argument('--workers',type=int,default=5);p.add_argument('--seed',type=int,default=411)
    a=p.parse_args()
    if a.workers<1 or a.seed not in range(411,421):p.error('Invalid workers/seed')
    if a.action=='preflight':
        for seed in range(411,421):inputs(seed)
        print('All 600 original scenario inputs verified')
    elif a.action=='restore-data':restore(a.seed)
    else:
        with ProcessPoolExecutor(max_workers=a.workers) as pool:list(pool.map(worker,range(411,421)))


if __name__=='__main__':main()
