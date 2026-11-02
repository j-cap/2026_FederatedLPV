r"""Supporting matched-control test of frozen OE-LPV models (no fitting).

Windows CMD: python code\experiments\paper_oe_lqi_validation.py preflight
See docs/oe_lqi_validation.html for the frozen protocol and execution handoff.
"""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
import multiprocessing
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/src'))
import numpy as np
from threadpoolctl import threadpool_limits
from experiment_10a_basis_coverage_audit import raw_basis
from federated_lpv.oe_lqi_validation import design_nodes, interpolate, simulate
from federated_lpv.control import augmented_tracking_matrices
from federated_lpv.vehicle import VehicleParameters

CONFIG = ROOT / 'code/config/paper_oe_lqi_validation.json'
PREFIX = 'paper_oe_lqi_validation'
OUT = ROOT / 'results/tables'
RAW = ROOT / 'results/data' / PREFIX
METHODS = ['NominalLTI', 'RestrictedLocal', 'SmoothLocal', 'Global', 'Federated', 'ExactLPV']
SCENARIOS = ['moderate', 'hard']


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    path = Path(path)
    content = path.read_bytes()
    if path.suffix in ['.py', '.json', '.yml']:
        content = content.replace(b'\r\n', b'\n')
    return hashlib.sha256(content).hexdigest()


def atomic_json(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.json.tmp')
    with temp.open('w', encoding='utf-8', newline='\n') as stream:
        json.dump(dict(payload=payload, payload_sha256=digest(payload)), stream, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(temp, path)


def envelope(path):
    saved = load(path)
    if digest(saved['payload']) != saved['payload_sha256']:
        raise ValueError(f'Payload hash mismatch: {path}')
    return saved['payload']


def runtime():
    return dict(python=f'{sys.version_info.major}.{sys.version_info.minor}',
                packages={name: importlib.metadata.version(name) for name in ['numpy', 'scipy', 'pandas', 'threadpoolctl']})


def source_path(seed):
    return ROOT / f'results/tables/paper_oe_lpv_confirmation_result_seed{seed}.json'


def result_path(seed, out=OUT):
    return Path(out) / f'{PREFIX}_seed{seed}.json'


def profiles(cfg):
    with np.load(ROOT / cfg['profiles'], allow_pickle=False) as data:
        return {s: {k: data[f'{s}_{k}'].copy() for k in ['time', 'speed', 'reference', 'curvature']} for s in SCENARIOS}


def coefficient(models, method, index):
    if method in ['RestrictedLocal', 'SmoothLocal', 'ExactLPV']:
        value = models[method][index]
    elif method == 'Global':
        value = models[method]
    elif method == 'Federated':
        value = models[method][models['assignments'][index]]
    else:
        raise ValueError(method)
    return np.asarray(value, float)


def preflight():
    cfg = load(CONFIG)
    if runtime() != cfg['runtime']:
        raise ValueError(f'Numeric environment differs: {runtime()}; expected {cfg["runtime"]}')
    for name, expected in cfg['source_sha256'].items():
        if file_hash(ROOT / name) != expected:
            raise ValueError(f'Frozen source changed: {name}')
    if cfg['methods'] != METHODS or cfg['scenarios'] != SCENARIOS:
        raise ValueError('Unexpected protocol methods/scenarios')
    for seed in cfg['seeds']:
        source = envelope(source_path(seed))
        if source['seed'] != seed or source['status'] != 'execution_complete' or source['blocked_conditions']:
            raise ValueError(f'Incomplete/blocked source seed {seed}')
        if source['grid'] != cfg['speed_grid'] or len(source['clients']) != 30:
            raise ValueError(f'Unexpected grid/client count: {seed}')
        models = source['conditions']['ClosedOE4']['models']
        if len(set(c['client'] for c in source['clients'])) != 30 or len(models['assignments']) != 30:
            raise ValueError('Invalid client/assignment identities')
        for i in range(30):
            for method in METHODS[1:]:
                a = coefficient(models, method, i)
                if a.shape != (3 if method == 'RestrictedLocal' else 7, 6) or not np.isfinite(a).all():
                    raise ValueError(f'Invalid coefficients: {seed}/{i}/{method}')
            p = list(source['clients'][i]['parameters'].values())
            if not np.isfinite(p).all() or min(p) <= 0:
                raise ValueError('Invalid saved plant parameters')
    for profile in profiles(cfg).values():
        t, v, r = (profile[k] for k in ['time', 'speed', 'reference'])
        if len(t) != 1201 or not np.isfinite(np.concatenate(list(profile.values()))).all():
            raise ValueError('Invalid profile arrays')
        np.testing.assert_allclose(np.diff(t), cfg['sample_time'], atol=1e-12, rtol=0)
        np.testing.assert_allclose(r, v*profile['curvature'], atol=0, rtol=0)
        if min(v) < cfg['speed_grid'][0] or max(v) > cfg['speed_grid'][-1]:
            raise ValueError('Out-of-domain profile')
    return cfg


def provenance(cfg, seed):
    return dict(protocol_sha256=file_hash(CONFIG), source_sha256=cfg['source_sha256'],
                source_payload_sha256=load(source_path(seed))['payload_sha256'], runtime=runtime())


def committed_code(cfg):
    files = ['code/config/paper_oe_lqi_validation.json', *cfg['source_sha256']]
    for name in files:
        subprocess.run(['git', 'ls-files', '--error-unmatch', name], cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    subprocess.run(['git', 'diff', '--exit-code', 'HEAD', '--', *files], cwd=ROOT, check=True, stdout=subprocess.DEVNULL)


def nominal_schedule(cfg):
    nominal = load(ROOT / 'code/config/paper_fixed_lti_identification.json')['controller_frozen']
    matrix = np.r_[np.asarray(nominal['a']).ravel(), nominal['b']]
    schedule = design_nodes(np.tile(matrix, (len(cfg['speed_grid']), 1)), cfg)
    if schedule['status'] != 'designed':
        raise ValueError('Frozen nominal design unexpectedly failed')
    for node in schedule['nodes']:
        np.testing.assert_allclose(node['gain'], nominal['gain'], rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(node['prefilter'], nominal['prefilter'], rtol=1e-8, atol=1e-10)
        # Use exactly the saved acquisition gain and prefilter, not a new fit.
        node.update(gain=nominal['gain'], prefilter=nominal['prefilter'])
    return schedule


def spectral_diagnostic(schedule, exact, cfg, basis):
    if schedule['status'] != 'designed':
        return dict(status='unavailable_after_synthesis_failure')
    grid = np.linspace(cfg['speed_grid'][0], cfg['speed_grid'][-1], 81)
    gains, _ = interpolate(schedule, cfg['speed_grid'], grid)
    matrices = raw_basis(grid, 7, basis) @ exact
    try:
        values = []
        for matrix, gain in zip(matrices, gains, strict=True):
            aa, bb = augmented_tracking_matrices(matrix[:4].reshape(2, 2), matrix[4:].reshape(2, 1), cfg['sample_time'])
            values.append(float(max(abs(np.linalg.eigvals(aa-bb@gain[None, :])))))
        if not np.isfinite(values).all():
            raise ValueError('nonfinite radius')
        return dict(status='complete', speed=grid.tolist(), radius=values, maximum=max(values),
                    scope='Frozen-speed saved full-reference LPV map; diagnostic only, not a stability guarantee or acceptance gate.')
    except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
        return dict(status='diagnostic_failed', reason=str(exc))


def array_hashes(arrays):
    if arrays is None:
        return None
    result = {}
    for key, value in arrays.items():
        value = np.ascontiguousarray(value, dtype='<f8')
        result[key] = dict(shape=list(value.shape), sha256=hashlib.sha256(value.tobytes()).hexdigest())
    return result


def raw_path(raw, seed, client_index, method, scenario):
    return Path(raw) / f'seed{seed}' / f'client{client_index:02d}_{method}_{scenario}.npz'


def save_arrays(path, arrays):
    if arrays is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.npz.tmp')
    with temp.open('wb') as stream:
        np.savez_compressed(stream, **{k: np.asarray(v, dtype='<f8') for k, v in arrays.items()})
    os.replace(temp, path)


def validate_checkpoint(payload, cfg, seed):
    if payload['seed'] != seed or payload['provenance'] != provenance(cfg, seed):
        raise ValueError(f'Checkpoint provenance mismatch: {seed}')
    clients = payload['clients']
    if [c['index'] for c in clients] != list(range(len(clients))) or len(clients) > 30:
        raise ValueError('Invalid checkpoint client order')
    for c in clients:
        if set(c['schedules']) != set(METHODS) or len(c['rows']) != len(METHODS)*len(SCENARIOS):
            raise ValueError('Incomplete client checkpoint')
        if {(r['method'], r['scenario']) for r in c['rows']} != {(m,s) for m in METHODS for s in SCENARIOS}:
            raise ValueError('Duplicate or missing result cases')
    if payload['status'] == 'execution_complete' and len(clients) != 30:
        raise ValueError('Incomplete fleet marked complete')


def run_fleet(seed):
    cfg = load(CONFIG); source = envelope(source_path(seed)); path = result_path(seed)
    payload = envelope(path) if path.exists() else dict(seed=seed, status='running', provenance=provenance(cfg, seed),
        protocol=cfg, host=dict(platform=platform.platform(), python=platform.python_version()),
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(), clients=[])
    validate_checkpoint(payload, cfg, seed)
    if payload['status'] == 'execution_complete':
        return dict(seed=seed, status='already_complete', clients=30)
    basis = load(ROOT / 'code/config/experiment_10a.json'); models = source['conditions']['ClosedOE4']['models']
    profile_set = profiles(cfg)
    with threadpool_limits(limits=1):
        fixed = nominal_schedule(cfg)
        for i in range(len(payload['clients']), 30):
            original = source['clients'][i]
            coefficients = {m: coefficient(models,m,i) for m in METHODS[1:]}
            schedules = {'NominalLTI': fixed}
            for method, values in coefficients.items():
                schedules[method] = design_nodes(raw_basis(np.asarray(cfg['speed_grid']), len(values), basis) @ values, cfg)
            client = dict(index=i, client=original['client'], family=original['family'], parameters=original['parameters'],
                observed_speeds=original['speeds'], assignment=models['assignments'][i],
                coefficients={m:v.tolist() for m,v in coefficients.items()}, schedules=schedules,
                spectral_diagnostics={m:spectral_diagnostic(s,coefficients['ExactLPV'],cfg,basis) for m,s in schedules.items()}, rows=[])
            parameters = VehicleParameters(**client['parameters'])
            for method, schedule in schedules.items():
                for scenario, profile in profile_set.items():
                    outcome, arrays = simulate(parameters, schedule, profile, cfg)
                    save_arrays(raw_path(RAW,seed,i,method,scenario), arrays)
                    client['rows'].append(dict(method=method, scenario=scenario, outcome=outcome, arrays=array_hashes(arrays)))
            payload['clients'].append(client)
            if len(payload['clients']) == 30:
                payload['status'] = 'execution_complete'
            validate_checkpoint(payload,cfg,seed); atomic_json(path,payload)
            print(json.dumps(dict(seed=seed, clients_saved=i+1, total=30)), flush=True)
    return dict(seed=seed, status=payload['status'], clients=len(payload['clients']))


def restore_client(client, seed, profile_set, cfg, raw):
    restored = 0
    for row in client['rows']:
        path = raw_path(raw,seed,client['index'],row['method'],row['scenario'])
        if row['arrays'] is None:
            continue
        if path.exists():
            with np.load(path,allow_pickle=False) as arrays:
                if array_hashes(dict(arrays)) != row['arrays']:
                    raise ValueError(f'Existing raw array hash mismatch: {path}; preserve it for diagnosis')
            continue
        outcome, arrays = simulate(VehicleParameters(**client['parameters']), client['schedules'][row['method']],
                                   profile_set[row['scenario']], cfg)
        if outcome != row['outcome'] or array_hashes(arrays) != row['arrays']:
            raise ValueError(f'Reconstruction mismatch: {path}; no output overwritten. Use original numeric runtime.')
        save_arrays(path,arrays); restored += 1
    return restored


def restore_fleet(seed):
    cfg=load(CONFIG); payload=envelope(result_path(seed)); validate_checkpoint(payload,cfg,seed)
    with threadpool_limits(limits=1):
        count=sum(restore_client(c,seed,profiles(cfg),cfg,RAW) for c in payload['clients'])
    return dict(seed=seed, restored=count)


@contextmanager
def execution_lock():
    RAW.mkdir(parents=True,exist_ok=True); path=RAW/'execution.lock'
    try:
        fd=os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(f'Execution lock exists: {path}. If a previous process crashed, stop all its workers before removing this file.') from exc
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as stream:
            stream.write(str(os.getpid()))
        yield
    finally:
        path.unlink()


def smoke_worker():
    """Nominal technical fixture only; never reads client parameters or runs fleets."""
    cfg=load(CONFIG)
    with threadpool_limits(limits=1), tempfile.TemporaryDirectory() as directory:
        schedule=nominal_schedule(cfg); profile_set=profiles(cfg)
        client=dict(index=0, parameters=dict(mass=1500.,yaw_inertia=2500.,front_length=1.2,rear_length=1.6,
                    front_stiffness=80000.,rear_stiffness=80000.), schedules={'NominalLTI':schedule}, rows=[])
        for scenario, profile in profile_set.items():
            outcome, arrays=simulate(VehicleParameters(**client['parameters']),schedule,profile,cfg)
            if outcome['status'] != 'complete':
                raise AssertionError(outcome)
            client['rows'].append(dict(method='NominalLTI',scenario=scenario,outcome=outcome,arrays=array_hashes(arrays)))
        path=Path(directory)/'utf8.json'; atomic_json(path,dict(client=client,label='β — fixture'))
        saved=envelope(path)['client']
        count=restore_client(saved,0,profile_set,cfg,Path(directory))
        if count!=2 or restore_client(saved,0,profile_set,cfg,Path(directory))!=0:
            raise AssertionError('Restore/resume fixture failed')
    return dict(status='smoke_passed',scientific_fleets_run=0, nominal_trajectories=2,
                spawned_process=True,utf8_checkpoint=True,verified_reconstruction=True)


def status(cfg):
    total=0; complete=0; cases=Counter()
    for seed in cfg['seeds']:
        path=result_path(seed)
        if not path.exists():
            print(json.dumps(dict(seed=seed,status='not_started',clients=0))); continue
        payload=envelope(path); validate_checkpoint(payload,cfg,seed)
        total+=len(payload['clients']); complete+=payload['status']=='execution_complete'
        for client in payload['clients']:
            for row in client['rows']:
                outcome=row['outcome']
                cases[outcome['status']+('_feasible' if outcome['feasible'] else '_infeasible')]+=1
        print(json.dumps(dict(seed=seed,status=payload['status'],clients=len(payload['clients']))))
    print(json.dumps(dict(complete_fleets=complete,total_fleets=10,completed_clients=total,expected_clients=300,
                         recorded_cases=sum(cases.values()),expected_cases=3600,outcomes=dict(cases))))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['preflight','smoke','run','status','restore'])
    parser.add_argument('--workers',type=int,default=5)
    args=parser.parse_args()
    if not 1<=args.workers<=10:
        parser.error('--workers must be between 1 and 10')
    cfg=preflight()
    if args.command=='preflight':
        print(json.dumps(dict(status='preflight_passed',fleets=10,clients=300,methods=METHODS,scenarios=SCENARIOS,
                              planned_trajectories=3600,new_identification_fits=0,scientific_simulations_run=0)))
    elif args.command=='status':
        status(cfg)
    elif args.command=='smoke':
        with ProcessPoolExecutor(max_workers=1,mp_context=multiprocessing.get_context('spawn')) as pool:
            print(json.dumps(pool.submit(smoke_worker).result()))
    else:
        committed_code(cfg)
        if args.command=='restore' and any(not result_path(s).exists() for s in cfg['seeds']):
            raise ValueError('Restore requires all ten checkpoints; run/resume first.')
        with execution_lock(), ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            worker=run_fleet if args.command=='run' else restore_fleet
            futures={pool.submit(worker,seed):seed for seed in cfg['seeds']}
            for future in as_completed(futures):
                print(json.dumps(future.result()),flush=True)
        status(cfg)


if __name__=='__main__':
    multiprocessing.freeze_support()
    main()
