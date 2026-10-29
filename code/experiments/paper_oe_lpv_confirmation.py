"""Frozen ten-fleet confirmation. Run locally; analyze only after pushing checkpoints."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import html
import json
from multiprocessing import get_context
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'code/src'), str(ROOT / 'code/experiments')]
import numpy as np
from scipy.stats import t
from threadpoolctl import threadpool_limits
import paper_oe_lpv_integration as lpv
from federated_lpv import sample_fleet, VehicleParameters
from federated_lpv.vehicle import family_centers
from experiment_10a_basis_coverage_audit import collect_targets, discrete_jacobian
from experiment_10e_federated_latent_groups import label_free_categories

acq, parent = lpv.acquisition, lpv.parent
PREFIX = 'paper_oe_lpv_confirmation'
CONFIG = ROOT / f'code/config/{PREFIX}.json'
OUT = ROOT / 'results/tables'
DATA = ROOT / 'results/data' / PREFIX
REPORT = ROOT / 'report/oe_lpv_confirmation.html'


def preflight():
    protocol = parent.load(CONFIG)
    cfg, basis, mixture, acquisition_cfg, prior = lpv.preflight()
    controller_protocol, settings, _, acquisition_lock = acq.preflight()
    for path, expected in protocol['source_sha256'].items():
        if parent.file_hash(ROOT / path) != expected:
            raise ValueError(f'Frozen confirmation dependency changed: {path}')
    if protocol['seeds'] != list(range(1311, 1321)) or protocol['seeds'] != acquisition_cfg['confirmation_seeds']:
        raise ValueError('Expected exactly the ten reserved confirmation fleets')
    lock = dict(development_provenance=prior, protocol=protocol,
        runner_sha256=parent.file_hash(__file__), config_sha256=parent.file_hash(CONFIG),
        confirmation_unlocked=True)
    return protocol, cfg, basis, mixture, acquisition_cfg, controller_protocol, settings, acquisition_lock['controller'], lock


def path_for(seed, kind='result', directory=None):
    return (OUT if directory is None else directory) / f'{PREFIX}_{kind}_seed{seed}.json'


def save(path, payload):
    parent.save_json(path, dict(payload=payload, payload_sha256=hashlib.sha256(parent.canonical(payload)).hexdigest()))


def read(path, seed, lock):
    envelope = parent.load(path)
    payload = envelope['payload']
    if hashlib.sha256(parent.canonical(payload)).hexdigest() != envelope['payload_sha256']:
        raise ValueError(f'Corrupt checkpoint: {path}')
    if payload['seed'] != seed or payload['provenance'] != lock:
        raise ValueError(f'Stale checkpoint: {path}; never overwrite or reuse silently')
    return payload


@contextmanager
def execution_lock(directory):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'execution.lock'
    try:
        with path.open('x', encoding='utf-8') as stream:
            stream.write('An execution is active. Remove only after verifying that it has stopped.\n')
    except FileExistsError as exc:
        raise RuntimeError(f'Execution lock exists: {path}. If the prior process crashed, verify it has stopped before deleting this file.') from exc
    try:
        yield
    finally:
        path.unlink()


def make_clients(seed, acquisition_cfg, basis, mixture):
    """Same sampling and coverage assignment as development; no reference-based selection."""
    clients = sample_fleet(seed, clients_per_family=10)
    tags = label_free_categories(clients, np.random.default_rng(seed + acquisition_cfg['coverage_rng_offset']), mixture['coverage_blocks'])
    frozen = []
    for client in clients:
        reference = collect_targets(client, acquisition_cfg['curvature_per_m'], basis)[1]
        frozen.append(dict(client=client.client_id, family=client.family, parameters=asdict(client.parameters),
            speeds=list(map(float, mixture['coverage_blocks'][tags[client.client_id]])), reference=np.asarray(reference).tolist()))
    if len(frozen) != 30 or sum(len(c['speeds']) for c in frozen) != 83:
        raise ValueError('Unexpected fleet/coverage size')
    return frozen


def cell_key(cell):
    return cell['client_index'], float(cell['speed'])


def validate_cells(result):
    expected = {(i, float(v)) for i, c in enumerate(result['clients']) for v in c['speeds']}
    keys = [cell_key(c) for c in result['cells']]
    if len(keys) != len(set(keys)) or not set(keys) <= expected:
        raise ValueError('Duplicate or unexpected acquisition cells')
    for cell in result['cells']:
        client = result['clients'][cell['client_index']]
        if cell['client'] != client['client'] or cell['parameters'] != client['parameters']:
            raise ValueError('Acquisition client identity changed')
    return expected - set(keys)


def raw_path(seed, index, speed, directory=None):
    return (DATA if directory is None else directory) / f'seed{seed}_client{index}_speed{int(speed)}.npz'


def fit_cell(seed, index, client, speed, cfg, protocol, settings, controller, basis, raw_directory=None):
    start = time.perf_counter()
    acquisition_seed = parent.acquisition_seed(seed, index, speed)
    reference = np.asarray(client['reference'][list(basis['speed_grid']).index(speed)])
    row = dict(client_index=index, client=client['client'], parameters=client['parameters'], speed=float(speed),
        acquisition_seed=acquisition_seed, reference=reference.tolist())
    try:
        traces = [acq.generate_fixed_lqi(VehicleParameters(**client['parameters']), speed, acquisition_seed,
            cfg, protocol, controller, validation) for validation in [False, True]]
        arrays = parent.pack_traces(*traces)
        row['array_sha256'] = {k: acq.array_hash(v) for k, v in arrays.items()}
        parent.write_raw(raw_path(seed, index, speed, raw_directory), arrays)
        training, validation = traces
        y, u = acq.inputs(training, cfg)  # Known applied steering; never observed_steering.
        three, four = acq.add_nominal_start(y, u, settings['estimator'], controller)
        records, center = acq.observations(training, cfg, 'clean')
        target, diagnostic = acq.fit_records(records, center, cfg)
        clean = dict(target=target.tolist(), offset=(center[:2] - target[:4].reshape(2, 2) @ center[:2] - target[4:] * center[2]).tolist(),
                     converged=diagnostic['fit_valid'])
        physical = [acq.physical_diagnostics(trace, speed, cfg) for trace in traces]
        row.update(status='simulated', models=dict(ClosedOE4=acq.score(four, validation, reference, cfg),
            ClosedCleanOLS=acq.score(clean, validation, reference, cfg)), closed_optimizer=four,
            closed_three_start_optimizer=three, training=physical[0], validation=physical[1],
            physical_pass=all(p['physical_pass'] for p in physical), settling_pass=all(p['settling_pass'] for p in physical),
            tracking_train=acq.tracking(training, cfg), tracking_validation=acq.tracking(validation, cfg),
            training_measurements_sha256=acq.array_hash(y), training_input_sha256=acq.array_hash(u))
    except (FloatingPointError, np.linalg.LinAlgError) as exc:
        row.update(status='numerical_failure', reason=f'{type(exc).__name__}: {exc}', models={})
    row['elapsed_s'] = time.perf_counter() - start
    return row


def run_fleet(seed):
    protocol, cfg, basis, mixture, acquisition_cfg, controller_protocol, settings, controller, lock = preflight()
    result_path, acquisition_path = path_for(seed), path_for(seed, 'acquisition')
    if result_path.exists():
        result = read(result_path, seed, lock)
        acquired = read(acquisition_path, seed, lock)
        if validate_cells(acquired) or result['acquisition_sha256'] != parent.file_hash(acquisition_path):
            raise ValueError('Completed result no longer matches acquisition checkpoint')
        return dict(seed=seed, status=result['status'], resumed=True)
    if acquisition_path.exists():
        acquired = read(acquisition_path, seed, lock)
    else:
        acquired = dict(seed=seed, provenance=lock, status='acquiring',
            clients=make_clients(seed, acquisition_cfg, basis, mixture), cells=[])
        save(acquisition_path, acquired)
    pending = validate_cells(acquired)
    with threadpool_limits(limits=1):
        for i, client in enumerate(acquired['clients']):
            for speed in client['speeds']:
                if (i, speed) not in pending:
                    continue
                acquired['cells'].append(fit_cell(seed, i, client, speed, acquisition_cfg,
                    controller_protocol, settings, controller, basis))
                save(acquisition_path, acquired)
                print(json.dumps(dict(seed=seed, completed_cells=len(acquired['cells']), total_cells=83)), flush=True)
        acquired['status'] = 'acquisition_complete'
        save(acquisition_path, acquired)
        result = integrate(acquired, cfg, basis, mixture, acquisition_cfg)
        result['acquisition_sha256'] = parent.file_hash(acquisition_path)
        save(result_path, result)
    return dict(seed=seed, status=result['status'], blocked_conditions=result['blocked_conditions'])


def integrate(acquired, cfg, basis, mixture, acquisition_cfg):
    if validate_cells(acquired):
        raise ValueError('Incomplete acquisition cannot be integrated')
    conditions, inputs, blocked = {}, {}, {}
    for condition in cfg['conditions']:
        if any(c['status'] != 'simulated' for c in acquired['cells']):
            blocked[condition] = 'At least one numerical acquisition/identification failure; entire fleet blocked.'
            continue
        try:
            speeds, targets, diagnostics = lpv.assemble_inputs(acquired, acquired, condition, acquisition_cfg)
        except ValueError as exc:
            blocked[condition] = str(exc)
            continue
        try:
            fitted = lpv.fit_models(speeds, targets, acquired['seed'], cfg, basis, mixture)
        except (FloatingPointError, np.linalg.LinAlgError) as exc:
            blocked[condition] = f'Numerical LPV fit failure: {type(exc).__name__}: {exc}'
            continue
        lpv.add_diagnostic_models(fitted, acquired['clients'], speeds, targets, basis, mixture)
        errors, sensitivity = lpv.evaluate(fitted, acquired['clients'], speeds, basis)
        conditions[condition] = dict(**fitted, errors=errors, sensitivity_errors=sensitivity, local_identification=diagnostics)
        inputs[condition] = [target.tolist() for target in targets]
    return dict(status='execution_complete' if not blocked else 'execution_blocked', seed=acquired['seed'],
        provenance=acquired['provenance'], clients=acquired['clients'], grid=list(basis['speed_grid']),
        conditions=conditions, inputs=inputs, blocked_conditions=blocked, confirmation_unlocked=True,
        control_benefit_tested=False)


def restore_cell(seed, cell, cfg, protocol, controller, directory=None):
    if 'array_sha256' not in cell:
        return False
    path = raw_path(seed, cell['client_index'], cell['speed'], directory)
    exists = path.exists()
    if exists:
        with np.load(path, allow_pickle=False) as data:
            arrays = {k: data[k] for k in data.files}
    else:
        traces = [acq.generate_fixed_lqi(VehicleParameters(**cell['parameters']), cell['speed'], cell['acquisition_seed'],
            cfg, protocol, controller, validation) for validation in [False, True]]
        arrays = parent.pack_traces(*traces)
    if set(arrays) != set(cell['array_sha256']) or any(acq.array_hash(arrays[k]) != sha for k, sha in cell['array_sha256'].items()):
        raise ValueError(f'Reconstruction mismatch: {path}')
    if not exists:
        parent.write_raw(path, arrays)
    return not exists


def restore_fleet(seed):
    _, _, _, _, cfg, protocol, _, controller, lock = preflight()
    result = read(path_for(seed, 'acquisition'), seed, lock)
    restored = sum(restore_cell(seed, c, cfg, protocol, controller) for c in result['cells'])
    return dict(seed=seed, restored=restored)


def paired_intervals(units, protocol):
    view = units[(units.condition == 'ClosedOE4') & (units.region == 'unseen')].pivot(index='seed', columns='method', values='matrix')
    if set(view.index) != set(protocol['seeds']):
        return []  # Never silently infer from a surviving subset.
    out = []
    for baseline in protocol['primary_baselines']:
        differences = (view[baseline] - view.Federated).to_numpy() * 100
        margin = t.ppf(1 - protocol['family_alpha'] / (2 * len(protocol['primary_baselines'])), len(differences)-1) * np.std(differences, ddof=1) / np.sqrt(len(differences))
        out.append(dict(baseline=baseline, mean_difference_pp=float(np.mean(differences)),
            lower_pp=float(np.mean(differences)-margin), upper_pp=float(np.mean(differences)+margin),
            wins=int(np.sum(differences > 0)), fleets=len(differences), differences_pp=differences.tolist()))
    return out


def analyze():
    protocol, cfg, _, _, _, _, _, _, lock = preflight()
    results, hashes, acquisitions = [], {}, []
    for seed in protocol['seeds']:
        for kind in ['acquisition', 'result']:
            path = path_for(seed, kind)
            if not path.exists() or not parent.tracked_unchanged(path):
                raise ValueError('Complete, commit and push all 20 execution checkpoints before analyze.')
            hashes[path.relative_to(ROOT).as_posix()] = parent.file_hash(path)
        result = read(path_for(seed), seed, lock)
        acquired = read(path_for(seed, 'acquisition'), seed, lock)
        if validate_cells(acquired) or result['acquisition_sha256'] != parent.file_hash(path_for(seed, 'acquisition')):
            raise ValueError('Acquisition/result mismatch')
        results.append(result)
        acquisitions.append(acquired)
    usable = [r for r in results if r['conditions']]
    intervals, comparisons = [], []
    units = None
    if usable:
        # A blocked secondary/primary condition must not break failure reporting.
        frames = []
        for condition in cfg['conditions']:
            available = [dict(r, conditions={condition:r['conditions'][condition]}) for r in usable if condition in r['conditions']]
            if available:
                u, s, p, c = lpv.summarize(available, {**cfg, 'conditions':[condition]})
                frames.append((u,s,p)); comparisons.extend(c)
        units, sensitivity, per_speed = [lpv.pd.concat([f[i] for f in frames], ignore_index=True) for i in range(3)]
        intervals = paired_intervals(units, protocol)
        for suffix, frame in [('fleet_summary', units), ('smooth_sensitivity', sensitivity), ('speed_summary', per_speed)]:
            frame.to_csv(OUT / f'{PREFIX}_{suffix}.csv', index=False, lineterminator='\n')
    equivalence = all('ClosedOE4' in r['conditions'] and r['conditions']['ClosedOE4']['equivalence']['passed'] for r in results)
    manifest = dict(status='confirmation_analysis_complete', provenance=lock, checkpoint_sha256=hashes,
        primary_intervals=intervals, descriptive_comparisons=comparisons,
        primary_fleets=sum('ClosedOE4' in r['conditions'] for r in results), requested_fleets=10,
        all_primary_equivalence_pass=equivalence,
        primary_confirmation_supported=bool(equivalence and len(intervals)==2 and all(v['lower_pp'] > 0 for v in intervals)),
        failures=[dict(seed=r['seed'], conditions=r['blocked_conditions']) for r in results if r['blocked_conditions']],
        acquisition_diagnostics=acquisition_diagnostics(acquisitions),
        client_wins=client_wins(results), control_benefit_tested=False)
    parent.save_json(OUT / f'{PREFIX}_analysis.json', manifest)
    write_report(results, units, manifest, REPORT)
    print(REPORT)


def acquisition_diagnostics(acquisitions):
    rows=[]
    for acquired in acquisitions:
        cells=acquired['cells'];simulated=[c for c in cells if c['status']=='simulated']
        errors=[c['models']['ClosedOE4']['matrix_error'] for c in simulated if c['models']['ClosedOE4']['fit_valid']]
        rows.append(dict(seed=acquired['seed'], total=len(cells), simulated=len(simulated),
            valid_primary=len(errors), accuracy_pass=sum(c['models']['ClosedOE4']['accuracy_pass'] for c in simulated),
            original_zero_limiter_physical_pass=sum(c['physical_pass'] for c in simulated),
            matrix_median_pct=float(100*np.median(errors)) if errors else None,
            matrix_p95_pct=float(100*np.percentile(errors,95)) if errors else None,
            limiter_fraction_mean=float(np.mean([c['tracking_validation']['limiter_fraction'] for c in simulated])) if simulated else None))
    return rows


def client_wins(results):
    rows=[]
    for result in results:
        for condition,fitted in result['conditions'].items():
            for baseline in ['RestrictedLocal','SmoothLocal','Global']:
                wins=0
                for i,client in enumerate(result['clients']):
                    mask=lpv.region_mask(result['grid'],client['speeds'],'unseen')
                    errors=fitted['errors']
                    wins+=int(np.mean(np.array(errors['Federated'][i]['matrix'])[mask]) < np.mean(np.array(errors[baseline][i]['matrix'])[mask]))
                rows.append(dict(seed=result['seed'],condition=condition,baseline=baseline,wins=wins,clients=len(result['clients'])))
    return rows


def write_report(results, units, manifest, destination):
    table = acq.table
    body = '<h1>Independent OE–LPV confirmation</h1><p>Ten reserved fleets, fixed protocol. Positive differences favor Federated. Fleet is the independent unit.</p>'
    body += '<h2>Primary comparisons</h2><p>Simultaneous 95% family confidence via two Bonferroni-adjusted paired Student-t intervals (each 97.5%). Approximate inference assumes independent fleet differences with a sufficiently regular distribution; all ten differences and win counts are saved. A positive lower bound for both comparisons and all ten valid, numerically equivalent fits are required for the prespecified primary claim.</p>'
    body += table(['Baseline','Mean difference [pp]','Lower [pp]','Upper [pp]','Fleet wins'],
        [[v['baseline'],v['mean_difference_pp'],v['lower_pp'],v['upper_pp'],v['wins']] for v in manifest['primary_intervals']])
    body += f'<p>Primary confirmation supported: <strong>{manifest["primary_confirmation_supported"]}</strong>. Valid primary fleets: {manifest["primary_fleets"]}/10. No failed fleet is removed to obtain a confirmation claim.</p>'
    if units is not None:
        rows = units[(units.condition=='ClosedOE4') & (units.region=='unseen')]
        body += '<h2>Individual fleet results</h2>' + table(['Fleet','Method','Unseen matrix [%]','A [%]','B [%]','Unstable fraction [%]'],
            [[int(r.seed),r.method,100*r.matrix,100*r.a,100*r.b,100*r.unstable_fraction] for r in rows.itertuples()])
        def draw(ax):
            for method in lpv.PRIMARY:
                v=rows[rows.method==method].sort_values('seed')
                ax.plot(v.seed,100*v.matrix,marker='o',label=method)
            ax.set(xlabel='Independent fleet seed',ylabel='Unobserved-speed matrix error [%]')
            ax.legend();ax.grid(alpha=.2)
        body += lpv.svg_figure(draw)
    body += '<h2>Local identification and acquisition</h2>' + table(['Fleet','Cells','Simulated','Valid OE','Accuracy passes','Median matrix [%]','p95 [%]','Mean validation limiter [%]'],
        [[v['seed'],v['total'],v['simulated'],v['valid_primary'],v['accuracy_pass'],v['matrix_median_pct'],v['matrix_p95_pct'],100*v['limiter_fraction_mean'] if v['limiter_fraction_mean'] is not None else None] for v in manifest.get('acquisition_diagnostics',[])])
    body += '<h2>Client-level unseen-speed wins</h2><p>Descriptive counts; clients within a fleet are not independent inference units.</p>' + table(['Fleet','Condition','Baseline','Client wins','Clients'],
        [[v['seed'],v['condition'],v['baseline'],v['wins'],v['clients']] for v in manifest.get('client_wins',[])])
    groups=[];unstable=[]
    for result in results:
        for condition,fitted in result['conditions'].items():
            for g in fitted['groups']:
                groups.append([result['seed'],condition,g['group'],g['clients'],g['rank'],g['condition'],str(g['unique_speeds'])])
            for i,error in enumerate(fitted['errors']['Federated']):
                for j,rho in enumerate(error['spectral_radius']):
                    if rho>=1:
                        unstable.append([result['seed'],condition,result['clients'][i]['client'],fitted['models']['assignments'][i],result['grid'][j],rho])
    body += '<h2>Group coverage</h2>' + table(['Fleet','Condition','Group','Clients','Rank / 7','Condition number','Observed speeds'],groups)
    body += '<h2>Unstable learned model instances</h2><p>Every recipient/model instance with spectral radius ≥1 is listed. Shared group-speed models may appear for multiple clients. These are model diagnostics, not closed-loop stability tests. No clipping or fallback is applied.</p>' + table(['Fleet','Condition','Client','Group','Speed','Spectral radius'],unstable)
    body += '<h2>Failures and reproducibility</h2><pre>'+html.escape(json.dumps(manifest['failures'],indent=2))+'</pre><p>Clean-state OLS and seven smoothness strengths are secondary diagnostics saved in the checkpoints and CSVs; they cannot change the primary settings. Historical zero-limiter criteria remain unchanged: this confirmation permits limiter activity with known applied input, but requires bounds and settling. No controller benefit is tested.</p><p><a href="../docs/oe_lpv_confirmation.html">Frozen protocol and commands</a></p>'
    destination.write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>OE–LPV confirmation</title><style>body{font:16px/1.6 system-ui;background:#edf3f7;color:#203345}main{max-width:1200px;margin:auto;background:white;padding:28px}table{border-collapse:collapse;font-size:13px}td,th{padding:7px;border-bottom:1px solid #ddd;text-align:right}.scroll{overflow:auto}svg{width:100%;height:auto}</style><main>'+body+'</main></html>',encoding='utf-8',newline='\n')


def smoke_worker():
    _, cfg, basis, _, acquisition_cfg, protocol, settings, controller, lock = preflight()
    # Nominal technical fixture only; no reserved fleet is generated or inspected.
    with tempfile.TemporaryDirectory(prefix='oe_confirmation_smoke_') as tmp, threadpool_limits(limits=1):
        directory=Path(tmp)
        vehicle=family_centers()['nominal']
        speed=10.
        a,b,_=discrete_jacobian(vehicle,speed,acquisition_cfg['curvature_per_m'],acquisition_cfg['sample_time'])
        reference=np.r_[a.ravel(),b.ravel()]
        client=dict(client='technical_nominal',parameters=asdict(vehicle),speeds=[speed],reference=[np.asarray(reference).tolist()])
        fixture_basis={**basis,'speed_grid':[speed]}
        cell=fit_cell(1300,0,client,speed,acquisition_cfg,protocol,settings,controller,fixture_basis,directory)
        if cell['status']!='simulated' or not cell['models']['ClosedOE4']['fit_valid']:
            raise AssertionError('Technical OE smoke failed')
        path=raw_path(1300,0,speed,directory);path.unlink()
        assert restore_cell(1300,cell,acquisition_cfg,protocol,controller,directory)
        assert not restore_cell(1300,cell,acquisition_cfg,protocol,controller,directory)
        save(directory/'checkpoint.json',dict(seed=1300,provenance=lock,cells=[cell]))
        assert read(directory/'checkpoint.json',1300,lock)['cells']==[cell]
    lpv.smoke_worker()
    return dict(status='smoke_passed',scientific_fleets_run=False,trajectory_reconstruction_hashes_verified=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['preflight','smoke','run','status','restore','analyze'])
    parser.add_argument('--workers',type=int,default=5)
    args=parser.parse_args()
    if args.workers<1:parser.error('--workers must be positive')
    protocol,*_,lock=preflight()
    if args.command=='preflight':
        print(json.dumps(dict(status='preflight_passed',fleets=10,local_fits=830,trajectories=1660,seeds=protocol['seeds'],confirmation_unlocked=True)))
    elif args.command=='smoke':
        with ProcessPoolExecutor(max_workers=1,mp_context=get_context('spawn')) as pool:
            print(json.dumps(pool.submit(smoke_worker).result()))
    elif args.command=='analyze':analyze()
    elif args.command=='status':
        for seed in protocol['seeds']:
            p=path_for(seed);a=path_for(seed,'acquisition')
            r=read(p,seed,lock) if p.exists() else None
            acquired=read(a,seed,lock) if a.exists() else None
            if acquired:validate_cells(acquired)
            print(json.dumps(dict(seed=seed,status=r['status'] if r else 'pending',completed_cells=len(acquired['cells']) if acquired else 0)))
    else:
        for path in [CONFIG,Path(__file__)]:
            if not parent.tracked_unchanged(path):raise ValueError('Commit and push the frozen protocol and runner before execution.')
        with execution_lock(DATA):
            fn=run_fleet if args.command=='run' else restore_fleet
            seeds=protocol['seeds'] if args.command=='run' else [s for s in protocol['seeds'] if path_for(s,'acquisition').exists()]
            with ProcessPoolExecutor(max_workers=min(args.workers,10),mp_context=get_context('spawn')) as pool:
                for future in as_completed([pool.submit(fn,seed) for seed in seeds]):
                    print(json.dumps(future.result()),flush=True)
        if args.command=='run':print('Execution finished. Commit and push all 20 acquisition/result JSON checkpoints before analyze.',flush=True)


if __name__=='__main__':main()
