r"""Structured continuous-time local OE diagnostic, isolated from the frozen paper.

Windows CMD: python code\experiments\paper_physical_ct_oe.py preflight
Commands: preflight, smoke, restore, run, status, analyze, control.
Default cohort: three existing pilot fleets. See docs/physical_ct_oe.html.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
import hashlib
import html
import importlib.metadata
import io
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'code/src'), str(ROOT/'code/experiments')]
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from federated_lpv.physical_ct_oe import fit_physical_oe, discrete_matrices, continuous_matrices
from federated_lpv.structured_oe import propagate
from federated_lpv.vehicle import VehicleParameters
from federated_lpv.trajectory_calibration import array_hash, window_indices
import paper_fixed_lti_identification as acquisition
import paper_oe_lpv_integration as lpv
import paper_oe_lqi_validation as storage
from experiment_10a_basis_coverage_audit import raw_basis

PREFIX = 'paper_physical_ct_oe'
CONFIG = ROOT/'code/config'/f'{PREFIX}.json'
OUT = ROOT/'results/tables'
METHOD = 'PhysicalCTLocal'
load, file_hash, envelope, atomic_json = storage.load, storage.file_hash, storage.envelope, storage.atomic_json


def source_paths(seed):
    if seed < 1311:
        return (OUT/f'paper_fixed_lti_identification_seed{seed}.json',
                OUT/f'paper_oe_lpv_integration_seed{seed}.json')
    return (OUT/f'paper_oe_lpv_confirmation_acquisition_seed{seed}.json',
            OUT/f'paper_oe_lpv_confirmation_result_seed{seed}.json')


def sources(seed):
    paths = source_paths(seed)
    acquired, integrated = (envelope(p) for p in paths)
    if acquired['seed'] != seed or integrated['seed'] != seed:
        raise ValueError('Source seed mismatch')
    if len(acquired['cells']) != 83 or len(integrated['clients']) != 30:
        raise ValueError('Incomplete original fleet')
    if 'ClosedOE4' not in integrated['conditions']:
        raise ValueError('Missing original primary results')
    keys = [(c['client_index'], float(c['speed'])) for c in acquired['cells']]
    expected = {(i, float(v)) for i, c in enumerate(integrated['clients']) for v in c['speeds']}
    if len(set(keys)) != 83 or set(keys) != expected:
        raise ValueError('Duplicate, missing or extra acquisition cells')
    return acquired, integrated


def runtime():
    return dict(python=f'{sys.version_info.major}.{sys.version_info.minor}',
        packages={name: importlib.metadata.version(name) for name in
                  ['numpy', 'scipy', 'pandas', 'scikit-learn', 'threadpoolctl']})


def preflight(cohort):
    cfg = load(CONFIG)
    for name, sha in cfg['source_sha256'].items():
        if file_hash(ROOT/name) != sha:
            raise ValueError(f'Frozen diagnostic dependency changed: {name}')
    original_cfg = load(ROOT/'code/config/paper_trajectory_calibration.json')
    if runtime()['python'] != original_cfg['python_minor'] or any(
        runtime()['packages'][name] != original_cfg['package_versions'][name]
        for name in runtime()['packages']):
        raise ValueError('Activate federated_lpv from environment.yml; reconstruction uses the original numeric versions')
    for key in ['sample_time', 'curvature_per_m', 'state_noise_deg']:
        if cfg['estimator'][key] != original_cfg[key]:
            raise ValueError(f'Original acquisition setting differs: {key}')
    for seed in cfg['cohorts'][cohort]:
        acquired, integrated = sources(seed)
        for i, client in enumerate(integrated['clients']):
            if [client['parameters'][k] for k in ['front_length', 'rear_length']] != cfg['estimator']['geometry_m']:
                raise ValueError('This protocol assumes common known geometry')
            for cell in (c for c in acquired['cells'] if c['client_index'] == i):
                if cell['client'] != client['client'] or cell['parameters'] != client['parameters']:
                    raise ValueError('Source client identity mismatch')
                if cell['status'] != 'simulated' or not cell['models']['ClosedOE4']['fit_valid']:
                    raise ValueError('Invalid original acquisition cell')
                if not all(lpv.bounded(cell[part], original_cfg) and cell[part]['settling_pass']
                           for part in ['training', 'validation']):
                    raise ValueError('Original acquisition failed bounds or settling')
                ref = client['reference'][integrated['grid'].index(cell['speed'])]
                np.testing.assert_array_equal(ref, cell['reference'])
    return cfg


def provenance(seed, cfg):
    files = [CONFIG, Path(__file__), ROOT/'code/src/federated_lpv/physical_ct_oe.py']
    return dict(protocol_sources={p.relative_to(ROOT).as_posix(): file_hash(p) for p in files},
        frozen_sources=cfg['source_sha256'],
        inputs={p.relative_to(ROOT).as_posix(): file_hash(p) for p in source_paths(seed)},
        runtime=runtime())


def result_path(seed, control=False, directory=OUT):
    suffix = 'control_' if control else ''
    return Path(directory)/f'{PREFIX}_{suffix}seed{seed}.json'


def validate_result(payload, seed, cfg, integrated, control=False):
    if payload['seed'] != seed or payload['provenance'] != provenance(seed, cfg):
        raise ValueError(f'Stale diagnostic checkpoint: {seed}; archive it before changing the protocol')
    clients = payload['clients']
    if [c['index'] for c in clients] != list(range(len(clients))) or len(clients) > 30:
        raise ValueError('Invalid checkpoint client order')
    if payload['status'] not in ['running', 'execution_complete']:
        raise ValueError('Unexpected checkpoint status')
    if payload['status'] == 'execution_complete' and len(clients) != 30:
        raise ValueError('Incomplete checkpoint marked complete')
    for c in clients:
        source = integrated['clients'][c['index']]
        if c['client'] != source['client'] or c['observed_speeds'] != source['speeds']:
            raise ValueError('Checkpoint client/coverage changed')
        if not control and c['fit']['converged']:
            if len(c['grid_targets']) != len(integrated['grid']):
                raise ValueError('Incomplete fitted grid')
            if np.asarray(c['grid_targets']).shape != (len(integrated['grid']), 6):
                raise ValueError('Invalid fitted target shape')


@contextmanager
def execution_lock(cohort, command):
    path = ROOT/'results/data'/PREFIX/f'{cohort}_{command}.lock'
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open('x', encoding='utf-8') as stream:
            stream.write(f'PID {os.getpid()}. Remove only after verifying no workers remain.\n')
    except FileExistsError as exc:
        raise RuntimeError(f'Execution lock exists: {path}; stop prior workers before removing a stale lock') from exc
    try:
        yield
    finally:
        path.unlink()


def original_raw_path(seed, cell):
    prefix = 'paper_fixed_lti_identification' if seed < 1311 else 'paper_oe_lpv_confirmation'
    return ROOT/'results/data'/prefix/f'seed{seed}_client{cell["client_index"]}_speed{int(cell["speed"])}.npz'


def traces(seed, cell, restore=False):
    """Recover only the original arrays and require their committed byte hashes."""
    cfg = load(ROOT/'code/config/paper_trajectory_calibration.json')
    path = original_raw_path(seed, cell)
    existed = path.exists()
    if existed:
        with np.load(path, allow_pickle=False) as saved:
            arrays = {k: saved[k] for k in saved.files}
    elif restore:
        protocol = load(acquisition.CONFIG)
        pair = [acquisition.generate_fixed_lqi(VehicleParameters(**cell['parameters']), cell['speed'],
                    cell['acquisition_seed'], cfg, protocol, protocol['controller_frozen'], valid)
                for valid in [False, True]]
        arrays = acquisition.parent.pack_traces(*pair)
    else:
        raise FileNotFoundError(f'Missing {path}. Run restore for this cohort, or run with --restore-missing.')
    if set(arrays) != set(cell['array_sha256']) or any(
        array_hash(arrays[key]) != sha for key, sha in cell['array_sha256'].items()):
        raise ValueError(f'Original trajectory hash mismatch: {path}. Use the original arrays or reconstruct on the original machine/environment; floating-point platform differences are not silently accepted.')
    if not existed:
        acquisition.parent.write_raw(path, arrays)
    pair = tuple({k[len(prefix):]: v for k, v in arrays.items() if k.startswith(prefix)}
                 for prefix in ['train_', 'validation_'])
    y, u = acquisition.inputs(pair[0], cfg)
    if array_hash(y) != cell['training_measurements_sha256'] or array_hash(u) != cell['training_input_sha256']:
        raise ValueError('Training-window/input hash mismatch')
    return pair, not existed


def restore_fleet(seed):
    acquired, _ = sources(seed)
    count = 0
    with threadpool_limits(limits=1):
        for cell in acquired['cells']:
            _, added = traces(seed, cell, restore=True)
            count += added
    return dict(seed=seed, original_cells_verified=83, restored=count)


def dynamic_metrics(prediction, truth, cfg):
    """Historical error plus identity-free increments, poles and pulse response.

    Pulse comparison concerns local linearizations, not an unseen nonlinear
    driving trajectory. Statewise normalization avoids mixing beta and yaw units.
    """
    errors = {k: v.tolist() for k, v in lpv.error_values(prediction, truth).items()}
    identity = np.r_[np.eye(2).ravel(), np.zeros(2)]
    errors['increment'] = (np.linalg.norm(prediction-truth, axis=1)/
                           np.linalg.norm(truth-identity, axis=1)).tolist()
    poles, responses = [], []
    u = np.zeros(201); u[:20] = np.deg2rad(.3)
    for model, ref in zip(prediction, truth):
        a, ar = model[:4].reshape(2, 2), ref[:4].reshape(2, 2)
        eig, exact = np.linalg.eigvals(a), np.linalg.eigvals(ar)
        poles.append(float(min(np.linalg.norm(eig-exact), np.linalg.norm(eig-exact[::-1]))/np.sqrt(2)))
        with np.errstate(over='ignore', invalid='ignore'):
            x = propagate(a, u[:, None]*model[4:], np.zeros(2))
            xr = propagate(ar, u[:, None]*ref[4:], np.zeros(2))
            value = np.linalg.norm(x-xr, axis=0)/np.maximum(np.linalg.norm(xr, axis=0), 1e-15)
        responses.append(value.tolist() if np.isfinite(value).all() else [None, None])
    errors['pole_rms_absolute'] = poles
    errors['pulse_relative_by_state'] = responses
    return errors


def evaluate_client(seed, index, source, acquired, integrated, cfg, restore):
    start = time.perf_counter()
    acfg = load(ROOT/'code/config/paper_trajectory_calibration.json')
    cells = {(c['client_index'], float(c['speed'])): c for c in acquired['cells']}
    segments, validations, selected_cells = [], [], []
    for v in source['speeds']:
        cell = cells[index, float(v)]
        pair, _ = traces(seed, cell, restore)
        y, u = acquisition.inputs(pair[0], acfg)
        segments.append(dict(speed=v, measured_state=y, applied_input=u))
        validations.append(pair[1]); selected_cells.append(cell)
    # Deliberate boundary: fit sees no client metadata, truth, family or holdout.
    fitted = fit_physical_oe(segments, cfg['estimator'])
    result = dict(index=index, client=source['client'], observed_speeds=source['speeds'], fit=fitted,
        training_hashes=[dict(speed=s['speed'], y=array_hash(s['measured_state']), u=array_hash(s['applied_input']))
                         for s in segments])
    if fitted['converged']:
        grid_targets = []
        for v in integrated['grid']:
            a, b = discrete_matrices(fitted['parameters'], v, cfg['estimator']['geometry_m'],
                                     cfg['estimator']['curvature_per_m'], cfg['estimator']['sample_time'])
            grid_targets.append(np.r_[a.ravel(), b])
        prediction = np.asarray(grid_targets); truth = np.asarray(source['reference'])
        validation_rows = []
        for trajectory, trace, cell in zip(fitted['trajectories'], validations, selected_cells):
            # Exactly the previous affine rollout diagnostic: training offset,
            # known applied validation input, true initial state per 1-s window.
            score = acquisition.oe.affine_prediction(trajectory['target'], trajectory['offset'], trace, acfg)
            validation_rows.append(dict(speed=cell['speed'], physical_ct_affine_nrmse=score,
                original_oe_affine_nrmse=cell['models']['ClosedOE4']['affine_rollout_nrmse']))
        # Truth is read only AFTER selection, for descriptive parameter recovery.
        p = source['parameters']
        reference_parameters = [p['front_stiffness']/p['mass'], p['rear_stiffness']/p['mass'],
                                p['mass']/p['yaw_inertia'], .9]
        result.update(grid_targets=prediction.tolist(), errors=dynamic_metrics(prediction, truth, cfg),
            validation=validation_rows, reference_parameters=reference_parameters,
            relative_parameter_error=(np.asarray(fitted['parameters'])/reference_parameters-1).tolist())
    result['elapsed_s'] = time.perf_counter()-start
    return result


def run_fleet(seed, restore=False):
    cfg = load(CONFIG); acquired, integrated = sources(seed)
    path = result_path(seed)
    payload = envelope(path) if path.exists() else dict(seed=seed, status='running',
        provenance=provenance(seed, cfg), clients=[],
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())
    validate_result(payload, seed, cfg, integrated)
    with threadpool_limits(limits=1):
        for index in range(len(payload['clients']), 30):
            client = evaluate_client(seed, index, integrated['clients'][index], acquired, integrated, cfg, restore)
            payload['clients'].append(client)
            payload['status'] = 'execution_complete' if len(payload['clients']) == 30 else 'running'
            atomic_json(path, payload)
            print(json.dumps(dict(seed=seed, saved_clients=len(payload['clients']),
                                  converged=client['fit']['converged'], elapsed_s=client['elapsed_s'])), flush=True)
    return dict(seed=seed, status=payload['status'], converged=sum(c['fit']['converged'] for c in payload['clients']))


def control_fleet(seed):
    """Optional matched transient control for structured and frozen comparators."""
    from federated_lpv.oe_lqi_validation import design_nodes, simulate
    from paper_oe_transient_benchmark import profile
    cfg = load(CONFIG); _, integrated = sources(seed)
    fitted = envelope(result_path(seed)); validate_result(fitted, seed, cfg, integrated)
    if fitted['status'] != 'execution_complete':
        raise ValueError('Finish and checkpoint identification before control')
    control_cfg = load(storage.CONFIG)
    spec = load(ROOT/'code/config/paper_oe_transient_benchmark.json')
    basis = load(ROOT/'code/config/experiment_10a.json')
    path = result_path(seed, True)
    payload = envelope(path) if path.exists() else dict(seed=seed, status='running',
        provenance=provenance(seed, cfg), identification_sha256=file_hash(result_path(seed)), clients=[])
    validate_result(payload, seed, cfg, integrated, True)
    if payload['identification_sha256'] != file_hash(result_path(seed)):
        raise ValueError('Identification checkpoint changed after controller evaluation')
    methods = ['NominalLTI', *cfg['comparators'], METHOD]
    with threadpool_limits(limits=1):
        nominal = storage.nominal_schedule(control_cfg)
        for i in range(len(payload['clients']), 30):
            client = integrated['clients'][i]
            rows = []
            schedules = {}
            for method in methods:
                if method == METHOD:
                    c = fitted['clients'][i]
                    schedules[method] = (design_nodes(np.array(c['grid_targets']), control_cfg)
                        if c['fit']['converged'] else dict(status='synthesis_failed', nodes=[]))
                elif method == 'NominalLTI':
                    schedules[method] = nominal
                else:
                    coeff = lpv.model_for(integrated['conditions']['ClosedOE4']['models'], method, i)
                    schedules[method] = design_nodes(raw_basis(np.array(integrated['grid']), len(coeff), basis)@coeff, control_cfg)
                for v in spec['speeds_mps']:
                    outcome, _ = simulate(VehicleParameters(**client['parameters']), schedules[method],
                                          profile(spec, v), control_cfg)
                    rows.append(dict(method=method, speed=v, outcome=outcome))
            payload['clients'].append(dict(index=i, client=client['client'], observed_speeds=client['speeds'],
                                           schedules=schedules, rows=rows))
            payload['status'] = 'execution_complete' if len(payload['clients']) == 30 else 'running'
            atomic_json(path, payload)
            print(json.dumps(dict(seed=seed, control_clients_saved=len(payload['clients']))), flush=True)
    return dict(seed=seed, status=payload['status'], cases=30*len(methods)*3)


def strict_mean(values):
    values = np.asarray(values, dtype=float)
    return float(np.mean(values)) if len(values) and np.isfinite(values).all() else np.nan


def analyze(cohort, cfg):
    basis = load(ROOT/'code/config/experiment_10a.json')
    rows, fits, validation, controls, hashes = [], [], [], [], {}
    methods = [*cfg['comparators'], METHOD]
    for seed in cfg['cohorts'][cohort]:
        _, source = sources(seed)
        saved = envelope(result_path(seed)); validate_result(saved, seed, cfg, source)
        if saved['status'] != 'execution_complete':
            raise ValueError(f'Incomplete cohort: seed {seed}')
        hashes[result_path(seed).name] = file_hash(result_path(seed))
        for i, client in enumerate(source['clients']):
            fitted = saved['clients'][i]; fit = fitted['fit']; grid = np.array(source['grid'])
            fits.append(dict(seed=seed, client=client['client'], observed_speeds=str(client['speeds']),
                converged=fit['converged'], any_near_bound=any(fit.get('near_bound', [])),
                condition=fit.get('information', {}).get('condition'),
                rank=fit.get('information', {}).get('rank'), elapsed_s=fitted['elapsed_s'],
                **{name: value for name, value in zip(['p1', 'p2', 'p3', 'mu'], fit.get('parameters', [None]*4))},
                **{name+'_relative_error': value for name, value in zip(['p1', 'p2', 'p3', 'mu'], fitted.get('relative_parameter_error', [None]*4))}))
            for v in fitted.get('validation', []):
                validation.append(dict(seed=seed, client=client['client'], **v))
            for method in methods:
                valid = method != METHOD or fit['converged']
                if valid:
                    if method == METHOD:
                        metrics = fitted['errors']
                    else:
                        coeff = lpv.model_for(source['conditions']['ClosedOE4']['models'], method, i)
                        prediction = raw_basis(grid, len(coeff), basis)@coeff
                        metrics = dynamic_metrics(prediction, np.array(client['reference']), cfg)
                for j, v in enumerate(grid):
                    row = dict(seed=seed, client=client['client'], method=method, speed=v,
                               seen=bool(v in client['speeds']), valid=valid)
                    for key in ['matrix', 'a', 'b', 'increment', 'pole_rms_absolute', 'spectral_radius']:
                        row[key] = metrics[key][j] if valid else None
                    for k, name in enumerate(['pulse_beta', 'pulse_yaw']):
                        row[name] = metrics['pulse_relative_by_state'][j][k] if valid else None
                    rows.append(row)
        path = result_path(seed, True)
        if path.exists():
            cp = envelope(path); validate_result(cp, seed, cfg, source, True)
            if cp['identification_sha256'] != file_hash(result_path(seed)):
                raise ValueError('Stale control checkpoint')
            if cp['status'] == 'execution_complete':
                hashes[path.name] = file_hash(path)
                for client in cp['clients']:
                    for row in client['rows']:
                        o = row['outcome']; m = o['metrics'] or {}
                        controls.append(dict(seed=seed, client=client['client'], method=row['method'],
                            speed=row['speed'], complete=o['status']=='complete', feasible=o['feasible'],
                            tracking_rmse_deg_s=m.get('tracking_rmse_deg_s'),
                            beta_peak_deg=m.get('beta_peak_deg'),
                            lateral_acceleration_peak_mps2=m.get('lateral_acceleration_peak_mps2')))
    frame, fit_frame = pd.DataFrame(rows), pd.DataFrame(fits)
    metric_names = ['matrix', 'a', 'b', 'increment', 'pole_rms_absolute', 'pulse_beta', 'pulse_yaw']
    # Average speeds within vehicle, then vehicles within fleet, then fleets.
    # A missing fit/metric invalidates its full fleet mean; never survivor means.
    unseen = frame.loc[~frame.seen]
    units = unseen.groupby(['seed', 'method', 'client'])[metric_names].agg(strict_mean).reset_index()
    fleets = units.groupby(['seed', 'method'])[metric_names].agg(strict_mean).reset_index()
    summary = fleets.groupby('method')[metric_names].agg(strict_mean).reindex(methods).reset_index()
    counts = fit_frame.groupby('seed').agg(expected=('client', 'size'), converged=('converged', 'sum'),
                                         near_bound=('any_near_bound', 'sum')).reset_index()
    differences = []
    for seed in cfg['cohorts'][cohort]:
        f = fleets.loc[fleets.seed == seed].set_index('method')
        for baseline in cfg['comparators']:
            differences.append(dict(seed=seed, baseline=baseline,
                matrix_gain_pp=100*(f.loc[baseline, 'matrix']-f.loc[METHOD, 'matrix']),
                increment_gain_pp=100*(f.loc[baseline, 'increment']-f.loc[METHOD, 'increment'])))
    outputs = dict(grid_errors=frame, parameters=fit_frame, validation=pd.DataFrame(validation),
                   fleet_summary=fleets, summary=summary, paired_differences=pd.DataFrame(differences))
    control_frame = pd.DataFrame(controls)
    control_summary = pd.DataFrame()
    if len(control_frame):
        control_units = control_frame.groupby(['seed', 'method', 'client']).agg(
            tracking_rmse_deg_s=('tracking_rmse_deg_s', strict_mean),
            feasible=('feasible', 'all'), complete=('complete', 'all')).reset_index()
        control_fleets = control_units.groupby(['seed', 'method']).agg(
            tracking_rmse_deg_s=('tracking_rmse_deg_s', strict_mean),
            feasible_clients=('feasible', 'sum'), complete_clients=('complete', 'sum')).reset_index()
        control_summary = control_fleets
        outputs.update(control_cases=control_frame, control_fleet_summary=control_fleets)
    for suffix, data in outputs.items():
        data.to_csv(OUT/f'{PREFIX}_{cohort}_{suffix}.csv', index=False)
    audit = dict(cohort=cohort, status='analysis_complete', checkpoints=hashes,
        all_fits_converged=bool(fit_frame.converged.all()), expected_clients=len(fit_frame),
        converged_clients=int(fit_frame.converged.sum()),
        control_fleets_complete=sorted(set(control_frame.seed)) if len(control_frame) else [],
        note='Reused data diagnostic. No independent confirmation, no filtering and no confidence claims from local sensitivity.')
    acquisition.parent.save_json(OUT/f'{PREFIX}_{cohort}_analysis.json', audit)
    report = ROOT/'report'/f'physical_ct_oe_{cohort}.html'
    report.parent.mkdir(exist_ok=True)
    write_report(report, summary, fleets, counts, fit_frame, frame, pd.DataFrame(validation), control_summary, audit)
    print(summary.to_string(index=False)); print(report)
    return audit


def write_report(path, summary, fleets, counts, parameters, errors, validation, controls, audit):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    def svg(fig):
        buf = io.StringIO(); fig.savefig(buf, format='svg', bbox_inches='tight'); plt.close(fig)
        return buf.getvalue()[buf.getvalue().find('<svg'):]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for method, data in errors.loc[~errors.seen].groupby('method', sort=False):
        grouped = data.groupby(['seed', 'speed'])[['matrix', 'increment']].agg(strict_mean)
        grouped = grouped.groupby('speed').agg(strict_mean)
        for ax, key in zip(axes, ['matrix', 'increment']):
            ax.plot(grouped.index, 100*grouped[key], marker='o', ms=3, label=method)
            ax.set(xlabel='Unobserved speed [m/s]', ylabel='Relative error [%]', title=key.title())
            ax.grid(alpha=.2)
    axes[1].legend(fontsize=8); chart = svg(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    for ax, key in zip(axes, ['mu', 'condition']):
        vals = parameters[key].dropna()
        ax.hist(np.log10(vals) if key == 'condition' else vals, bins=15, color='#25748d')
        ax.set(xlabel='log10 projected sensitivity condition' if key == 'condition' else 'Estimated friction mu', ylabel='Vehicles')
    info_chart = svg(fig)
    def table(df):
        return '<div class="scroll">'+df.to_html(index=False, float_format=lambda x:f'{x:.5g}', na_rep='unavailable', border=0)+'</div>'
    display = summary.copy()
    for key in ['matrix', 'a', 'b', 'increment', 'pulse_beta', 'pulse_yaw']:
        display[key] *= 100
    text = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Physical continuous-time OE diagnostic</title><style>body{{font:16px/1.55 system-ui;background:#eef3f6;color:#163047;margin:0}}main{{max-width:1200px;margin:auto;padding:30px}}section{{background:white;padding:24px;margin:20px 0;border-radius:10px}}table{{border-collapse:collapse;font-size:13px;white-space:nowrap;width:100%}}td,th{{padding:8px;border-bottom:1px solid #dae1e6;text-align:right}}.scroll{{overflow:auto}}svg{{width:100%;height:auto}}h1,h2{{line-height:1.2}}</style><main>
<h1>Does analytical vehicle structure remove the need for sharing?</h1><p>Cohort: {html.escape(audit['cohort'])}. Existing trajectories and frozen comparison methods; no new independent confirmation.</p>
<section><h2>Validity first</h2>{table(counts)}<p>{audit['converged_clients']} / {audit['expected_clients']} fits converged. Bound proximity and weak parameter sensitivity remain diagnostics even when optimization converges. Failed fits stay in the denominator; incomplete fleet metrics are unavailable.</p></section>
<section><h2>Unobserved-speed identification</h2><p>Four physical dynamic parameters per vehicle, fitted jointly over its two or three observed speeds. Four free nuisance parameters per training segment. Methods use the same measurements and known applied steering. PhysicalCTLocal additionally assumes the analytical tanh-tire structure and known geometry.</p>{table(display)}<p>Relative errors are percentages; pole RMS is absolute in the discrete-time plane. Matrix reproduces the historical metric. Increment normalizes by [A−I,B], removing the identity contribution to its denominator; it is still coordinate-dependent. Pulse errors compare each state's local linear response to a 0.3-degree, 0.2-second steering pulse over 2.01 seconds. They are not unseen nonlinear trajectory errors.</p>{chart}</section>
<section><h2>Fleet results</h2><p>Below: relative fractions, not percentages. Speeds are averaged within vehicles, vehicles within fleets, then fleets equally. No statistical confirmation claim is made from these reused cohorts.</p>{table(fleets)}</section>
<section><h2>Parameter information</h2><p>Sensitivities use log physical parameters after allowing offsets and initial states to adapt. Their condition numbers are local diagnostics, not confidence intervals. Inspect friction estimates and bound hits before attributing prediction gains to physical parameter recovery.</p>{info_chart}{table(parameters)}</section>
<section><h2>Observed-speed validation</h2><p>Independent chirp trajectories at the observed speeds; training-fitted affine offsets, one-second windows with true initial states, original state scales. Simulator states are evaluation targets only. This compares PhysicalCTLocal with the existing local affine OE fits; it does not test a state observer.</p>{table(validation)}</section>
<section><h2>Optional closed-loop comparison</h2><p>The control command applies the existing 13/21/29 m/s transient, unchanged LQI weights, gain grid, limits and noise-free feedback. All methods are synthesized and simulated with the same implementation. Means include finite infeasible runs; numerical/synthesis failures make the fleet tracking mean unavailable.</p>{table(controls) if len(controls) else '<p>Not run yet. Complete and checkpoint identification first.</p>'}</section>
<section><h2>Decision</h2><p>First ask whether the structured local model matches or improves the frozen grouped model on unobserved speeds. Then inspect increment, pulse and pole errors, validation, sensitivity and controls. Success challenges the necessity of the flexible 42-coefficient representation for this plant. Failure alone does not prove a need for federation: distinguish nonlinear mismatch, weak information, bounds and optimization.</p></section></main></html>'''
    path.write_text(text, encoding='utf-8', newline='\n')


def smoke_worker():
    cfg = load(CONFIG); settings = cfg['estimator']; rng = np.random.default_rng(8271)
    truth = np.array([47., 59., .56, 1.02]); segments = []
    with threadpool_limits(limits=1):
        for v in [12., 28.]:
            a, b = discrete_matrices(truth, v, settings['geometry_m'], settings['curvature_per_m'], settings['sample_time'])
            u = .012*rng.normal(size=700)
            x = propagate(a, u[:, None]*b+[.00013, -.00021], [.002, .09])
            segments.append(dict(speed=v, measured_state=x, applied_input=u))
        result = fit_physical_oe(segments, settings)
    if not result['converged']:
        raise AssertionError('Synthetic recovery failed')
    np.testing.assert_allclose(result['parameters'], truth, rtol=2e-5)
    assert result['information']['rank'] == 4
    with tempfile.TemporaryDirectory() as directory:
        p = Path(directory)/'checkpoint.json'; atomic_json(p, result)
        assert envelope(p) == result
        for segment in segments:
            ar, br = continuous_matrices(truth, segment['speed'], settings['geometry_m'], settings['curvature_per_m'])
            assert np.isfinite(ar).all() and np.isfinite(br).all()
    return dict(status='smoke_passed', windows_spawn=True, parameters=result['parameters'],
                joint_rank=4, single_speed_ranks=[x['relative_rank'] for x in result['single_speed_information']],
                scientific_results_written=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['preflight', 'smoke', 'restore', 'run', 'status', 'analyze', 'control'])
    parser.add_argument('--cohort', choices=['pilot', 'evaluation'], default='pilot')
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--restore-missing', action='store_true', help='Reconstruct and hash-check original arrays if absent')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    cfg = preflight(args.cohort); seeds = cfg['cohorts'][args.cohort]
    if args.command == 'preflight':
        missing = sum(not original_raw_path(s, c).exists() for s in seeds for c in sources(s)[0]['cells'])
        print(json.dumps(dict(status='preflight_passed', cohort=args.cohort, fleets=len(seeds),
            clients=30*len(seeds), original_speed_cells=83*len(seeds), missing_raw_cells=missing,
            unknown_dynamic_parameters=4, existing_sources_unchanged=True)))
    elif args.command == 'smoke':
        with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context('spawn')) as pool:
            print(json.dumps(pool.submit(smoke_worker).result()))
    elif args.command == 'status':
        for seed in seeds:
            p = result_path(seed)
            if p.exists():
                saved = envelope(p); validate_result(saved, seed, cfg, sources(seed)[1])
                print(json.dumps(dict(seed=seed, status=saved['status'], saved_clients=len(saved['clients']),
                    converged=sum(c['fit']['converged'] for c in saved['clients']))))
            else:
                print(json.dumps(dict(seed=seed, status='not_started')))
    elif args.command == 'analyze':
        print(json.dumps(analyze(args.cohort, cfg)))
    else:
        # Same lock for restore/run prevents overlapping writes of original arrays.
        stage = 'control' if args.command == 'control' else 'identification'
        with execution_lock(args.cohort, stage), ProcessPoolExecutor(max_workers=min(args.workers, len(seeds)),
                mp_context=multiprocessing.get_context('spawn')) as pool:
            if args.command == 'run':
                jobs = [pool.submit(run_fleet, seed, args.restore_missing) for seed in seeds]
            else:
                job = restore_fleet if args.command == 'restore' else control_fleet
                jobs = [pool.submit(job, seed) for seed in seeds]
            for future in as_completed(jobs):
                print(json.dumps(future.result()), flush=True)


if __name__ == '__main__':
    main()
