r"""Matched yaw-only observer/control diagnostic using existing physical CT fits.

Windows: python code\experiments\paper_physical_ct_observer.py preflight
No new identification or acquisition. Five methods, 90 vehicles, three speeds.
"""
from __future__ import annotations
import os
for variable in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[variable] = '1'
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import io
import json
import multiprocessing
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'code/src'), str(ROOT/'code/experiments')]
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import paper_physical_ct_oe as old
from paper_physical_ct_coverage_resume import atomic_json
from paper_oe_transient_benchmark import profile
from analyze_physical_ct_control_pilot import rk4_origin_matrices
from federated_lpv.ct_observer_control import (observer_model, kalman_design,
    simulate_observed, origin_closed_loop, trace_hash, yaw_noise)
from federated_lpv.oe_lqi_validation import interpolate
from federated_lpv.vehicle import VehicleParameters, family_centers

PREFIX = 'paper_physical_ct_observer'
CONFIG = ROOT/'code/config'/f'{PREFIX}.json'
OUT = ROOT/'results/tables'
REPORT = ROOT/'report/physical_ct_observer.html'
METRICS = ['tracking_rmse_deg_s', 'tracking_peak_deg_s', 'beta_estimation_rmse_deg',
    'beta_estimation_peak_deg', 'yaw_estimation_rmse_deg_s', 'beta_peak_deg',
    'steering_rate_rms_deg_s', 'steering_rate_peak_deg_s', 'steering_peak_deg',
    'limiter_fraction', 'lateral_acceleration_peak_mps2', 'mean_normalized_innovation_squared']


def path(seed):
    return OUT/f'{PREFIX}_seed{seed}.json'


def inputs(seed):
    return (old.sources(seed)[1], old.envelope(old.result_path(seed)),
        old.envelope(OUT/f'paper_physical_ct_yaw_only_seed{seed}.json'),
        old.envelope(OUT/f'paper_physical_ct_yaw_only_control_seed{seed}.json'))


def preflight():
    cfg = old.load(CONFIG)
    for name, sha in cfg['source_sha256'].items():
        if old.file_hash(ROOT/name) != sha:
            raise ValueError(f'Frozen observer input changed: {name}')
    cc = old.load(old.storage.CONFIG)
    if cfg['observer']['sample_time'] != cc['sample_time']:
        raise ValueError('Observer/controller sampling mismatch')
    nominal = old.storage.nominal_schedule(cc)
    if nominal['status'] != 'designed':
        raise ValueError('Nominal controller unavailable')
    for seed in cfg['seeds']:
        source, full, yaw, controls = inputs(seed)
        for p in [full, yaw, controls]:
            if p['seed'] != seed or p['status'] != 'execution_complete' or len(p['clients']) != 30:
                raise ValueError('Incomplete input fleet')
        for i, client in enumerate(source['clients']):
            for p in [full, yaw, controls]:
                if p['clients'][i]['index'] != i or p['clients'][i]['client'] != client['client']:
                    raise ValueError('Input client mismatch')
            if not all(p['clients'][i]['fit']['converged'] for p in [full, yaw]):
                raise ValueError('Expected completed physical fits')
            if any(v in client['speeds'] for v in cfg['speeds_mps']):
                raise ValueError('Control test speed must be locally unobserved')
    return cfg


def provenance(cfg):
    # Historical numerical modules/helpers are pinned in source_sha256.
    # Explicit paths keep the identity independent of imports and spawn aliases.
    files = [CONFIG, Path(__file__), ROOT/'code/src/federated_lpv/ct_observer_control.py']
    return dict(files={p.relative_to(ROOT).as_posix(): old.file_hash(p) for p in sorted(set(files))},
                frozen=cfg['source_sha256'], runtime=old.runtime())


def validate(payload, seed, cfg, expected_provenance=None):
    if payload['seed'] != seed or payload['provenance'] != (expected_provenance or provenance(cfg)):
        raise ValueError('Stale observer checkpoint: source/config/runtime changed')
    if payload['status'] not in ['running', 'execution_complete']:
        raise ValueError('Unknown checkpoint status')
    clients = payload['clients']
    if [c['index'] for c in clients] != list(range(len(clients))) or len(clients) > 30:
        raise ValueError('Missing/duplicate/out-of-order clients')
    if payload['status'] == 'execution_complete' and len(clients) != 30:
        raise ValueError('Incomplete fleet marked complete')
    source = old.sources(seed)[1]
    expected = {(m, v) for m in cfg['methods'] for v in cfg['speeds_mps']}
    cc = old.load(old.storage.CONFIG)
    for client in clients:
        if client['client'] != source['clients'][client['index']]['client']:
            raise ValueError('Checkpoint client mismatch')
        rows = client['rows']
        if len(rows) != len(expected) or {(r['method'], r['speed']) for r in rows} != expected:
            raise ValueError('Missing/duplicate method-speed cases')
        for speed in cfg['speeds_mps']:
            if len({r['noise_sha256'] for r in rows if r['speed'] == speed}) != 1:
                raise ValueError('Unmatched sensor-noise realizations')
        for row in rows:
            outcome = row['outcome']; metrics = outcome['metrics']
            if outcome['status'] == 'complete':
                if outcome['completed_steps'] != 1200 or not metrics or not np.isfinite(list(metrics.values())).all():
                    raise ValueError('Invalid complete trajectory')
                if not all(k in metrics for k in METRICS) or not row.get('trace_sha256'):
                    raise ValueError('Missing metrics/reconstruction hash')
                feasible = (metrics['beta_violation_samples'] == 0 and
                    metrics['steering_peak_deg'] <= cc['steering_limit_deg']+1e-8 and
                    metrics['steering_rate_peak_deg_s'] <= cc['steering_rate_limit_deg_s']+1e-8)
                if outcome['feasible'] != feasible:
                    raise ValueError('Incorrect feasibility flag')
            elif outcome['status'] not in ['synthesis_failed', 'rollout_failed'] or metrics is not None or outcome['feasible']:
                raise ValueError('Failed cases cannot carry full-duration metrics')


def nominal_matrices():
    fixed = old.load(ROOT/'code/config/paper_fixed_lti_identification.json')['controller_frozen']
    return np.array(fixed['a']), np.array(fixed['b'])


def case(seed, index, method, speed, cfg, data=None):
    source, full, yaw, controls = data if data is not None else inputs(seed)
    cc = old.load(old.storage.CONFIG)
    spec = old.load(ROOT/'code/config/paper_oe_transient_benchmark.json')
    maneuver = profile(spec, speed); settings = cfg['observer']
    identity = cfg['methods'][method]
    if identity['controller'] == 'NominalLTI':
        schedule = old.storage.nominal_schedule(cc)
    else:
        schedule = controls['clients'][index]['schedules'][identity['controller']]
    noise = yaw_noise(seed, index, speed, len(maneuver['time']), settings)
    row = dict(method=method, speed=speed, noise_sha256=trace_hash({'noise': noise}),
               observer=None, true_origin_rho=None, trace_sha256=None)
    arrays = None
    try:
        if schedule['status'] != 'designed':
            raise ValueError('Saved controller schedule unavailable')
        if identity['observer'] == 'NominalLTI':
            a, b = nominal_matrices()
        else:
            fitted = full if identity['observer'] == 'FullStateCT' else yaw
            theta = fitted['clients'][index]['fit']['parameters']
            a, b = observer_model(theta, speed, settings)
        observer = kalman_design(a, b, settings, len(noise))
        row['observer'] = observer['diagnostics']
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
        row['outcome'] = dict(status='synthesis_failed', reason=str(exc),
                             metrics=None, feasible=False, completed_steps=0)
        return row, None
    plant = VehicleParameters(**source['clients'][index]['parameters'])
    outcome, arrays = simulate_observed(plant, schedule, maneuver, observer, noise, cc)
    row['outcome'] = outcome; row['trace_sha256'] = trace_hash(arrays)
    # Truth is used after the rollout, exclusively for the local diagnostic.
    ap, bp = rk4_origin_matrices(plant, speed, cc['sample_time'])
    gain = interpolate(schedule, cc['speed_grid'], [speed])[0][0]
    jacobian = origin_closed_loop(ap, bp, a, b, gain,
        observer['diagnostics']['steady_gain'], cc['sample_time'])
    row['true_origin_rho'] = float(max(abs(np.linalg.eigvals(jacobian))))
    return row, arrays


def run_fleet(seed):
    cfg = old.load(CONFIG); prov = provenance(cfg); data = inputs(seed)
    payload = old.envelope(path(seed)) if path(seed).exists() else dict(seed=seed,
        status='running', provenance=prov, clients=[],
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())
    validate(payload, seed, cfg, prov)
    with threadpool_limits(limits=1):
        for index in range(len(payload['clients']), 30):
            client = data[0]['clients'][index]; rows = []
            for method in cfg['methods']:
                for speed in cfg['speeds_mps']:
                    row, arrays = case(seed, index, method, speed, cfg, data)
                    if (seed == cfg['preview']['seed'] and client['client'] in cfg['preview']['clients']
                        and speed == cfg['preview']['speed'] and arrays is not None):
                        row['preview'] = {k: v[::cfg['preview']['stride']].tolist()
                            for k, v in arrays.items()}
                    rows.append(row)
            payload['clients'].append(dict(index=index, client=client['client'],
                observed_speeds=client['speeds'], rows=rows))
            payload['status'] = 'execution_complete' if len(payload['clients']) == 30 else 'running'
            atomic_json(path(seed), payload)
            print(json.dumps(dict(seed=seed, saved_vehicles=len(payload['clients']),
                                  saved_cases=15*len(payload['clients']))), flush=True)
    validate(payload, seed, cfg, prov)
    return dict(seed=seed, status=payload['status'], cases=450)


def aggregate(frame, cfg):
    frame = frame.copy()
    for name in METRICS:
        if name not in frame:
            frame[name] = np.nan
    clients = frame.groupby(['seed', 'client', 'method'])[METRICS].agg(old.strict_mean).reset_index()
    fleets = clients.groupby(['seed', 'method'])[METRICS].agg(old.strict_mean).reset_index()
    summary = fleets.groupby('method')[METRICS].agg(old.strict_mean).reindex(cfg['methods']).reset_index()
    counts = frame.groupby('method').agg(cases=('status', 'size'),
        complete=('status', lambda s: int((s == 'complete').sum())),
        synthesis_failed=('status', lambda s: int((s == 'synthesis_failed').sum())),
        rollout_failed=('status', lambda s: int((s == 'rollout_failed').sum())),
        feasible=('feasible', 'sum'), origin_unstable=('true_origin_rho', lambda s: int((s >= 1).sum())),
        origin_available=('true_origin_rho', 'count')).reset_index()
    summary = summary.merge(counts, on='method', validate='one_to_one')
    speeds = frame.groupby(['seed', 'method', 'speed'])[METRICS].agg(old.strict_mean).reset_index()
    pairs = []
    for left, right in cfg['paired_comparisons']:
        a = fleets[fleets.method == left].set_index('seed')
        b = fleets[fleets.method == right].set_index('seed')
        for seed in cfg['seeds']:
            for metric in METRICS:
                values = [a.loc[seed, metric], b.loc[seed, metric]]
                pairs.append(dict(seed=seed, first=left, second=right, metric=metric,
                    second_minus_first=values[1]-values[0], available=bool(np.isfinite(values).all())))
    return clients, fleets, summary, speeds, pd.DataFrame(pairs)


def figure_svg(fig):
    import matplotlib.pyplot as plt
    buffer = io.StringIO(); fig.savefig(buffer, format='svg', bbox_inches='tight', metadata={'Date': None})
    plt.close(fig); text = buffer.getvalue()
    return text[text.index('<svg'):]


def write_report(summary, fleets, speeds, paired, previews, cfg):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    def table(frame):
        return '<div class="scroll">'+frame.to_html(index=False, border=0,
            float_format=lambda x: f'{x:.6g}', na_rep='unavailable')+'</div>'
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout='constrained')
    for method in cfg['methods']:
        values = speeds[speeds.method == method].groupby('speed')[METRICS].agg(old.strict_mean)
        for ax, metric, label in zip(axes,
            ['tracking_rmse_deg_s', 'beta_estimation_rmse_deg', 'steering_rate_rms_deg_s'],
            ['Yaw tracking RMSE [deg/s]', 'Sideslip estimation RMSE [deg]', 'Steering-rate RMS [deg/s]']):
            ax.plot(values.index, values[metric], '-o', label=method)
            ax.set(xlabel='Speed [m/s]', ylabel=label); ax.grid(alpha=.2)
    axes[0].legend(fontsize=6); charts = figure_svg(fig)
    for client in cfg['preview']['clients']:
        fig, axes = plt.subplots(2, 2, figsize=(11, 6), layout='constrained')
        for row in previews.get(client, []):
            p = {k: np.array(v) for k, v in row['preview'].items()}
            time = p['time']; label = row['method']
            axes[0, 0].plot(time, np.rad2deg(p['state'][:, 1]), label=label)
            axes[0, 1].plot(time, np.rad2deg(p['state'][:, 1]-p['reference']), label=label)
            axes[1, 0].plot(time, np.rad2deg(p['estimate'][:, 0]-p['state'][:, 0]), label=label)
            axes[1, 1].plot(time[:len(p['steering'])], np.rad2deg(p['steering']), label=label)
        if previews.get(client):
            p = previews[client][0]['preview']
            axes[0, 0].plot(p['time'], np.rad2deg(p['reference']), 'k--', label='Reference')
        for ax, label in zip(axes.flat, ['Yaw rate [deg/s]', 'Yaw tracking error [deg/s]',
                                         'Estimated minus true sideslip [deg]', 'Applied steering [deg]']):
            ax.set(xlabel='Time [s]', ylabel=label); ax.grid(alpha=.2)
        if previews.get(client):
            axes[0, 0].legend(fontsize=6)
        fig.suptitle(f'Prespecified example: {client}, 21 m/s')
        charts += figure_svg(fig)
    text = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Yaw-only observer and control comparison</title><style>body{{font:16px/1.6 system-ui;background:#eef3f6;color:#173044}}main{{max-width:1300px;margin:auto;padding:28px}}section{{background:white;padding:24px;margin:20px 0;border-radius:10px}}.scroll{{overflow:auto}}table{{border-collapse:collapse;white-space:nowrap;font-size:12px}}th,td{{padding:7px;border-bottom:1px solid #ddd}}svg{{width:100%;height:auto}}</style><main>
<h1>Do identified models help when sideslip must be estimated?</h1>
<section><h2>What this test isolates</h2><p>90 reused pilot vehicles × 5 combinations × 3 speeds = 1350 new rollouts. All use yaw measurements only, with 0.03°/s Gaussian sensor noise, known applied steering and known prescribed speed. The same noise sequence is used for every method within a vehicle/speed case. No true-state feedback. Existing controller gains and CT fits are unchanged.</p>
<p>Nominal/Nominal uses the fixed 20 m/s model throughout. FullStateCT/Nominal and YawOnlyCT/Nominal change the controller only. FullStateCT/Matched and YawOnlyCT/Matched also use their fitted physical model for observation, evaluated at the actual fixed speed and straight-running trim. OE nuisance offsets never enter the observer. At this trim friction does not enter the linear observer model.</p>
<p>The sensor covariance R is known. Q is a shared engineering allowance for model discrepancy: per-step standard deviations 0.002° for sideslip and 0.01°/s for yaw rate at 0.01 s. No process noise is injected into the plant and no method-specific tuning is performed. P0=0 reflects the known zero initial state. Measurement correction precedes control; prediction uses the applied steering. The integrator uses measured yaw. Innovation statistics are descriptive under model mismatch.</p>
<p>These fits were acquired previously under full-state feedback. This comparison does not establish a fully yaw-only acquisition-to-learning workflow. FullStateCT remains a privileged training comparator. No new federated estimator is tested. A single paired noise realization per case supports a pilot comparison, not a sensor-noise robustness claim.</p></section>
<section><h2>Failures and aggregate performance</h2>{table(summary)}<p>Each method has 270 planned cases. Finite bound violations remain in the means; any missing full-duration metric invalidates the corresponding fleet and overall mean. Feasibility checks steering/sideslip bounds, not tracking quality. Local origin instability counts use the true plant Jacobian with the steady observer gain; they are diagnostics, not nonlinear stability guarantees.</p></section>
<section><h2>Speed dependence and prespecified examples</h2>{charts}<p>Example clients were fixed before execution (fleet 1301, first vehicle of each family, 21 m/s). Plots use every fifth sample; all metrics use every sample. Reconstruct full traces with the replay command before interpreting high-frequency behavior. Failed trajectories show only their finite prefixes.</p></section>
<section><h2>Paired fleet differences</h2><p>Second minus first: negative means a smaller metric for the second configuration. Observer comparisons hold controller gains fixed. Full-state-feedback historical results are contextual only because sensor noise also changed.</p>{table(paired[paired.metric.isin(['tracking_rmse_deg_s','beta_estimation_rmse_deg','steering_rate_rms_deg_s'])])}</section>
<section><h2>Fleet means</h2>{table(fleets)}</section>
<section><h2>Decision</h2><p>Inspect failures, sideslip error and steering activity before interpreting small tracking differences. Next, if the nominal observer/controller operates satisfactorily, acquire new yaw-only trajectories under that common feedback architecture and refit. This run itself performs no acquisition, fitting, covariance search or automatic confirmation extension.</p></section></main></html>'''
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(text+'\n', encoding='utf-8', newline='\n')


def analyze(cfg):
    records = []; previews = {}; hashes = {}; prov = provenance(cfg)
    for seed in cfg['seeds']:
        payload = old.envelope(path(seed)); validate(payload, seed, cfg, prov)
        if payload['status'] != 'execution_complete':
            raise ValueError('Finish all fleets before analysis')
        hashes[path(seed).relative_to(ROOT).as_posix()] = old.file_hash(path(seed))
        for client in payload['clients']:
            for row in client['rows']:
                outcome = row['outcome']
                records.append(dict(seed=seed, client=client['client'], method=row['method'],
                    speed=row['speed'], status=outcome['status'], feasible=outcome['feasible'],
                    true_origin_rho=row['true_origin_rho'],
                    observer_error_rho=(row['observer'] or {}).get('error_rho'),
                    observability_rank=(row['observer'] or {}).get('observability_rank'),
                    observability_condition=(row['observer'] or {}).get('observability_condition'),
                    **(outcome['metrics'] or {})))
                if 'preview' in row:
                    previews.setdefault(client['client'], []).append(row)
    frame = pd.DataFrame(records)
    clients, fleets, summary, speeds, paired = aggregate(frame, cfg)
    for name, table in [('cases', frame), ('clients', clients), ('fleets', fleets),
                         ('summary', summary), ('speeds', speeds), ('paired', paired)]:
        table.to_csv(OUT/f'{PREFIX}_{name}.csv', index=False)
    atomic_json(OUT/f'{PREFIX}_analysis.json', dict(status='analysis_complete',
        expected_cases=1350, recorded_cases=len(frame), checkpoint_sha256=hashes,
        provenance=prov, results_are_pilot=True, acquisition_changed=False, refits=0))
    write_report(summary, fleets, speeds, paired, previews, cfg)
    print(summary.to_string(index=False)); print(REPORT)


def smoke_worker():
    cfg = old.load(CONFIG); cc = old.load(old.storage.CONFIG)
    schedule = old.storage.nominal_schedule(cc); a, b = nominal_matrices()
    settings = cfg['observer']; spec = old.load(ROOT/'code/config/paper_oe_transient_benchmark.json')
    outcomes = []
    with threadpool_limits(limits=1):
        for speed in [10., 20., 30.]:
            maneuver = profile(spec, speed)
            noise = yaw_noise(991, 0, speed, len(maneuver['time']), settings)
            observer = kalman_design(a, b, settings, len(noise))
            result, arrays = simulate_observed(family_centers()['nominal'], schedule, maneuver, observer, noise, cc)
            if result['status'] != 'complete' or result['completed_steps'] != 1200:
                raise AssertionError('Synthetic observer rollout failed')
            outcomes.append(result)
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)/'unicode_checkpoint.json'
            atomic_json(target, dict(note='yaw rate °/s; β', outcomes=outcomes))
            assert old.envelope(target)['outcomes'] == outcomes
    return dict(status='observer_smoke_passed', synthetic_speeds=[10,20,30],
        windows_spawn=True, scientific_results_written=False)


def replay(args, cfg):
    payload = old.envelope(path(args.seed)); validate(payload, args.seed, cfg)
    client = next((c for c in payload['clients'] if c['client'] == args.client), None)
    if client is None:
        raise ValueError('Requested client has no completed checkpoint')
    expected = next(r for r in client['rows'] if r['method'] == args.method and r['speed'] == args.speed)
    with threadpool_limits(limits=1):
        row, arrays = case(args.seed, client['index'], args.method, args.speed, cfg)
    if arrays is None or row['trace_sha256'] != expected['trace_sha256'] or row['outcome'] != expected['outcome']:
        raise ValueError('Replay differs from committed checkpoint; original results preserved')
    target = ROOT/'results/data'/PREFIX/f'seed{args.seed}_{args.client}_{args.method.replace("/", "_")}_{args.speed:g}.npz'
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, **arrays)
    print(json.dumps(dict(status='replay_hash_verified', path=str(target), sha256=row['trace_sha256'])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['preflight','smoke','run','status','analyze','replay'])
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--seed', type=int, default=1301)
    parser.add_argument('--client', default='nominal_00')
    parser.add_argument('--speed', type=float, default=21.)
    parser.add_argument('--method', default='YawOnlyCT/Matched')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    cfg = preflight()
    if args.command == 'preflight':
        print(json.dumps(dict(status='observer_preflight_passed', fleets=3, vehicles=90,
            methods=list(cfg['methods']), planned_cases=1350, refits=0,
            raw_identification_arrays_required=False, acquisition_changed=False)))
    elif args.command == 'smoke':
        with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context('spawn')) as pool:
            print(json.dumps(pool.submit(smoke_worker).result()))
    elif args.command == 'run':
        with old.execution_lock(PREFIX, 'run'), ProcessPoolExecutor(
            max_workers=min(args.workers,3), mp_context=multiprocessing.get_context('spawn')) as pool:
            for future in as_completed([pool.submit(run_fleet, seed) for seed in cfg['seeds']]):
                print(json.dumps(future.result()), flush=True)
    elif args.command == 'status':
        for seed in cfg['seeds']:
            if path(seed).exists():
                p = old.envelope(path(seed)); validate(p, seed, cfg)
                print(json.dumps(dict(seed=seed, status=p['status'], saved_vehicles=len(p['clients']))))
            else:
                print(json.dumps(dict(seed=seed, status='not_started')))
    elif args.command == 'replay':
        if args.seed not in cfg['seeds'] or args.speed not in cfg['speeds_mps'] or args.method not in cfg['methods']:
            parser.error('Replay seed/speed/method outside protocol')
        replay(args, cfg)
    else:
        analyze(cfg)


if __name__ == '__main__':
    main()
