r"""Small yaw-only nonlinear identification and matched control pilot.

Windows: python code\experiments\paper_tire_relaxation_pilot.py preflight
         python code\experiments\paper_tire_relaxation_pilot.py smoke
         python code\experiments\paper_tire_relaxation_pilot.py run --workers 3
"""
from __future__ import annotations
import os
for variable in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[variable] = '1'
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import hashlib
import importlib.metadata
import io
import json
import multiprocessing
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'code/src'))
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import paper_tire_relaxation_gate as gate
from federated_lpv import tire_relaxation_pilot as model

PREFIX = 'paper_tire_relaxation_pilot'
CONFIG = ROOT/'code/config'/f'{PREFIX}.json'
OUT = ROOT/'results/tables'
RAW = ROOT/'results/data'/PREFIX
REPORT = ROOT/'report/tire_relaxation_pilot.html'
atomic_json = gate.atomic_json
read_checkpoint = gate.read_checkpoint


def canonical_hash(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def identity(cfg):
    files = [CONFIG, Path(__file__), Path(model.__file__), Path(gate.__file__),
        ROOT/'code/src/federated_lpv/tire_relaxation.py',
        ROOT/'code/src/federated_lpv/physical_ct_oe.py',
        ROOT/'code/src/federated_lpv/structured_oe.py']
    return dict(files={p.relative_to(ROOT).as_posix(): canonical_hash(p) for p in files},
        source=cfg['source_sha256'], runtime=dict(python=f'{sys.version_info.major}.{sys.version_info.minor}',
        packages={p: importlib.metadata.version(p) for p in ['numpy', 'scipy', 'pandas', 'threadpoolctl']}))


def preflight():
    cfg = json.loads(CONFIG.read_text(encoding='utf-8'))
    for name, sha in cfg['source_sha256'].items():
        if canonical_hash(ROOT/name) != sha:
            raise ValueError(f'Frozen source changed: {name}')
    if len(cfg['clients']) != 3 or len(cfg['methods']) != 6:
        raise ValueError('Expected three clients and six comparison methods')
    if cfg['integration_max_step'] > cfg['fit']['bounds'][4][0]/max(cfg['evaluation_speeds_mps'])/2:
        raise ValueError('Integration step violates shortest allowed relaxation time')
    for c in cfg['clients']:
        for v in cfg['training_speeds_mps']:
            old = read_checkpoint(gate.path((c['client'], v, 'broadband')))
            if old['status'] != 'execution_complete' or not old['data']['operating_checks_passed']:
                raise ValueError('Missing/infeasible source training record')
    return cfg


def settings(cfg):
    return dict(cfg['fit'], **{k: cfg[k] for k in ['sample_time', 'integration_max_step', 'yaw_noise_std_deg_s']})


def control_settings(cfg):
    return dict(cfg['control'], **{k: cfg[k] for k in ['sample_time', 'integration_max_step', 'yaw_noise_std_deg_s']})


def path(name):
    return OUT/f'{PREFIX}_{name}.json'


def validate(p, cfg, name):
    if p['client'] != name or p['provenance'] != identity(cfg):
        raise ValueError('Protocol/source/runtime changed; checkpoint preserved')
    if p['status'] not in ['running', 'execution_complete']:
        raise ValueError('Invalid checkpoint status')
    if p['status'] == 'execution_complete':
        if p['training'] is None or set(p['training']) != {str(v) for v in cfg['training_speeds_mps']}:
            raise ValueError('Complete checkpoint lacks training provenance')
        for method in ['LocalCT2', 'FixedCT4', 'JointCT4']:
            entry = p['fits'].get(method, {})
            if len(entry.get('attempts', [])) != len(cfg['fit']['length_starts_m']) or 'selected' not in entry:
                raise ValueError('Complete checkpoint lacks optimizer attempts/selection')
        for stage, probes in [('prediction', cfg['validation_inputs']), ('control', cfg['control_reference_hz'])]:
            expected = {f'{m}_v{v}_{probe}' for m in cfg['methods'] for v in cfg['evaluation_speeds_mps'] for probe in probes}
            if set(p[stage]) != expected:
                raise ValueError(f'Complete checkpoint has missing/extra {stage} conditions')


def store_trace(name, key, arrays):
    gate.save_arrays(RAW/name/f'{key}.npz', arrays)
    return dict(trace_sha256=gate.trace_hash(arrays),
        preview={k: np.asarray(v)[::10].tolist() for k, v in arrays.items()})


def training(c, cfg):
    records = []; metadata = {}; gcfg = gate.load_config()
    for v in cfg['training_speeds_mps']:
        job = (c['client'], v, 'broadband'); saved = read_checkpoint(gate.path(job))['data']
        rebuilt, arrays = gate.generate(job, gcfg)
        # Explicitly permit numerically equivalent cross-platform reconstruction;
        # preserve the original hash and record whether the replay is byte exact.
        for k, value in saved['preview'].items():
            np.testing.assert_allclose(rebuilt['preview'][k], value, rtol=1e-9, atol=1e-11)
        for k, value in saved['metrics'].items():
            np.testing.assert_allclose(rebuilt['metrics'][k], value, rtol=1e-9, atol=1e-11)
        records.append(dict(speed=v, applied_input=arrays['steering'], measured_yaw=arrays['measured_yaw']))
        metadata[str(v)] = dict(source_trace_sha256=saved['trace_sha256'],
            exact_source_trace_match=rebuilt['trace_sha256'] == saved['trace_sha256'],
            **store_trace(c['client'], f'training_v{v}', arrays))
    return records, metadata


def signal(cfg, probe):
    local = dict(cfg, inputs=cfg['validation_inputs'])
    return gate.steering(local, probe)


def reference(cfg, speed, probe):
    t = np.arange(round(cfg['duration_s']/cfg['sample_time'])+1)*cfg['sample_time']
    taper = np.sin(np.pi/2*np.clip(np.minimum(t, cfg['duration_s']-t)/cfg['taper_s'], 0, 1))**2
    ref = np.sin(2*np.pi*cfg['control_reference_hz'][probe]*t)*taper
    return t, cfg['reference_peak_ay_mps2']/speed*ref/max(abs(ref))


def noise(cfg, cindex, v, probe_index, role, n):
    return np.random.default_rng(np.random.SeedSequence([cfg['noise_seed'], role, cindex, int(v), probe_index])).normal(
        0., np.deg2rad(cfg['yaw_noise_std_deg_s']), n)


def parameters(p, method, c, cfg):
    if method == 'PreviousCT2': return c['previous_estimated_ratios']
    if method == 'ExactCT2': return c['physical_ratios']
    if method == 'ExactCT4': return c['physical_ratios']+cfg['relaxation_lengths_m']
    selected = p['fits'][method]['selected']
    return selected.get('parameters') if selected['converged'] else None


def prediction_case(par, c, ci, v, probe, cfg):
    t, u = signal(cfg, probe); geom = c['geometry_m']
    true = model.batch_predict(c['physical_ratios']+cfg['relaxation_lengths_m'], [u], [v], geom,
        cfg['sample_time'], cfg['integration_max_step'])[0, 0]
    eps = noise(cfg, ci, v, list(cfg['validation_inputs']).index(probe), 1, len(t))
    arrays = dict(time=t, steering=u, state=true, noise=eps, measured_yaw=true[:, 1]+eps)
    if par is None:
        return dict(status='fit_failed', metrics=None), arrays
    pred = model.batch_predict(par, [u], [v], geom, cfg['sample_time'], cfg['integration_max_step'])[0, 0]
    arrays['predicted_state'] = pred
    rms = lambda x: float(np.sqrt(np.mean(x*x)))
    return dict(status='complete', metrics=dict(yaw_rmse_deg_s=float(np.rad2deg(rms(pred[:, 1]-true[:, 1]))),
        measured_yaw_rmse_deg_s=float(np.rad2deg(rms(pred[:, 1]-arrays['measured_yaw']))),
        beta_rmse_deg=float(np.rad2deg(rms(pred[:, 0]-true[:, 0]))),
        beta_peak_deg=float(np.rad2deg(max(abs(true[:, 0])))),
        ay_peak_mps2=float(max(abs(true[:, 2]+true[:, 3]))))), arrays


def control_case(par, c, ci, v, probe, cfg):
    t, ref = reference(cfg, v, probe)
    eps = noise(cfg, ci, v, list(cfg['control_reference_hz']).index(probe), 2, len(t))
    if par is None:
        return dict(status='fit_failed', metrics=None, feasible=False), None
    try:
        d = model.design(par, v, c['geometry_m'], control_settings(cfg), len(t))
    except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
        return dict(status='synthesis_failed', reason=str(exc), metrics=None, feasible=False), None
    result, arrays = model.control_rollout(np.r_[c['physical_ratios'], cfg['relaxation_lengths_m']],
        c['geometry_m'], v, ref, eps, d, control_settings(cfg))
    result['design'] = d['diagnostics']
    origin = model.closed_loop_matrix(np.r_[c['physical_ratios'], cfg['relaxation_lengths_m']],
        v, c['geometry_m'], d, control_settings(cfg))
    result['design']['actual_plant_origin_rho'] = float(max(abs(np.linalg.eigvals(origin))))
    return result, arrays


def checkpoint(p, target, stage):
    atomic_json(target, p)
    print(json.dumps(dict(client=p['client'], saved=stage)), flush=True)


def worker(name):
    cfg = preflight(); ci = next(i for i, c in enumerate(cfg['clients']) if c['client'] == name)
    c = cfg['clients'][ci]; target = path(name)
    with threadpool_limits(limits=1):
        if target.exists():
            p = read_checkpoint(target); validate(p, cfg, name)
        else:
            p = dict(client=name, status='running', provenance=identity(cfg), training=None,
                fits={}, profiles={}, prediction={}, control={}, implementation_commit=subprocess.check_output(
                    ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())
        if p['status'] == 'execution_complete':
            return dict(client=name, status='already_complete')
        records, trainmeta = training(c, cfg)
        if p['training'] is not None:
            for v, info in trainmeta.items():
                if info['trace_sha256'] != p['training'][v]['trace_sha256']:
                    raise ValueError('Training reconstruction differs from saved pilot data')
        else:
            p['training'] = trainmeta; checkpoint(p, target, 'training')
        for method, free in [('LocalCT2', list(range(4))), ('FixedCT4', [4, 5]), ('JointCT4', list(range(6)))]:
            entry = p['fits'].setdefault(method, dict(attempts=[]))
            for index in range(len(entry['attempts']), len(cfg['fit']['length_starts_m'])):
                ct = np.asarray(c['previous_estimated_ratios'])
                if method != 'FixedCT4': ct = ct*np.asarray(cfg['fit']['ct_start_factors'][index])
                start = ct if method == 'LocalCT2' else np.r_[ct, cfg['fit']['length_starts_m'][index]]
                entry['attempts'].append(model.fit_attempt(records, c['geometry_m'], settings(cfg), start, free))
                checkpoint(p, target, f'{method}_start{index}')
            entry['selected'] = model.select_attempt(entry['attempts'])
            checkpoint(p, target, f'{method}_selected')
        joint = p['fits']['JointCT4']['selected']
        if joint['converged']:
            for index in [4, 5]:
                for factor in cfg['fit']['profile_factors']:
                    key = f'parameter{index}_factor{factor}'
                    if key in p['profiles']: continue
                    start = np.asarray(joint['parameters']); value = start[index]*factor
                    low, high = cfg['fit']['bounds'][index]
                    if not low <= value <= high:
                        fit = dict(converged=False, message='Profile point outside declared bounds')
                    elif factor == 1:
                        fit = copy.deepcopy(joint)
                    else:
                        start[index] = value
                        fit = model.fit_attempt(records, c['geometry_m'], settings(cfg), start,
                                                [j for j in range(6) if j != index])
                    p['profiles'][key] = dict(fixed_index=index, factor=factor, fixed_value=float(value), fit=fit,
                        delta_sse=fit['weighted_sse']-joint['weighted_sse'] if 'weighted_sse' in fit else None)
                    checkpoint(p, target, key)
        for stage, probes, operation in [('prediction', cfg['validation_inputs'], prediction_case),
                                          ('control', cfg['control_reference_hz'], control_case)]:
            for v in cfg['evaluation_speeds_mps']:
                for probe in probes:
                    for method in cfg['methods']:
                        key = f'{method}_v{v}_{probe}'
                        if key in p[stage]: continue
                        par = parameters(p, method, c, cfg)
                        result, arrays = operation(par, c, ci, v, probe, cfg)
                        result.update(method=method, speed=v, probe=probe,
                            observed_speed=v in cfg['training_speeds_mps'])
                        if arrays is not None: result.update(store_trace(name, f'{stage}_{key}', arrays))
                        p[stage][key] = result; checkpoint(p, target, f'{stage}_{key}')
        p['status'] = 'execution_complete'; checkpoint(p, target, 'execution_complete')
    return dict(client=name, status=p['status'])


def restore(cfg, selected):
    """Regenerate ignored arrays; require their saved numeric-byte hashes exactly."""
    for ci, c in enumerate(cfg['clients']):
        if c['client'] not in selected: continue
        p = read_checkpoint(path(c['client'])); validate(p, cfg, c['client'])
        # Regenerate into a temporary output directory; never replace a mismatched
        # original artifact or alter a frozen checkpoint during restoration.
        gcfg = gate.load_config()
        tasks = []
        for v in cfg['training_speeds_mps']:
            _, arrays = gate.generate((c['client'], v, 'broadband'), gcfg)
            tasks.append((f'training_v{v}', arrays, p['training'][str(v)]))
        for stage, operation in [('prediction', prediction_case), ('control', control_case)]:
            for key, record in p[stage].items():
                if 'trace_sha256' not in record: continue
                _, arrays = operation(parameters(p, record['method'], c, cfg), c, ci,
                                      record['speed'], record['probe'], cfg)
                tasks.append((f'{stage}_{key}', arrays, record))
        for key, arrays, saved in tasks:
            if arrays is None or gate.trace_hash(arrays) != saved['trace_sha256']:
                raise ValueError(f'Exact restore mismatch: {key}; use original numerical platform')
        for key, arrays, _ in tasks:
            dest = RAW/c['client']/f'{key}.npz'
            if dest.exists():
                with np.load(dest, allow_pickle=False) as loaded:
                    if gate.trace_hash(dict(loaded)) != gate.trace_hash(arrays):
                        raise ValueError(f'Existing raw file differs: {dest}')
            else: gate.save_arrays(dest, arrays)
        print(json.dumps(dict(client=c['client'], status='exact_arrays_verified', traces=len(tasks))))


def analyze(cfg):
    checkpoints = []
    for c in cfg['clients']:
        p = read_checkpoint(path(c['client'])); validate(p, cfg, c['client'])
        if p['status'] != 'execution_complete': raise ValueError('Run all clients before analysis')
        checkpoints.append(p)
    tables = {}
    for stage in ['prediction', 'control']:
        rows = []
        for p in checkpoints:
            for r in p[stage].values():
                rows.append(dict(client=p['client'], **{k: r[k] for k in ['method', 'speed', 'probe', 'status', 'observed_speed']},
                    feasible=r.get('feasible'), actual_plant_origin_rho=r.get('design', {}).get('actual_plant_origin_rho'),
                    **(r['metrics'] or {})))
        tables[stage] = pd.DataFrame(rows)
        primary = 'yaw_rmse_deg_s' if stage == 'prediction' else 'tracking_rmse_deg_s'
        if primary not in tables[stage]: tables[stage][primary] = np.nan
        tables[stage].to_csv(OUT/f'{PREFIX}_{stage}.csv', index=False)
    fitrows = []; profiles = []
    for p in checkpoints:
        c = next(c for c in cfg['clients'] if c['client'] == p['client'])
        truth = np.r_[c['physical_ratios'], cfg['relaxation_lengths_m']]
        for method, entry in p['fits'].items():
            for i, a in enumerate(entry['attempts']):
                row = dict(client=p['client'], method=method, start=i, converged=a['converged'],
                    selected=entry['selected'].get('selected_start') == i,
                    weighted_mse=a.get('weighted_mse'), nfev=a.get('nfev'),
                    near_bound=any(a.get('near_bound', [])))
                for j, value in enumerate(a.get('parameters', [])):
                    row[f'parameter{j}'] = value; row[f'parameter{j}_error_pct'] = 100*(value/truth[j]-1)
                fitrows.append(row)
        for key, r in p['profiles'].items():
            profiles.append(dict(client=p['client'], **{k: r[k] for k in ['fixed_index', 'factor', 'fixed_value', 'delta_sse']},
                converged=r['fit']['converged'], near_bound=any(r['fit'].get('near_bound', []))))
    tables['fits'] = pd.DataFrame(fitrows); tables['profiles'] = pd.DataFrame(profiles)
    for name in ['fits', 'profiles']: tables[name].to_csv(OUT/f'{PREFIX}_{name}.csv', index=False)
    summary = []
    for stage, metric in [('prediction', 'yaw_rmse_deg_s'), ('control', 'tracking_rmse_deg_s')]:
        for (method, probe), g in tables[stage].groupby(['method', 'probe']):
            complete = int((g.status == 'complete').sum()); expected = len(cfg['clients'])*len(cfg['evaluation_speeds_mps'])
            summary.append(dict(stage=stage, method=method, probe=probe, complete=complete, expected=expected,
                feasible=int(g.feasible.fillna(False).astype(bool).sum()) if stage == 'control' else None,
                mean_rmse_deg_s=float(g[metric].mean()) if complete == expected else None,
                finite_case_mean_rmse_deg_s=float(g[metric].mean()) if metric in g and g[metric].notna().any() else None))
    tables['summary'] = pd.DataFrame(summary); tables['summary'].to_csv(OUT/f'{PREFIX}_summary.csv', index=False)
    diagnostic_rows = []
    for (method, probe), g in tables['control'].groupby(['method', 'probe']):
        row = dict(method=method, probe=probe, complete=int((g.status == 'complete').sum()))
        for metric in ['beta_estimation_rmse_deg', 'steering_rms_deg', 'limiter_fraction', 'ay_peak_mps2', 'actual_plant_origin_rho']:
            row['finite_mean_'+metric] = float(g[metric].mean()) if metric in g and g[metric].notna().any() else None
        diagnostic_rows.append(row)
    tables['control_diagnostics'] = pd.DataFrame(diagnostic_rows)
    tables['control_diagnostics'].to_csv(OUT/f'{PREFIX}_control_diagnostics.csv', index=False)
    render(checkpoints, tables, cfg)
    atomic_json(OUT/f'{PREFIX}_analysis.json', dict(status='analysis_complete', provenance=identity(cfg),
        checkpoint_sha256={path(c['client']).name: canonical_hash(path(c['client'])) for c in cfg['clients']},
        prediction_cases=len(tables['prediction']), control_cases=len(tables['control']), scientific_pass=None))
    print(REPORT)


def render(checkpoints, tables, cfg):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({'svg.hashsalt': PREFIX, 'axes.spines.top': False, 'axes.spines.right': False})
    def svg(fig):
        stream = io.StringIO(); fig.savefig(stream, format='svg', metadata={'Date': None}); plt.close(fig)
        text = stream.getvalue(); return text[text.index('<svg'):]
    def table(df): return '<div class="scroll">'+df.to_html(index=False, border=0, float_format=lambda x: f'{x:.5g}')+'</div>'
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    for ax, stage in zip(axes, ['prediction', 'control']):
        metric = 'yaw_rmse_deg_s' if stage == 'prediction' else 'tracking_rmse_deg_s'
        for method, group in tables[stage].groupby('method'):
            means = group.groupby('speed')[metric].mean()
            counts = group.groupby('speed').status.apply(lambda x: (x == 'complete').sum())
            means[counts != 2*len(cfg['clients'])] = np.nan
            ax.plot(means.index, means, marker='o', label=method)
        ax.set(xlabel='Speed [m/s]', ylabel='Yaw RMSE [deg/s]', title=stage.capitalize()); ax.grid(alpha=.2)
    axes[0].legend(fontsize=8); overview = svg(fig)
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, layout='constrained')
    nominal = next(p for p in checkpoints if p['client'] == 'nominal_00')
    for method in cfg['methods']:
        rec = nominal['control'][f'{method}_v21_dynamic']; preview = rec.get('preview')
        if preview is None: continue
        t = np.asarray(preview['time']); x = np.asarray(preview['state'])
        axes[0].plot(t, np.rad2deg(x[:, 1]), label=method)
        axes[1].plot(t, np.rad2deg(x[:, 1]-preview['reference']))
        axes[2].plot(t[:len(preview['steering'])], np.rad2deg(preview['steering']))
    tt, rr = reference(cfg, 21, 'dynamic'); axes[0].plot(tt, np.rad2deg(rr), 'k--', label='Reference')
    for ax, label in zip(axes, ['Yaw rate [deg/s]', 'Tracking error [deg/s]', 'Steering [deg]']):
        ax.set_ylabel(label); ax.grid(alpha=.2)
    axes[0].legend(fontsize=8, ncol=3); axes[-1].set_xlabel('Time [s]'); traces = svg(fig)
    html = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Tire relaxation: local identification and control pilot</title><style>body{{font:16px/1.6 system-ui;background:#edf2f5;color:#163247}}main{{max-width:1150px;margin:auto;padding:24px}}section{{background:white;padding:24px;margin:20px 0;border-radius:10px}}svg{{width:100%;height:auto}}.scroll{{overflow:auto}}table{{border-collapse:collapse;font-size:13px;white-space:nowrap}}th,td{{padding:8px;border-bottom:1px solid #ddd}}h1,h2{{line-height:1.25}}</style><main>
<h1>Tire relaxation: local identification and closed-loop comparison</h1>
<p>Three selected vehicles, nine reused yaw-only training records, five evaluation speeds. Exploratory pilot; no federation or population-level inference.</p>
<section><h2>Questions and matched comparisons</h2><p>PreviousCT2 uses the existing parameters. LocalCT2 refits all four ratios on the augmented-plant data. FixedCT4 fits two relaxation lengths with the previous CT ratios fixed. JointCT4 estimates all six parameters jointly. ExactCT2 and ExactCT4 use generating ratios; both still use yaw-only observers. The exact four-state reference is not a guaranteed optimum of nonlinear tracking.</p><p>All estimators use known zero initial states, the same training yaw and steering, and training-only start selection. All controllers use the same beta/yaw/integral cost, limits, measurement noise realization and physical plant. Augmented force states have no direct LQI penalty. Their observer covariance is a fixed design allowance. Results compare complete controller/observer designs; they do not attribute gains solely to the feedback gain.</p></section>
<section><h2>Completeness before performance</h2>{table(tables['summary'])}<p>Unconditional means are withheld if any expected run lacks full-duration metrics. Finite infeasible runs remain in means. A finite-case mean is descriptive and must be read with completeness and feasibility counts.</p>{overview}<p>Each plotted point averages three vehicles and two probes only when all six cases complete. Training speeds are 13, 21 and 29 m/s; 17 and 25 m/s test interpolation. Prediction references use different frequencies/phases from training. These are synthetic fixed-speed probes, not a standardized maneuver.</p></section>
<section><h2>Observer, effort and operating diagnostics</h2>{table(tables['control_diagnostics'])}<p>These are means over finite cases, with counts stated explicitly. The actual-plant origin radius uses the true linearized plant, steady observer gain and inactive limits only for evaluation. It is not a nonlinear stability certificate.</p></section>
<section><h2>Local fits and parameter ambiguity</h2>{table(tables['fits'])}<p>All starts are shown, including unsuccessful attempts and bound flags. Parameter errors use simulator truth only after fitting. Small prediction error alone does not establish physical parameter recovery; friction may remain weakly constrained under gentle excitation.</p>{table(tables['profiles'])}<p>Each profile fixes one selected relaxation length at 0.75, 1 or 1.25 times its fitted value and reoptimizes the other five parameters from the selected fit. These sparse local profiles are not confidence intervals or globally minimized profiles. Negative delta SSE signals that the original multistart solution may be suboptimal and requires review; no automatic scientific pass is assigned.</p></section>
<section><h2>Predetermined time-series example</h2><p>nominal_00 at 21 m/s, dynamic reference (0.7 Hz, peak reference v·r = 1 m/s²). This example was selected in the protocol, independently of results. Every controller sees noisy yaw only.</p>{traces}</section>
<section><h2>What to decide next</h2><p>First assess whether JointCT4 gives stable fits and improves held-out prediction over the refitted LocalCT2 baseline. Then compare tracking, sideslip-estimation error, steering effort, feasibility and failures against ExactCT4. Similar control results mean the added physics is not yet important for this task, even if prediction improves. Weak local identification does not by itself demonstrate that a fleet can supply useful information.</p><p>Full condition tables are saved beside the checkpoints. Source hashes, numerical environment, selected parameters, all optimizer attempts, profile results, control designs, trace hashes and previews are saved. Run the restore command on the original numerical platform for strict array reconstruction.</p></section></main></html>'''
    REPORT.write_text('\n'.join(line.rstrip() for line in html.splitlines())+'\n', encoding='utf-8', newline='\n')


def smoke():
    """Short synthetic checks only; no scientific pilot case is executed."""
    from federated_lpv.tire_relaxation import simulate
    cfg = preflight(); s = settings(cfg); p = np.array([50., 55., .6, .9, .4, .7])
    t = np.arange(61)*.01; u = .004*np.sin(2*np.pi*.8*t[:-1]); geom = [1.2, 1.6]
    x = model.batch_predict(p, [u], [21.], geom, .01, .001)[0, 0]
    old = simulate(p, u, 21., geom, .01, .001)[0]
    np.testing.assert_allclose(x, old, rtol=1e-13, atol=1e-13)
    q = np.log(p); _, jac = model.response_jacobian(q, p, list(range(6)), [u], [21.], geom, s)
    for index in [0, 3, 4, 5]:
        plus = p.copy(); minus = p.copy(); plus[index] *= np.exp(1e-4); minus[index] *= np.exp(-1e-4)
        yp = model.batch_predict([plus, minus], [u], [21.], geom, .01, .001)[:, 0, :, 1]
        np.testing.assert_allclose(jac[:, index], (yp[0]-yp[1])/2e-4, rtol=2e-5, atol=1e-8)
    records = [dict(speed=21., applied_input=u, measured_yaw=x[:, 1])]
    fit = model.fit_attempt(records, geom, dict(s, max_nfev=12), np.r_[p[:4], .45, .65], [4, 5])
    if not fit['converged'] or fit['weighted_mse'] > 1e-10: raise AssertionError(fit)
    np.testing.assert_allclose(fit['parameters'][4:], p[4:], rtol=1e-3)
    ccfg = control_settings(cfg); reference_short = .02*np.sin(2*np.pi*.5*t)
    for par in [p[:4], p]:
        d = model.design(par, 21., geom, ccfg, len(t))
        result, arrays = model.control_rollout(p, geom, 21., reference_short, np.zeros(len(t)), d, ccfg)
        if result['status'] != 'complete': raise AssertionError(result)
        d2 = model.design(par, 21., geom, ccfg, len(t))
        _, replay = model.control_rollout(p, geom, 21., reference_short, np.zeros(len(t)), d2, ccfg)
        assert gate.trace_hash(arrays) == gate.trace_hash(replay)
    with tempfile.TemporaryDirectory() as directory:
        dest = Path(directory)/'checkpoint.json'; atomic_json(dest, dict(message='β → μ; Prüfung', fit=fit))
        assert read_checkpoint(dest)['fit'] == fit
    return dict(status='smoke_passed', synthetic_fit_mse=fit['weighted_mse'],
        four_state_reference_max_difference=float(np.max(abs(x-old))), scientific_cases_run=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['preflight', 'smoke', 'run', 'status', 'analyze', 'restore'])
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--client', choices=['nominal_00', 'heavy_00', 'handling_00'])
    args = parser.parse_args(); cfg = preflight()
    selected = [c['client'] for c in cfg['clients'] if args.client is None or c['client'] == args.client]
    if args.command == 'preflight':
        print(json.dumps(dict(status='preflight_passed', clients=3, reused_training_records=9,
            estimator_starts=27, additional_profile_fits=12, held_out_predictions=180,
            closed_loop_runs=180, known_zero_initial_state=True, yaw_only=True, fleet_learning=False))); return
    if args.command == 'smoke':
        with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context('spawn')) as pool:
            print(json.dumps(pool.submit(smoke).result()))
        return
    if args.command == 'status':
        for name in selected:
            if path(name).exists():
                p = read_checkpoint(path(name)); validate(p, cfg, name)
                print(json.dumps(dict(client=name, status=p['status'], fits=len(p['fits']),
                    predictions=len(p['prediction']), controls=len(p['control']))))
            else: print(json.dumps(dict(client=name, status='missing')))
        return
    if not 1 <= args.workers <= 3: raise ValueError('Use 1–3 workers')
    RAW.mkdir(parents=True, exist_ok=True); lock = RAW/'run.lock'
    try: fd = os.open(lock, os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError(f'Run lock exists: {lock}; remove only after confirming no pilot process is running')
    os.write(fd, str(os.getpid()).encode()); os.close(fd)
    try:
        if args.command == 'analyze': analyze(cfg); return
        if args.command == 'restore': restore(cfg, selected); return
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
            for result in as_completed([pool.submit(worker, name) for name in selected]):
                print(json.dumps(result.result()), flush=True)
    finally: lock.unlink()
    print('Execution complete. Commit/push the three client checkpoints before analyze; see docs/tire_relaxation_pilot.html.')


if __name__ == '__main__': main()
