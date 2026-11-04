r"""Fixed-speed transient tracking with saved OE-LPV controller schedules.

ISO-7401-inspired simulation, not an ISO-compliant steering-input test.
Windows: python code\experiments\paper_oe_transient_benchmark.py preflight
See docs/oe_transient_benchmark.html. No identification or controller synthesis.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
import html
import io
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'paper'), str(ROOT/'code/experiments'), str(ROOT/'code/src')]
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import prepare_oe_control_timeseries as previous

run = previous.run
require = previous.require
PREFIX = 'paper_oe_transient_benchmark'
CONFIG = ROOT/'code/config'/f'{PREFIX}.json'
LABELS = dict(previous.LABELS, RestrictedLocal='Restricted Local', SmoothLocal='Smooth Local',
              ExactLPV='Exact LPV (privileged reference)')


def profile(spec, speed):
    """C2 turn-on/off: one sin^3 cycle, preceded/followed by straight running.

    a_ref is a nominal demand, not measured acceleration or a feasibility bound.
    At constant v, r_ref=a_ref/v and kappa_ref=a_ref/v^2.
    """
    dt = spec['sample_time']
    t = np.arange(round(spec['duration_s']/dt)+1, dtype=float)*dt
    start, stop = spec['active_interval_s']
    demand = np.zeros_like(t)
    inside = (t > start) & (t < stop)
    phase = 2*np.pi*(t[inside]-start)/(stop-start)
    demand[inside] = spec['nominal_lateral_peak_mps2']*np.sin(phase)**3
    return dict(time=t, speed=np.full_like(t, speed), reference=demand/speed,
                curvature=demand/speed**2, nominal_lateral_acceleration=demand)


def prepare():
    spec = run.load(CONFIG)
    for name, expected in spec['source_sha256'].items():
        require(run.file_hash(ROOT/name) == expected, f'Changed benchmark source: {name}')
    old_spec, cfg, payloads, selection, sources = previous.prepare()
    require(spec['methods'] == cfg['methods'], 'All six frozen methods must be evaluated')
    require(spec['seeds'] == cfg['seeds'], 'Confirmation fleets changed')
    require(spec['display_methods'] == old_spec['methods'], 'Display methods changed')
    require(spec['selection'] == old_spec['expected_selection'], 'Display selection changed')
    require(spec['sample_time'] == cfg['sample_time'], 'Sampling time changed')
    require(0 < spec['active_interval_s'][0] < spec['active_interval_s'][1] < spec['duration_s'],
            'Invalid active interval')
    require(spec['nominal_lateral_peak_mps2'] > 0, 'Invalid lateral demand')
    require(len(set(spec['speeds_mps'])) == 3, 'Expected three distinct speeds')
    for speed in spec['speeds_mps']:
        require(cfg['speed_grid'][0] < speed < cfg['speed_grid'][-1], 'Speed outside design envelope')
        for payload in payloads:
            for client in payload['clients']:
                require(not any(np.isclose(speed, v, atol=1e-10, rtol=0) for v in client['observed_speeds']),
                        f'Speed {speed} was observed locally')
        p = profile(spec, speed)
        np.testing.assert_allclose(np.diff(p['time']), cfg['sample_time'], atol=1e-12, rtol=0)
        np.testing.assert_allclose(p['speed']*p['reference'], p['nominal_lateral_acceleration'], atol=1e-14)
        require(np.ptp(p['speed']) == 0, 'Speed must remain constant')
    cfg = dict(cfg, duration_s=spec['duration_s'])
    sources.update(spec['source_sha256'])
    sources[CONFIG.relative_to(ROOT).as_posix()] = run.file_hash(CONFIG)
    return spec, cfg, payloads, selection, sources


def provenance(spec, seed):
    return dict(protocol_sha256=run.file_hash(CONFIG), runtime=run.runtime(),
                source_checkpoint_sha256=run.file_hash(run.result_path(seed)),
                source_sha256=spec['source_sha256'])


def result_path(base, seed):
    return Path(base)/'results/tables'/f'{PREFIX}_seed{seed}.json'


def raw_path(base, seed, index, method, speed):
    return Path(base)/'results/data'/PREFIX/f'seed{seed}'/f'client{index:02d}_{method}_v{speed:g}.npz'


def evaluate(client, method, p, cfg, spec):
    outcome, arrays = run.simulate(run.VehicleParameters(**client['parameters']),
                                   client['schedules'][method], p, cfg)
    extra = None
    if outcome['status'] == 'complete':
        t = arrays['time']
        error = np.rad2deg(arrays['state'][:, 1]-arrays['reference'])
        start, stop = spec['active_interval_s']
        active = (t >= start) & (t <= stop)
        tail = t >= stop
        extra = dict(active_tracking_rmse_deg_s=float(np.sqrt(np.mean(error[active]**2))),
                     tail_tracking_rmse_deg_s=float(np.sqrt(np.mean(error[tail]**2))),
                     terminal_abs_yaw_deg_s=float(abs(np.rad2deg(arrays['state'][-1, 1]))))
    return dict(method=method, speed_mps=float(p['speed'][0]), outcome=outcome,
                extra_metrics=extra, arrays=run.array_hashes(arrays)), arrays


def validate(payload, spec, seed, original):
    require(payload['seed'] == seed and payload['provenance'] == provenance(spec, seed),
            f'Checkpoint provenance mismatch: {seed}')
    clients = payload['clients']
    require([c['index'] for c in clients] == list(range(len(clients))) and len(clients) <= 30,
            'Invalid checkpoint client order')
    require(payload['status'] in ['running', 'execution_complete'], 'Invalid checkpoint status')
    require(payload['status'] != 'execution_complete' or len(clients) == 30, 'Incomplete fleet')
    expected = {(m, float(v)) for m in spec['methods'] for v in spec['speeds_mps']}
    for client in clients:
        source = original['clients'][client['index']]
        for key in ['client', 'family', 'observed_speeds']:
            require(client[key] == source[key], 'Client identity/coverage mismatch')
        require(len(client['rows']) == len(expected) and
                {(r['method'], r['speed_mps']) for r in client['rows']} == expected,
                'Missing or duplicate benchmark cases')
        for row in client['rows']:
            o = row['outcome']
            require(o['status'] in ['complete', 'synthesis_failed', 'rollout_failed'], 'Invalid outcome')
            if o['status'] == 'complete':
                require(o['completed_steps'] == round(spec['duration_s']/spec['sample_time']), 'Incomplete rollout')
                require(o['metrics'] is not None and row['extra_metrics'] is not None, 'Missing complete metrics')
                values = list(o['metrics'].values())+list(row['extra_metrics'].values())
                require(np.isfinite(values).all() and min(values) >= 0, 'Invalid metrics')
                m = o['metrics']
                cfg = original['protocol']
                feasible = (m['beta_violation_samples'] == 0 and
                            m['steering_peak_deg'] <= cfg['steering_limit_deg']+1e-8 and
                            m['steering_rate_peak_deg_s'] <= cfg['steering_rate_limit_deg_s']+1e-8)
                require(o['feasible'] == feasible, 'Feasibility mismatch')
                p = profile(spec, row['speed_mps'])
                for key in ['time', 'speed', 'reference']:
                    require(row['arrays'][key] == run.array_hashes({key:p[key]})[key], 'Profile hash mismatch')
                n = len(p['time'])
                expected_shapes = {k:[n] for k in ['time', 'speed', 'reference', 'integral']}
                expected_shapes['state'] = [n, 2]
                expected_shapes.update({k:[n-1] for k in ['steering', 'requested_steering',
                    'lateral_acceleration', 'limited', 'angle_limited', 'rate_limited']})
                require({k:v['shape'] for k,v in row['arrays'].items()} == expected_shapes, 'Array shape mismatch')
            else:
                require(o['metrics'] is None and row['extra_metrics'] is None and not o['feasible'],
                        'Failures must not have full-duration metrics')


def run_fleet(seed, base):
    spec = run.load(CONFIG)
    original = run.envelope(run.result_path(seed))
    cfg = dict(original['protocol'], duration_s=spec['duration_s'])
    path = result_path(base, seed)
    payload = run.envelope(path) if path.exists() else dict(seed=seed, status='running',
        provenance=provenance(spec, seed),
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(), clients=[])
    validate(payload, spec, seed, original)
    with threadpool_limits(limits=1):
        for client in original['clients'][len(payload['clients']):]:
            saved = {k:client[k] for k in ['index', 'client', 'family', 'observed_speeds']}
            saved['rows'] = []
            for speed in spec['speeds_mps']:
                p = profile(spec, speed)
                for method in spec['methods']:
                    row, _ = evaluate(client, method, p, cfg, spec)
                    saved['rows'].append(row)
            payload['clients'].append(saved)
            if len(payload['clients']) == 30:
                payload['status'] = 'execution_complete'
            validate(payload, spec, seed, original)
            run.atomic_json(path, payload)
            if len(payload['clients']) % 5 == 0:
                print(json.dumps(dict(seed=seed, clients_saved=len(payload['clients']), total=30)), flush=True)
    return dict(seed=seed, status=payload['status'], cases=540)


@contextmanager
def execution_lock(base):
    path = Path(base)/'results/data'/PREFIX/'execution.lock'
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(f'Execution lock exists: {path}. Stop any live workers before removing a stale lock.') from exc
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            stream.write(str(os.getpid()))
        yield
    finally:
        path.unlink()


def audited_results(spec, originals, base):
    results = []
    for original in originals:
        seed = original['seed']
        p = run.envelope(result_path(base, seed))
        validate(p, spec, seed, original)
        require(p['status'] == 'execution_complete', f'Fleet {seed} is incomplete')
        results.append(p)
    return results


def records(results):
    rows = []
    for p in results:
        for c in p['clients']:
            for r in c['rows']:
                o = r['outcome']
                rows.append(dict(seed=p['seed'], index=c['index'], client=c['client'], family=c['family'],
                    method=r['method'], speed_mps=r['speed_mps'], observed_locally=False,
                    status=o['status'], feasible=o['feasible'], reason=o.get('reason', ''),
                    **(o['metrics'] or {}), **(r['extra_metrics'] or {})))
    return pd.DataFrame(rows)


METRICS = ['tracking_rmse_deg_s', 'active_tracking_rmse_deg_s', 'tail_tracking_rmse_deg_s',
           'tracking_peak_deg_s', 'beta_peak_deg', 'steering_peak_deg', 'steering_rate_peak_deg_s',
           'lateral_acceleration_peak_mps2', 'limiter_fraction']


def summaries(frame):
    """Failures stay in counts. Finite infeasible trajectories stay in means.

    Overall means weight available fleet means equally. Paired differences only
    use full 30-client fleets for both methods, with an explicit availability count.
    """
    fleets = []
    groups = [('all', frame)] + [(f, frame[frame.family == f]) for f in sorted(frame.family.unique())]
    for family, group in groups:
        for (seed, speed, method), part in group.groupby(['seed', 'speed_mps', 'method'], sort=True):
            finite = part[part.status == 'complete']
            row = dict(seed=int(seed), family=family, speed_mps=float(speed), method=method,
                cases=len(part), finite=len(finite), feasible=int(part.feasible.sum()),
                synthesis_failed=int((part.status == 'synthesis_failed').sum()),
                rollout_failed=int((part.status == 'rollout_failed').sum()))
            row.update({k:float(finite[k].mean()) if len(finite) else np.nan for k in METRICS})
            fleets.append(row)
    fleet = pd.DataFrame(fleets)
    overall = []
    for (family, speed, method), part in fleet.groupby(['family', 'speed_mps', 'method'], sort=True):
        row = dict(family=family, speed_mps=float(speed), method=method, fleets=len(part),
                   fleets_with_finite=int((part.finite > 0).sum()))
        row.update({k:int(part[k].sum()) for k in ['cases', 'finite', 'feasible', 'synthesis_failed', 'rollout_failed']})
        row.update({k:float(part[k].mean()) for k in METRICS})
        overall.append(row)
    pairs = []
    all_fleets = fleet[fleet.family == 'all']
    for speed in sorted(frame.speed_mps.unique()):
        at_speed = all_fleets[all_fleets.speed_mps == speed]
        fed = at_speed[at_speed.method == 'Federated'].set_index('seed')
        for method in sorted(set(frame.method)-{'Federated'}):
            other = at_speed[at_speed.method == method].set_index('seed')
            valid = (fed.finite == 30) & (other.finite == 30)
            diff = other.loc[valid, 'tracking_rmse_deg_s']-fed.loc[valid, 'tracking_rmse_deg_s']
            pairs.append(dict(speed_mps=float(speed), baseline=method, complete_paired_fleets=int(valid.sum()),
                total_fleets=len(fed), baseline_minus_federated_mean_deg_s=float(diff.mean()),
                min_fleet_difference_deg_s=float(diff.min()), max_fleet_difference_deg_s=float(diff.max())))
    return fleet, pd.DataFrame(overall), pd.DataFrame(pairs)


def replay_cases(spec, cfg, originals, results, selection, base=None, all_cases=False):
    traces = {}
    checked = 0
    chosen = {(c['seed'], c['index']) for c in selection['cases']}
    with threadpool_limits(limits=1):
        for original, result in zip(originals, results, strict=True):
            for source, saved in zip(original['clients'], result['clients'], strict=True):
                selected = (original['seed'], source['index']) in chosen
                for row in saved['rows']:
                    display = selected and row['method'] in spec['display_methods']
                    if not all_cases and not display:
                        continue
                    actual, arrays = evaluate(source, row['method'], profile(spec, row['speed_mps']), cfg, spec)
                    require(actual == row, f"Exact replay mismatch: {original['seed']}/{source['client']}/{row['method']}/{row['speed_mps']}")
                    checked += 1
                    if display:
                        traces[(source['family'], row['speed_mps'], row['method'])] = arrays
                    if base is not None and arrays is not None:
                        path = raw_path(base, original['seed'], source['index'], row['method'], row['speed_mps'])
                        if path.exists():
                            with np.load(path, allow_pickle=False) as saved_arrays:
                                require(run.array_hashes(dict(saved_arrays)) == row['arrays'], f'Changed raw file: {path}')
                        else:
                            run.save_arrays(path, arrays)
    return traces, checked


def make_panels(spec, cfg, traces, selection):
    previous.style()
    plt = previous.plt
    panels = {}
    families = [c['family'] for c in selection['cases']]
    # Shared scales across recipients, speeds and displayed methods; failures are
    # explicitly annotated, never silently replaced by a successful case.
    valid = [a for a in traces.values() if a is not None]
    error_peak = max([float(np.max(abs(np.rad2deg(a['state'][:,1]-a['reference'])))) for a in valid]+[.001])
    yaw_peak = max([float(np.max(abs(np.rad2deg(a['state'][:,1])))) for a in valid] +
                   [float(np.max(abs(np.rad2deg(profile(spec, v)['reference'])))) for v in spec['speeds_mps']])
    steer_peak = max([float(np.max(abs(np.rad2deg(a['steering'])))) for a in valid if len(a['steering'])]+[.1])
    for family in families:
        for speed in spec['speeds_mps']:
            p = profile(spec, speed)
            for signal, label, lim in [('response', 'Yaw rate\n[deg/s]', 1.08*yaw_peak),
                                       ('error', 'Yaw-rate error\n[deg/s]', 1.08*error_peak),
                                       ('steering', 'Applied steering\n[deg]', 1.12*steer_peak)]:
                fig = plt.figure(figsize=(3.48, 1.65))
                ax = fig.add_axes([.23, .25, .75, .69])
                ax.set(xlabel='Time [s]', ylabel=label, xlim=(0, spec['duration_s']), ylim=(-lim, lim))
                ax.set_xticks(np.arange(0, spec['duration_s']+.01, 2))
                ax.ticklabel_format(axis='y', style='plain', useOffset=False)
                ax.grid(color='#dddddd', linewidth=.45); ax.set_axisbelow(True)
                ax.spines[['top','right']].set_visible(False)
                ax.axhline(0, color='#aaaaaa', linewidth=.4)
                if signal == 'response':
                    ax.plot(p['time'], np.rad2deg(p['reference']), color='black', linestyle=':', linewidth=1.3)
                if signal == 'steering' and lim >= cfg['steering_limit_deg']:
                    for bound in [-cfg['steering_limit_deg'], cfg['steering_limit_deg']]:
                        ax.axhline(bound, color='#999999', linestyle=':', linewidth=.7)
                failed = []
                for method in spec['display_methods']:
                    a = traces[(family, float(speed), method)]
                    if a is None or len(a['time']) < len(p['time']):
                        failed.append(LABELS[method])
                    if a is None:
                        continue
                    opts = dict(color=previous.COLORS[method], linestyle=previous.STYLES[method])
                    if signal == 'steering':
                        if len(a['steering']):
                            ax.step(a['time'], np.rad2deg(np.r_[a['steering'], a['steering'][-1]]), where='post', **opts)
                    else:
                        y = a['state'][:,1] - (a['reference'] if signal == 'error' else 0)
                        ax.plot(a['time'], np.rad2deg(y), **opts)
                if failed:
                    ax.text(.02, .96, 'Incomplete: '+', '.join(failed), transform=ax.transAxes, va='top', fontsize=6)
                panels[f'{family}_v{speed:g}_{signal}'] = fig
    from matplotlib.lines import Line2D
    legend = plt.figure(figsize=(7.16, .30))
    handles = [Line2D([], [], color='black', linestyle=':', label='Reference')]
    handles += [Line2D([], [], color=previous.COLORS[m], linestyle=previous.STYLES[m], label=LABELS[m])
                for m in spec['display_methods']]
    legend.legend(handles=handles, loc='center', ncol=4, frameon=False, fontsize=8, handlelength=2.6)
    panels['legend'] = legend
    return panels


def trace_table(traces, selection):
    rows = []
    cases = {c['family']:c for c in selection['cases']}
    for (family, speed, method), a in traces.items():
        if a is None:
            continue
        case = cases[family]
        for k,t in enumerate(a['time']):
            row = dict(seed=case['seed'], client=case['client'], family=family, method=method,
                       speed_mps=speed, time_s=float(t), reference_deg_s=float(np.rad2deg(a['reference'][k])),
                       yaw_rate_deg_s=float(np.rad2deg(a['state'][k,1])), beta_deg=float(np.rad2deg(a['state'][k,0])),
                       yaw_error_deg_s=float(np.rad2deg(a['state'][k,1]-a['reference'][k])))
            row.update({key:float(a[array][k]*scale) if k < len(a[array]) else np.nan for key,array,scale in
                        [('steering_deg','steering',180/np.pi), ('lateral_acceleration_mps2','lateral_acceleration',1),
                         ('angle_limited','angle_limited',1), ('rate_limited','rate_limited',1)]})
            rows.append(row)
    return pd.DataFrame(rows)


def render(spec, cfg, traces, selection, summary, pairs, base):
    figures = Path(base)/'results/figures'; figures.mkdir(parents=True, exist_ok=True)
    report = Path(base)/'report'/f'{PREFIX}.html'; report.parent.mkdir(parents=True, exist_ok=True)
    files, svgs = [], {}
    for name, fig in make_panels(spec, cfg, traces, selection).items():
        stem = PREFIX+'_'+name+('' if name == 'legend' else '_panel')
        pdf = figures/(stem+'.pdf')
        fig.savefig(pdf, metadata={'CreationDate':None, 'ModDate':None}); files.append(pdf)
        fig.savefig(figures/(stem+'.png'), dpi=300)
        buf = io.StringIO(); fig.savefig(buf, format='svg', metadata={'Date':None})
        svgs[name] = buf.getvalue()[buf.getvalue().index('<svg'):]
        previous.plt.close(fig)
    sections = []
    for case in selection['cases']:
        family = case['family']
        parts = [f'<h3>{v:g} m/s</h3>' for v in spec['speeds_mps']]
        for signal in ['response', 'error', 'steering']:
            parts += [f'<section>{svgs[f"{family}_v{v:g}_{signal}"]}</section>' for v in spec['speeds_mps']]
        sections.append(f'<h2>{family.title()} recipient: {html.escape(case["client"])}, fleet {case["seed"]}</h2>'
                        f'<p>Locally observed speeds: {case["observed_speeds"]} m/s.</p><div class="grid">'+''.join(parts)+'</div>')
    columns = ['speed_mps','method','cases','finite','feasible','synthesis_failed','rollout_failed',
               'tracking_rmse_deg_s','active_tracking_rmse_deg_s','beta_peak_deg','lateral_acceleration_peak_mps2','limiter_fraction']
    table = summary[summary.family == 'all'][columns].to_html(index=False, float_format=lambda v:f'{v:.5g}')
    body = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Fixed-speed transient benchmark</title>
<style>body{{font:16px/1.5 system-ui;max-width:1400px;margin:30px auto;padding:0 20px;color:#17232d}}
.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:8px 14px}}h3{{text-align:center;margin:0}}svg{{width:100%;height:auto}}
.legend{{max-width:850px;margin:auto}}.table{{overflow-x:auto}}td,th{{padding:5px 9px;text-align:right;border-bottom:1px solid #ddd}}
table{{border-collapse:collapse;font-size:12px}}@media(max-width:800px){{.grid{{display:block}}}}</style>
<h1>Fixed-speed transient yaw-rate tracking</h1>
<p>ISO-7401-inspired simulation adaptation, <strong>not an ISO-compliant test</strong>. The standard prescribes steering;
this evaluation prescribes yaw rate and lets the controller determine steering. No path-tracking or obstacle-clearance claim.</p>
<p>Three separate runs at 13, 21 and 29 m/s. Each lasts 12 s, with a smooth sin³ left–right demand during 2–10 s.
Nominal lateral demand peaks at ±2.5 m/s²; r_ref = a_ref/v. Speed is constant, initial states are zero,
and feedback is noise-free. The fixed-speed plant uses nonlinear tanh lateral tires and static axle loads.</p>
<p>All six saved controller schedules are reused without synthesis, fitting or tuning. These speeds were not used for local
identification by any vehicle. This evaluates behavior across the envelope, not transitions between speeds.
The same previously selected two recipients are shown; selection did not use these new outcomes.</p>
<div class="legend">{svgs['legend']}</div>{''.join(sections)}
<p>Individual vector panels contain no titles or panel letters. All panels of each signal share their scale.
The error is r − r_ref. Steering axes show the response range; the unchanged bounds are ±12° and 120°/s.
The CSV also contains sideslip and actual lateral acceleration. Input quantities have no terminal sample.</p>
<h2>Fleet-wide results: all six methods</h2><p>Failures are counted first. Finite infeasible runs remain in the means.
Metric columns are equal-weight means of available fleet means, conditional on finite runs; beta and acceleration columns
average per-run peaks and are not fleet maxima. Full client metrics and per-family tables are exported separately.
The full-duration RMSE includes the straight-running intervals; active RMSE uses 2–10 s.</p><div class="table">{table}</div>
<h2>Paired fleet differences</h2><p>Positive values favor Federated. Only fleets with all 30 finite trajectories in both methods
enter a difference; availability is shown explicitly. No confirmatory significance claim.</p>
<div class="table">{pairs.to_html(index=False, float_format=lambda v:f'{v:.5g}')}</div>
<details><summary>Per-family summaries</summary><div class="table">{summary.to_html(index=False, float_format=lambda v:f'{v:.5g}')}</div></details>
<p>Supporting evaluation on already analyzed confirmation fleets. Existing variable-speed stress tests remain unchanged.
Exact LPV uses privileged model information and is not a guaranteed performance lower bound.</p>
<p>Background: <a href="https://www.iso.org/standard/54144.html">ISO 7401:2011 — lateral transient response</a>.</p></html>'''
    report.write_text(body, encoding='utf-8', newline='\n'); files.append(report)
    return files


def analyze(context, base):
    spec, cfg, originals, selection, sources = context
    results = audited_results(spec, originals, base)
    frame = records(results); fleet, summary, pairs = summaries(frame)
    traces, checked = replay_cases(spec, cfg, originals, results, selection)
    require(checked == 18, 'Incomplete selected-case replay audit')
    tables = Path(base)/'results/tables'; tables.mkdir(parents=True, exist_ok=True)
    files = []
    for name, data in [('metrics',frame), ('fleets',fleet), ('summary',summary), ('paired',pairs),
                       ('traces',trace_table(traces,selection))]:
        path = tables/f'{PREFIX}_{name}.csv'
        data.to_csv(path, index=False, encoding='utf-8', lineterminator='\n'); files.append(path)
    files += render(spec, cfg, traces, selection, summary, pairs, base)
    manifest = dict(version=1, sources=sources, runtime=run.runtime(),
        matplotlib=previous.importlib.metadata.version('matplotlib'), specification=spec, selection=selection,
        cases=len(frame), exact_selected_replays=checked, outcomes=frame.status.value_counts().to_dict(),
        checkpoints={result_path(base,s).relative_to(base).as_posix():run.file_hash(result_path(base,s)) for s in spec['seeds']},
        outputs={p.relative_to(base).as_posix():run.file_hash(p) for p in files})
    previous.write_json(tables/f'{PREFIX}_manifest.json', manifest)
    return dict(status='analyzed', cases=len(frame), exact_selected_replays=checked,
                report=str(Path(base)/'report'/f'{PREFIX}.html'))


def smoke():
    """Technical fixture only: three nominal rollouts, no fleet outcomes inspected."""
    spec = run.load(CONFIG); cfg = run.load(run.CONFIG)
    frozen = run.load(ROOT/'code/config/paper_fixed_lti_identification.json')['controller_frozen']
    schedule = dict(status='designed', nodes=[dict(gain=frozen['gain'], prefilter=frozen['prefilter']) for _ in cfg['speed_grid']])
    client = dict(parameters=dict(mass=1500.,yaw_inertia=2500.,front_length=1.2,rear_length=1.6,
                                 front_stiffness=80000.,rear_stiffness=80000.),
                  schedules={m:schedule for m in spec['methods']})
    traces = {}; fixture_rows = []
    with threadpool_limits(limits=1), tempfile.TemporaryDirectory() as temp:
        base = Path(temp)
        selection = dict(cases=[dict(seed=0,index=i,client=f'{f}_fixture',family=f,observed_speeds=[10,12])
                                for i,f in enumerate(['heavy','handling'])])
        for speed in spec['speeds_mps']:
            p = profile(spec, speed)
            row, arrays = evaluate(client, 'NominalLTI', p, cfg, spec)
            require(row['outcome']['status'] == 'complete', 'Nominal smoke rollout failed')
            again, replay = evaluate(client, 'NominalLTI', p, cfg, spec)
            require(again == row, 'Smoke replay mismatch')
            for case in selection['cases']:
                for method in spec['display_methods']:
                    traces[(case['family'],float(speed),method)] = arrays
            for method in spec['methods']:
                fixture_rows.append(dict(seed=0,family='fixture',method=method,speed_mps=float(speed),
                    status='complete',feasible=row['outcome']['feasible'],**row['outcome']['metrics'],**row['extra_metrics']))
        _, summary, pairs = summaries(pd.DataFrame(fixture_rows))
        files = render(spec,cfg,traces,selection,summary,pairs,base)
        require(len([p for p in files if p.suffix == '.pdf']) == 19, 'Missing smoke panels')
        require((base/'report'/f'{PREFIX}.html').read_text(encoding='utf-8').count('<svg') == 19, 'Missing preview SVGs')
        run.atomic_json(base/'fixture.json',dict(row=row))
        require(run.envelope(base/'fixture.json')['row'] == row, 'Checkpoint round trip failed')
    return dict(status='smoke_passed', nominal_profiles=3, exact_replays=3, pdf_panels=18, legends=1,
                confirmation_fleet_runs=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['preflight','smoke','run','analyze','restore','check'])
    parser.add_argument('--workers', type=int, default=5)
    parser.add_argument('--output-dir', type=Path, help='Isolated output root; uses results/ and report/ beneath it')
    parser.add_argument('--all', action='store_true', help='Restore every trajectory, rather than the 18 selected traces')
    args = parser.parse_args()
    require(args.workers > 0, 'workers must be positive')
    require(not args.all or args.command == 'restore', '--all only applies to restore')
    context = prepare(); spec,cfg,originals,selection,sources = context
    base = args.output_dir.resolve() if args.output_dir else ROOT
    if args.command == 'preflight':
        result = dict(status='preflight_passed', fleets=10, clients=300, methods=6, speeds_mps=spec['speeds_mps'],
            cases=5400, all_speeds_locally_unobserved=True, new_fits=0, new_controller_designs=0,
            selection=spec['selection'], runtime=run.runtime())
    elif args.command == 'smoke':
        with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context('spawn')) as pool:
            result = pool.submit(smoke).result()
    elif args.command == 'run':
        # Frozen protocol must be committed; unrelated working-tree files are irrelevant.
        for name in sources:
            subprocess.run(['git','ls-files','--error-unmatch',name],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
        subprocess.run(['git','diff','--exit-code','HEAD','--',*sources],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
        with execution_lock(base), ProcessPoolExecutor(max_workers=min(args.workers,10),
                mp_context=multiprocessing.get_context('spawn')) as pool:
            futures = [pool.submit(run_fleet,s,str(base)) for s in spec['seeds']]
            for future in as_completed(futures):
                print(json.dumps(future.result()),flush=True)
        audited_results(spec,originals,base)
        result = dict(status='execution_complete',cases=5400,next='Commit and push seed checkpoints, then run analyze')
    elif args.command == 'analyze':
        result = analyze(context,base)
    elif args.command == 'restore':
        results = audited_results(spec,originals,base)
        with execution_lock(base):
            _,checked = replay_cases(spec,cfg,originals,results,selection,base=base,all_cases=args.all)
        result = dict(status='restored_and_verified',exact_replays=checked)
    else:
        results = audited_results(spec,originals,base)
        manifest = run.load(base/'results/tables'/f'{PREFIX}_manifest.json')
        require(manifest['sources'] == sources and manifest['selection'] == selection, 'Stale manifest sources/selection')
        require(manifest['cases'] == 5400 and manifest['exact_selected_replays'] == 18, 'Incomplete manifest')
        for name, expected in (manifest['checkpoints'] | manifest['outputs']).items():
            require(run.file_hash(base/name) == expected, f'Changed/missing output: {name}')
        result = dict(status='verified',cases=5400,new_simulations=0)
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
