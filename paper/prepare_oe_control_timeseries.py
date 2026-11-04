r"""Select and replay six frozen cases; export separate IEEE-sized vector panels.

Windows CMD: python paper\prepare_oe_control_timeseries.py --preflight
             python paper\prepare_oe_control_timeseries.py --generate
No refitting or controller synthesis. All text I/O explicitly uses UTF-8.
"""
from __future__ import annotations

import argparse
import csv
import html
import importlib.metadata
import io
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'code/experiments'), str(ROOT / 'code/src')]
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from threadpoolctl import threadpool_limits
import paper_oe_lqi_validation as run

CONFIG = ROOT / 'code/config/paper_oe_control_timeseries.json'
PREFIX = 'paper_oe_control_timeseries'
LABELS = {'NominalLTI': 'Nominal LTI', 'Global': 'Global', 'Federated': 'Federated'}
COLORS = {'NominalLTI': '#666666', 'Global': '#de7e24', 'Federated': '#167a9e'}
STYLES = {'NominalLTI': '--', 'Global': '-.', 'Federated': '-'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8', newline='\n')


def rank_typical(vectors, identifiers):
    """Unweighted Euclidean distance after component-wise spread normalization."""
    values = np.asarray(vectors, dtype=float)
    require(values.ndim == 2 and len(values) == len(identifiers), 'Invalid selection matrix')
    require(np.isfinite(values).all(), 'Nonfinite selection metric')
    center = np.median(values, axis=0)
    spread = np.std(values, axis=0, ddof=0)
    active = spread > 64 * np.finfo(float).eps * np.maximum(1., np.abs(center))
    scaled = (values[:, active] - center[active]) / spread[active]
    distance = np.sqrt(np.sum(scaled**2, axis=1))
    order = sorted(range(len(values)), key=lambda i: (float(distance[i]), identifiers[i]))
    return dict(selected_id=identifiers[order[0]], median=center.tolist(),
                population_std=spread.tolist(), active_components=active.tolist(),
                candidates=[dict(id=identifiers[i], distance=float(distance[i]),
                                 vector=values[i].tolist()) for i in order])


def row_for(client, method, scenario):
    matches = [r for r in client['rows'] if (r['method'], r['scenario']) == (method, scenario)]
    require(len(matches) == 1, 'Missing or duplicate result row')
    return matches[0]


def vector(client, spec):
    values = []
    for scenario in spec['selection_scenarios']:
        for method in spec['methods']:
            outcome = row_for(client, method, scenario)['outcome']
            require(outcome['status'] == 'complete' and outcome['metrics'] is not None,
                    'Selection requires all six complete responses; no case exclusion')
            values.append(outcome['metrics']['tracking_rmse_deg_s'])
    return values


def select_cases(payloads, spec):
    payloads = sorted(payloads, key=lambda p: p['seed'])
    fleet_vectors = [np.mean([vector(c, spec) for c in p['clients']], axis=0) for p in payloads]
    fleet = rank_typical(fleet_vectors, [p['seed'] for p in payloads])
    selected = next(p for p in payloads if p['seed'] == fleet['selected_id'])
    cases = []
    for family in spec['families']:
        members = sorted([c for c in selected['clients'] if c['family'] == family], key=lambda c: c['index'])
        require(len(members) == 10, 'Expected ten clients per selected family')
        ranking = rank_typical([vector(c, spec) for c in members], [c['index'] for c in members])
        client = next(c for c in members if c['index'] == ranking['selected_id'])
        cases.append(dict(seed=selected['seed'], family=family, index=client['index'],
                          client=client['client'], observed_speeds=client['observed_speeds'], ranking=ranking))
    return dict(component_order=[f'{s}/{m}' for s in spec['selection_scenarios'] for m in spec['methods']],
                units='degrees/s', fleet=fleet, cases=cases)


def prepare():
    spec = run.load(CONFIG)
    for name, expected in spec['anchors'].items():
        require(run.file_hash(ROOT / name) == expected, f'Changed anchor: {name}')
    cfg = run.preflight()  # Includes original numerical runtime and all frozen source hashes.
    analysis_path = ROOT / 'results/tables/paper_oe_lqi_validation_analysis.json'
    analysis = run.load(analysis_path)
    audit = analysis['audit']
    require(audit['status'] == 'passed' and audit['cases'] == 3600, 'Incomplete control audit')
    require(audit['protocol_sha256'] == run.file_hash(run.CONFIG), 'Audit protocol mismatch')
    sources = dict(spec['anchors'])
    sources[CONFIG.relative_to(ROOT).as_posix()] = run.file_hash(CONFIG)
    sources[Path(__file__).relative_to(ROOT).as_posix()] = run.file_hash(Path(__file__))
    sources.update(cfg['source_sha256'])
    payloads = []
    for seed in cfg['seeds']:
        path = run.result_path(seed); name = path.relative_to(ROOT).as_posix()
        require(run.file_hash(path) == audit['result_file_sha256'][name], f'Changed checkpoint: {name}')
        sources[name] = run.file_hash(path)
        p = run.envelope(path)
        run.validate_checkpoint(p, cfg, seed)
        require(p['status'] == 'execution_complete' and p['protocol'] == cfg, 'Incomplete or changed protocol')
        require(len({c['client'] for c in p['clients']}) == 30, 'Invalid client identities')
        payloads.append(p)
    selection = select_cases(payloads, spec)
    actual = dict(seed=selection['fleet']['selected_id'],
                  clients={c['family']: c['client'] for c in selection['cases']})
    require(actual == spec['expected_selection'], 'Representative selection changed; preserve the frozen figure protocol')
    return spec, cfg, payloads, selection, sources


def verify_replay(outcome, arrays, saved, context):
    require(outcome == saved['outcome'], f'Replay metrics/status mismatch: {context}; no figures exported')
    require(run.array_hashes(arrays) == saved['arrays'],
            f'Replay array hash mismatch: {context}; use the original numeric runtime, no tolerance fallback')


def replay(spec, cfg, payloads, selection):
    profile = run.profiles(cfg)[spec['display_scenario']]
    traces, checks = {}, []
    with threadpool_limits(limits=1):
        for case in selection['cases']:
            p = next(p for p in payloads if p['seed'] == case['seed'])
            client = next(c for c in p['clients'] if c['index'] == case['index'])
            traces[case['family']] = {}
            for method in spec['methods']:
                saved = row_for(client, method, spec['display_scenario'])
                outcome, arrays = run.simulate(run.VehicleParameters(**client['parameters']),
                                               client['schedules'][method], profile, cfg)
                verify_replay(outcome, arrays, saved, f"{case['seed']}/{case['client']}/{method}")
                traces[case['family']][method] = arrays
                checks.append(dict(seed=case['seed'], client=case['client'], family=case['family'],
                                   method=method, outcome=outcome, arrays=saved['arrays'], exact_match=True))
    return profile, traces, checks


def style():
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['DejaVu Serif'], 'font.size': 8,
                         'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
                         'legend.fontsize': 6.5, 'lines.linewidth': 1.15, 'axes.linewidth': .6,
                         'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
                         'svg.hashsalt': PREFIX})


def make_panels(profile, traces, cfg, spec):
    """Six independent panels; reference widths equal the two-column figure width."""
    style(); panels = {}
    def panel(name, ylabel, wide=False):
        fig = plt.figure(figsize=(7.16, 1.05) if wide else (3.48, 1.85))
        ax = fig.add_axes([.095, .35, .89, .60] if wide else [.19, .24, .79, .60])
        ax.set(xlabel='Time [s]', ylabel=ylabel, xlim=(profile['time'][0], profile['time'][-1]))
        ax.set_xticks(np.arange(0, cfg['duration_s'] + .1, 2))
        ax.grid(axis='both', color='#dddddd', linewidth=.45)
        ax.set_axisbelow(True); ax.spines[['top', 'right']].set_visible(False)
        panels[name] = (fig, ax)
        return ax
    panel('speed', 'Prescribed\nspeed [m/s]', True).plot(profile['time'], profile['speed'], color='#222222')
    panel('reference', 'Yaw-rate ref.\n[deg/s]', True).plot(
        profile['time'], np.rad2deg(profile['reference']), color='#222222')
    peak = max(float(np.max(abs(np.rad2deg(a['state'][:, 1] - a['reference']))))
               for family in traces.values() for a in family.values())
    error_limit = max(1e-6, 1.08 * peak)
    for family in spec['families']:
        error = panel(f'{family}_error', 'Yaw-rate error\n[deg/s]')
        steering = panel(f'{family}_steering', 'Applied steering\n[deg]')
        error.set_ylim(-error_limit, error_limit)
        steering.set_ylim(-1.08 * cfg['steering_limit_deg'], 1.08 * cfg['steering_limit_deg'])
        error.axhline(0, color='#999999', linewidth=.5)
        for bound in [-cfg['steering_limit_deg'], cfg['steering_limit_deg']]:
            steering.axhline(bound, color='#999999', linewidth=.7, linestyle=':')
        for method in spec['methods']:
            a = traces[family][method]
            opts = dict(color=COLORS[method], linestyle=STYLES[method], label=LABELS[method])
            error.plot(a['time'], np.rad2deg(a['state'][:, 1] - a['reference']), **opts)
            # Applied input is held on [t_k,t_{k+1}); extend the last value to final time.
            steering.step(a['time'], np.rad2deg(np.r_[a['steering'], a['steering'][-1]]), where='post', **opts)
        for ax in [error, steering]:
            ax.legend(loc='lower center', bbox_to_anchor=(.5, 1.02), ncol=3, frameon=False,
                      handlelength=2., columnspacing=.85, borderaxespad=0)
    return panels


def write_traces(path, traces, selection, spec):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ['seed', 'client', 'family', 'method', 'scenario', 'time_s', 'speed_m_s',
              'reference_deg_s', 'yaw_rate_deg_s', 'yaw_error_deg_s', 'steering_deg']
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n'); writer.writeheader()
        for case in selection['cases']:
            for method in spec['methods']:
                a = traces[case['family']][method]
                for k, t in enumerate(a['time']):
                    writer.writerow(dict(seed=case['seed'], client=case['client'], family=case['family'],
                        method=method, scenario=spec['display_scenario'], time_s=float(t), speed_m_s=float(a['speed'][k]),
                        reference_deg_s=float(np.rad2deg(a['reference'][k])), yaw_rate_deg_s=float(np.rad2deg(a['state'][k, 1])),
                        yaw_error_deg_s=float(np.rad2deg(a['state'][k, 1]-a['reference'][k])),
                        steering_deg=float(np.rad2deg(a['steering'][k])) if k < len(a['steering']) else ''))


def render(profile, traces, cfg, spec, selection, checks, figures, report):
    panels = make_panels(profile, traces, cfg, spec); files = []; svgs = {}
    figures.mkdir(parents=True, exist_ok=True); report.parent.mkdir(parents=True, exist_ok=True)
    for name, (fig, ax) in panels.items():
        stem = f'{PREFIX}_{name}_panel'
        for suffix in ['pdf', 'png']:
            path = figures / f'{stem}.{suffix}'
            kwargs = dict(metadata={'CreationDate': None, 'ModDate': None}) if suffix == 'pdf' else dict(dpi=300)
            fig.savefig(path, **kwargs); files.append(path)
        buf = io.StringIO(); fig.savefig(buf, format='svg', metadata={'Date': None})
        svgs[name] = buf.getvalue()[buf.getvalue().index('<svg'):]
        plt.close(fig)
    cards = []
    for key in ['speed', 'reference', 'heavy_error', 'handling_error', 'heavy_steering', 'handling_steering']:
        title = {'speed': 'Prescribed longitudinal speed', 'reference': 'Yaw-rate demand'}.get(key, key.replace('_', ' ').title())
        cards.append(f'<section class="{"wide" if key in ["speed", "reference"] else ""}"><h2>{title}</h2>{svgs[key]}</section>')
    case_text = '; '.join(f"{c['family']}: fleet {c['seed']}, {c['client']}" for c in selection['cases'])
    rows = ''.join('<tr>' + ''.join(f'<td>{html.escape(str(v))}</td>' for v in
        [c['family'], LABELS[c['method']], f"{c['outcome']['metrics']['tracking_rmse_deg_s']:.6f}",
         f"{c['outcome']['metrics']['limiter_fraction']:.4f}", c['exact_match']]) + '</tr>' for c in checks)
    content = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Closed-loop time-series draft</title>
<style>body{{font:16px/1.5 system-ui;max-width:1150px;margin:32px auto;padding:0 20px;color:#17232d}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}.wide{{grid-column:1/-1}}svg{{width:100%;height:auto}}
h2{{font-size:16px;margin:0}}section{{min-width:0}}td,th{{padding:6px 14px;border-bottom:1px solid #ddd;text-align:left}}
@media(max-width:700px){{.grid{{display:block}}section{{margin:20px 0}}}}</style>
<h1>Closed-loop behavior: first figure draft</h1><p>{html.escape(case_text)}. Displayed maneuver: hard.</p>
<p>Outcome-based representative examples selected using both maneuvers and all three methods.
The shared references appear once. Errors use r − r_ref; dotted steering lines mark ±12° limits.
Both recipients use identical axes. No smoothing or time-window selection is applied.</p>
<div class="grid">{''.join(cards)}</div><h2>Saved-case metrics and replay verification</h2>
<table><tr><th>Family</th><th>Method</th><th>Yaw RMSE [deg/s]</th><th>Limiter fraction</th><th>Exact replay</th></tr>{rows}</table>
<p>{html.escape(spec['scope'])}</p><details><summary>Selection rule and candidate distances</summary>
<p>{html.escape(spec['selection'])}</p><pre>{html.escape(json.dumps(selection, indent=2))}</pre></details>
<p>Panels are separate vector PDFs without embedded titles or panel letters. Headings above are preview-only.
The final steering CSV cell is empty because there is no input interval after the terminal state.</p></html>'''
    report.write_text(content, encoding='utf-8', newline='\n'); files.append(report)
    return files


def destinations(directory):
    if directory is not None:
        root = Path(directory).resolve()
        return root, root, root / f'{PREFIX}.html'
    return ROOT/'results/figures', ROOT/'results/tables', ROOT/'report'/f'{PREFIX}.html'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    for flag in ['preflight', 'generate', 'check']:
        modes.add_argument('--'+flag, action='store_true')
    parser.add_argument('--output-dir', type=Path, help='Place all draft outputs here, away from committed outputs')
    args = parser.parse_args()
    spec, cfg, payloads, selection, sources = prepare()
    figures, tables, report = destinations(args.output_dir)
    manifest_path = tables / f'{PREFIX}_manifest.json'
    if args.preflight:
        print(json.dumps(dict(status='preflight_passed', selection=spec['expected_selection'],
                              display_scenario=spec['display_scenario'], replay_cases=6, new_fits=0), indent=2))
        return
    if args.check:
        manifest = run.load(manifest_path)
        require(manifest['sources'] == sources and manifest['selection'] == selection, 'Stale source/selection manifest')
        require(len(manifest['replay']) == 6 and all(c['exact_match'] for c in manifest['replay']), 'Incomplete replay audit')
        for name, expected in manifest['outputs'].items():
            path = (Path(args.output_dir).resolve() if args.output_dir else ROOT) / name
            require(run.file_hash(path) == expected, f'Missing or changed output: {name}')
        print(json.dumps(dict(status='verified', panels=6, replay_cases=0)))
        return
    profile, traces, checks = replay(spec, cfg, payloads, selection)
    # No exports occur until all six trajectories and outcomes match exactly.
    files = render(profile, traces, cfg, spec, selection, checks, figures, report)
    csv_path = tables / f'{PREFIX}_traces.csv'
    write_traces(csv_path, traces, selection, spec); files.append(csv_path)
    base = Path(args.output_dir).resolve() if args.output_dir else ROOT
    # PNG previews are regenerable and ignored by git; verify portable PDF/HTML/CSV outputs.
    outputs = {p.relative_to(base).as_posix(): run.file_hash(p) for p in files if p.suffix != '.png'}
    write_json(manifest_path, dict(version=1, specification=spec, selection=selection, sources=sources,
        replay=checks, runtime=run.runtime(), matplotlib=importlib.metadata.version('matplotlib'), outputs=outputs))
    print(json.dumps(dict(status='generated', panels=6, replay_cases=6, exact_replays=6,
                         report=str(report), manifest=str(manifest_path)), indent=2))


if __name__ == '__main__':
    main()
