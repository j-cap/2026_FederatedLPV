"""Audit the final continuations and consolidate existing pilots; no fitting.

python code/experiments/analyze_federated_relaxation_consolidation.py
"""
from __future__ import annotations
import io
from pathlib import Path
import numpy as np
import pandas as pd
import paper_federated_relaxation_peers_finish as run

base = run.base
PREFIX = 'paper_federated_relaxation_consolidation'
REPORT = run.ROOT / 'report/federated_relaxation_consolidation.html'
METRICS = ['yaw_rmse_deg_s', 'measured_yaw_rmse_deg_s', 'beta_rmse_deg']


def check_state(state, cfg):
    h = state['history']; last = h[-1]
    assert state['status'] == 'converged' and not state['pending']
    assert len(h) == state['round'] and [r['round'] for r in h] == list(range(1, state['round']+1))
    objectives = np.array([r['objective'] for r in h])
    assert np.all(np.diff(objectives) <= 1e-6*np.maximum(1, abs(objectives[:-1])))
    z = np.array(state['log_parameters']); eta = np.array(state['centers'])
    assert state['assignments'] == [0]*len(z)
    np.testing.assert_allclose(eta[0], z.mean(axis=0), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(last['penalty'], np.sum(((z-eta[0])*cfg['sharing']['weights'])**2), rtol=1e-12)
    np.testing.assert_allclose(last['objective'], last['data_sse']+last['penalty'], rtol=1e-12)
    relative = abs(objectives[-1]-objectives[-2])/max(1, abs(objectives[-2]))
    assert relative < cfg['sharing']['personal_relative_tolerance']
    assert last['max_log_step'] < cfg['sharing']['personal_step_tolerance']
    assert last['all_local_optimizers_converged'] and not last['assignments_changed']
    return relative


def main():
    cfg, protocol = run.preflight()
    # Lock all prior scientific results/reports used in this consolidation.
    paths = list(run.OUT.glob('paper_federated_relaxation_*.json'))
    paths += list(run.OUT.glob('paper_federated_relaxation_*.csv'))
    paths += list((run.ROOT/'report').glob('federated_relaxation*.html'))
    paths += [run.ROOT/'report/tire_relaxation_pilot_review.html',
              run.OUT/'paper_tire_relaxation_pilot_review_audit.json']
    paths = [p for p in paths if 'consolidation' not in p.name]
    hashes = {p.relative_to(run.ROOT).as_posix(): base.canonical_hash(p) for p in paths}
    rows = []; opt = []; replay = []; objectives = []; refreshed = 0
    for seed in [1301, 1302, 1303]:
        original, complementary, previous = run.sources(seed, cfg, protocol)
        check_state(complementary['optimization'], cfg)
        fleet = base.fleet_for(cfg, seed)
        meta = {c['client']: (i, c) for i, c in enumerate(fleet['clients'])}
        p = None
        if seed in protocol['seeds']:
            p = base.storage.read_checkpoint(run.path(seed))
            run.validate(p, original, previous, cfg, protocol, execution=False)
            assert p['status'] == 'execution_complete' and len(p['evaluation']) == 42
        for speed, condition in previous['conditions'].items():
            before = condition['optimization']
            state = p['optimization'] if p and speed == '13' else before
            relative = check_state(state, cfg)
            opt.append(dict(seed=seed, training_speed=int(speed), reused=state is before,
                original_rounds=before['round'], total_rounds=state['round'],
                added_rounds=state['round']-before['round'], status=state['status'],
                before_objective=before['history'][-1]['objective'], after_objective=state['history'][-1]['objective'],
                max_parameter_change_pct=float(np.max(abs(np.expm1(np.array(state['log_parameters'])-before['log_parameters']))))*100))
            objectives.append(dict(seed=seed, training_speed=int(speed),
                max_log_step=state['history'][-1]['max_log_step'], relative_objective_change=relative))
        # Recompute the final training SSE on all 18 continued client records.
        if p:
            data_sse = 0.
            for i, c in enumerate(fleet['clients']):
                a, _ = base.generate(c, seed, 'training', c['training_speed'], cfg)
                a['state'] = base.batch_predict(c['physical_parameters'], [a['steering']], [13],
                    c['geometry_m'], cfg['sample_time'], cfg['integration_max_step'])[0, 0]
                a['measured_yaw'] = a['state'][:, 1]+a['noise']
                record = dict(speed=13, applied_input=a['steering'], measured_yaw=a['measured_yaw'])
                client = base.fl.Client([record], c['geometry_m'], base.settings(cfg))
                data_sse += client.loss_message(p['optimization']['log_parameters'][i])
            saved_sse = p['optimization']['history'][-1]['data_sse']
            np.testing.assert_allclose(data_sse, saved_sse, rtol=1e-8, atol=1e-8)
            objectives[-3]['replayed_data_sse'] = data_sse
            objectives[-3]['saved_data_sse'] = saved_sse
        for key, oldcase in original['evaluation'].items():
            i, c = meta[oldcase['client']]
            before = previous['evaluation'][key]
            after = p['evaluation'][key] if p and key in p['evaluation'] else before
            assert after['source_trace_sha256'] == oldcase['trace_sha256']
            methods = {m: oldcase['methods'][m] for m in ['LocalOE', 'NominalPrior']}
            methods.update(Complementary=complementary['evaluation'][key]['metrics'],
                SameSpeedBefore=before['metrics'], SameSpeedAfter=after['metrics'])
            for method, metrics in methods.items():
                assert metrics['status'] == 'complete' and np.isfinite([metrics[m] for m in METRICS]).all()
                rows.append(dict(seed=seed, client=c['client'], training_speed=c['training_speed'],
                    speed=oldcase['speed'], probe=oldcase['probe'], region=run.peers.REGIONS[oldcase['region']], method=method, **metrics))
            if after is not before:
                refreshed += 1
                a, _ = base.generate(c, seed, oldcase['probe'], oldcase['speed'], cfg)
                y = base.batch_predict(np.exp(p['optimization']['log_parameters'][i]), [a['steering']], [oldcase['speed']],
                    c['geometry_m'], cfg['sample_time'], cfg['integration_max_step'])[0, 0]
                rms = lambda x: float(np.rad2deg(np.sqrt(np.mean(x*x))))
                values = [rms(y[:, 1]-a['state'][:, 1]), rms(y[:, 1]-a['measured_yaw']), rms(y[:, 0]-a['state'][:, 0])]
                saved = [after['metrics'][m] for m in METRICS]
                np.testing.assert_allclose(values, saved, rtol=1e-8, atol=1e-10)
                a['predicted_SameSpeed'] = y
                replay.append(dict(seed=seed, case=key, max_metric_difference=float(np.max(abs(np.array(values)-saved))),
                    exact_trace_match=base.storage.trace_hash(a) == after['trace_sha256']))
        print(f'Fleet {seed}: checkpoint, convergence and unchanged comparisons verified', flush=True)
    cases = pd.DataFrame(rows)
    assert refreshed == 84 and len(cases) == 1890
    keys = ['seed', 'client', 'speed', 'probe', 'method']
    pd.testing.assert_frame_equal(cases.sort_values(keys).reset_index(drop=True),
        pd.read_csv(run.OUT/f'{run.PREFIX}_cases.csv').sort_values(keys).reset_index(drop=True),
        check_exact=False, rtol=1e-10, atol=1e-12)
    summary = cases.groupby(['seed', 'training_speed', 'region', 'method'], as_index=False).yaw_rmse_deg_s.mean()
    pd.testing.assert_frame_equal(summary, pd.read_csv(run.OUT/f'{run.PREFIX}_summary.csv'), check_exact=False, rtol=1e-10, atol=1e-12)
    pd.testing.assert_frame_equal(pd.DataFrame(opt), pd.read_csv(run.OUT/f'{run.PREFIX}_optimization.csv'), check_exact=False, rtol=1e-10, atol=1e-12)
    saved_paired = pd.read_csv(run.OUT/f'{run.PREFIX}_paired.csv')
    for r in saved_paired.to_dict('records'):
        g = cases[(cases.seed == r['seed']) & (cases.training_speed == r['training_speed']) & (cases.region == r['region'])]
        pivot = g.pivot(index=['client', 'speed', 'probe'], columns='method', values='yaw_rmse_deg_s')
        a, b = pivot[r['method']], pivot[r['baseline']]
        assert len(pivot) == r['cases'] and int((a < b).sum()) == r['improved_cases']
        np.testing.assert_allclose([a.mean(), b.mean(), 100*(1-a.mean()/b.mean())],
            [r['method_mean'], r['baseline_mean'], r['reduction_pct']], rtol=1e-9, atol=1e-10)
    saved_analysis = base.storage.read_checkpoint(run.OUT/f'{run.PREFIX}_analysis.json')
    assert saved_analysis['all_same_speed_converged'] is True
    assert saved_analysis['sources'] == protocol['source_checkpoints']
    assert saved_analysis['finish_checkpoints'] == {str(s): base.canonical_hash(run.path(s)) for s in protocol['seeds']}
    primary = cases[cases.region == 'target_unseen_anchor']
    pivot = primary.pivot(index=['seed', 'client', 'speed', 'probe'], columns='method', values='yaw_rmse_deg_s')
    means = pivot.mean(); vehicles = pivot.groupby(['seed', 'client']).mean()
    tables = dict(optimization=pd.DataFrame(opt), stopping=pd.DataFrame(objectives), replays=pd.DataFrame(replay))
    for name, key in [('fleets', 'seed'), ('speeds', 'training_speed'), ('regions', 'region')]:
        frame = (cases if key == 'region' else primary).pivot_table(index=key, columns='method', values='yaw_rmse_deg_s')
        frame['complementary_reduction_vs_same_pct'] = 100*(1-frame.Complementary/frame.SameSpeedAfter)
        frame['complementary_reduction_vs_nominal_pct'] = 100*(1-frame.Complementary/frame.NominalPrior)
        tables[name] = frame.reset_index()
    tables['overall'] = means.rename('mean_yaw_rmse_deg_s').reset_index()
    tables['overall']['reduction_vs_local_pct'] = 100*(1-tables['overall'].mean_yaw_rmse_deg_s/means.LocalOE)
    info = dict(status='audit_passed', new_fits=0, new_test_conditions=0, same_speed_converged=9,
        continued_conditions=2, refreshed_cases=refreshed, total_comparison_rows=len(cases),
        means_deg_s=means.to_dict(), complementary_reduction_vs_same_pct=100*(1-means.Complementary/means.SameSpeedAfter),
        prior_complementary_reduction_vs_same_pct=100*(1-means.Complementary/means.SameSpeedBefore),
        complementary_reduction_vs_nominal_pct=100*(1-means.Complementary/means.NominalPrior),
        absolute_peer_speed_gain_deg_s=means.SameSpeedAfter-means.Complementary,
        same_speed_continuation_reduction_pct=100*(1-means.SameSpeedAfter/means.SameSpeedBefore),
        nominal_fraction_of_complementary_gain=(means.LocalOE-means.NominalPrior)/(means.LocalOE-means.Complementary),
        same_speed_fraction_of_complementary_gain=(means.LocalOE-means.SameSpeedAfter)/(means.LocalOE-means.Complementary),
        complementary_case_wins_vs_same=int((pivot.Complementary < pivot.SameSpeedAfter).sum()), cases=len(pivot),
        complementary_vehicle_wins_vs_same=int((vehicles.Complementary < vehicles.SameSpeedAfter).sum()), vehicles=len(vehicles),
        replayed_prediction_cases=len(replay), exact_trace_matches=sum(r['exact_trace_match'] for r in replay),
        max_replay_metric_difference=max(r['max_metric_difference'] for r in replay), training_sse_clients=18,
        original_artifacts_modified=False, source_hashes=hashes, audit_runtime=base.identity(cfg)['runtime'],
        review_script_sha256=base.canonical_hash(Path(__file__)))
    assert hashes == {n: base.canonical_hash(run.ROOT/n) for n in hashes}
    render(tables, info)
    for name, frame in tables.items(): frame.to_csv(run.OUT/f'{PREFIX}_{name}.csv', index=False)
    base.storage.atomic_json(run.OUT/f'{PREFIX}.json', info)
    print({k:v for k,v in info.items() if k not in ['source_hashes', 'audit_runtime']})
    print(REPORT)


def render(t, info):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib.rcParams['svg.hashsalt'] = PREFIX
    methods = ['LocalOE', 'NominalPrior', 'SameSpeedAfter', 'Complementary']
    labels = ['Local OE', 'Fixed nominal prior', 'Same-speed peers', 'Complementary peers']
    colors = ['#667788', '#b47922', '#8061a8', '#087d87']
    figures = []
    fig, ax = plt.subplots(figsize=(9, 3.7), layout='constrained')
    values = [info['means_deg_s'][m] for m in methods]
    ax.barh(labels[::-1], values[::-1], color=colors[::-1])
    for j, v in enumerate(values[::-1]): ax.text(v+0.00008, j, f'{v:.6f}', va='center')
    ax.set(xlabel='Mean clean yaw prediction RMSE [deg/s]', xlim=(0, max(values)*1.2), title='Prediction at the two unobserved anchor speeds')
    figures.append(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    for ax, key, frame in [(axes[0], 'seed', t['fleets']), (axes[1], 'training_speed', t['speeds'])]:
        for m, label, color in zip(methods, labels, colors): ax.plot(frame[key].astype(str), frame[m], 'o-', color=color, label=label)
        ax.set(xlabel='Fleet' if key == 'seed' else 'Target training speed [m/s]', ylabel='Clean yaw RMSE [deg/s]'); ax.grid(alpha=.2)
    axes[0].legend(fontsize=8); figures.append(fig)
    svgs = []
    for fig in figures:
        b = io.StringIO(); fig.savefig(b, format='svg', metadata={'Date':None}); plt.close(fig)
        svg = b.getvalue(); svgs.append(svg[svg.index('<svg'):])
    tab = lambda frame: '<div class="scroll">'+frame.to_html(index=False, border=0, float_format=lambda x:f'{x:.6g}')+'</div>'
    html = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Personalized federated identification — consolidated evidence</title>
<style>body{{margin:0;background:#eef3f6;color:#173348;font:16px/1.65 system-ui}}main{{max-width:1120px;margin:auto;padding:28px}}section{{background:white;border-radius:12px;padding:25px;margin:22px 0}}h1,h2{{line-height:1.25}}h1{{font-size:34px}}.lead{{font-size:20px}}.decision{{border-left:6px solid #087d87}}svg{{width:100%;height:auto}}.scroll{{overflow:auto}}table{{border-collapse:collapse;font-size:13px}}th,td{{padding:9px;border-bottom:1px solid #ddd}}a{{color:#087d87}}</style><main>
<h1>Numerical closure achieved.<br>The main benefit is regularization.</h1>
<p class="lead">Both remaining fits converged. Complementary-speed peers still improve mean unseen-anchor prediction, but the increment is small and concentrated at low training speed. The evidence supports a conditional identification benefit; it does not yet establish an important control benefit.</p>
<section><h2>1. The two unfinished fits are closed</h2>{tab(t['optimization'])}
<p>Only fleets 1301 and 1302 at 13 m/s continued, from round 30 to rounds 32 and 46. All nine SameSpeed conditions now meet the original stopping criteria. All three Complementary conditions also meet those criteria. This is numerical convergence under the declared tolerances, not a proof of global optimality.</p>
<p>SameSpeed mean unseen-anchor error changes by {info['same_speed_continuation_reduction_pct']:.3f}%. The Complementary advantage changes from {info['prior_complementary_reduction_vs_same_pct']:.2f}% to {info['complementary_reduction_vs_same_pct']:.2f}%. No data, initialization, prior weight or stopping tolerance changed.</p></section>
<section><h2>2. What each comparison actually establishes</h2>{svgs[0]}{tab(t['overall'])}
<p>Each of 27 targets supplies one local 12 s record at 13, 21 or 29 m/s. The primary metric averages two independent test probes at the other two anchor speeds: 108 paired cases, nested within 27 vehicles and three reused development fleets. These 108 cases are not 108 independent fleet replications.</p>
<ul><li><b>Local OE → fixed nominal prior:</b> a 45.7% reduction without learning from peers. Regularization alone already solves much of the limited-data estimation problem.</li>
<li><b>Fixed nominal prior → Complementary:</b> {info['complementary_reduction_vs_nominal_pct']:.2f}% lower mean error with a fleet-learned prior. This is the relevant additional gain over a strong local baseline; it varies across fleets.</li>
<li><b>SameSpeed → Complementary:</b> {info['complementary_reduction_vs_same_pct']:.2f}% lower mean error, an absolute {info['absolute_peer_speed_gain_deg_s']:.6f} deg/s. This isolates the tested change in peer operating speeds while keeping target data, peer identities, duration, input and paired noise fixed.</li></ul>
<p>The fixed nominal prior accounts arithmetically for {100*info['nominal_fraction_of_complementary_gain']:.1f}% of the absolute Complementary improvement over Local OE; SameSpeed accounts for {100*info['same_speed_fraction_of_complementary_gain']:.1f}%. These ratios compare error reductions; they are not causal decompositions or additive contributions.</p>
<p>SameSpeed and Complementary both use personalized models. Their difference is peer data: eight same-speed peers versus two same-speed peers and six at the other anchor speeds. No target gains its own additional-speed training record. Changing speed also changes response amplitude, parameter sensitivity and output signal-to-noise ratio; this does not isolate coverage geometry from those effects.</p></section>
<section><h2>3. The average hides a conditional benefit</h2>{svgs[1]}{tab(t['fleets'])}{tab(t['speeds'])}
<p>Complementary reduces unseen-anchor error versus SameSpeed by 0.13%, 3.09% and 10.35% in fleets 1301–1303. By target training speed, it improves 13 m/s targets by 6.79%, but increases errors by 0.13% at 21 m/s and 1.63% at 29 m/s. It wins in {info['complementary_case_wins_vs_same']}/{info['cases']} trajectory comparisons and {info['complementary_vehicle_wins_vs_same']}/{info['vehicles']} vehicle means. A positive pooled average is not a uniform individual benefit.</p>
<p>Against the fixed nominal prior, Complementary remains worse in fleet 1301 and better in fleets 1302 and 1303. Therefore the present evidence does not establish consistent dominance of learned sharing over a well-chosen physical prior.</p>
<details><summary>All speed regions</summary>{tab(t['regions'])}<p>Observed: target's own anchor. Unseen anchors: the other two among 13/21/29 m/s. Intermediate: 17/25 m/s. Outside the reference range: 9/33 m/s.</p></details></section>
<section><h2>4. Consolidation across the experimental sequence</h2>
<div class="scroll"><table><tr><th>Question</th><th>Evidence</th><th>Consequence for the paper</th></tr>
<tr><td>Does the four-state extension capture useful dynamics?</td><td>The earlier yaw-only physical OE pilot improved predictions and estimated the relaxation lengths well. Friction remained weakly determined.</td><td>A defensible identification model; do not claim reliable recovery of all six physical parameters.</td></tr>
<tr><td>Must heterogeneous vehicles share a single model?</td><td>The initial limited-speed pilot found a much larger Global prediction error than personalized methods.</td><td>Supports personalization in this population, not a claim that any fleet learning outperforms local regularization.</td></tr>
<tr><td>Do learned groups add value?</td><td>The tested grouped variant was worse than the single learned prior in the original pilot.</td><td>There is no positive evidence here for making group discovery the central contribution. Its original optimization budget also limits claims of general inferiority.</td></tr>
<tr><td>Does a learned prior help beyond a fixed one?</td><td>About 7% mean improvement in the matched development data, with one of three fleets worse.</td><td>A modest and population-dependent additional benefit.</td></tr>
<tr><td>Is complementary speed coverage the main source of improvement?</td><td>The converged matched comparison adds 4.10% over SameSpeed; almost all total improvement over Local OE remains with same-speed peers.</td><td>Complementary coverage is a secondary, conditional benefit in this regime.</td></tr>
<tr><td>Does improved identification improve tracking?</td><td>The earlier four-state controller/observer pilot improved prediction but worsened yaw tracking relative to the two-state design. The present peer experiments contain no closed-loop tests.</td><td>The central control benefit remains unresolved. Prediction gains cannot substitute for it.</td></tr></table></div>
<p>These experiments answer different questions under different protocols. Their numerical errors must not be pooled into one ranking. Links below preserve the original evidence and limitations.</p>
<p>All present peers share the true relaxation lengths (0.4/0.7 m) and known geometry (1.2/1.6 m). That is favorable for transfer. The plant uses tanh tires with force relaxation; these are gentle fixed-speed simulated maneuvers, not a validation of general nonlinear vehicle operation. Only three development fleets were reused, with no independent confirmation in this sequence. Do not tune a preferred story against these test outcomes and then call it confirmatory.</p></section>
<section class="decision"><h2>5. What we can conclude now</h2>
<p><b>Under limited local speed coverage, a fleet-learned physical prior can improve yaw-only model identification, but most of the gain over unregularized local OE is already achieved by a fixed nominal prior. Complementary peer speeds provide a small additional benefit concentrated at low target speed.</b></p>
<p>This is a useful, qualified identification result. It is not yet sufficient to support a strong claim that federated learning is necessary for, or materially improves, closed-loop vehicle control. The old group-specific LPV narrative is not established by these physical-model pilots and should not be carried over unchanged.</p>
<p>Consolidate and discuss the scientific scope before commissioning further experiments: either accept a narrower identification/regularization contribution with explicit limits, or make a meaningful control consequence the requirement for a control-paper direction. No new experiment or manuscript rewrite is included in this review.</p></section>
<section><h2>6. Audit and reproducibility</h2>
<p>Verified frozen source/configuration locks and checkpoint digests, unchanged round-30 history prefixes, server means, prior penalties, objective accounting and stopping conditions. The complete 1890-row comparison and saved summary, paired and optimization tables reconstruct from checkpoints. Only 84 SameSpeed predictions changed; the other 294 and all baseline/complementary predictions were reused.</p>
<p>Replayed all 84 updated predictions and recomputed the final training SSE for all 18 continued clients, without fitting. Maximum prediction-metric difference: {info['max_replay_metric_difference']:.3g}. Full trace hashes match in {info['exact_trace_matches']}/84 replays. Numerical agreement across platforms is distinct from byte-exact restoration; no exact restoration is claimed here. Original artifacts were preserved.</p>
<details><summary>Stopping checks and replay details</summary>{tab(t['stopping'])}{tab(t['replays'])}</details>
<p>Reproduce this read-only scientific audit and generated report:</p><pre>python code/experiments/analyze_federated_relaxation_consolidation.py</pre>
<p>To restore ignored original execution arrays on the original runtime, use the existing hash-checked command (covered by its synthetic restoration test; not executed here as a cross-platform exact recovery):</p><pre>python code/experiments/paper_federated_relaxation_peers_finish.py restore</pre>
<p><a href="federated_relaxation_peers_finish.html">Final run report</a> · <a href="federated_relaxation_peers_review.html">Matched peer-speed audit</a> · <a href="federated_relaxation_coverage_review.html">Initial limited-speed audit</a> · <a href="federated_relaxation_continue_review.html">Personalized continuation audit</a> · <a href="tire_relaxation_pilot_review.html">Four-state prediction/control audit</a></p></section></main></html>'''
    REPORT.write_text('\n'.join(line.rstrip() for line in html.splitlines())+'\n', encoding='utf-8', newline='\n')


if __name__ == '__main__': main()
