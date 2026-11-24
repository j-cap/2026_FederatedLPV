"""Audit and interpret the completed observer pilot without changing its protocol.

Reconstructs only the 15 prespecified preview cases. Original Windows hashes
remain authoritative; cross-platform replays are checked numerically and are
never relabeled as byte-identical recovered trajectories.
"""
from __future__ import annotations
import os
for variable in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[variable] = '1'
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'code/src'), str(ROOT/'code/experiments')]
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import paper_physical_ct_observer as run
from federated_lpv.ct_observer_control import (observer_model, kalman_design,
    origin_closed_loop, trace_hash, yaw_noise)
from federated_lpv.oe_lqi_validation import interpolate
from federated_lpv.vehicle import VehicleParameters

PREFIX = 'paper_physical_ct_observer_review'
REPORT = ROOT/'report/physical_ct_observer_review.html'
METRICS = ['tracking_rmse_deg_s', 'beta_estimation_rmse_deg', 'steering_rate_rms_deg_s']
PAIRS = [*run.old.load(run.CONFIG)['paired_comparisons'],
         ['Nominal/Nominal', 'YawOnlyCT/Matched']]


def audit(cfg):
    cc = run.old.load(run.old.storage.CONFIG); prov = run.provenance(cfg)
    records = []; replays = []; traces = {}; checkpoint_hashes = {}
    designs = {}; checked = 0; noise_checked = 0; full_trace_hash_matches = 0
    max_metric_difference = 0.; max_preview_difference = 0.
    for seed in cfg['seeds']:
        saved = run.old.envelope(run.path(seed)); run.validate(saved, seed, cfg, prov)
        if saved['status'] != 'execution_complete':
            raise ValueError('All three fleets must be complete')
        checkpoint_hashes[run.path(seed).relative_to(ROOT).as_posix()] = run.old.file_hash(run.path(seed))
        data = run.inputs(seed); source, full, yaw, controls = data
        for client in saved['clients']:
            index = client['index']; plant = VehicleParameters(**source['clients'][index]['parameters'])
            for row in client['rows']:
                method = row['method']; speed = row['speed']; identity = cfg['methods'][method]
                noise = yaw_noise(seed, index, speed, 1201, cfg['observer'])
                if trace_hash({'noise': noise}) != row['noise_sha256']:
                    raise ValueError('Sensor-noise hash mismatch')
                noise_checked += 1
                if row['outcome']['status'] == 'synthesis_failed':
                    raise ValueError('This review expects the observed completed-run dataset')
                if identity['observer'] == 'NominalLTI':
                    key = ('nominal',); a, b = run.nominal_matrices()
                else:
                    key = (seed, index, speed, identity['observer'])
                    fit = full if identity['observer'] == 'FullStateCT' else yaw
                    a, b = observer_model(fit['clients'][index]['fit']['parameters'], speed, cfg['observer'])
                if key not in designs:
                    designs[key] = kalman_design(a, b, cfg['observer'], 1)['diagnostics']
                expected = designs[key]
                if set(expected) != set(row['observer']):
                    raise ValueError('Observer diagnostic fields differ')
                for name, value in expected.items():
                    if value is None:
                        assert row['observer'][name] is None
                    else:
                        np.testing.assert_allclose(row['observer'][name], value, rtol=1e-10, atol=1e-12)
                schedule = (run.old.storage.nominal_schedule(cc) if identity['controller'] == 'NominalLTI'
                            else controls['clients'][index]['schedules'][identity['controller']])
                gain = interpolate(schedule, cc['speed_grid'], [speed])[0][0]
                ap, bp = run.rk4_origin_matrices(plant, speed, cc['sample_time'])
                matrix = origin_closed_loop(ap, bp, a, b, gain, expected['steady_gain'], cc['sample_time'])
                rho = float(max(abs(np.linalg.eigvals(matrix))))
                np.testing.assert_allclose(rho, row['true_origin_rho'], rtol=1e-10, atol=1e-12)
                checked += 1
                o = row['outcome']
                records.append(dict(seed=seed, client=client['client'], family=client['client'].rsplit('_', 1)[0],
                    method=method, speed=speed, status=o['status'], feasible=o['feasible'],
                    true_origin_rho=row['true_origin_rho'], observer_error_rho=expected['error_rho'],
                    observability_rank=expected['observability_rank'],
                    observability_condition=expected['observability_condition'], **(o['metrics'] or {})))
                if 'preview' in row:
                    reconstructed, arrays = run.case(seed, index, method, speed, cfg, data)
                    if reconstructed['outcome']['status'] != o['status'] or reconstructed['outcome']['feasible'] != o['feasible']:
                        raise ValueError('Reconstructed status/feasibility mismatch')
                    differences = []
                    for name, value in o['metrics'].items():
                        other = reconstructed['outcome']['metrics'][name]
                        np.testing.assert_allclose(other, value, rtol=1e-10, atol=1e-10)
                        differences.append(abs(other-value))
                    preview_differences = []
                    for name, value in row['preview'].items():
                        other = arrays[name][::cfg['preview']['stride']]
                        np.testing.assert_allclose(other, value, rtol=1e-10, atol=1e-10)
                        preview_differences.append(float(np.max(abs(other-value))) if other.size else 0.)
                    exact = reconstructed['trace_sha256'] == row['trace_sha256']
                    full_trace_hash_matches += int(exact)
                    max_metric_difference = max(max_metric_difference, max(differences))
                    max_preview_difference = max(max_preview_difference, max(preview_differences))
                    replays.append(dict(seed=seed, client=client['client'], method=method, speed=speed,
                        original_trace_sha256=row['trace_sha256'], reconstructed_trace_sha256=reconstructed['trace_sha256'],
                        exact_hash_match=exact, numerical_check_passed=True,
                        max_metric_abs_difference=max(differences), max_preview_abs_difference=max(preview_differences)))
                    traces[(client['client'], method)] = arrays
        print(json.dumps(dict(seed=seed, audited_cases=checked, replayed_previews=len(replays))), flush=True)
    frame = pd.DataFrame(records)
    if len(frame) != 1350 or len(replays) != 15:
        raise ValueError('Unexpected completed case/preview count')
    clients, fleets, summary, speeds, paired = run.aggregate(frame, cfg)
    for name, expected in [('clients', clients), ('fleets', fleets), ('summary', summary),
                           ('speeds', speeds), ('paired', paired)]:
        actual = pd.read_csv(run.OUT/f'{run.PREFIX}_{name}.csv')
        pd.testing.assert_frame_equal(actual, expected, check_dtype=False, atol=1e-12, rtol=1e-10)
    metadata = dict(status='audit_complete', checked_cases=checked, matched_noise_hash_checks=noise_checked,
        unique_observer_designs=len(designs), reconstructed_preview_cases=len(replays),
        exact_reconstructed_trace_hash_matches=full_trace_hash_matches,
        max_replayed_metric_abs_difference=max_metric_difference,
        max_replayed_preview_abs_difference=max_preview_difference,
        numerical_replay_tolerance=dict(rtol=1e-10, atol=1e-10),
        checkpoint_sha256=checkpoint_hashes, protocol_sha256=run.old.file_hash(run.CONFIG),
        audit_script_sha256=run.old.file_hash(__file__), runtime=run.old.runtime(),
        original_results_modified=False, refits=0, new_scientific_cases=0,
        replay_scope='15 prespecified cases reconstructed numerically; full-resolution originals are not available here. Raw exact-hash replay remains strict. No original checkpoint/hash was replaced.')
    return frame, clients, fleets, summary, speeds, pd.DataFrame(replays), traces, metadata


def paired_table(frame, clients):
    result = []
    for level, data, keys in [('case', frame, ['seed','client','speed']),
                              ('vehicle', clients, ['seed','client'])]:
        for first, second in PAIRS:
            a = data[data.method == first].set_index(keys)
            b = data[data.method == second].set_index(keys)
            for metric in METRICS:
                delta = b[metric]-a[metric]
                result.append(dict(level=level, first=first, second=second, metric=metric,
                    pairs=len(delta), second_lower=int((delta < 0).sum()),
                    mean_first=float(a[metric].mean()), mean_second=float(b[metric].mean()),
                    mean_second_minus_first=float(delta.mean()),
                    percent_change_of_means=float(100*delta.mean()/a[metric].mean()),
                    median_delta=float(delta.median()), min_delta=float(delta.min()), max_delta=float(delta.max())))
    return pd.DataFrame(result)


def tables(frame, clients):
    speeds = frame.groupby(['method','speed'])[METRICS].mean().reset_index()
    families = frame.groupby(['method','family'])[METRICS].mean().reset_index()
    distribution = frame.groupby('method')[METRICS].quantile([.5,.95,1]).rename_axis(['method','quantile']).reset_index()
    a = clients[clients.method == 'FullStateCT/Matched'].set_index(['seed','client'])
    b = clients[clients.method == 'YawOnlyCT/Matched'].set_index(['seed','client'])
    tail = (b[METRICS]-a[METRICS]).sort_values('beta_estimation_rmse_deg', ascending=False).reset_index()
    return speeds, families, distribution, tail


def figures(frame, clients, traces, cfg):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = dict(zip(cfg['methods'], ['#626975','#8e9abb','#285c9e','#dda960','#b75725']))
    charts = []
    fig, axes = plt.subplots(1, 3, figsize=(13,4), layout='constrained')
    for method in cfg['methods']:
        values = frame[frame.method == method].groupby('speed')[METRICS].mean()
        for ax, metric, label in zip(axes, METRICS,
            ['Yaw tracking RMSE [deg/s]', 'Sideslip estimation RMSE [deg]', 'Steering-rate RMS [deg/s]']):
            ax.plot(values.index, values[metric], '-o', color=colors[method], label=method)
            ax.set(xlabel='Speed [m/s]', ylabel=label, xticks=[13,21,29]); ax.grid(alpha=.2)
    axes[0].legend(fontsize=6); charts.append(run.figure_svg(fig))
    fig, axes = plt.subplots(1,2, figsize=(10,4), layout='constrained')
    for ax, metric, label in zip(axes, METRICS[:2], ['Yaw tracking RMSE [deg/s]', 'Sideslip estimation RMSE [deg]']):
        a = clients[clients.method == 'FullStateCT/Matched'].set_index(['seed','client'])[metric]
        b = clients[clients.method == 'YawOnlyCT/Matched'].set_index(['seed','client'])[metric]
        ax.scatter(a, b, s=18, alpha=.7, color='#285c9e')
        lo = min(a.min(),b.min()); hi = max(a.max(),b.max())
        ax.plot([lo,hi],[lo,hi],'k--',linewidth=1)
        ax.set(xlabel='Full-state CT: '+label, ylabel='Yaw-only CT: '+label); ax.grid(alpha=.2)
    charts.append(run.figure_svg(fig))
    for client in cfg['preview']['clients']:
        fig, axes = plt.subplots(2,2, figsize=(11,6), layout='constrained')
        for method in cfg['methods']:
            p = traces[(client, method)]; time = p['time']; deg = np.rad2deg
            axes[0,0].plot(time,deg(p['state'][:,1]-p['reference']),label=method,color=colors[method],linewidth=.9)
            axes[0,1].plot(time,deg(p['estimate'][:,0]-p['state'][:,0]),label=method,color=colors[method],linewidth=.9)
            axes[1,0].plot(time[:-1],deg(p['steering']),label=method,color=colors[method],linewidth=.8)
            if method.endswith('/Matched'):
                axes[1,1].plot(time,deg(p['estimate'][:,0]),label=method+' estimate',color=colors[method])
                axes[1,1].plot(time,deg(p['state'][:,0]),'--',label=method+' truth',color=colors[method],linewidth=.8)
        for ax,label in zip(axes.flat,['Yaw tracking error [deg/s]','Sideslip estimation error [deg]',
                                       'Applied steering [deg]','Sideslip [deg]']):
            ax.set(xlabel='Time [s]',ylabel=label);ax.grid(alpha=.2)
        axes[0,0].legend(fontsize=6);axes[1,1].legend(fontsize=6)
        fig.suptitle(f'Prespecified example: fleet 1301, {client}, 21 m/s')
        if client == 'nominal_00':
            fig.savefig(Path(tempfile.gettempdir())/'physical_ct_observer_review_preview.png',dpi=130)
        charts.append(run.figure_svg(fig))
    return charts


def report(summary, fleet, speed, families, distribution, paired, tail, charts, meta):
    def table(data):
        return '<div class="scroll">'+data.to_html(index=False,border=0,float_format=lambda v:f'{v:.6g}')+'</div>'
    headline = summary[['method',*METRICS,'complete','feasible','origin_unstable']]
    tracking_pairs = paired[(paired.level == 'vehicle') & (paired.metric == METRICS[0])]
    tail_fraction = 100*tail.beta_estimation_rmse_deg.head(10).sum()/tail.beta_estimation_rmse_deg.sum()
    text = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Observer pilot: audited interpretation</title><style>body{{font:16px/1.6 system-ui;background:#eef3f6;color:#173044}}main{{max-width:1300px;margin:auto;padding:28px}}section{{background:white;padding:24px;margin:20px 0;border-radius:10px}}.scroll{{overflow:auto}}table{{border-collapse:collapse;white-space:nowrap;font-size:12px}}th,td{{padding:8px;border-bottom:1px solid #ddd}}svg{{width:100%;height:auto}}.lead{{font-size:19px}}pre{{white-space:pre-wrap}}</style><main>
<h1>Learned local models support yaw-only feedback</h1>
<p class="lead">Matched model-based observation improves tracking and sideslip estimates relative to a fixed 20 m/s nominal observer. Removing sideslip from CT fitting still causes almost no tracking penalty. This experiment does not establish a need for federation or isolate learning from speed scheduling.</p>
<section><h2>All 1350 cases completed</h2><p>270 cases per configuration; all satisfy the inherited steering/sideslip bounds. No steering limiter activation, synthesis failure or numerical failure. All evaluated true-origin controller–observer spectral radii are below one. These are local linear diagnostics, not nonlinear stability certificates.</p>{table(headline)}
<p>Means weight speed, vehicle and fleet equally. FullStateCT refers to training with sideslip measurements; all five configurations use yaw measurements only online. Matched observers use straight-running models reconstructed from the learned physical coefficients. Controllers use unchanged saved gains.</p></section>
<section><h2>What changes when the observer uses the learned model?</h2><p>With the yaw-only CT controller held fixed, changing Nominal to Matched reduces mean tracking RMSE by 47.32% (0.03464 → 0.01825°/s), and sideslip-estimation RMSE by 93.80% (0.23750 → 0.01474°). Tracking improves for all 90 vehicle averages, although only 249 of 270 individual speed cases improve. Mean steering-rate RMS changes by −1.10%; the observer improvement does not require a higher mean steering-rate RMS in this comparison.</p>
<p>Changing both controller and observer from Nominal/Nominal to YawOnlyCT/Matched reduces mean tracking RMSE by 64.61%, an absolute 0.03332°/s. Steering-rate RMS rises by 26.22% (2.980 → 3.761°/s). Report this control-effort tradeoff with the tracking benefit. No method was tuned separately.</p>{charts[0]}{table(speed)}
<p>The strongest tracking improvement is at 13 m/s: Nominal/Nominal has 0.10601°/s mean error, YawOnlyCT/Matched 0.02385°/s. At 29 m/s the corresponding values are 0.02383 and 0.01430°/s. Thus the overall gain is consistent across vehicles but strongly speed-dependent, with a large contribution from the low-speed condition.</p></section>
<section><h2>Yaw-only versus full-state identification</h2><p>With matched observers, mean tracking RMSE is 0.018216°/s for FullStateCT and 0.018251°/s for YawOnlyCT: a difference of only 0.0000354°/s (0.194%). YawOnlyCT gives lower tracking error for 40 of 90 vehicle averages and 119 of 270 cases. This is a descriptive near-tie, not a formal equivalence test.</p>
<p>Sideslip estimation is more sensitive: mean RMSE increases from 0.010263° to 0.014735° (+43.57%, absolute +0.004472°). The case-level 95th percentile rises from 0.01812° to 0.03720°. The maximum is 0.03584° versus 0.05584°. The increase is not uniform: yaw-only fitting improves the sideslip estimate for 39 of 90 vehicle averages. The ten largest positive vehicle differences account for {tail_fraction:.2f}% of the net mean increase; the median vehicle difference is +0.001718°.</p>{charts[1]}{table(distribution)}
<p>A worse sideslip estimate need not worsen yaw tracking proportionally. The observer also directly corrects the measured yaw channel and the integrator acts on measured yaw error. These results show that the resulting sideslip errors have limited tracking consequences for this maneuver and controller; they do not prove that sideslip information is generally unnecessary.</p>{table(tail.head(10))}</section>
<section><h2>What the nominal comparison does not isolate</h2><p>The nominal observer uses one fixed 20 m/s LTI model at 13, 21 and 29 m/s. The matched observer uses both recipient-specific learned parameters and the correct test speed. Its advantage therefore combines physical-parameter learning and known speed dependence. A speed-scheduled observer built from unchanged nominal physical parameters is the missing comparison before attributing the whole gain to identification. Hold the learned controller fixed when comparing these observers.</p>
<p>The very high mean normalized innovation squared with the fixed nominal observer (about 57) versus about 0.86 with matched observers is consistent with substantial nominal model mismatch under the chosen covariance allowance. It is not a calibrated statistical consistency test: Q was prescribed, not estimated, and the plant is nonlinear.</p>
<p>The local yaw-only CT baseline remains strong. This test provides no evidence that federation is needed, nor does it compare a newly learned federated observer. Previous training data were collected with full-state feedback. A future complete sensing workflow must acquire new trajectories with the common nominal observer/controller, then refit from measured yaw and applied steering.</p></section>
<section><h2>Prespecified full-resolution examples</h2><p>These are the original three selected clients, not selected winners or worst cases. The following traces were numerically reconstructed on this platform. All saved preview samples and all full-duration metrics agree within the audit tolerance; their raw hashes are not byte-identical to the original Windows traces.</p>{''.join(charts[2:])}</section>
<section><h2>Paired vehicle and fleet checks</h2>{table(tracking_pairs)}{table(fleet[['seed','method',*METRICS]])}{table(families)}<p>All three fleets show the same mean ordering. These are three reused pilot fleets and one paired noise realization per case; no confirmatory confidence intervals or universal robustness claim are made.</p></section>
<section><h2>Recommended next step</h2><p>First add the small speed-scheduled nominal-observer comparison, holding the learned controller fixed. It distinguishes knowledge of speed dependence from learned recipient parameters without refitting. Then, if the acquisition controller/observer passes its operating checks, generate new yaw-only acquisition trajectories and refit. Do not add plant complexity or change covariance settings merely to obtain a larger method gap.</p></section>
<section><h2>Audit and reconstruction limits</h2><p>Verified all checkpoint integrity/provenance records, 1350 noise hashes, observer designs, true-origin spectral radii, and regenerated aggregate tables. Reconstructed the 15 prespecified preview cases without fitting or tuning. Maximum absolute difference across recorded metric values: {meta['max_replayed_metric_abs_difference']:.3g}; maximum saved-preview numeric difference: {meta['max_replayed_preview_abs_difference']:.3g}. These maxima span different physical units and are numerical reproducibility diagnostics only.</p>
<p>Exact hash matches: {meta['exact_reconstructed_trace_hash_matches']}/15. The original strict replay command rejects these cross-platform differences, as intended. Numerical agreement is checked at rtol=1e-10 and atol=1e-10, while original results and hashes remain unchanged. Full-resolution original arrays for all 1350 cases are not available here; only the selected 15 are independently reconstructed.</p>
<p><a href="physical_ct_observer.html">Protocol-generated report</a> · <a href="../docs/physical_ct_observer.html">Protocol and local commands</a></p></section></main></html>'''
    REPORT.write_text('\n'.join(line.rstrip() for line in text.splitlines())+'\n',encoding='utf-8',newline='\n')


def main():
    cfg = run.preflight()
    with threadpool_limits(limits=1):
        frame, clients, fleets, summary, _, replays, traces, meta = audit(cfg)
    paired = paired_table(frame, clients)
    speeds, families, distribution, tail = tables(frame, clients)
    for name, data in [('paired',paired),('speeds',speeds),('families',families),
                       ('distribution',distribution),('tail',tail),('replays',replays)]:
        data.to_csv(run.OUT/f'{PREFIX}_{name}.csv',index=False)
    charts = figures(frame,clients,traces,cfg)
    report(summary,fleets,speeds,families,distribution,paired,tail,charts,meta)
    # Whitespace-only normalization of generated SVG markup; frozen runner stays unchanged.
    original_report = run.REPORT.read_text(encoding='utf-8')
    run.REPORT.write_text('\n'.join(line.rstrip() for line in original_report.splitlines())+'\n',
                          encoding='utf-8',newline='\n')
    run.atomic_json(run.OUT/f'{PREFIX}.json',meta)
    print(json.dumps(meta,indent=2));print(REPORT)


if __name__ == '__main__':
    main()
