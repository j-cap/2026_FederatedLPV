"""Audit saved payload results and render a separate decision-focused review.

No refits or new scientific cases. Acquisition and prespecified preview replays
are numerical cross-platform checks, never replacements for original traces.
"""
from __future__ import annotations
import os
for name in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[name] = '1'
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'code/src'), str(ROOT/'code/experiments')]
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import paper_ct_payload_adaptation as run
from federated_lpv.payload_adaptation import yaw_prediction, loaded_parameters, fitting_segment

PREFIX = 'paper_ct_payload_adaptation_review'
REPORT = ROOT/'report/ct_payload_adaptation_review.html'
METRICS = run.METRICS[:3]


def compare_replay(original, replay, arrays, stride):
    for field in ['status', 'feasible', 'completed_steps']:
        assert original['outcome'][field] == replay['outcome'][field]
    errors = []
    for name, value in original['outcome']['metrics'].items():
        other = replay['outcome']['metrics'][name]
        np.testing.assert_allclose(other, value, rtol=1e-10, atol=1e-10)
        errors.append(abs(other-value))
    preview_error = 0.
    for name, value in original.get('preview', {}).items():
        other = arrays[name][::stride]
        np.testing.assert_allclose(other, value, rtol=1e-10, atol=1e-10)
        preview_error = max(preview_error, float(np.max(abs(other-value))))
    return dict(original_noise_sha256=original['noise_sha256'], replay_noise_sha256=replay['noise_sha256'],
        exact_noise_hash_match=original['noise_sha256'] == replay['noise_sha256'],
        original_trace_sha256=original['trace_sha256'],
        replay_trace_sha256=replay['trace_sha256'],
        exact_hash_match=original['trace_sha256'] == replay['trace_sha256'],
        max_metric_difference=max(errors), max_preview_difference=preview_error)


def audit(cfg):
    records = []; fit_records = []; replays = []; hashes = {}; traces = {}
    max_loss_error = 0.; noise_checks = 0; noise_exact = 0; extra_replays = 0
    for client, percent in run.jobs(cfg):
        p = run.base.old.envelope(run.path(client, percent))
        run.validate(p, client, percent, cfg)
        assert p['status'] == 'execution_complete'
        hashes[run.path(client, percent).relative_to(ROOT).as_posix()] = run.base.old.file_hash(run.path(client, percent))
        index, _, previous, _ = run.material(client, percent, cfg)
        acq, arrays = run.acquire(client, percent, cfg)
        replays.append(dict(client=client, loading_percent=percent, kind='acquisition', method='Previous', speed=cfg['acquisition']['speed_mps'],
            **compare_replay(p['acquisition'], acq, arrays, 10)))
        assert acq['operational_checks_passed'] == p['acquisition']['operational_checks_passed']
        # Evaluate all saved grid points and selected estimates together; do not refit.
        grid = np.linspace(*cfg['fit']['bounds'], cfg['fit']['grid_points'])
        fractions = np.r_[grid, [p['fits'][str(t)]['fraction'] for t in cfg['prefix_seconds']]]
        segment = fitting_segment(arrays, cfg['prefix_seconds'][-1], cfg['fit']['sample_time'])
        pred = yaw_prediction(loaded_parameters(previous, fractions, cfg['fit']['payload_radius_m']),
            segment['applied_input'], segment['speed'], cfg['fit']['geometry_m'], cfg['fit']['sample_time'])
        for j, seconds in enumerate(cfg['prefix_seconds']):
            fit = p['fits'][str(seconds)]
            assert fit['converged'] and not fit['boundary']
            np.testing.assert_allclose(fit['parameters'], loaded_parameters(previous, fit['fraction'], cfg['fit']['payload_radius_m']), rtol=1e-13, atol=1e-13)
            np.testing.assert_array_equal(fit['grid'], grid)
            n = round(seconds/cfg['fit']['sample_time'])+1
            loss = np.mean(((pred[:, :n]-segment['measured_yaw'][None, :n])/np.deg2rad(cfg['fit']['yaw_noise_std_deg_s']))**2, axis=1)
            np.testing.assert_allclose(loss[:len(grid)], fit['grid_loss'], rtol=1e-9, atol=1e-9)
            np.testing.assert_allclose(loss[len(grid)+j], fit['weighted_mse'], rtol=1e-9, atol=1e-9)
            np.testing.assert_allclose(fit['previous_model_weighted_mse'], fit['grid_loss'][0])
            candidates = [fit['grid_loss'][0], fit['grid_loss'][-1]] + [a['weighted_mse'] for a in fit['attempts'] if a['converged']]
            np.testing.assert_allclose(fit['weighted_mse'], min(candidates), rtol=1e-12, atol=1e-12)
            assert fit['weighted_mse'] <= min(fit['grid_loss'])+1e-9
            max_loss_error = max(max_loss_error, float(np.max(abs(loss[:len(grid)]-fit['grid_loss']))), abs(loss[len(grid)+j]-fit['weighted_mse']))
            fit_records.append(dict(client=client, loading_percent=percent, prefix_s=seconds,
                error_pp=100*fit['fraction']-percent, weighted_mse=fit['weighted_mse']))
        for row in p['rows']:
            expected_noise = run.sensor_noise(index, percent, 1, row['speed'], row['outcome']['completed_steps']+1, cfg)
            noise_matches = run.trace_hash({'noise':expected_noise}) == row['noise_sha256']
            noise_checks += 1; noise_exact += int(noise_matches)
            theta = previous if row['method'] == 'Previous' else loaded_parameters(previous,
                percent/100 if row['method'] == 'KnownChange' else p['fits'][row['method'][5:]]['fraction'], cfg['fit']['payload_radius_m'])
            np.testing.assert_allclose(row['parameters'], theta, rtol=1e-13, atol=1e-13)
            out = row['outcome']
            records.append(dict(client=client, family=client.rsplit('_',1)[0], loading_percent=percent,
                method=row['method'], speed=row['speed'], status=out['status'], feasible=out['feasible'],
                true_origin_rho=row['true_origin_rho'], **out['metrics']))
            if 'preview' in row or not noise_matches:
                extra_replays += int('preview' not in row)
                replay, full = run.evaluate(client, percent, row['method'], row['speed'], cfg,
                    p['fits'][row['method'][5:]] if row['method'].startswith('Local') else None)
                replays.append(dict(client=client, loading_percent=percent, kind='control' if 'preview' in row else 'noise_hash_followup', method=row['method'], speed=row['speed'],
                    **compare_replay(row, replay, full, 5)))
                if 'preview' in row:
                    traces[row['method']] = full
        print(json.dumps(dict(audited_client=client, loading_percent=percent, control_rows=len(records))), flush=True)
    frame = pd.DataFrame(records); fits = pd.DataFrame(fit_records)
    assert len(frame) == 567 and len(fits) == 135 and len(replays) == 34+extra_replays
    original = pd.read_csv(run.OUT/f'{run.PREFIX}_cases.csv')
    pd.testing.assert_frame_equal(frame, original, check_dtype=False, rtol=1e-10, atol=1e-12)
    units, vehicles, summary, paired, tolerances = run.tables(frame, cfg)
    for name, expected in [('units',units),('vehicles',vehicles),('summary',summary),('paired',paired),('tolerances',tolerances)]:
        pd.testing.assert_frame_equal(pd.read_csv(run.OUT/f'{run.PREFIX}_{name}.csv'), expected, check_dtype=False, rtol=1e-10, atol=1e-12)
    for name, digest in hashes.items():
        assert run.base.old.file_hash(ROOT/name) == digest
    replay_frame = pd.DataFrame(replays)
    metadata = dict(status='audit_complete', control_rows=567, fits=135, loading_cases=27,
        replayed_acquisitions=27, replayed_prespecified_control_previews=7,
        checked_control_noise_hashes=noise_checks, exact_control_noise_hash_matches=noise_exact,
        additional_control_replays_after_noise_hash_difference=extra_replays, verified_training_grid_values=135*len(grid),
        verified_selected_training_losses=135, max_training_loss_abs_difference=float(max_loss_error),
        exact_trace_hash_matches=int(replay_frame.exact_hash_match.sum()),
        max_replay_metric_abs_difference=float(replay_frame.max_metric_difference.max()),
        max_replay_preview_abs_difference=float(replay_frame.max_preview_difference.max()),
        replay_tolerance=dict(rtol=1e-10,atol=1e-10), loss_tolerance=dict(rtol=1e-9,atol=1e-9),
        original_checkpoints_modified=False, refits=0, new_scientific_cases=0,
        checkpoint_sha256=hashes, protocol_sha256=run.base.old.file_hash(run.CONFIG),
        audit_script_sha256=run.base.old.file_hash(__file__), runtime=run.base.old.runtime(),
        note='Reconstructed numerical replays, not recovered full-resolution originals. Original hashes remain unchanged. No inferential independence is assigned to repeated loading levels or speeds.')
    return frame, fits, units, summary, tolerances, replay_frame, traces, metadata


def compare(frame, units, cfg):
    result = []
    for level, data, keys in [('case',frame,['client','loading_percent','speed']),('vehicle_loading',units,['client','loading_percent'])]:
        previous = data[data.method=='Previous'].set_index(keys)
        known = data[data.method=='KnownChange'].set_index(keys)
        for t in cfg['prefix_seconds']:
            local = data[data.method==f'Local{t}'].set_index(keys)
            for metric in METRICS:
                delta = local[metric]-known[metric]; gap = previous[metric].mean()-known[metric].mean()
                result.append(dict(level=level, prefix_s=t, metric=metric, pairs=len(delta),
                    previous_mean=previous[metric].mean(), known_mean=known[metric].mean(), local_mean=local[metric].mean(),
                    remaining_mean_gap=delta.mean(), min_remaining_gap=delta.min(), max_remaining_gap=delta.max(),
                    local_better_than_previous=int((local[metric]<previous[metric]).sum()),
                    local_better_than_known=int((delta<0).sum()),
                    mean_gap_closed_percent=100*(previous[metric].mean()-local[metric].mean())/gap if gap>1e-12 else np.nan))
    return pd.DataFrame(result)


def render(frame, fits, summary, comparison, traces, metadata, cfg):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({'svg.hashsalt':PREFIX, 'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False})
    from io import StringIO
    def svg(fig):
        stream=StringIO();fig.savefig(stream,format='svg',metadata={'Date':None});plt.close(fig)
        return stream.getvalue()[stream.getvalue().find('<svg'):]
    def table(data):
        return '<div class="scroll">'+data.to_html(index=False,border=0,float_format=lambda x:f'{x:.6g}')+'</div>'
    colors={'Previous':'#687781','KnownChange':'#c77722','Local4':'#147c93','Local40':'#7852a5'}
    values=summary.set_index('method');times=cfg['prefix_seconds']
    fig,axes=plt.subplots(1,3,figsize=(12.5,3.6),layout='constrained')
    for ax,metric,label in zip(axes,METRICS,['Yaw tracking RMSE [°/s]','Sideslip estimation RMSE [°]','Steering-rate RMS [°/s]']):
        ax.plot(times,[values.loc[f'Local{t}',metric] for t in times],'-o',color=colors['Local4'],label='Local adaptation')
        for method in ['Previous','KnownChange']:
            ax.axhline(values.loc[method,metric],color=colors[method],ls='--',label=method)
        ax.set(xlabel='Available identification data [s]',ylabel=label);ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.savefig(Path(tempfile.gettempdir())/'ct_payload_review_overview.png',dpi=150)
    overview=svg(fig)
    fig,axes=plt.subplots(1,2,figsize=(12.5,3.8),layout='constrained')
    known=frame[frame.method=='KnownChange'].set_index(['client','loading_percent','speed'])
    for j,t in enumerate(times):
        local=frame[frame.method==f'Local{t}'].set_index(known.index.names)
        delta=(local.tracking_rmse_deg_s-known.tracking_rmse_deg_s).to_numpy()
        # Fixed small horizontal offsets expose points without suggesting uncertainty intervals.
        axes[0].scatter(j+np.linspace(-.15,.15,len(delta)),1000*delta,s=9,alpha=.5,color=colors['Local4'])
        axes[0].plot(j,1000*delta.mean(),'k_',ms=15)
        e=fits[fits.prefix_s==t].error_pp.to_numpy()
        axes[1].scatter(j+np.linspace(-.15,.15,len(e)),e,s=16,alpha=.65,color=colors['Local4'])
        axes[1].plot(j,e.mean(),'k_',ms=15)
    for ax,label in zip(axes,['Local − KnownChange yaw RMSE [10⁻³ °/s]','Estimated − actual added mass [percentage points]']):
        ax.axhline(0,color='#555',lw=.8);ax.set(xticks=range(5),xticklabels=times,xlabel='Available identification data [s]',ylabel=label);ax.grid(alpha=.2)
    distributions=svg(fig)
    fig,axes=plt.subplots(3,1,figsize=(11,7),sharex=True,layout='constrained')
    for method in colors:
        a=traces[method];style='--' if method=='KnownChange' else '-'
        axes[0].plot(a['time'],np.rad2deg(a['state'][:,1]-a['reference']),style,color=colors[method],label=method,lw=1)
        axes[1].plot(a['time'],np.rad2deg(a['estimate'][:,0]-a['state'][:,0]),style,color=colors[method],lw=1)
        axes[2].plot(a['time'][:len(a['steering'])],np.rad2deg(a['steering']),style,color=colors[method],lw=1)
    for ax,label in zip(axes,['Yaw tracking error [°/s]','Sideslip estimation error [°]','Applied steering [°]']):
        ax.set(ylabel=label);ax.grid(alpha=.2)
    axes[0].legend(ncol=4,fontsize=9);axes[-1].set_xlabel('Time [s]')
    preview=svg(fig)
    byload=frame[frame.method.isin(colors)].groupby(['loading_percent','method'])[METRICS].mean().reset_index()
    biases=fits.assign(absolute_error_pp=lambda x:abs(x.error_pp)).groupby('prefix_s').agg(mean_bias_pp=('error_pp','mean'),mean_absolute_error_pp=('absolute_error_pp','mean'),max_absolute_error_pp=('absolute_error_pp','max')).reset_index()
    compact=summary[summary.method.isin(colors)][['method',*METRICS,'complete','feasible']]
    p=comparison[(comparison.level=='case')&(comparison.prefix_s==4)].set_index('metric')
    yaw=p.loc[METRICS[0]];beta=p.loc[METRICS[1]]
    effort=100*(values.loc['Local4',METRICS[2]]/values.loc['Previous',METRICS[2]]-1)
    body=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Centered payload adaptation — decision review</title>
<style>body{{margin:0;background:#eef2f5;color:#173044;font:16px/1.65 system-ui}}main{{max-width:1150px;margin:auto;padding:30px}}section{{background:white;border-radius:12px;padding:25px;margin:22px 0}}h1{{line-height:1.2;font-size:34px}}h2{{line-height:1.3}}.lead{{font-size:20px}}.decision{{border-left:6px solid #147c93}}.scroll{{overflow:auto}}table{{border-collapse:collapse;font-size:13px;white-space:nowrap}}th,td{{padding:9px;text-align:right;border-bottom:1px solid #dde4e9}}th:first-child,td:first-child{{text-align:left}}svg{{width:100%;height:auto}}.note{{color:#526573;font-size:14px}}a{{color:#147c93}}code{{background:#eef2f5;padding:2px 5px}}</style><main>
<h1>Centered payload adaptation:<br>local identification leaves little room for a fleet prior</h1>
<p class="lead">Updating the model helps, particularly the sideslip estimate. But four seconds of local yaw-only data recover almost all of the available benefit in this controlled scenario.</p>
<section class="decision"><h2>Decision for this scenario</h2><p><strong>Do not build a fleet prior for the present centered-loading test.</strong> At four seconds, the earliest tested prefix, local identification closes {yaw.mean_gap_closed_percent:.1f}% of the mean yaw-tracking gap and {beta.mean_gap_closed_percent:.1f}% of the mean sideslip-estimation gap between the previous model and the known-change reference. Longer prefixes provide virtually no additional control benefit.</p><p>This rules out a useful persistent adaptation gap in these tests. It does not rule out a benefit before four seconds, with less informative driving, or under unknown geometry and multiple simultaneous changes. Those would require a separately motivated experiment.</p></section>
<section><h2>What was tested</h2><p>Nine previously identified vehicles from one pilot fleet; three centered payload amounts (10%, 20%, 30%); five nested data prefixes (4, 8, 16, 32, 40 s); three held-out speeds (13, 21, 29 m/s). The 27 loading cases are repeated conditions on nine vehicles. There are 135 scalar fits and 567 closed-loop tests, with paired yaw noise across methods.</p><p><strong>Previous:</strong> saved pre-change model, controller and observer. <strong>KnownChange:</strong> true loading fraction applied to the same imperfect identified model. <strong>Local4 / Local40:</strong> loading fraction estimated from four / forty seconds of measured yaw, known speed and applied steering. No measured sideslip enters the new identification or feedback. KnownChange is a reference, not an exact-plant oracle or guaranteed lower bound.</p><p>All methods know the change is centered loading with a 1 m payload radius of gyration; only its amount is unknown to Local. Tire stiffness, friction and geometry remain fixed. Acquisition uses the previous controller/observer at 20 m/s and smooth prescribed curves with nominal lateral demand 1.5 m/s². Four seconds cover the first positive lobe of an eight-second left–right curve.</p><p class="note">The load is applied before straight-running initialization. Controllers are tested after fitting, in separate reset rollouts. These results do not measure online cumulative tracking performance during adaptation, estimator execution latency, or ordinary unplanned driving. No fleet prior is trained here.</p></section>
<section><h2>The benefit is real, but local adaptation captures it immediately at the tested resolution</h2>{table(compact)}{overview}<p>Means weight speeds, loading levels and vehicles equally. Local4 reduces mean yaw RMSE by {100*(1-values.loc['Local4',METRICS[0]]/values.loc['Previous',METRICS[0]]):.1f}% and sideslip-estimation RMSE by {100*(1-values.loc['Local4',METRICS[1]]/values.loc['Previous',METRICS[1]]):.1f}%. Its steering-rate RMS is {effort:.1f}% higher than Previous. This effort increase is also present in KnownChange; improved error does not imply improvement in every control metric.</p><p>The absolute mean tracking improvement is only {yaw.previous_mean-yaw.local_mean:.6f}°/s. The residual Local4–KnownChange difference is {yaw.remaining_mean_gap:.8f}°/s. The 99.4% statement is a ratio of aggregate mean differences, not a guarantee for each case.</p></section>
<section><h2>No persistent tracking gap is hidden by the average</h2>{distributions}<p>All 27 vehicle–loading combinations improve on Previous when averaged over the three speeds. At the individual speed level, Local4 improves yaw RMSE in 68 of 81 cases; small regressions remain. The largest positive Local4–KnownChange difference is {yaw.max_remaining_gap:.8f}°/s. All 27 loading cases satisfy each prespecified descriptive tolerance (0.005, 0.01, 0.02°/s) from four seconds onward at all three speeds, including every later prefix. These tolerances are not vehicle acceptance standards.</p><p>Sideslip estimation is less uniform than the tracking averages suggest. Local4 improves it in 78 of 81 cases. Its largest excess RMSE over KnownChange is 0.01184° (heavy_00, 30% loading, 13 m/s: 0.02503° versus 0.01319°), still far below Previous at 0.20299°. In the three handling_01 / 10% loading cases, both updated models estimate sideslip worse than Previous. Thus nearly complete mean gap closure does not mean identical or universally better state estimates.</p><p>Dots show individual cases: 81 per prefix for tracking, 27 for loading estimation; black marks show means. Repeated conditions are not independent replicates. No significance claim or fleet-population confidence interval is made.</p>{table(biases)}<p>The mean absolute loading error at four seconds is about 0.69 percentage points. More data do not systematically remove the roughly −0.54 percentage-point mean bias. This is consistent with residual error in the pre-change model, but this audit does not identify the bias mechanism. Exact parameter recovery is not needed to obtain nearly the same tracking performance as KnownChange.</p><details><summary>Performance by loading amount</summary>{table(byload)}</details></section>
<section><h2>Prespecified time-series example</h2><p>Vehicle nominal_00, 20% payload, 21 m/s. This case was fixed in the experiment configuration before execution. All four curves use the same noise. Local4, Local40 and KnownChange almost overlap; that overlap is itself the result.</p>{preview}<p class="note">Full-rate numerical reconstruction of the saved case; all plotted metrics agree with the original results within the audit tolerance. This example illustrates the mechanism; the complete-case comparisons above support the conclusion.</p></section>
<section><h2>Integrity and operating checks</h2><p>All 27 acquisition cases pass the prescribed checks. All 135 fits converge without a boundary solution. All 567 control tests finish and satisfy the defined operating bounds, with no steering-limiter activation and no unstable true-origin linearization. These finite tests and local linearizations are not a global stability or vehicle-safety guarantee.</p><p>The review verifies all checkpoint identities, paired control-noise hashes within all 81 vehicle–loading–speed comparisons, all 567 parameter transforms, all aggregate tables, all 135 selected training losses and 5,535 grid losses. It reconstructs all 27 acquisition traces and seven prespecified control previews without refitting. Of 567 regenerated control-noise hashes, {metadata['exact_control_noise_hash_matches']} match exactly. The remaining seven share one vehicle–loading–speed condition; all seven methods were additionally replayed and their saved metrics checked numerically. The cause of this cross-platform hash difference is not established by this audit. Maximum objective discrepancy: {metadata['max_training_loss_abs_difference']:.3g}; maximum replay metric discrepancy: {metadata['max_replay_metric_abs_difference']:.3g}; maximum preview sample discrepancy: {metadata['max_replay_preview_abs_difference']:.3g}.</p><p>Exact full-trace hash matches: {metadata['exact_trace_hash_matches']}/{34+metadata['additional_control_replays_after_noise_hash_difference']}. Numerical agreement is checked separately from byte identity. Original checkpoints, hashes, protocol and full-resolution reconstruction rules are unchanged.</p></section>
<section><h2>What this means for the research direction</h2><p>The strong physical model and a single well-excited unknown make this loading change locally identifiable very quickly. A donor prior would have to outperform this baseline over an earlier, practically important window; the present results give no evidence that such a window matters.</p><p>Keep this as a negative diagnostic for centered payload adaptation. Do not make coverage or noise artificially harsher just to obtain a learning advantage. If pursuing the broader adaptation direction, first establish a real use case for unknown change structure or weak excitation. Alternatively, discuss a specific missing-physics problem with a correspondingly strong augmented local physical baseline before implementing another experiment.</p><p><a href="ct_payload_adaptation.html">Original complete report</a> · <a href="../docs/ct_payload_adaptation.html">Frozen protocol and reproduction commands</a></p></section></main></html>'''
    REPORT.write_text('\n'.join(line.rstrip() for line in body.splitlines())+'\n',encoding='utf-8',newline='\n')


def main():
    cfg=run.preflight()
    with threadpool_limits(limits=1):
        frame,fits,units,summary,tolerances,replays,traces,metadata=audit(cfg)
    comparison=compare(frame,units,cfg)
    for name,data in [('comparisons',comparison),('fit_errors',fits),('replays',replays)]:
        data.to_csv(run.OUT/f'{PREFIX}_{name}.csv',index=False)
    run.base.atomic_json(run.OUT/f'{PREFIX}_audit.json',metadata)
    render(frame,fits,summary,comparison,traces,metadata,cfg)
    print(json.dumps({k:v for k,v in metadata.items() if k not in ['checkpoint_sha256','runtime']},indent=2))
    print(REPORT)


if __name__=='__main__':
    main()
