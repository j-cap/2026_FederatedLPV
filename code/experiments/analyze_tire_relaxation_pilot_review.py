"""Audit and scientific interpretation of the frozen local relaxation pilot.

No optimizer, new test condition, controller tuning or original artifact writes.
Existing rollouts are replayed numerically; byte-exact recovery is reported apart.
"""
from __future__ import annotations
import os
for variable in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[variable] = '1'
from concurrent.futures import ProcessPoolExecutor, as_completed
import io
import json
import multiprocessing
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'code/src'), str(ROOT/'code/experiments')]
import numpy as np
import pandas as pd
from scipy.linalg import solve_discrete_lyapunov
from threadpoolctl import threadpool_limits
import paper_tire_relaxation_pilot as run
from federated_lpv import tire_relaxation_pilot as model
from federated_lpv.tire_relaxation import sampled_matrices

PREFIX = 'paper_tire_relaxation_pilot_review'
REPORT = ROOT/'report/tire_relaxation_pilot_review.html'


def compare(a, b):
    if isinstance(b, dict):
        assert set(a) == set(b)
        for key in b: compare(a[key], b[key])
    elif isinstance(b, (list, float, int)) and not isinstance(b, bool):
        np.testing.assert_allclose(a, b, rtol=1e-8, atol=1e-9)
    else: assert a == b


def linear_diagnostic(c, speed, d, cfg):
    """Steady linear noise/reference transfer, evaluated with the true plant.

    z=[plant state, observer prior, integral]. Neither these calculations nor
    plant truth enter fitted models or controller design. Limits/tapers omitted.
    """
    plant = np.r_[c['physical_ratios'], cfg['relaxation_lengths_m']]
    a, b = np.array(d['a']), np.array(d['b'])[:, None]
    gain = np.array(d['gain']); ks = gain[:-1][None, :]
    l = np.array(d['steady_observer_gain'])[:, None]; n = len(b)
    cp = np.array([[0., 1., 0., 0.]]); cm = np.zeros((1, n)); cm[0, 1] = 1.
    ap, bp = sampled_matrices(plant, speed, c['geometry_m'], cfg['sample_time']); bp = bp[:, None]
    j = np.eye(n)-l@cm
    m = np.block([[ap-bp@ks@l@cp, -bp@ks@j, -bp*gain[-1]],
        [(a-b@ks)@l@cp, (a-b@ks)@j, -b*gain[-1]],
        [cfg['sample_time']*cp, np.zeros((1, n)), np.ones((1, 1))]])
    rho = float(max(abs(np.linalg.eigvals(m))))
    np.testing.assert_allclose(rho, d['actual_plant_origin_rho'], rtol=1e-8, atol=1e-9)
    if rho >= 1: raise ValueError('Stationary covariance unavailable for unstable linearized loop')
    gn = np.r_[(-bp@ks@l).ravel(), ((a-b@ks)@l).ravel(), cfg['sample_time']]
    gr = np.r_[bp.ravel()*d['prefilter'], b.ravel()*d['prefilter'], -cfg['sample_time']]
    noise_var = np.deg2rad(cfg['yaw_noise_std_deg_s'])**2
    p = solve_discrete_lyapunov(m, np.outer(gn, gn)*noise_var)
    np.testing.assert_allclose(p, m@p@m.T+np.outer(gn, gn)*noise_var, rtol=1e-7, atol=1e-12)
    command_map = np.r_[(-ks@l@cp).ravel(), (-ks@j).ravel(), -gain[-1]]
    direct = float((-ks@l)[0, 0])
    noise_rms = float(np.rad2deg(np.sqrt(p[1, 1])))
    steering_noise_rms = float(np.rad2deg(np.sqrt(command_map@p@command_map+direct**2*noise_var)))
    records = []
    for probe, hz in cfg['control_reference_hz'].items():
        transfer = np.linalg.solve(np.exp(2j*np.pi*hz*cfg['sample_time'])*np.eye(len(m))-m, gr)[1]
        records.append(dict(probe=probe, reference_hz=hz, yaw_gain=float(abs(transfer)),
            yaw_phase_deg=float(np.angle(transfer, deg=True)),
            steady_sinusoidal_tracking_rmse_deg_s=float(np.rad2deg(abs(transfer-1)*cfg['reference_peak_ay_mps2']/speed/np.sqrt(2))),
            stationary_yaw_noise_rms_deg_s=noise_rms, stationary_steering_noise_rms_deg=steering_noise_rms,
            instantaneous_steering_per_yaw_noise=direct, actual_plant_origin_rho=rho))
    return records


def worker(ci):
    cfg = run.preflight(); c = cfg['clients'][ci]; name = c['client']
    saved = run.read_checkpoint(run.path(name)); run.validate(saved, cfg, name)
    assert saved['status'] == 'execution_complete'
    replays = []; fitchecks = []; linear = []; dynamics = []; paired_noise = {}
    max_preview = 0.; max_metric = 0.
    with threadpool_limits(limits=1):
        records = []
        for v in cfg['training_speeds_mps']:
            _, arrays = run.gate.generate((name, v, 'broadband'), run.gate.load_config())
            old = saved['training'][str(v)]
            for key, values in old['preview'].items():
                diff = float(np.max(abs(arrays[key][::10]-values))); max_preview = max(max_preview, diff)
                np.testing.assert_allclose(arrays[key][::10], values, rtol=1e-9, atol=1e-10)
            replays.append(dict(client=name, stage='training', case=f'v{v}',
                exact_trace_match=run.gate.trace_hash(arrays) == old['trace_sha256'], max_preview_difference=diff))
            records.append(dict(speed=v, applied_input=arrays['steering'], measured_yaw=arrays['measured_yaw']))
        u, y, speeds = model.observed_records(records); sd = np.deg2rad(cfg['yaw_noise_std_deg_s'])
        for method, entry in saved['fits'].items():
            chosen = model.select_attempt(entry['attempts'])
            compare(chosen, entry['selected'])
            for index, attempt in enumerate(entry['attempts']):
                if not attempt['converged']: continue
                prediction = model.batch_predict(attempt['parameters'], u, speeds, c['geometry_m'],
                    cfg['sample_time'], cfg['integration_max_step'])[0, :, :, 1]
                mse = float(np.mean(((prediction-y)/sd)**2))
                np.testing.assert_allclose(mse, attempt['weighted_mse'], rtol=1e-9, atol=1e-10)
                fitchecks.append(dict(client=name, method=method, start=index, stage='multistart',
                    mse_absolute_difference=abs(mse-attempt['weighted_mse'])))
            par = np.asarray(chosen['parameters']); free = chosen['free_indices']
            _, jac = model.response_jacobian(np.log(par[free]), par, free, u, speeds, c['geometry_m'], run.settings(cfg))
            singular = np.linalg.svd(jac/sd, compute_uv=False)
            np.testing.assert_allclose(singular, chosen['information']['singular_values'], rtol=1e-8, atol=1e-8)
            if method == 'JointCT4':
                _, _, vh = np.linalg.svd(jac/sd, full_matrices=False)
                dynamics.append(dict(client=name, **{f'parameter{j}_error_pct': 100*(par[j]/np.r_[c['physical_ratios'],cfg['relaxation_lengths_m']][j]-1) for j in range(6)},
                    sigma_front_m=float(par[4]), sigma_rear_m=float(par[5]), mu=float(par[3]),
                    weakest_direction_mu_abs=float(abs(vh[-1, 3])),
                    relaxation_projected_smin=chosen['information']['relaxation_after_ct_projection']['smallest_singular']))
        for key, profile in saved['profiles'].items():
            a = profile['fit']
            if not a['converged']: continue
            prediction = model.batch_predict(a['parameters'], u, speeds, c['geometry_m'], cfg['sample_time'], cfg['integration_max_step'])[0, :, :, 1]
            mse = float(np.mean(((prediction-y)/sd)**2))
            np.testing.assert_allclose(mse, a['weighted_mse'], rtol=1e-9, atol=1e-10)
            delta = mse*y.size-saved['fits']['JointCT4']['selected']['weighted_sse']
            np.testing.assert_allclose(delta, profile['delta_sse'], rtol=1e-8, atol=1e-8)
            fitchecks.append(dict(client=name, method='JointCT4', start=key, stage='profile', mse_absolute_difference=abs(mse-a['weighted_mse'])))
        print(json.dumps(dict(client=name, audited='training_and_fits')), flush=True)
        for stage, operation in [('prediction', run.prediction_case), ('control', run.control_case)]:
            for key, old in saved[stage].items():
                par = run.parameters(saved, old['method'], c, cfg)
                result, arrays = operation(par, c, ci, old['speed'], old['probe'], cfg)
                for field in ['status', 'metrics', 'feasible', 'design']:
                    if field in result: compare(result[field], old[field])
                if result['metrics']:
                    max_metric = max(max_metric, max(abs(result['metrics'][k]-old['metrics'][k]) for k in result['metrics']))
                preview_difference = 0.
                if arrays is not None:
                    for field, values in old['preview'].items():
                        diff = float(np.max(abs(arrays[field][::10]-values)))
                        preview_difference = max(preview_difference, diff)
                        np.testing.assert_allclose(arrays[field][::10], values, rtol=1e-8, atol=1e-9)
                    max_preview = max(max_preview, preview_difference)
                    noise_key = (stage, old['speed'], old['probe'])
                    nhash = run.gate.trace_hash({'noise': arrays['noise']})
                    if noise_key in paired_noise: assert paired_noise[noise_key] == nhash
                    paired_noise[noise_key] = nhash
                    replays.append(dict(client=name, stage=stage, case=key,
                        exact_trace_match=run.gate.trace_hash(arrays) == old['trace_sha256'],
                        max_preview_difference=preview_difference))
                if stage == 'control' and old['probe'] == 'dynamic':
                    linear.extend([dict(client=name, speed=old['speed'], method=old['method'], **r)
                        for r in linear_diagnostic(c, old['speed'], old['design'], cfg)])
            print(json.dumps(dict(client=name, audited=stage, cases=len(saved[stage]))), flush=True)
    return dict(replays=replays, fitchecks=fitchecks, linear=linear, dynamics=dynamics,
        max_preview=max_preview, max_metric=max_metric,
        checkpoint_sha256=run.canonical_hash(run.path(name)))


def check_tables(cfg):
    rows = {'prediction': [], 'control': [], 'fits': [], 'profiles': []}
    for c in cfg['clients']:
        saved = run.read_checkpoint(run.path(c['client'])); run.validate(saved, cfg, c['client'])
        for stage in ['prediction', 'control']:
            for r in saved[stage].values():
                rows[stage].append(dict(client=c['client'], **{k: r[k] for k in ['method', 'speed', 'probe', 'status', 'observed_speed']},
                    feasible=r.get('feasible'), actual_plant_origin_rho=r.get('design', {}).get('actual_plant_origin_rho'), **(r['metrics'] or {})))
    for stage in ['prediction', 'control']:
        pd.testing.assert_frame_equal(pd.DataFrame(rows[stage]), pd.read_csv(run.OUT/f'{run.PREFIX}_{stage}.csv'), check_dtype=False, atol=1e-10, rtol=1e-10)
    tables = {name: pd.read_csv(run.OUT/f'{run.PREFIX}_{name}.csv') for name in ['prediction', 'control', 'fits', 'profiles', 'summary']}
    for _, row in tables['summary'].iterrows():
        data = tables[row.stage]; group = data[(data.method == row.method)&(data.probe == row.probe)]
        metric = 'yaw_rmse_deg_s' if row.stage == 'prediction' else 'tracking_rmse_deg_s'
        assert (group.status == 'complete').sum() == row.complete == row.expected == 15
        np.testing.assert_allclose(group[metric].mean(), row.mean_rmse_deg_s, rtol=1e-10)
        if row.stage == 'control': assert group.feasible.sum() == row.feasible
    return tables


def render(tables, results, audit):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({'svg.hashsalt': PREFIX, 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    colors = {'LocalCT2': '#d07928', 'JointCT4': '#187d8d', 'ExactCT4': '#78529a', 'PreviousCT2': '#63707b'}
    def picture(fig, name):
        fig.savefig(Path(tempfile.gettempdir())/f'{PREFIX}_{name}.png', dpi=130)
        stream = io.StringIO(); fig.savefig(stream, format='svg', metadata={'Date': None}); plt.close(fig)
        s = stream.getvalue(); return s[s.index('<svg'):]
    def table(df): return '<div class="scroll">'+df.to_html(index=False, border=0, float_format=lambda x:f'{x:.5g}')+'</div>'
    summary = tables['summary']; pred = tables['prediction']; control = tables['control']
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    for ax, stage, probes in zip(axes, ['prediction', 'control'], [['slow', 'broadband'], ['slow', 'dynamic']]):
        for j, method in enumerate(['LocalCT2', 'JointCT4', 'ExactCT4']):
            data = summary[(summary.stage == stage)&(summary.method == method)].set_index('probe')
            ax.bar(np.arange(2)+(j-1)*.25, [data.loc[p].mean_rmse_deg_s for p in probes], width=.23, color=colors[method], label=method)
        ax.set(xticks=[0,1], xticklabels=probes, ylabel='Mean yaw RMSE [deg/s]', title='Held-out prediction' if stage == 'prediction' else 'Closed-loop tracking')
        ax.grid(axis='y', alpha=.2)
    axes[0].legend(fontsize=9); comparison = picture(fig, 'comparison')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    for method in colors:
        g = control[control.method == method].groupby('speed')
        axes[0].plot(g.beta_estimation_rmse_deg.mean().index, g.beta_estimation_rmse_deg.mean(), marker='o', color=colors[method], label=method)
        axes[1].plot(g.steering_rate_peak_deg_s.mean().index, g.steering_rate_peak_deg_s.mean(), marker='o', color=colors[method], label=method)
    axes[0].set(ylabel='Mean sideslip-estimation RMSE [deg]', title='Observer accuracy')
    axes[1].set(ylabel='Mean peak steering rate [deg/s]', title='Steering activity')
    for ax in axes: ax.set_xlabel('Speed [m/s]'); ax.grid(alpha=.2)
    axes[0].legend(fontsize=9); tradeoff = picture(fig, 'tradeoff')
    linear = results['linear']; focus = linear[(linear.client == 'nominal_00')&(linear.speed == 21)&(linear.probe == 'dynamic')]
    focus = focus[['method','yaw_gain','yaw_phase_deg','stationary_yaw_noise_rms_deg_s','instantaneous_steering_per_yaw_noise']]
    joint = results['dynamics'][['client','sigma_front_m','sigma_rear_m','mu','parameter0_error_pct','parameter1_error_pct','parameter2_error_pct','weakest_direction_mu_abs']]
    aggregate = control.groupby('method').agg(tracking_rmse_deg_s=('tracking_rmse_deg_s','mean'),
        sideslip_estimation_rmse_deg=('beta_estimation_rmse_deg','mean'),steering_rms_deg=('steering_rms_deg','mean'),
        mean_peak_steering_rate_deg_s=('steering_rate_peak_deg_s','mean'),limited_runs=('limiter_fraction',lambda x:int((x>0).sum()))).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    for ax, index, label in zip(axes, [4,5], ['Front length', 'Rear length']):
        for name, g in tables['profiles'][tables['profiles'].fixed_index == index].groupby('client'):
            ax.plot(g.factor, g.delta_sse, marker='o', label=name)
        ax.set(title=label, xlabel='Fixed length / selected estimate', ylabel='Increase in weighted SSE'); ax.grid(alpha=.2)
    axes[0].legend(fontsize=9); profiles = picture(fig, 'profiles')
    text = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Tire relaxation pilot: scientific review</title>
<style>body{{margin:0;background:#edf2f5;color:#173348;font:16px/1.65 system-ui}}main{{max-width:1120px;margin:auto;padding:28px}}section{{padding:25px;margin:22px 0;background:white;border-radius:12px}}h1{{font-size:34px;line-height:1.2}}h2{{line-height:1.3}}.lead{{font-size:20px}}.decision{{border-left:6px solid #d07928}}.scroll{{overflow:auto}}table{{border-collapse:collapse;font-size:13px;white-space:nowrap}}td,th{{padding:9px;border-bottom:1px solid #ddd;text-align:right}}td:first-child,th:first-child{{text-align:left}}svg{{width:100%;height:auto}}a{{color:#187d8d}}</style><main>
<h1>Better identification and observation,<br>but worse yaw tracking with the present design</h1>
<p class="lead">The four-state model captures the missing force dynamics. A local yaw-only fit already provides accurate predictions. Its current LQI/observer implementation does not improve the control task.</p>
<section class="decision"><h2>Decision</h2><p><b>Keep the extension as a validated pilot model, but do not expand to a federated study on the strength of these results.</b> JointCT4 beats LocalCT2 in all 30 paired prediction cases and loses in all 30 paired control cases. ExactCT4 tracks almost identically to JointCT4, so a better parameter estimate is unlikely to remove the current tracking disadvantage.</p><p>The next unresolved issue is the controller/observer design, not additional plant complexity. If we continue this direction, freeze identification and first diagnose the reference response and measurement-noise response separately. This review contains analytical transfer calculations of the saved designs, not new noise ablations, tuning runs or experiments.</p></section>
<section><h2>1. All planned cases completed</h2><p>All 27 multistart fits and all 12 additional profile fits converged. All 180 prediction cases and 180 control cases completed; all control cases met the declared sideslip and lateral-acceleration limits. Maximum sideslip was {control.beta_peak_deg.max():.3f}° and maximum lateral acceleration {control.ay_peak_mps2.max():.3f} m/s². These remain gentle fixed-speed simulation tests.</p><p>Three selected vehicles are not independent population confirmation. The 17 and 25 m/s tests are interpolation between training speeds 13, 21 and 29 m/s; there is no test of incomplete fleet coverage or out-of-range extrapolation here.</p></section>
<section><h2>2. Local identification works for these trajectories; friction does not</h2>{table(joint)}<p>Generating relaxation lengths are 0.4 m front and 0.7 m rear. The joint estimates differ by at most 7.17% front and 2.76% rear. The three CT dynamic ratios differ by at most 1.53%. All three starts agree closely, and selected training weighted MSE is 0.989–1.027.</p><p>In contrast, μ is estimated as 1.50, 0.728 and 0.535, although the generating value is 0.9. The nominal estimate reaches its upper bound. The least visible normalized log-parameter direction has absolute μ component above 0.999 for every vehicle. Therefore these data support useful dynamic models, not reliable recovery of all six physical parameters. All LocalCT2 fits also hit the μ upper bound, while their other CT ratios shift substantially to compensate for missing force dynamics.</p>{profiles}<p>Fixing a relaxation length at ±25% of its selected value increases weighted SSE by 44–331 after local reoptimization of the other five parameters. No sampled profile point improves on the selected solution. These coarse, single-start profiles, some with active bounds, are evidence against a very broad local valley at this scale, not globally optimized profiles or calibrated confidence intervals.</p></section>
<section><h2>3. Prediction improves; control does not</h2>{comparison}<p>For broadband held-out steering, LocalCT2 → JointCT4 reduces mean clean-yaw prediction RMSE from 0.04720 to 0.001484 deg/s (96.9%). For slow steering, it falls from 0.01276 to 0.001171 deg/s (90.8%). Improvement occurs at both observed speeds and intermediate speeds. ExactCT4 gives zero clean prediction error by construction because it uses the generating model and integrator.</p><p>Dynamic-reference tracking instead worsens from 0.06808 to 0.09440 deg/s (+38.7%), and slow-reference tracking from 0.01633 to 0.02527 deg/s (+54.7%). The absolute increases are 0.02632 and 0.00894 deg/s. PreviousCT2 also tracks better than JointCT4 in every matched case.</p><p>ExactCT4 gives 0.09349 and 0.02502 deg/s: JointCT4 is only about 1% worse. Thus the fitted four-state model already reproduces the known-parameter four-state design's behavior closely. The exact model is a parameter reference, not a guaranteed optimum for yaw tracking under this LQI cost and observer rule.</p></section>
<section><h2>4. Better sideslip estimates come with more steering activity</h2>{table(aggregate)}{tradeoff}<p>Across both probes, JointCT4 lowers mean sideslip-estimation RMSE from 0.05765° (LocalCT2) to 0.001726°. Actual sideslip magnitudes are similar; the large difference is observer accuracy. Mean steering RMS increases from 0.4183° to 0.4703°.</p><p>Mean peak steering rate rises from 10.0 to 109.1 deg/s. Fifteen of the 30 JointCT4 runs reach a limiter, with a mean limited-sample fraction of 0.217%; no LocalCT2 run limits. All remain within the declared operating bounds. Low limiter fractions should not hide the much higher steering activity. The original 0.1 s preview traces undersample the 0.01 s control sequence and should not be used alone to assess this activity.</p></section>
<section><h2>5. What the saved linear designs suggest</h2><p>For nominal_00 at 21 m/s and the 0.7 Hz reference, the steady, unsaturated linearized closed loop gives:</p>{table(focus)}<p>JointCT4 has more phase lag (about −1.75° versus −0.86° for LocalCT2) and slightly more amplitude overshoot. Its predicted stationary yaw component due to measurement noise is also higher: 0.02010 versus 0.01198 deg/s. The instantaneous steering response to a yaw-noise increment is approximately −4.49 versus −0.65 rad/rad.</p><p>These calculations identify two plausible contributors: the reference transfer and noise amplification both change. They use the true linearized plant, saved model-based gain, steady Kalman gain, independent white yaw noise and inactive limits. Startup, nonlinear tire behavior, tapering and rate limiting are omitted. They therefore do not assign exact portions of the nonlinear tracking error or prove that changing only the observer resolves the result.</p><p>The common LQI weights do not define a minimum-yaw-RMSE controller across model orders, and the augmented observer adds an untuned force-process covariance. A more accurate model need not win this particular controller comparison automatically. No control weights or covariances were selected using these test results.</p></section>
<section><h2>6. Implications for the paper direction</h2><p>The extension addresses the earlier model–plant match objection: the refitted instantaneous model cannot explain the transient yaw records nearly as well. However, the richer model is already learned accurately by each vehicle with these three-speed records, and it brings no yaw-tracking improvement under the present design. The experiment therefore establishes neither a control advantage nor a need for fleet learning.</p><p>I would stop enlarging the plant and fleet for now. If we pursue the control route, use one bounded controller diagnosis with the saved models: separate deterministic reference tracking from measurement-noise response, then decide whether a common, practically justified controller/observer design criterion is needed. If that fails to show a useful advantage, reconsider the task rather than manufacture more model mismatch.</p></section>
<section><h2>Audit and artifacts</h2><p>Checked the source/runtime identity, checkpoint digests, all saved fit/profile objective values, selected-fit sensitivity singular values, original summary means and both 180-row case tables. Replayed all nine training records, 180 predictions and 180 closed-loop runs without refitting. Paired evaluation noise was verified across methods.</p><p>Maximum reproduced metric difference: {audit['max_metric_difference']:.3g}; maximum preview difference: {audit['max_preview_difference']:.3g}. Exact trace-hash matches: {audit['exact_trace_matches']} of {audit['replayed_traces']}. Numerical agreement is distinguished from byte-identical recovery across platforms. Original execution checkpoints, source protocol and report are unchanged.</p><p><a href="tire_relaxation_pilot.html">Original complete pilot report</a> · <a href="../docs/tire_relaxation_pilot.html">Frozen protocol</a></p></section></main></html>'''
    REPORT.write_text('\n'.join(line.rstrip() for line in text.splitlines())+'\n', encoding='utf-8', newline='\n')


def main():
    cfg = run.preflight(); tables = check_tables(cfg)
    original_paths = [run.path(c['client']) for c in cfg['clients']]+list(run.OUT.glob(f'{run.PREFIX}_*.csv'))+[run.OUT/f'{run.PREFIX}_analysis.json',run.REPORT]
    hashes = {p.relative_to(ROOT).as_posix():run.canonical_hash(p) for p in original_paths}
    parts = []
    with ProcessPoolExecutor(max_workers=3, mp_context=multiprocessing.get_context('spawn')) as pool:
        for completed in as_completed([pool.submit(worker, i) for i in range(3)]): parts.append(completed.result())
    results = {name:pd.DataFrame([row for part in parts for row in part[name]]) for name in ['replays','fitchecks','linear','dynamics']}
    for name, frame in results.items():
        frame.sort_values(list(frame.columns[:min(3,len(frame.columns))])).to_csv(run.OUT/f'{PREFIX}_{name}.csv',index=False)
    for name, sha in hashes.items(): assert run.canonical_hash(ROOT/name) == sha
    audit = dict(status='audit_complete', original_hashes=hashes,
        review_script_sha256=run.canonical_hash(__file__), replayed_traces=len(results['replays']),
        exact_trace_matches=int(results['replays'].exact_trace_match.sum()),
        checked_fit_and_profile_objectives=len(results['fitchecks']),
        max_objective_mse_difference=float(results['fitchecks'].mse_absolute_difference.max()),
        max_metric_difference=max(p['max_metric'] for p in parts),max_preview_difference=max(p['max_preview'] for p in parts),
        new_scientific_cases=0, refits=0, controller_retuning=False, original_artifacts_modified=False,
        comparison_tolerance=dict(rtol=1e-8,atol=1e-9),
        additional_analysis='Steady linearized reference/noise transfer of saved controllers; no noise-ablation experiment')
    run.atomic_json(run.OUT/f'{PREFIX}_audit.json',audit);render(tables,results,audit)
    print(json.dumps({k:v for k,v in audit.items() if k!='original_hashes'},indent=2));print(REPORT)


if __name__ == '__main__': main()
