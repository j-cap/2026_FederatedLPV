"""Audit completed coverage/control diagnostic without refitting or new rollouts.

Reproduce: python code/experiments/analyze_physical_ct_coverage_control.py
Preserves frozen inputs, checkpoint envelopes and the original analysis tables.
"""
from __future__ import annotations
import io
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_physical_ct_coverage_resume as compat
from analyze_physical_ct_control_pilot import rk4_origin_matrices
from federated_lpv.control import augmented_tracking_matrices
from federated_lpv.oe_lqi_validation import interpolate
from federated_lpv.vehicle import VehicleParameters
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

base = compat.base
PREFIX = 'paper_physical_ct_coverage_control_review'
REPORT = base.ROOT / 'report/physical_ct_coverage_control_review.html'
LABELS = dict(full='Full coverage', single='Single speed',
              adjacent_pair='Adjacent pair', multi_20s='Multi-speed, 20 s')
METRICS = ['tracking_rmse_deg_s', 'beta_peak_deg', 'steering_rms_deg']


def close(a, b):
    np.testing.assert_allclose(a, b, rtol=2e-10, atol=2e-12)


def check_csv(frame, suffix, keys):
    saved = pd.read_csv(base.OUT / f'{base.PREFIX}_{suffix}.csv')
    frame = frame[saved.columns]
    pd.testing.assert_frame_equal(frame.sort_values(keys).reset_index(drop=True),
        saved.sort_values(keys).reset_index(drop=True), check_exact=False,
        rtol=2e-10, atol=2e-12, check_dtype=False)


def audit():
    compat.install()
    cfg = base.preflight()
    control = base.old.load(base.old.storage.CONFIG)
    maneuver = base.old.load(base.ROOT / 'code/config/paper_oe_transient_benchmark.json')
    settings = base.old.load(base.old.CONFIG)['estimator']
    archived = base.old.load(base.OUT / f'{base.PREFIX}_analysis.json')
    prior = base.old.load(base.OUT / 'paper_physical_ct_coverage_review.json')
    assert archived['script_sha256'] == base.old.file_hash(base.__file__)
    assert archived['protocol_sha256'] == base.old.file_hash(base.CONFIG)
    assert archived['analysis_adapter'] == compat.adapter_identity()
    assert prior['analysis_sha256'] == base.old.file_hash(base.ROOT / 'code/experiments/analyze_physical_ct_coverage_review.py')
    archived_hashes = {k.replace('\\', '/'): v for k, v in archived['checkpoint_sha256'].items()}
    hashes = {}
    records, fits, nodes = [], [], []
    reused = 0
    for seed in cfg['seeds']:
        fit_payload = base.old.envelope(base.path(seed))
        cp = base.old.envelope(base.path(seed, True))
        for payload, is_control in [(fit_payload, False), (cp, True)]:
            compat.validate(payload, seed, cfg, is_control)
            assert payload['status'] == 'execution_complete'
            path = base.path(seed, is_control)
            key = path.relative_to(base.ROOT).as_posix()
            hashes[key] = base.old.file_hash(path)
            assert hashes[key] == archived_hashes[key]
            if not is_control:
                # The previous audit already checked training hashes and the
                # complete sampled single-speed equivalence families.
                assert hashes[key] == prior['source_sha256'][key]
        source = base.old.sources(seed)[1]
        original_control = base.old.envelope(base.old.result_path(seed, True))
        grid = np.array(source['grid'])
        for fitted, record in zip(fit_payload['records'], cp['records'], strict=True):
            s = fitted['spec']
            assert s == record['spec'] and fitted['client'] == record['client']
            i = s['index']; client = source['clients'][i]
            assert fitted['fit']['converged']
            target = base.models(fitted['fit']['parameters'], grid, settings)
            close(target, fitted['grid_targets'])
            errors = base.old.dynamic_metrics(target, np.array(client['reference']), {})
            for key in errors:
                close(errors[key], fitted['errors'][key])
            common = ~np.isin(grid, client['speeds'])
            fit_row = dict(seed=seed, client=record['client'], id=s['id'], kind=s['kind'],
                original_speeds=str(client['speeds']), selected_speeds=str([w['speed'] for w in s['windows']]),
                seconds=.01*fitted['transitions'], mu=fitted['fit']['parameters'][3],
                common_unseen_matrix_pct=100*np.array(errors['matrix'])[common].mean(),
                common_unseen_increment_pct=100*np.array(errors['increment'])[common].mean(),
                all_grid_matrix_pct=100*np.array(errors['matrix']).mean())
            fits.append(fit_row)
            schedule = record['schedule']
            assert schedule['status'] == 'designed'
            close([n['speed'] for n in schedule['nodes']], grid)
            close([n['matrix'] for n in schedule['nodes']], target)
            if s['reuse'] == 'full':
                original = original_control['clients'][i]
                assert schedule == original['schedules'][base.old.METHOD]
                expected = [dict(speed=r['speed'], outcome=r['outcome'])
                            for r in original['rows'] if r['method'] == base.old.METHOD]
                assert record['rows'] == expected
                reused += 1
            for n in schedule['nodes']:
                a = np.array(n['matrix'][:4]).reshape(2, 2)
                b = np.array(n['matrix'][4:]).reshape(2, 1)
                aa, bb = augmented_tracking_matrices(a, b, control['sample_time'])
                gain = np.array(n['gain'])
                rho = float(max(abs(np.linalg.eigvals(aa-bb@gain[None, :]))))
                close(rho, n['design_rho'])
                assert rho < 1 and n['status'] == 'designed'
                eq = np.linalg.solve(np.block([[np.eye(2)-a, -b],
                    [np.array([[0., 1.]]), np.zeros((1, 1))]]), [0, 0, 1])
                close(n['prefilter'], eq[2]+gain[:2]@eq[:2])
                nodes.append(dict(seed=seed, id=s['id'], speed=n['speed'],
                    design_rho=rho, K_beta=gain[0], K_r=gain[1], K_I=gain[2], F=n['prefilter']))
            speeds = [row['speed'] for row in record['rows']]
            assert speeds == maneuver['speeds_mps']
            assert not set(speeds).intersection(client['speeds'])
            gains, prefilters = interpolate(schedule, control['speed_grid'], speeds)
            for row, gain, ff in zip(record['rows'], gains, prefilters, strict=True):
                o = row['outcome']; m = o['metrics']; v = row['speed']
                assert o['status'] == 'complete'
                assert o['completed_steps'] == round(maneuver['duration_s']/control['sample_time'])
                assert m and np.isfinite(list(m.values())).all() and min(m.values()) >= 0
                bounded = (m['steering_peak_deg'] <= control['steering_limit_deg']+1e-8
                    and m['steering_rate_peak_deg_s'] <= control['steering_rate_limit_deg_s']+1e-8
                    and m['beta_violation_samples'] == 0)
                assert o['feasible'] == bounded
                a, b = rk4_origin_matrices(VehicleParameters(**client['parameters']), v, control['sample_time'])
                aa, bb = augmented_tracking_matrices(a, b, control['sample_time'])
                eig = np.linalg.eigvals(aa-bb@gain[None, :])
                dominant = eig[np.argmax(abs(eig))]
                rho = float(abs(dominant))
                reference = base.profile(maneuver, v)['reference']
                peak_reference = float(np.rad2deg(np.max(abs(reference))))
                records.append(dict(**fit_row, speed=v, status=o['status'], feasible=o['feasible'], **m,
                    origin_rho=rho, origin_unstable=rho > 1+1e-9,
                    dominant_real=float(dominant.real), dominant_imag=float(dominant.imag),
                    K_beta=gain[0], K_r=gain[1], K_I=gain[2], F=ff,
                    reference_peak_deg_s=peak_reference,
                    rmse_percent_of_peak_reference=100*m['tracking_rmse_deg_s']/peak_reference))
    frame = pd.DataFrame(records); fit_frame = pd.DataFrame(fits)
    assert len(frame) == 1764 and len(fit_frame) == 588 and reused == 111
    check_csv(frame, 'control_cases', ['seed', 'id', 'speed'])
    # Match archived fit metrics without replacing any original tables.
    saved_fit = pd.read_csv(base.OUT / f'{base.PREFIX}_fits.csv')
    selected = ['seed', 'id', 'kind', 'mu', 'seconds', 'common_unseen_matrix_pct',
                'common_unseen_increment_pct', 'all_grid_matrix_pct']
    pd.testing.assert_frame_equal(fit_frame[selected].sort_values(['seed', 'id']).reset_index(drop=True),
        saved_fit[selected].sort_values(['seed', 'id']).reset_index(drop=True),
        check_exact=False, rtol=2e-10, atol=2e-12, check_dtype=False)
    clients, fleets, summary = base.aggregate(frame, METRICS)
    for suffix, data, keys in [('control_clients', clients, ['seed','client','kind']),
                              ('control_fleets', fleets, ['seed','kind'])]:
        check_csv(data, suffix, keys)
    counts = frame.groupby('kind').agg(cases=('status', 'size'),
        complete=('status', lambda x: int((x == 'complete').sum())), feasible=('feasible', 'sum'),
        limited_cases=('limiter_fraction', lambda x: int((x > 0).sum())),
        origin_unstable=('origin_unstable', 'sum')).reset_index()
    summary = summary.merge(counts, on='kind')
    check_csv(summary, 'control_summary', ['kind'])
    _, _, model_summary = base.aggregate(fit_frame,
        ['common_unseen_matrix_pct', 'common_unseen_increment_pct'])
    summary = summary.merge(model_summary, on='kind')
    wide = clients.pivot(index=['seed','client'], columns='kind', values='tracking_rmse_deg_s')
    paired = []
    for k in base.KINDS[1:]:
        difference = wide[k]-wide.full
        paired.append(dict(kind=k, vehicles=len(difference),
            mean_difference_deg_s=difference.mean(), median_difference_deg_s=difference.median(),
            maximum_increase_deg_s=difference.max(), maximum_decrease_deg_s=difference.min(),
            lower_rmse_than_full=int((difference < 0).sum()),
            relative_mean_change_pct=100*difference.mean()/wide.full.mean()))
    paired = pd.DataFrame(paired)
    # Marginal case quantiles are descriptive and do not use the equal-vehicle
    # weighting of the primary means. Also report vehicle worst-variant means.
    dist = frame.groupby('kind').agg(median_rmse=('tracking_rmse_deg_s','median'),
        p95_rmse=('tracking_rmse_deg_s', lambda x: x.quantile(.95)),
        max_rmse=('tracking_rmse_deg_s','max'), max_origin_rho=('origin_rho','max'),
        max_steering_rate_rms=('steering_rate_rms_deg_s','max'),
        max_rate_limiter_fraction=('rate_limiter_fraction','max'),
        max_steering_total_variation=('steering_total_variation_deg','max'),
        max_rmse_pct_peak_reference=('rmse_percent_of_peak_reference','max')).reset_index()
    worst = frame.groupby(['seed','client','kind','id']).tracking_rmse_deg_s.mean().reset_index()
    worst = worst.groupby(['seed','client','kind']).tracking_rmse_deg_s.max().reset_index()
    _, _, worst_summary = base.aggregate(worst, ['tracking_rmse_deg_s'])
    dist = dist.merge(worst_summary.rename(columns={'tracking_rmse_deg_s':'mean_vehicle_worst_variant_rmse'}), on='kind')
    speed_clients = frame.groupby(['seed','client','kind','speed'])[METRICS].mean().reset_index()
    speeds = speed_clients.groupby(['seed','kind','speed'])[METRICS].mean().groupby(['kind','speed']).mean().reset_index()
    block_clients = frame.groupby(['seed','client','kind','original_speeds'])[METRICS].mean().reset_index()
    blocks = block_clients.groupby(['seed','kind','original_speeds'])[METRICS].mean().groupby(['kind','original_speeds']).mean().reset_index()
    full = frame[frame.kind == 'full'][['seed','client','speed',*METRICS,'steering_rate_rms_deg_s', 'steering_total_variation_deg']]
    frame = frame.merge(full, on=['seed','client','speed'], suffixes=('', '_full'), validate='many_to_one')
    frame['delta_rmse_deg_s'] = frame.tracking_rmse_deg_s-frame.tracking_rmse_deg_s_full
    variant_count = frame.groupby(['seed','client','kind','speed']).id.transform('size')
    frame['primary_mean_weight'] = 1/(3*30*3*variant_count)
    close(frame.groupby('kind').primary_mean_weight.sum(), np.ones(4))
    tail = frame[frame.origin_unstable | (frame.limiter_fraction > 0)].copy()
    pair = frame[frame.kind == 'adjacent_pair']
    tail_contribution = float((pair.loc[pair.origin_unstable, 'delta_rmse_deg_s']
        *pair.loc[pair.origin_unstable, 'primary_mean_weight']).sum())
    total_gap = float((pair.delta_rmse_deg_s*pair.primary_mean_weight).sum())
    audit_info = dict(status='completed_coverage_control_audit',fit_conditions=len(fit_frame),
        control_cases=len(frame),controller_nodes_checked=len(nodes),
        new_rollouts_in_checkpoints=3*(len(fit_frame)-reused),reused_case_records=3*reused,
        limited_cases=int((frame.limiter_fraction>0).sum()),origin_unstable_cases=int(frame.origin_unstable.sum()),
        adjacent_pair_mean_gap_deg_s=total_gap,adjacent_pair_unstable_case_contribution_deg_s=tail_contribution,
        adjacent_pair_gap_fraction_from_unstable_cases=tail_contribution/total_gap,
        checkpoint_sha256=hashes,script_sha256=base.old.file_hash(__file__),
        prior_identification_audit_sha256=base.old.file_hash(base.OUT/'paper_physical_ct_coverage_review.json'),
        origin_jacobian_helper_sha256=base.old.file_hash(base.ROOT/'code/experiments/analyze_physical_ct_control_pilot.py'),
        scope='Three reused pilot fleets; no new fitting, tuning, acquisition or nonlinear rollouts. '
              'Origin Jacobian uses the simulator RK4 derivative with inactive limits; no nonlinear stability guarantee. '
              'Stored rollout metrics audited but not recomputed from unavailable raw trajectories. '
              'Original identification checkpoints match the earlier grid/equivalence audit; cropped raw-data hashes remain unrecomputed.')
    for suffix, data in [('cases',frame),('summary',summary),('paired',paired),('distribution',dist),
                         ('fleets',fleets),('speeds',speeds),('blocks',blocks),('tail',tail)]:
        data.to_csv(base.OUT/f'{PREFIX}_{suffix}.csv',index=False)
    base.old.acquisition.parent.save_json(base.OUT/f'{PREFIX}.json',audit_info)
    render(frame,fit_frame,clients,fleets,summary,paired,dist,speeds,blocks,tail,audit_info)
    print(summary.to_string(index=False));print(paired.to_string(index=False))
    print(dist.to_string(index=False))
    print(tail[['seed','client','selected_speeds','speed','mu','tracking_rmse_deg_s',
        'origin_rho','dominant_real','rate_limiter_fraction','steering_rate_rms_deg_s']].to_string(index=False))
    print('Adjacent-pair gap fraction from origin-unstable cases:',tail_contribution/total_gap)
    print(REPORT)


def render(frame,fits,clients,fleets,summary,paired,dist,speeds,blocks,tail,audit_info):
    import matplotlib
    matplotlib.use('Agg');matplotlib.rcParams['svg.hashsalt']=PREFIX
    import matplotlib.pyplot as plt
    colors = dict(zip(base.KINDS,['#177c75','#467eae','#b65a31','#7958a0']))
    fig, axes = plt.subplots(2,2,figsize=(13,8.5),layout='constrained')
    for kind in base.KINDS:
        f = fleets[fleets.kind==kind]
        axes[0,0].plot(f.seed, f.tracking_rmse_deg_s,marker='o',label=LABELS[kind],color=colors[kind])
        s = speeds[speeds.kind==kind]
        axes[0,1].plot(s.speed,s.tracking_rmse_deg_s,marker='o',label=LABELS[kind],color=colors[kind])
        c = frame[frame.kind==kind]
        axes[1,0].scatter(c.common_unseen_increment_pct,c.tracking_rmse_deg_s,s=11,alpha=.35,color=colors[kind])
        axes[1,1].scatter(c.origin_rho,c.steering_rate_rms_deg_s,s=11,alpha=.35,color=colors[kind])
    axes[0,0].set(title='Fleet means remain close',xlabel='Pilot fleet',ylabel='Tracking RMSE [deg/s]',xticks=[1301,1302,1303])
    axes[0,0].ticklabel_format(useOffset=False,axis='x');axes[0,0].legend(fontsize=9)
    axes[0,1].set(title='All three test speeds; all variants included',xlabel='Speed [m/s]',ylabel='Mean tracking RMSE [deg/s]',xticks=[13,21,29])
    axes[1,0].set(title='Model error is not a tracking-quality certificate',xlabel='Common unseen increment error [%]',ylabel='Case tracking RMSE [deg/s]')
    axes[1,1].axvline(1,color='gray',linestyle='--',linewidth=1)
    axes[1,1].set(title='Two locally unstable cases have high steering rates',xlabel='True-plant origin spectral radius',ylabel='Steering-rate RMS [deg/s]',yscale='log')
    for _,r in tail.iterrows():
        axes[1,0].annotate(f"{r['seed']} / {r['client']}",(r.common_unseen_increment_pct,r.tracking_rmse_deg_s),xytext=(-8,7),textcoords='offset points',ha='right',fontsize=8)
    axes[1,0].margins(y=.12)
    for ax in axes.flat:ax.grid(alpha=.2)
    buf=io.StringIO();fig.savefig(buf,format='svg',metadata={'Date':None});plt.close(fig)
    svg=buf.getvalue();svg=svg[svg.index('<svg'):]
    def table(df):
        return '<div class="scroll">'+df.to_html(index=False,border=0,float_format=lambda x:f'{x:.6g}')+'</div>'
    summary_display=summary[['kind','common_unseen_matrix_pct','common_unseen_increment_pct',
        'tracking_rmse_deg_s','cases','feasible','limited_cases','origin_unstable']].replace({'kind':LABELS})
    tail_cols=['seed','client','selected_speeds','speed','mu','common_unseen_increment_pct',
        'tracking_rmse_deg_s','tracking_rmse_deg_s_full','origin_rho','dominant_real',
        'rate_limiter_fraction','steering_rate_rms_deg_s','steering_rate_rms_deg_s_full',
        'steering_total_variation_deg','steering_total_variation_deg_full']
    text=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Completed coverage diagnostic: model accuracy and control</title>
<style>body{{font:16px/1.6 system-ui;background:#eef3f6;color:#193142;margin:0}}main{{max-width:1350px;margin:auto;padding:30px}}section{{background:white;padding:24px;margin:20px 0;border-radius:10px}}h1,h2{{line-height:1.25}}.lead{{font-size:20px}}.scroll{{overflow:auto}}table{{border-collapse:collapse;font-size:13px;white-space:nowrap}}td,th{{padding:8px;border-bottom:1px solid #ddd;text-align:right}}svg{{width:100%;height:auto}}.note{{border-left:5px solid #b65a31}}code{{font-size:14px}}</style><main>
<h1>Restricted coverage hurts identification, but does not create a broad tracking deficit</h1>
<p class="lead">Single-speed local CT models still track almost as well as full-coverage models. Two low-speed adjacent-pair fits expose a genuine controller robustness problem that the original feasibility flag does not detect.</p>
<section><h2>1. Complete, matched results</h2><p>All 588 fitted conditions and 1,764 control case records are complete. There are 477 new fits and 1,431 new nonlinear rollouts; 111 fitted conditions and 333 control case records are reused. All 588 schedules were synthesized. The three fleets contain 90 vehicles. Every condition is tested at the same previously unobserved speeds, 13, 21 and 29 m/s, with the same 12-s left-right maneuver, LQI weights, gain interpolation, steering limits and noise-free full-state feedback.</p>
<p>Full coverage uses the original 2–3 speeds and 40–60 s of identification data. Single speed uses each available speed separately for 20 s. Adjacent pair uses each neighboring pair for 40 s. Multi-speed 20 s divides 20 s total across the original speeds using central contiguous windows. All available variants are included; none is chosen by test performance. Means average variants within vehicles, then vehicles within fleets, then fleets equally.</p>{table(summary_display)}
<p>Matrix error uses [A,B]; increment error removes the identity contribution through the denominator [A−I,B]. Both use the same original-unobserved grid speeds for every condition of a vehicle. The 20-s comparison controls duration, but cropping changes waveform portions and the number of nuisance states; it is not an equal-information comparison.</p></section>
<section><h2>2. The main comparison</h2>{svg}{table(paired)}
<p>Single-speed fitting increases mean matrix error from 0.386% to 0.964%, and increment error from 1.555% to 3.729%. Yet mean tracking RMSE changes from 0.012832 to 0.012842°/s: only 0.0000097°/s, or about 0.076%. Multi-speed 20 s reduces the single-speed matrix-error mean by 37.6%, but its tracking RMSE is similarly close at 0.012812°/s. These are descriptive differences, not a statistical equivalence or noninferiority result.</p>
<p>The selected single-speed solutions therefore work well for this maneuver despite the exact stiffness/friction ambiguity already established by the identification audit. They do not establish that every equally fitting single-speed solution would work well. Parameter identifiability, model extrapolation accuracy and performance of a particular feedback controller are distinct questions.</p>{table(fleets)}{table(speeds)}</section>
<section class="note"><h2>3. Two adjacent-pair cases are control-relevant</h2>
<p>Both occur when fitting only low-speed data and controlling at 29 m/s: fleet 1302, heavy_08, trained at 12/14 m/s; and fleet 1303, handling_00, trained at 10/12 m/s. Their fitted friction values are about 0.563 and 0.529, versus the simulated value 0.9. Their common-unseen increment errors are 19.49% and 24.71%.</p>{table(tail[tail_cols])}
<p>The true simulated plant's augmented closed-loop Jacobian at straight running has dominant real poles about −1.180 and −1.725. Thus the unsaturated origin equilibrium is locally unstable for these interpolated controllers, even though synthesis stabilizes each learned model at its design nodes. These diagnostics use the RK4 step's exact derivative at zero, matching the simulation. They are local diagnostics, not global nonlinear stability guarantees.</p>
<p>The corresponding saved rollouts activate the steering-rate limiter for 56.5% and 58.75% of their samples. Steering-rate RMS rises above 100°/s, with total steering variation above 1,075° over 12 s. Negative unstable poles and these metrics are consistent with rapid alternating steering; the raw time series were not saved here, so the waveform has not been independently inspected. This behavior is undesirable even though yaw-rate RMSE remains below 0.101°/s.</p>
<p>All 1,764 cases are marked feasible because that flag checks applied steering angle/rate and sideslip bounds; the limiter enforces the input bounds. It does not test tracking quality, absence of limiter activation or local stability. Consequently, “all feasible” must not be reported as “all controllers are satisfactory.” Only these two cases activate a limiter or have origin spectral radius above one.</p>
<p>These two cases account for {100*audit_info['adjacent_pair_gap_fraction_from_unstable_cases']:.2f}% of the weighted adjacent-pair-minus-full mean tracking gap. The percentage slightly exceeds 100% because the other cases make a small net contribution in the opposite direction. The official mean includes both. This is a decomposition using the independently computed local stability diagnostic, not removal of poor results.</p></section>
<section><h2>4. Tails, blocks and aggregation</h2>{table(dist)}
<p>Case medians, quantiles and maxima above describe all recorded variants without equal-vehicle weighting; the primary means use equal vehicle and fleet weights. The final column averages each vehicle's worst variant, after averaging that variant over the three speeds. Reused conditions and multiple variants of one vehicle are not independent repetitions.</p>{table(blocks)}
<p>The low-speed block has the largest adjacent-pair degradation. Full-rank sensitivity at two speeds is not enough to ensure reliable friction extrapolation under noisy finite data. Conversely, single-speed ambiguity can be masked by a favorable selected solution. The results do not justify a monotone claim that fewer speeds always produce worse control.</p></section>
<section><h2>5. What this means for the paper</h2>
<p><strong>This diagnostic does not establish a general need for federation under tighter speed coverage.</strong> The physics-based local model remains a strong baseline, including with one observed speed. The two pair failures establish a narrower, meaningful issue: uncertain low-speed extrapolation can produce controllers with undesirable high-speed behavior despite successful estimation and synthesis.</p>
<p>A sharing method would need a matched comparison using the same restricted client data and the same physical model class, together with a credible explanation of which quantities can be shared between vehicles. This experiment contains no such comparison. Existing unrestricted Federated results cannot demonstrate a sharing benefit under the new restrictions.</p>
<p>Before adding another plant or tightening coverage further, decide whether the intended contribution is reliable extrapolation with uncertainty, or identification under a realistic mismatch between the available vehicle model and the plant. If the aim is vehicle-model learning for control, the strong local CT baseline must remain in the evaluation. A more complicated tire model would need a physical motivation and a suitably strong local comparator; these results alone do not select that next experiment.</p>
<p>Scope remains three reused pilot fleets, a correctly specified analytical local model family, known applied inputs, and full-state measurements. Control uses noise-free state feedback and one prescribed maneuver at three fixed speeds. Model error is evaluated around the acquisition curvature; the control maneuver varies yaw rate around straight running. Good tracking here is not proof of accurate tire/friction recovery or robustness across other maneuvers.</p></section>
<section><h2>6. Reproducibility and audit limits</h2>
<p>Verified envelopes, source provenance, completeness, reused controls, all 588 model grids and their saved errors, all {audit_info['controller_nodes_checked']:,} synthesis matrices, learned-model closed-loop poles and prefilters, and the original fit/control summary CSV values. Computed true-plant origin Jacobians for all 1,764 cases. The original identification checkpoints match the previous audit, which also checked the sampled exact single-speed equivalence families.</p>
<p>No refits, controller retuning or nonlinear rollouts were performed. Saved trajectory metrics were checked for consistency and finiteness, but cannot be independently recomputed without the original trajectory arrays. The earlier cropped identification arrays also remain unavailable. Frozen experiment code, checkpoints and original analysis outputs are unchanged.</p>
<p>Reproduce: <code>python code/experiments/analyze_physical_ct_coverage_control.py</code>. Audit hashes and detailed CSV tables use the prefix <code>{PREFIX}</code>.</p></section></main></html>'''
    REPORT.write_text('\n'.join(line.rstrip() for line in text.splitlines())+'\n',encoding='utf-8',newline='\n')


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        audit()
