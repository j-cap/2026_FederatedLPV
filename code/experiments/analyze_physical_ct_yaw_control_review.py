"""Audit matched yaw-only/full-state CT controls; no new nonlinear rollouts."""
from __future__ import annotations
import io
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import paper_physical_ct_yaw_only as run
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
PREFIX='paper_physical_ct_yaw_control_review'
REPORT=run.ROOT/'report/physical_ct_yaw_control_review.html'


def close(a,b):np.testing.assert_allclose(a,b,rtol=2e-10,atol=2e-12)


def check_table(data,suffix,keys):
    saved=pd.read_csv(run.OUT/f'{run.PREFIX}_{suffix}.csv')
    pd.testing.assert_frame_equal(data[saved.columns].sort_values(keys).reset_index(drop=True),
        saved.sort_values(keys).reset_index(drop=True),check_exact=False,check_dtype=False,rtol=2e-10,atol=2e-12)


def audit():
    cfg=run.preflight();cc=run.old.load(run.old.storage.CONFIG)
    maneuver=run.old.load(run.ROOT/'code/config/paper_oe_transient_benchmark.json')
    archived=run.old.load(run.OUT/f'{run.PREFIX}_analysis.json')
    assert archived['analysis_sha256']==run.old.file_hash(run.__file__)
    assert archived['protocol_sha256']==run.old.file_hash(run.CONFIG)
    assert archived['control_cases']==540
    previous=run.old.load(run.OUT/'paper_physical_ct_yaw_only_review.json')
    records=[];nodes=[];hashes={}
    for seed in cfg['seeds']:
        cp=run.old.envelope(run.path(seed,True));run.validate(cp,seed,cfg,True)
        fitted=run.old.envelope(run.path(seed));run.validate(fitted,seed,cfg)
        assert cp['status']==fitted['status']=='execution_complete'
        for control in [False,True]:
            p=run.path(seed,control);key=p.relative_to(run.ROOT).as_posix();hashes[key]=run.old.file_hash(p)
            assert hashes[key]==archived['checkpoint_sha256'][key]
            if not control:assert hashes[key]==previous['checkpoint_sha256'][key]
        source=run.old.sources(seed)[1]
        full=run.old.envelope(run.old.result_path(seed))
        original=run.old.envelope(run.old.result_path(seed,True))
        for client in cp['clients']:
            i=client['index'];truth=run.VehicleParameters(**source['clients'][i]['parameters'])
            prior=original['clients'][i]
            assert client['schedules']['FullStateCT']==prior['schedules'][run.old.METHOD]
            oldrows=[dict(method='FullStateCT',speed=r['speed'],outcome=r['outcome']) for r in prior['rows'] if r['method']==run.old.METHOD]
            assert [r for r in client['rows'] if r['method']=='FullStateCT']==oldrows
            diagnostics={}
            for method,schedule in client['schedules'].items():
                assert schedule['status']=='designed' and len(schedule['nodes'])==11
                expected=np.array(full['clients'][i]['grid_targets'] if method=='FullStateCT' else fitted['clients'][i]['grid_targets'])
                close([n['matrix'] for n in schedule['nodes']],expected)
                close([n['speed'] for n in schedule['nodes']],cc['speed_grid'])
                for n in schedule['nodes']:
                    a=np.array(n['matrix'][:4]).reshape(2,2);b=np.array(n['matrix'][4:]).reshape(2,1)
                    aa,bb=run.augmented_tracking_matrices(a,b,cc['sample_time']);gain=np.array(n['gain'])
                    rho=float(max(abs(np.linalg.eigvals(aa-bb@gain[None,:]))))
                    close(rho,n['design_rho']);assert rho<1 and n['status']=='designed'
                    eq=np.linalg.solve(np.block([[np.eye(2)-a,-b],[np.array([[0.,1.]]),np.zeros((1,1))]]),[0,0,1])
                    close(n['prefilter'],eq[2]+gain[:2]@eq[:2])
                    nodes.append(dict(seed=seed,client=client['client'],method=method,speed=n['speed'],design_rho=rho))
                gains,ffs=run.interpolate(schedule,cc['speed_grid'],maneuver['speeds_mps'])
                for v,gain,ff in zip(maneuver['speeds_mps'],gains,ffs,strict=True):
                    a,b=run.rk4_origin_matrices(truth,v,cc['sample_time'])
                    aa,bb=run.augmented_tracking_matrices(a,b,cc['sample_time'])
                    rho=float(max(abs(np.linalg.eigvals(aa-bb@gain[None,:]))))
                    diagnostics[method,v]=dict(origin_rho=rho,K_beta=gain[0],K_r=gain[1],K_I=gain[2],F=ff)
            for row in client['rows']:
                o=row['outcome'];m=o['metrics']
                assert o['status']=='complete' and o['completed_steps']==round(maneuver['duration_s']/cc['sample_time'])
                assert np.isfinite(list(m.values())).all() and min(m.values())>=0
                bounded=(m['steering_peak_deg']<=cc['steering_limit_deg']+1e-8
                    and m['steering_rate_peak_deg_s']<=cc['steering_rate_limit_deg_s']+1e-8 and m['beta_violation_samples']==0)
                assert bounded==o['feasible']
                records.append(dict(seed=seed,client=client['client'],method=row['method'],speed=row['speed'],
                    observed_speeds=str(client['observed_speeds']),status=o['status'],feasible=o['feasible'],
                    **m,**diagnostics[row['method'],row['speed']]))
    frame=pd.DataFrame(records);assert len(frame)==540 and len(nodes)==1980
    check_table(frame,'control_cases',['seed','client','method','speed'])
    clients,fleets,summary=run.aggregate(frame,['tracking_rmse_deg_s','limiter_fraction','steering_rate_rms_deg_s'])
    counts=frame.groupby('method').agg(cases=('status','size'),complete=('status',lambda x:int((x=='complete').sum())),
        feasible=('feasible','sum'),limited=('limiter_fraction',lambda x:int((x>0).sum())),
        origin_unstable=('origin_rho',lambda x:int((x>1+1e-9).sum()))).reset_index()
    summary=summary.merge(counts,on='method')
    for suffix,data,keys in [('control_clients',clients,['seed','client','method']),('control_fleets',fleets,['seed','method']),
                             ('control_summary',summary,['method'])]:check_table(data,suffix,keys)
    w=frame.pivot(index=['seed','client','speed'],columns='method',values='tracking_rmse_deg_s')
    vehicle=w.groupby(['seed','client']).mean();d=w.YawOnlyCT-w.FullStateCT;dv=vehicle.YawOnlyCT-vehicle.FullStateCT
    paired=pd.DataFrame([dict(unit=unit,n=len(gap),yaw_wins=int((gap<0).sum()),full_wins=int((gap>0).sum()),
        mean_change_deg_s=gap.mean(),median_change_deg_s=gap.median(),minimum_change_deg_s=gap.min(),maximum_change_deg_s=gap.max())
        for unit,gap in [('vehicle_speed',d),('vehicle',dv)]])
    dist=frame.groupby('method').agg(median_rmse=('tracking_rmse_deg_s','median'),
        p95_rmse=('tracking_rmse_deg_s',lambda x:x.quantile(.95)),max_rmse=('tracking_rmse_deg_s','max'),
        max_origin_rho=('origin_rho','max'),max_steering_rate_rms=('steering_rate_rms_deg_s','max'),
        max_beta_deg=('beta_peak_deg','max'),max_steering_deg=('steering_peak_deg','max')).reset_index()
    speeds=frame.groupby(['seed','method','speed']).tracking_rmse_deg_s.mean().groupby(['method','speed']).mean().reset_index()
    blocks=frame.groupby(['seed','client','method','observed_speeds']).tracking_rmse_deg_s.mean().groupby(
        ['seed','method','observed_speeds']).mean().groupby(['method','observed_speeds']).mean().reset_index()
    casepairs=w.reset_index();casepairs['change_deg_s']=d.to_numpy()
    model=pd.read_csv(run.OUT/f'{run.PREFIX}_clients.csv').pivot(index=['seed','client'],columns='method',values='increment')
    gap=(100*(model.YawOnlyCT-model.FullStateCT)).rename('increment_error_change_pp').reset_index()
    vehiclepairs=vehicle.reset_index();vehiclepairs['change_deg_s']=dv.to_numpy();vehiclepairs=vehiclepairs.merge(gap,on=['seed','client'],validate='one_to_one')
    audit_info=dict(status='matched_yaw_control_audit_complete',new_rollouts=270,reused_rollouts=270,
        schedules=180,node_matrices_checked=len(nodes),complete_cases=int((frame.status=='complete').sum()),
        feasible_cases=int(frame.feasible.sum()),limited_cases=int((frame.limiter_fraction>0).sum()),
        origin_unstable_cases=int((frame.origin_rho>1+1e-9).sum()),
        relative_mean_rmse_change_pct=100*d.mean()/w.FullStateCT.mean(),
        checkpoint_sha256=hashes,analysis_sha256=run.old.file_hash(__file__),
        scope='Three reused pilot fleets; true noise-free state feedback for both methods. No observer, retuning or new identification. Full-state results reused exactly. Jacobians concern straight-running equilibria and inactive limits, not nonlinear stability guarantees.')
    for suffix,data in [('cases',frame),('summary',summary),('fleets',fleets),('speeds',speeds),('blocks',blocks),
                       ('distribution',dist),('paired',paired),('case_pairs',casepairs),('vehicle_pairs',vehiclepairs)]:
        data.to_csv(run.OUT/f'{PREFIX}_{suffix}.csv',index=False)
    run.old.acquisition.parent.save_json(run.OUT/f'{PREFIX}.json',audit_info)
    report(frame,summary,fleets,speeds,blocks,dist,paired,casepairs,vehiclepairs,audit_info)
    print(summary.to_string(index=False));print(paired.to_string(index=False));print(dist.to_string(index=False));print(REPORT)


def report(frame,summary,fleets,speeds,blocks,dist,paired,casepairs,vehiclepairs,audit):
    import matplotlib
    matplotlib.use('Agg');matplotlib.rcParams['svg.hashsalt']=PREFIX
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for speed,color in [(13,'#467eae'),(21,'#b47c17'),(29,'#177c75')]:
        z=casepairs[casepairs.speed==speed]
        axes[0,0].scatter(z.FullStateCT,z.YawOnlyCT,s=17,alpha=.65,color=color,label=f'{speed} m/s')
        v=np.sort(z.change_deg_s.to_numpy())
        axes[1,1].plot(v, np.arange(1,len(v)+1)/len(v),label=f'{speed} m/s',color=color)
    maximum=max(casepairs.FullStateCT.max(),casepairs.YawOnlyCT.max())*1.06
    axes[0,0].plot([0,maximum],[0,maximum],'--',color='gray')
    axes[0,0].set(xlabel='Full-state CT RMSE [deg/s]',ylabel='Yaw-only CT RMSE [deg/s]',title='All 270 paired vehicle/speed cases');axes[0,0].legend()
    for method,color in [('FullStateCT','#177c75'),('YawOnlyCT','#b65a31')]:
        z=speeds[speeds.method==method]
        axes[0,1].plot(z.speed,z.tracking_rmse_deg_s,marker='o',color=color,label=method)
    axes[0,1].set(xticks=[13,21,29],xlabel='Speed [m/s]',ylabel='Mean tracking RMSE [deg/s]',title='Mean tracking overlaps at each speed');axes[0,1].legend()
    axes[1,0].scatter(vehiclepairs.increment_error_change_pp,vehiclepairs.change_deg_s,s=20,alpha=.7)
    axes[1,0].axhline(0,color='gray',linestyle='--');axes[1,0].axvline(0,color='gray',linestyle='--')
    axes[1,0].set(xlabel='Yaw-only − full-state increment error [pp]',ylabel='Yaw-only − full-state RMSE [deg/s]',title='Higher model error need not worsen tracking')
    axes[1,1].axvline(0,color='gray',linestyle='--')
    axes[1,1].set(xlabel='Yaw-only − full-state RMSE [deg/s]',ylabel='Fraction of paired cases',title='Small changes occur in both directions');axes[1,1].legend()
    from matplotlib.ticker import MaxNLocator
    axes[1,1].xaxis.set_major_locator(MaxNLocator(5))
    for ax in axes.flat:ax.grid(alpha=.2)
    buf=io.StringIO();fig.savefig(buf,format='svg',metadata={'Date':None});plt.close(fig);svg=buf.getvalue();svg=svg[svg.index('<svg'):]
    def table(df):return '<div class="scroll">'+df.to_html(index=False,border=0,float_format=lambda x:f'{x:.7g}')+'</div>'
    text=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Matched yaw-only CT control review</title>
<style>body{{font:16px/1.6 system-ui;background:#edf2f5;color:#193142;margin:0}}main{{max-width:1280px;margin:auto;padding:30px}}section{{background:white;padding:24px;margin:20px 0;border-radius:10px}}h1,h2{{line-height:1.25}}.lead{{font-size:20px}}.scroll{{overflow:auto}}table{{border-collapse:collapse;font-size:13px;white-space:nowrap}}td,th{{padding:8px;border-bottom:1px solid #ddd}}svg{{width:100%;height:auto}}</style><main>
<h1>Removing sideslip from fitting leaves matched tracking almost unchanged</h1>
<p class="lead">All 270 new yaw-only-model rollouts complete. Neither method activates the steering limiter or produces a locally unstable straight-running equilibrium. The mean tracking difference is about 0.000020°/s.</p>
<section><h2>1. Matched execution and main result</h2><p>The comparison contains 90 pilot vehicles × three test speeds × two identification methods = 540 case records. The 270 full-state CT results are reused exactly. The 270 yaw-only-model runs use the same LQI design rules, 12-s left-right reference, 13/21/29 m/s speeds, initial states, gain interpolation, steering limits and noise-free true-state feedback. Identification, physical parameters and tuning settings are unchanged.</p>{table(summary)}
<p>Mean yaw-rate RMSE is 0.01283193°/s for full-state CT and 0.01281197°/s for yaw-only CT. The latter is lower by 0.1555%, but the absolute difference is only 0.00001996°/s. This is not persuasive evidence that removing a measurement improves control. It shows no practically apparent aggregate tracking penalty in this particular benchmark; no formal equivalence or noninferiority margin was specified.</p></section>
<section><h2>2. Paired cases, fleets and speeds</h2>{svg}{table(paired)}
<p>Yaw-only gives smaller errors in 134/270 individual cases and 46/90 vehicle means. Its pooled mean is slightly smaller in all three fleets, but changes occur in both directions. The largest individual increase is 0.000527°/s; the largest decrease is 0.001194°/s. The median case change is a small increase, despite the small decrease in the mean. Individual cases are nested within vehicles and three reused fleets, not 270 independent experiments.</p>{table(fleets)}{table(speeds)}{table(dist)}</section>
<section><h2>3. Steering effort and local stability</h2><p>All 540 cases meet the applied steering/rate and sideslip bounds. Unlike the two earlier restricted-coverage pair failures, none activates either limiter. Mean steering-rate RMS is 0.820845°/s for full-state CT and 0.820827°/s for yaw-only CT, with nearly identical maxima around 1.519°/s.</p>
<p>The maximum true-plant origin spectral radius is 0.966152 for full-state CT and 0.966193 for yaw-only CT. These Jacobians use the simulator RK4 derivative and interpolated gains at each test speed. They establish local linear stability at straight running for the tested controllers; they do not establish global nonlinear stability or robustness over arbitrary trajectories.</p>
<p>The definition of feasibility remains limited to physical bounds. Here, inspecting limiter activity and the local Jacobians independently finds no hidden counterpart of the earlier adjacent-pair problem.</p></section>
<section><h2>4. Relation to the identification result</h2><p>The preceding yaw-only identification audit found a 28.4% relative increase in unseen increment error and approximately doubled local sideslip pulse-response error, despite only a small held-out yaw-prediction change. The present runs show that those model differences do not produce a corresponding aggregate tracking loss for this LQI design and maneuver.</p>
<p>Matrix error and closed-loop tracking evaluate different properties. Tracking also depends on which model directions change the synthesized gains, integral action and feedback. This comparison does not isolate the causal contribution of those mechanisms. A slightly worse matrix estimate can produce slightly better tracking without being a better physical model.</p>{table(blocks)}<p>Identification used the original 2–3-speed blocks. This result does not extend to single-speed identification or the adjacent-pair restrictions.</p></section>
<section><h2>5. What has and has not been established</h2><p><strong>The structured local CT baseline remains strong after removing sideslip from its identification loss.</strong> We have no demonstrated need for fleet sharing from this sensing ablation alone. The result closes the specific matched state-feedback question without adding a new plant or changing heterogeneity.</p>
<p>The evaluation still provides true sideslip to the controller. Acquisition also used noisy sideslip feedback. Therefore this is not evidence that the complete workflow can operate without sideslip sensing, or that an observer will reconstruct sideslip accurately from these identified models.</p>
<p>If the practical sensing assumption is the next priority, the next distinct stage is observer-based operation: first check acquisition with one common nominal LTI observer/controller using only available measurements, then evaluate identified models under common observer-design rules. New observer settings should be established without selecting them from the final test outcomes. Adding geometry variation at the same time would obscure that comparison.</p></section>
<section><h2>6. Audit and reproducibility</h2><p>Verified execution checkpoint envelopes and provenance, all reused baseline schedules and outcomes, the 1,980 stored synthesis matrices against their fitted model sources, learned-model closed-loop poles and prefilters, case completeness and feasibility consistency, and all four original control CSV tables. Recomputed true-plant origin Jacobians for all 540 case records. Full identification checkpoints match the previous audit.</p>
<p>The 270 new scientific rollouts were executed before this audit and committed separately. This script performs no fitting, controller retuning or nonlinear rollouts. Individual time-series arrays are not part of the saved checkpoints; the review checks their stored metrics and does not reconstruct waveforms. Reproduce with <code>python code/experiments/analyze_physical_ct_yaw_control_review.py</code>.</p></section></main></html>'''
    REPORT.write_text('\n'.join(line.rstrip() for line in text.splitlines())+'\n',encoding='utf-8',newline='\n')


if __name__=='__main__':
    with threadpool_limits(limits=1):audit()
