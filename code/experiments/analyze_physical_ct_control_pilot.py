"""Audit committed control pilot and explain its distribution; no new rollouts.

python code/experiments/analyze_physical_ct_control_pilot.py
Leaves frozen experiment inputs, checkpoints and original analysis unchanged.
"""
from __future__ import annotations
import io
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'code/src'), str(ROOT/'code/experiments')]
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import paper_physical_ct_control as pilot
import paper_physical_ct_oe as old
from federated_lpv.vehicle import VehicleParameters, continuous_bicycle_matrices
from federated_lpv.control import augmented_tracking_matrices
from federated_lpv.oe_lqi_validation import interpolate

OUT = ROOT/'results/tables'
PREFIX = 'paper_physical_ct_control_review'
REPORT = ROOT/'report/physical_ct_control_review.html'
LABELS = {'PhysicalCTLocal':'Structured local CT','NominalLTI':'Nominal LTI',
          'RestrictedLocal':'Restricted Local','SmoothLocal':'Smooth Local',
          'Global':'Global','Federated':'Federated'}


def close(a,b):
    np.testing.assert_allclose(a,b,rtol=2e-10,atol=2e-12)


def rk4_origin_matrices(parameters,speed,dt):
    """Derivative of the implemented held-input RK4 plant step at x=u=0.

    tanh'(0)=1, so the continuous linear-tire model is the exact Jacobian
    at straight running. Use RK4's polynomial, not exact ZOH, to match the
    simulator. Saturation/antiwindup are inactive infinitesimally at zero.
    """
    a,b=continuous_bicycle_matrices(speed,parameters)
    h=dt*np.block([[a,b],[np.zeros((1,3))]])
    p=np.eye(3)+h+h@h/2+np.linalg.matrix_power(h,3)/6+np.linalg.matrix_power(h,4)/24
    return p[:2,:2],p[:2,2:]


def audit():
    cfg=pilot.preflight();control=old.load(old.storage.CONFIG)
    archived=old.load(OUT/'paper_physical_ct_control_analysis.json')
    if archived['protocol_sha256']!=old.file_hash(pilot.CONFIG) or archived['analysis_sha256']!=old.file_hash(pilot.__file__):
        raise ValueError('Original analysis provenance mismatch')
    records=[];diagnostics=[];nodes=[];hashes={};source_count=0
    basis=old.load(ROOT/'code/config/experiment_10a.json')
    for seed in cfg['seeds']:
        path=old.result_path(seed,True);result=old.envelope(path)
        source=old.sources(seed)[1]
        pilot.validate(result,seed,cfg,source)
        if result['status']!='execution_complete':raise ValueError('Incomplete pilot')
        key=path.relative_to(ROOT).as_posix();hashes[key]=old.file_hash(path)
        if archived['checkpoint_sha256'][key]!=hashes[key]:raise ValueError('Stale archived analysis')
        for other in [*old.source_paths(seed),old.result_path(seed)]:
            hashes[other.relative_to(ROOT).as_posix()]=old.file_hash(other)
        fitted=old.envelope(old.result_path(seed))
        models=source['conditions']['ClosedOE4']['models']
        nominal=old.storage.nominal_schedule(control)
        for client in result['clients']:
            i=client['index'];truth=VehicleParameters(**source['clients'][i]['parameters'])
            for method,schedule in client['schedules'].items():
                if method==old.METHOD:expected=np.array(fitted['clients'][i]['grid_targets'])
                elif method=='NominalLTI':expected=np.array([n['matrix'] for n in nominal['nodes']])
                else:
                    coeff=old.lpv.model_for(models,method,i)
                    expected=old.raw_basis(np.array(source['grid']),len(coeff),basis)@coeff
                close([n['matrix'] for n in schedule['nodes']],expected)
                source_count+=len(expected)
                for n in schedule['nodes']:
                    a=np.array(n['matrix'][:4]).reshape(2,2);b=np.array(n['matrix'][4:]).reshape(2,1)
                    aa,bb=augmented_tracking_matrices(a,b,control['sample_time'])
                    gain=np.array(n['gain']);rho=max(abs(np.linalg.eigvals(aa-bb@gain[None,:])))
                    close(rho,n['design_rho'])
                    eq=np.linalg.solve(np.block([[np.eye(2)-a,-b],[np.array([[0.,1.]]),np.zeros((1,1))]]),[0,0,1])
                    close(n['prefilter'],eq[2]+gain[:2]@eq[:2])
                    nodes.append(dict(seed=seed,client=client['client'],method=method,speed=n['speed'],
                        group=int(models['assignments'][i]),model_rho=float(max(abs(np.linalg.eigvals(a)))),
                        design_rho=float(rho),K_beta=gain[0],K_r=gain[1],K_I=gain[2],F=n['prefilter']))
                gains,prefilters=interpolate(schedule,control['speed_grid'],cfg['speeds_mps'])
                for speed,gain,prefilter in zip(cfg['speeds_mps'],gains,prefilters):
                    a,b=rk4_origin_matrices(truth,speed,control['sample_time'])
                    aa,bb=augmented_tracking_matrices(a,b,control['sample_time'])
                    eig=np.linalg.eigvals(aa-bb@gain[None,:]);rho=float(max(abs(eig)))
                    diagnostics.append(dict(seed=seed,client=client['client'],method=method,speed=speed,
                        group=int(models['assignments'][i]),origin_rho=rho,origin_unstable=bool(rho>1+1e-9),
                        K_beta=gain[0],K_r=gain[1],K_I=gain[2],F=prefilter))
            for row in client['rows']:
                o=row['outcome']
                records.append(dict(seed=seed,client=client['client'],observed_speeds=str(client['observed_speeds']),
                    method=row['method'],speed=row['speed'],status=o['status'],feasible=o['feasible'],**o['metrics']))
    frame=pd.DataFrame(records)
    if len(frame)!=1620:raise ValueError('Unexpected case count')
    keys=['seed','client','method','speed']
    saved=pd.read_csv(OUT/'paper_physical_ct_control_cases.csv')
    pd.testing.assert_frame_equal(frame.sort_values(keys).reset_index(drop=True),saved.sort_values(keys).reset_index(drop=True),check_exact=False,rtol=2e-10,atol=2e-12)
    units,fleets,summary,speed,paired=pilot.tables(frame,cfg)
    for suffix,df in [('clients',units),('fleets',fleets),('summary',summary),('speeds',speed),('paired',paired)]:
        pd.testing.assert_frame_equal(df,pd.read_csv(OUT/f'paper_physical_ct_control_{suffix}.csv'),check_exact=False,rtol=2e-10,atol=2e-12)
    diag=frame.merge(pd.DataFrame(diagnostics),on=keys,validate='one_to_one')
    node=pd.DataFrame(nodes)
    wide=frame.pivot(index=['seed','client','speed'],columns='method',values='tracking_rmse_deg_s')
    vehicle=wide.groupby(['seed','client']).mean()
    comparisons=[]
    for method in cfg['comparators']:
        gap=wide[method]-wide[old.METHOD]
        comparisons.append(dict(baseline=method,CT_case_wins=int((gap>0).sum()),cases=len(gap),
            CT_vehicle_wins=int((vehicle[method]>vehicle[old.METHOD]).sum()),vehicles=len(vehicle),
            mean_baseline_minus_CT_deg_s=gap.mean(),median_baseline_minus_CT_deg_s=gap.median()))
    comparisons=pd.DataFrame(comparisons)
    dist=frame.groupby('method').tracking_rmse_deg_s.agg(mean='mean',median='median',p95=lambda x:x.quantile(.95),maximum='max').reindex(cfg['methods']).reset_index()
    counts=diag.groupby('method').agg(origin_unstable=('origin_unstable','sum'),constraint_violations=('feasible',lambda x:int((~x).sum())),limited_cases=('limiter_fraction',lambda x:int((x>0).sum())))
    dist=dist.merge(counts,on='method')
    # Select by independently computed local stability, not by an error cutoff.
    bad=diag[(diag.method=='Federated')&diag.origin_unstable]
    bad_keys=pd.MultiIndex.from_frame(bad[['seed','client','speed']]);mask=wide.index.isin(bad_keys)
    total_gap=wide.Federated-wide[old.METHOD]
    remaining=wide.loc[~mask]
    metrics=dict(status='audit_complete',cases=len(frame),node_matrices_checked=source_count,
        full_duration_cases=int((frame.status=='complete').sum()),feasible_cases=int(frame.feasible.sum()),
        checkpoint_and_source_sha256=hashes,script_sha256=old.file_hash(__file__),
        federated_origin_unstable_cases=len(bad),federated_gap_fraction_from_origin_unstable=float(total_gap[mask].sum()/total_gap.sum()),
        remaining_cases=len(remaining),remaining_federated_rmse=float(remaining.Federated.mean()),
        remaining_CT_rmse=float(remaining[old.METHOD].mean()),
        remaining_mean_gap=float((remaining.Federated-remaining[old.METHOD]).mean()),
        scope='Descriptive audit of three reused pilot fleets. No refits, retuning, new rollouts or formal confirmation. Origin Jacobian uses simulator RK4 and inactive limits; it is not a nonlinear stability guarantee.')
    for suffix,df in [('distribution',dist),('paired',comparisons),('diagnostics',diag),('nodes',node)]:
        df.to_csv(OUT/f'{PREFIX}_{suffix}.csv',index=False)
    old.acquisition.parent.save_json(OUT/f'{PREFIX}.json',metrics)
    write_report(frame,dist,comparisons,diag,node,wide,metrics)
    print(dist.to_string(index=False));print(comparisons.to_string(index=False))
    print(json.dumps({k:v for k,v in metrics.items() if k not in ['checkpoint_and_source_sha256','script_sha256']}));print(REPORT)


def write_report(frame,dist,comparisons,diag,node,wide,audit):
    import matplotlib
    matplotlib.use('Agg')
    matplotlib.rcParams['svg.hashsalt']='physical_ct_control_review'
    import matplotlib.pyplot as plt
    selected=['NominalLTI','Global','Federated',old.METHOD]
    colors=['#66758c','#b47c17','#ae414c','#177c75']
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for method,color in zip(selected,colors):
        values=np.sort(wide[method].to_numpy())
        axes[0,0].plot(values,np.arange(1,len(values)+1)/len(values),label=LABELS[method],color=color)
        s=frame[frame.method==method].groupby('speed').tracking_rmse_deg_s
        axes[1,0].plot(s.mean().index,s.mean(),marker='o',color=color,label=LABELS[method])
        axes[1,1].plot(s.median().index,s.median(),marker='o',color=color,label=LABELS[method])
    axes[0,0].set(xscale='log',xlabel='Tracking RMSE [deg/s; logarithmic]',ylabel='Fraction of 270 cases',title='Full distribution: seven Federated cases form a tail')
    axes[0,0].legend(fontsize=8)
    for v,color in zip([13,21,29],['#467eae','#bb8833','#ae414c']):
        w=wide.xs(v,level='speed');axes[0,1].scatter(w.Federated,w[old.METHOD],s=18,alpha=.6,color=color,label=f'{v} m/s')
    axes[0,1].plot([.004,6],[.004,6],'--',color='gray',lw=1)
    axes[0,1].set(xscale='log',yscale='log',xlabel='Federated RMSE [deg/s]',ylabel='Structured local CT RMSE [deg/s]',title='Paired cases: below diagonal favors structured CT')
    axes[0,1].legend(fontsize=8)
    axes[1,0].set(yscale='log',xlabel='Speed [m/s]',ylabel='Mean RMSE [deg/s; logarithmic]',xticks=[13,21,29],title='Means include every completed case')
    axes[1,1].set(xlabel='Speed [m/s]',ylabel='Median RMSE [deg/s]',xticks=[13,21,29],title='Typical tracking differences remain small')
    for ax in axes.flat:ax.grid(alpha=.2)
    buf=io.StringIO();fig.savefig(buf,format='svg',metadata={'Date':None});plt.close(fig)
    svg=buf.getvalue();svg=svg[svg.index('<svg'):]
    def table(df):return '<div class="scroll">'+df.to_html(index=False,border=0,float_format=lambda x:f'{x:.6g}')+'</div>'
    bad=diag[(diag.method=='Federated')&diag.origin_unstable]
    cols=['seed','client','speed','tracking_rmse_deg_s','tracking_peak_deg_s','origin_rho','feasible']
    relevant=node[(node.seed==1302)&(node.client=='nominal_08')&(node.method.isin(['Federated',old.METHOD]))&(node.speed>=28)]
    maxima=frame.groupby('method')[['beta_peak_deg','steering_peak_deg','steering_rate_peak_deg_s','lateral_acceleration_peak_mps2']].max().reset_index()
    core=frame.groupby('method').agg(complete=('status',lambda x:int((x=='complete').sum())),feasible=('feasible','sum'),rmse_deg_s=('tracking_rmse_deg_s','mean'),steering_rms_deg=('steering_rms_deg','mean')).reset_index()
    review=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Structured CT closed-loop pilot — review</title>
<style>body{{font:16px/1.6 system-ui;background:#edf2f5;color:#193142;margin:0}}main{{max-width:1200px;margin:auto;padding:32px}}section{{background:white;padding:25px;margin:20px 0;border-radius:10px}}h1,h2{{line-height:1.25}}.lead{{font-size:20px}}.scroll{{overflow:auto}}table{{border-collapse:collapse;white-space:nowrap;font-size:13px;width:100%}}td,th{{padding:8px;text-align:right;border-bottom:1px solid #dce3e9}}svg{{width:100%;height:auto}}.note{{border-left:4px solid #b47c17;padding-left:16px}}code{{font-size:13px}}</style><main>
<h1>Structured local models are a strong control baseline</h1><p class="lead">The four-parameter local model gives the lowest mean tracking error in all three pilot fleets. Its large pooled advantage over Federated is dominated by one learned group at 29 m/s; typical differences are small.</p>
<section><h2>1. Completeness and matched comparison</h2><p>All 1,620 planned runs completed: 90 vehicles × six methods × three speeds. All 540 schedules were synthesized. This audit validates checkpoint provenance, reconstructs all {audit['node_matrices_checked']:,} stored synthesis matrices from frozen source models, checks saved model closed-loop poles and prefilters, and reproduces all six archived CSV tables. No controller retuning, identification or nonlinear rollouts were performed.</p><p>All methods use the same 12-s smooth left-right reference, fixed speeds 13/21/29 m/s, LQI weights, limits, zero initialization and noise-free full-state feedback. Known vehicle parameters are used only for the post-hoc plant-Jacobian diagnostic below, never for the learned controllers.</p>{table(core)}<p>Finite constraint-violating runs remain in all means. Restricted Local violates constraints in 33/270 cases; Smooth Local in 22/270. The other four methods satisfy the stated steering and sideslip constraints in every case. “Feasible” contains no tracking-quality or stability requirement.</p></section>
<section><h2>2. Typical performance and the tail</h2>{svg}{table(dist)}<p>Means and medians answer different questions. Restricted Local and Smooth Local also have low median errors, but severe tails. Their maximum sideslip excursions exceed the credible operating range of this model and cannot be interpreted as realistic vehicle maneuvers.</p>{table(comparisons)}<p>Wins count strictly smaller RMSE; a vehicle win compares its mean over all three speeds. The structured method improves the mean over each whole fleet against every comparator, but does not win every individual case.</p></section>
<section><h2>3. Why the Federated mean is large</h2><p>Seven nominal-family vehicles in fleet 1302 share learned group 0 and exactly the same Federated schedule. At 29 m/s their tracking RMSE is 5.37–5.51°/s. These seven cases account for {100*audit['federated_gap_fraction_from_origin_unstable']:.2f}% of the total paired Federated-minus-structured RMSE difference. All remain within the benchmark's steering and sideslip bounds.</p>{table(bad[cols])}<p>The group's learned 30-m/s model has an open-loop pole magnitude 1.01625. LQI synthesis stabilizes that learned model at the design node, but the resulting gain and prefilter differ sharply from the 28-m/s values. Linear interpolation at 29 m/s gives K<sub>β</sub>≈−23.13, K<sub>I</sub>≈−0.631 and F≈−10.609. The structured-model schedule has no such sign reversal in this interval.</p>{table(relevant[['method','speed','model_rho','design_rho','K_beta','K_r','K_I','F']])}<p>For each saved controller, the audit computes the augmented closed-loop Jacobian at straight running on the true simulated plant. It uses the RK4 step's exact derivative at zero, matching the simulator, and the interpolated gains. Exactly these seven Federated cases have spectral radius above one. This establishes a local linear instability of that equilibrium and supports a control-relevant explanation of the tail. It does not isolate the relative contribution of prefilter mismatch and unstable feedback to the finite-duration RMSE, or imply a nonlinear global stability result.</p><p class="note">As a descriptive decomposition only, the remaining {audit['remaining_cases']} paired cases average {audit['remaining_federated_rmse']:.6f}°/s for Federated and {audit['remaining_CT_rmse']:.6f}°/s for structured CT: a difference of {audit['remaining_mean_gap']:.6f}°/s. The primary averages above still include all 270 cases. These seven cases are identified by the computed local stability diagnostic; no cases are removed from the official result.</p></section>
<section><h2>4. Effort and physical limits</h2>{table(maxima)}<p>Structured CT, Global and Nominal LTI have mean steering RMS near 0.778° and no limiter activation. Structured CT's maximum sideslip is about 1.09°. Its improvement is therefore not accompanied by a material increase in steering effort in this test. Lateral acceleration is reported, but is not an extra feasibility gate. Peak reference yaw rate is 2.5/v rad/s (about 11.02, 6.82 and 4.94°/s at the three speeds).</p></section>
<section><h2>5. What this changes for the paper</h2><p><strong>The current vehicle benchmark does not establish that federation is needed for useful models or good closed-loop tracking under limited local speed coverage.</strong> The structured local estimator uses the same local acquisition data and four physically coupled parameters, and performs strongly without sharing. The original regression rank argument remains a statement about its chosen flexible representation; it is not evidence that the physical identification problem must require shared fleet information.</p><p>The control evidence strengthens Markus's baseline concern. It also shows why a pooled matrix-error mean alone is insufficient: Global tracks well despite its larger matrix error, while a single learned-group schedule can cause a localized control defect. Model accuracy, controller sensitivity, interpolation and closed-loop performance must be distinguished.</p><p>The structured result assumes the correct four-parameter model family, known geometry, full-state measurements and the present simulated fleet. It does not establish that local physical identification will always dominate sharing, nor that friction is well identified. These three fleets were already used in development, so this is a pilot result without confirmatory inference.</p></section>
<section><h2>6. Recommended next decision</h2><p>Keep this strong structured local baseline and the unmodified pilot results. Before expanding the campaign, decide whether the paper should focus on the present vehicle model or on a more general identification problem. For the vehicle scope, discuss the changed evidence with Markus and formulate a physically justified information limitation that remains after using model structure. Only then consider a matched structured local/pooled/group comparison. Increasing heterogeneity or weakening the local model solely to create a federation gain would not answer that question.</p><p>Separately, a future controller-design pass should validate interpolated gains on the relevant model/plant family; successful Riccati solves at grid nodes alone are insufficient. This audit does not repair the saved controllers or claim that a repair would recover the identification motivation.</p></section>
<section><h2>Reproduce and inspect</h2><p><code>python code/experiments/analyze_physical_ct_control_pilot.py</code></p><p>Saved alongside this report: distribution, paired-win, all-case Jacobian and all-node gain tables, plus source hashes in <code>{PREFIX}.json</code>. The original report remains <code>report/physical_ct_control_pilot.html</code>.</p></section></main></html>'''
    REPORT.write_text('\n'.join(line.rstrip() for line in review.splitlines())+'\n',encoding='utf-8',newline='\n')


if __name__=='__main__':
    with threadpool_limits(limits=1):audit()
