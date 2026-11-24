"""Audit the committed yaw-only pilot; no fitting, reconstruction or rollouts."""
from __future__ import annotations
import io
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import paper_physical_ct_yaw_only as run
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
PREFIX='paper_physical_ct_yaw_only_review'
REPORT=run.ROOT/'report/physical_ct_yaw_only_review.html'


def close(a,b):np.testing.assert_allclose(a,b,rtol=2e-10,atol=2e-12)


def check_table(data,suffix,keys):
    saved=pd.read_csv(run.OUT/f'{run.PREFIX}_{suffix}.csv')
    pd.testing.assert_frame_equal(data[saved.columns].sort_values(keys).reset_index(drop=True),
        saved.sort_values(keys).reset_index(drop=True),check_exact=False,check_dtype=False,rtol=2e-10,atol=2e-12)


def audit():
    cfg=run.preflight();archived=run.old.load(run.OUT/f'{run.PREFIX}_analysis.json')
    assert archived['analysis_sha256']==run.old.file_hash(run.__file__)
    assert archived['protocol_sha256']==run.old.file_hash(run.CONFIG)
    assert archived['control_cases'] in [0,540], 'Combined analysis must not contain partial controls'
    models=[];fits=[];validation=[];hashes={};input_pairs=0;start_count=0;all_nuisance=[]
    for seed in cfg['seeds']:
        p=run.old.envelope(run.path(seed));run.validate(p,seed,cfg)
        assert p['status']=='execution_complete'
        key=run.path(seed).relative_to(run.ROOT).as_posix();hashes[key]=run.old.file_hash(run.path(seed))
        assert hashes[key]==archived['checkpoint_sha256'][key]
        full=run.old.envelope(run.old.result_path(seed));source=run.old.sources(seed)[1]
        for c,old,src in zip(p['clients'],full['clients'],source['clients'],strict=True):
            fit=c['fit'];assert fit['converged']
            assert fit['scalar_observations']==2001*len(src['speeds'])
            assert len(fit['attempts'])==4
            assert fit['hidden_initial_beta_is_gauge'] and fit['nuisance_columns_per_segment']==3
            assert fit['free_nuisance_parameters']==sum(t['nuisance_rank'] for t in fit['trajectories'])
            for h,previous in zip(c['training_hashes'],old['training_hashes'],strict=True):
                assert h['speed']==previous['speed'] and h['u']==previous['u']
                assert h['original_full_measurements']==previous['y']
                assert len(h['yaw'])==64
                input_pairs+=1
            grid=np.array(source['grid']);mask=~np.isin(grid,src['speeds'])
            for method,f in [('FullStateCT',old['fit']),('YawOnlyCT',fit)]:
                attempts=[a for a in f['attempts'] if a['converged']]
                best=min(attempts,key=lambda x:(x['weighted_mse'],x['start']))
                assert best['start']==f['selected_start'];close(best['weighted_mse'],f['weighted_mse'])
                close(best['parameters'],f['parameters'])
                if method=='YawOnlyCT':start_count+=len(attempts)
                prediction=run.model_grid(f['parameters'],grid,cfg['estimator'])
                close(prediction,c['grid_targets'] if method=='YawOnlyCT' else old['grid_targets'])
                truth=np.array(src['reference']);errors=run.old.dynamic_metrics(prediction,truth,cfg)
                saved=c['comparisons'][method]
                for name in errors:close(errors[name],saved['errors'][name])
                extra=run.extra_errors(prediction,truth,cfg)
                close(extra['entry_absolute'],saved['entry_absolute']);close(extra['scaled_increment'],saved['scaled_increment'])
                sv=np.array(f['information']['log_parameter_singular_values']);rank=int(sum(sv>sv[0]*1e-8))
                params=np.array(f['parameters']);ref=np.array(c['reference_parameters'])
                physical=src['parameters']
                close(ref,[physical['front_stiffness']/physical['mass'],physical['rear_stiffness']/physical['mass'],physical['mass']/physical['yaw_inertia'],.9])
                spread=float(np.max(np.ptp(np.log([a['parameters'] for a in attempts]),axis=0)))
                row=dict(seed=seed,client=c['client'],method=method,block=str(src['speeds']),
                    relative_rank=rank,condition=sv[0]/sv[-1],near_bound=any(f['near_bound']),mu=params[3],
                    converged_starts=len(attempts),max_log_start_spread=spread,
                    min_nuisance_rank=min(t.get('nuisance_rank',4) for t in f['trajectories']))
                for j,name in enumerate(['Cf_over_m','Cr_over_m','m_over_Iz','mu']):
                    row[name]=params[j];row[name+'_relative_error_pct']=100*(params[j]/ref[j]-1)
                fits.append(row)
                for j,v in enumerate(grid):
                    row=dict(seed=seed,client=c['client'],method=method,block=str(src['speeds']),speed=v,seen=not mask[j])
                    for name in ['matrix','increment','a','b','pole_rms_absolute']:row[name]=errors[name][j]
                    row['scaled_increment']=extra['scaled_increment'][j]
                    for k,name in enumerate(['A11','A12','A21','A22','B1','B2']):row[name+'_absolute']=extra['entry_absolute'][j][k]
                    row['pulse_beta']=errors['pulse_relative_by_state'][j][0]
                    row['pulse_yaw']=errors['pulse_relative_by_state'][j][1]
                    models.append(row)
                val=[v for v in c['validation'] if v['method']==method]
                assert [v['speed'] for v in val]==src['speeds']
                for v in val:
                    assert v['status']=='complete' and v['scored_samples']==1800
                    nuisance=v['nuisance_prefix_only'];all_nuisance.append(nuisance['nuisance_rank'])
                    assert nuisance['initial_beta_is_gauge'] and nuisance['initial_state'][0]==0
                    close(nuisance['target'],run.model_grid(f['parameters'],[v['speed']],cfg['estimator'])[0])
                    if method=='YawOnlyCT':
                        match=next(x for x in c['validation'] if x['method']=='FullStateCT' and x['speed']==v['speed'])
                        assert v['yaw_sha256']==match['yaw_sha256'] and v['input_sha256']==match['input_sha256']
                    assert np.isfinite([v['measured_yaw_rmse_deg_s'],v['true_yaw_rmse_deg_s']]).all()
                    validation.append(dict(seed=seed,client=c['client'],method=method,block=str(src['speeds']),speed=v['speed'],
                        status=v['status'],measured_yaw_rmse_deg_s=v['measured_yaw_rmse_deg_s'],true_yaw_rmse_deg_s=v['true_yaw_rmse_deg_s']))
    model=pd.DataFrame(models);fit=pd.DataFrame(fits);val=pd.DataFrame(validation)
    assert len(fit)==180 and len(model)==1980 and len(val)==498 and input_pairs==249
    check_table(model,'grid_errors',['seed','client','method','speed']);check_table(val,'validation',['seed','client','method','speed'])
    metrics=['matrix','increment','scaled_increment','a','b','pole_rms_absolute',
        *[x+'_absolute' for x in ['A11','A12','A21','A22','B1','B2']],'pulse_beta','pulse_yaw']
    clients,fleets,summary=run.aggregate(model[~model.seen],metrics)
    vc,vf,vs=run.aggregate(val,['measured_yaw_rmse_deg_s','true_yaw_rmse_deg_s'])
    counts=val.groupby('method').agg(segments=('status','size'),complete=('status',lambda x:int((x=='complete').sum()))).reset_index()
    vs=vs.merge(counts,on='method')
    for suffix,data,keys in [('clients',clients,['seed','client','method']),('fleets',fleets,['seed','method']),
        ('summary',summary,['method']),('validation_clients',vc,['seed','client','method']),
        ('validation_fleets',vf,['seed','method']),('validation_summary',vs,['method'])]:check_table(data,suffix,keys)
    # Use the original equal-vehicle/fleet weighting also within each block.
    blockmap=fit[['seed','client','method','block']]
    bc=clients.merge(blockmap,on=['seed','client','method'],validate='one_to_one')
    blocks=bc.groupby(['seed','method','block'])[metrics].mean().groupby(['method','block']).mean().reset_index()
    paired=[]
    for data,names in [(clients,['matrix','increment','scaled_increment','pulse_beta','pulse_yaw']),
                       (vc,['measured_yaw_rmse_deg_s','true_yaw_rmse_deg_s'])]:
        for name in names:
            w=data.pivot(index=['seed','client'],columns='method',values=name);d=w.YawOnlyCT-w.FullStateCT
            paired.append(dict(metric=name,full_mean=w.FullStateCT.mean(),yaw_mean=w.YawOnlyCT.mean(),
                mean_change=d.mean(),relative_mean_change_pct=100*d.mean()/w.FullStateCT.mean(),
                yaw_vehicle_wins=int((d<0).sum()),vehicles=len(d),median_change=d.median(),
                yaw_p95=w.YawOnlyCT.quantile(.95),yaw_max=w.YawOnlyCT.max()))
    paired=pd.DataFrame(paired)
    parameter=[]
    for method in run.METHODS:
        z=fit[fit.method==method]
        for name in ['Cf_over_m','Cr_over_m','m_over_Iz','mu']:
            x=z[name+'_relative_error_pct']
            parameter.append(dict(method=method,parameter=name,median_absolute_error_pct=x.abs().median(),
                p95_absolute_error_pct=x.abs().quantile(.95),mean_signed_error_pct=x.mean(),max_absolute_error_pct=x.abs().max()))
    parameter=pd.DataFrame(parameter)
    conditioning=fit.groupby('method').agg(rank_four=('relative_rank',lambda x:int((x==4).sum())),
        near_bound=('near_bound','sum'),median_condition=('condition','median'),max_condition=('condition','max'),
        max_log_start_spread=('max_log_start_spread','max')).reset_index()
    audit_info=dict(status='yaw_only_identification_review_complete',fit_conditions=90,converged_starts=start_count,
        model_grid_points_checked=1980,paired_validation_segments=249,control_cases=0,
        original_training_input_and_full_measurement_hash_pairs_verified=input_pairs,
        yaw_hashes_recorded_not_recomputed=249,validation_metrics_recomputed_from_raw=False,
        validation_nuisance_ranks=sorted(set(all_nuisance)),checkpoint_sha256=hashes,
        script_sha256=run.old.file_hash(__file__),
        note='No fitting, retuning, raw reconstruction or nonlinear rollouts. Recomputed all model metrics and original summary CSVs. Validation RMSEs checked for completeness/provenance/paired inputs but not recomputed without original arrays. Acquisition still used sideslip feedback; no output-feedback deployment claim.')
    for suffix,data in [('fits',fit),('summary',summary),('paired',paired),('parameters',parameter),
                        ('conditioning',conditioning),('blocks',blocks),('validation_summary',vs)]:
        data.to_csv(run.OUT/f'{PREFIX}_{suffix}.csv',index=False)
    run.old.acquisition.parent.save_json(run.OUT/f'{PREFIX}.json',audit_info)
    report(clients,vc,fit,summary,paired,parameter,conditioning,blocks,vs,audit_info)
    print(paired.to_string(index=False));print(conditioning.to_string(index=False));print(parameter.to_string(index=False));print(REPORT)


def report(clients,validation,fits,summary,paired,parameters,conditioning,blocks,vs,audit):
    import matplotlib
    matplotlib.use('Agg');matplotlib.rcParams['svg.hashsalt']=PREFIX
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for ax,frame,name,factor,title in [(axes[0,0],clients,'increment',100,'Unseen increment error [%]'),
                                      (axes[0,1],validation,'true_yaw_rmse_deg_s',1,'Held-out yaw RMSE [deg/s; simulator reference]')]:
        w=frame.pivot(index=['seed','client'],columns='method',values=name)*factor
        ax.scatter(w.FullStateCT,w.YawOnlyCT,s=19,alpha=.7);end=w.to_numpy().max()*1.06
        ax.plot([0,end],[0,end],'--',color='gray');ax.set(xlabel='Full-state-trained CT',ylabel='Yaw-only-trained CT',title=title)
    labels=['10–14','16–20','22–26','28–30']
    for method,color in [('FullStateCT','#177c75'),('YawOnlyCT','#b65a31')]:
        z=blocks[blocks.method==method].sort_values('block')
        axes[1,0].plot(range(4),100*z.increment,marker='o',label=method,color=color)
        f=fits[fits.method==method]; axes[1,1].scatter(f.condition,f.mu,s=15,alpha=.6,label=method,color=color)
    axes[1,0].set(xticks=range(4),xticklabels=labels,xlabel='Original speed block [m/s]',ylabel='Mean unseen increment error [%]',title='Largest errors after low-speed identification')
    axes[1,1].axhline(.9,color='gray',linestyle='--');axes[1,1].set(xscale='log',xlabel='Log-parameter sensitivity condition number',ylabel='Estimated friction μ',title='Conditioning and friction recovery')
    axes[1,0].legend(fontsize=9);axes[1,1].legend(fontsize=9)
    for ax in axes.flat:ax.grid(alpha=.2)
    buf=io.StringIO();fig.savefig(buf,format='svg',metadata={'Date':None});plt.close(fig);svg=buf.getvalue();svg=svg[svg.index('<svg'):]
    def table(df):return '<div class="scroll">'+df.to_html(index=False,border=0,float_format=lambda x:f'{x:.6g}')+'</div>'
    display=summary[['method','matrix','increment','scaled_increment','a','b','pulse_beta','pulse_yaw']].copy()
    for col in display.columns[1:]:display[col]*=100
    entries=summary[['method',*[k+'_absolute' for k in ['A11','A12','A21','A22','B1','B2']]]]
    text=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Yaw-only identification: audited review</title>
<style>body{{font:16px/1.6 system-ui;background:#edf2f5;color:#193142;margin:0}}main{{max-width:1280px;margin:auto;padding:30px}}section{{background:white;padding:24px;margin:20px 0;border-radius:10px}}h1,h2{{line-height:1.25}}.lead{{font-size:20px}}.scroll{{overflow:auto}}table{{border-collapse:collapse;font-size:13px;white-space:nowrap}}td,th{{padding:8px;border-bottom:1px solid #ddd}}svg{{width:100%;height:auto}}.note{{border-left:5px solid #b65a31}}</style><main>
<h1>Yaw-only identification remains strong, with a modest loss of extrapolation accuracy</h1>
<p class="lead">All 90 local fits converge. Removing sideslip measurements raises mean unseen increment error from 1.555% to 1.996%, while held-out yaw prediction changes from 0.003081 to 0.003210°/s. Closed-loop consequences are evaluated in the separate matched-control review.</p>
<section><h2>1. What was isolated</h2><p>Three reused pilot fleets, 90 vehicles and 249 original speed segments. The same four-parameter physical CT model, known geometry/curvature, full local 2–3-speed blocks, bounds, four starts and optimizer settings are used. Full-state CT fits are reused; yaw-only fits receive only yaw measurements and applied steering at each known speed.</p><p>Unmeasured sideslip remains a simulated internal state. Fixing its initial value to zero is a gauge made possible by the free affine offsets; it adds no information about true initial sideslip. The recorded acquisition controller still used noisy sideslip feedback. This result is about removing sideslip from the identification objective, not a complete sideslip-sensor-free workflow.</p></section>
<section><h2>2. Model accuracy and prediction</h2>{svg}<p>All values in the following model table are percentages. Matrix errors use the same original-unobserved speed set per vehicle. Pulse errors compare local linear model responses under a common small steering pulse, separately for beta and yaw; they are not nonlinear driving tests.</p>{table(display)}
<p>Mean combined matrix error increases from 0.386% to 0.501% (+29.7% relatively). Identity-free increment error increases from 1.555% to 1.996% (+28.4%). Scaling states and input by the predeclared physical scales gives 1.147% versus 1.693% (+47.6%). Thus the loss is present after removing the identity and changing coordinate scaling; a single combined percentage understates its structure.</p>
<p>The unseen-speed linear pulse-response error for sideslip increases from 1.636% to 3.331%, while yaw-response error increases from 1.022% to 1.199%. This concerns perturbation responses of the local linear models under identical inputs and initial deviations; it is not a measured observer sideslip error. Full-state fitting has lower increment error for 62/90 vehicles and lower scaled-increment error for 70/90. Mean matrix and increment errors worsen in each of the three fleets. This is a consistent descriptive effect, not an independent confirmation or statistical equivalence test.</p>{table(entries)}
<p>All six mean absolute entry errors increase. The largest relative increases are in A12 (about 123%), A11 (85%), B1 (77%) and A21 (68%); A22 and B2 increase less (about 15% and 11%). Entries have different units/scales, so the table should be read entry by entry, not used to rank their control importance.</p></section>
<section><h2>3. Held-out yaw prediction remains close</h2>{table(vs)}
<p>All 249 independent validation segments are scored for both methods. Their dynamics are fixed at training estimates. Each model uses only the first 2 s of validation yaw/input data to initialize affine nuisance quantities, then predicts the remaining 18 s without measurement updates. No suffix measurements affect initialization or centering; no true sideslip is used.</p>
<p>Against true simulator yaw, the equal-vehicle/fleet mean increases by only 0.000129°/s (about 4.19%). Full-state fitting is better for 59/90 vehicle means. Against measured yaw, RMSE is 0.030225 versus 0.030238°/s, close to the imposed 0.03°/s measurement-noise standard deviation. Measurement noise therefore largely masks the small model-prediction difference; the evaluation-only true-yaw score is useful here.</p>
<p>Validation occurs at the observed identification speeds. These prediction results do not establish equally accurate prediction at unobserved speeds or unchanged closed-loop tracking. Good output prediction also does not imply equal recovery of the latent physical state or all matrix entries.</p></section>
<section><h2>4. Identifiability, optimization and coverage</h2>{table(conditioning)}
<p>All 360 yaw-only optimization attempts converge. All 90 selected joint fits have relative sensitivity rank four at 10⁻⁸, as do the full-state fits. Across converged starts, the maximum log-parameter range is about 0.00032 for yaw-only fitting (roughly 0.032% multiplicative spread). These results give no indication that different local optimizer solutions explain the observed accuracy loss. They do not prove global uniqueness.</p>
<p>Median sensitivity condition number increases from 55.2 to 59.1; the maximum rises from 520.5 to 606.8. One fit in each method lies near a parameter bound. Yaw-only friction estimates span about 0.714 to 1.5, although the simulator uses 0.9. Numerical full rank does not guarantee precise parameter recovery. Absolute sensitivity information and residual dependence also matter; these condition numbers are descriptive, not calibrated uncertainty bounds.</p>{table(parameters)}<p>Median absolute parameter errors rise from 0.45% to 1.57% for Cf/m, 0.49% to 1.93% for Cr/m, and 0.26% to 1.08% for m/Iz. Median friction error changes less, from 3.50% to 3.89%, and its upper tail is large for both methods. The loss of sideslip information therefore appears more clearly in stiffness/inertia separation and sideslip response than in the held-out yaw signal alone.</p>{table(blocks[['method','block','matrix','increment','scaled_increment']])}
<p>The block table contains ratios, not percentages. The 10–14 m/s block has the largest unseen increment errors: 3.123% for full-state fitting and 4.326% for yaw-only. Both methods still receive multiple observed speeds. This experiment does not remove the previously demonstrated exact single-speed stiffness/friction ambiguity.</p></section>
<section class="note"><h2>5. Implications and matched control</h2><p>Once available, see the <a href="physical_ct_yaw_control_review.html">separate matched-control review</a> for closed-loop results. This page audits identification only.</p><p><strong>The local CT baseline does not depend critically on directly observing sideslip in these identification conditions.</strong> Sideslip information improves physical-model accuracy, but removing it does not cause broad estimation failure or a large yaw-prediction deficit. This result alone provides no evidence of a federated advantage.</p>
<p>The separate matched state-feedback control diagnostic addresses the remaining control question: reuse the full-state CT rollouts and run the 270 yaw-only-model rollouts at 13/21/29 m/s. Keep the controller and true-state feedback unchanged for this comparison, so any difference comes from the learned models. Inspect limiter activity, steering-rate RMS and origin stability as well as yaw tracking; previous coverage results showed why feasibility alone is insufficient.</p>
<p>Only after that comparison should observer-based acquisition and evaluation be introduced. Changing geometry or fleet heterogeneity now would mix a new plant effect with the sensing change. An eventual practical workflow must remove the sideslip dependency from acquisition and runtime feedback too.</p></section>
<section><h2>6. Audit and limits</h2><p>Verified checkpoint envelopes, frozen provenance, client completeness, training-loss selection, original training input/full-measurement hash pairs, model grids, every stored model metric and eight published model and validation tables. Recomputed 1,980 model-grid points. Paired validation input/yaw hashes agree across methods; every record has 1,800 scored samples and nuisance rank three.</p><p>The raw original arrays are unavailable in this workspace. Therefore the saved yaw hashes and held-out RMSE values were not independently recomputed from raw trajectories; their completeness, pairing, provenance and aggregate arithmetic were checked. No new fitting, controller synthesis or nonlinear rollouts were performed. Frozen experiment files and original results remain unchanged.</p><p>Reproduce: <code>python code/experiments/analyze_physical_ct_yaw_only_review.py</code>. Detailed paired statistics:</p>{table(paired)}</section></main></html>'''
    REPORT.write_text('\n'.join(line.rstrip() for line in text.splitlines())+'\n',encoding='utf-8',newline='\n')


if __name__=='__main__':
    with threadpool_limits(limits=1):audit()
