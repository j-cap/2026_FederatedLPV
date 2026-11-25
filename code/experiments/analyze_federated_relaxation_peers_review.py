"""Audit and interpret the frozen matched peer-speed diagnostic; no fitting."""
import copy
import io
import numpy as np
import pandas as pd
import paper_federated_relaxation_peers as run

PREFIX=run.PREFIX+'_review'
REPORT=run.ROOT/'report/federated_relaxation_peers_review.html'


def main():
    cfg,protocol=run.preflight();rows=[];optimization=[];replays=[]
    paths=[p for p in run.OUT.glob(run.PREFIX+'*') if '_review' not in p.name]+[run.REPORT]
    hashes={p.relative_to(run.ROOT).as_posix():run.base.canonical_hash(p) for p in paths}
    for seed in protocol['seeds']:
        p=run.base.storage.read_checkpoint(run.path(seed));run.validate(p,cfg,protocol,execution=False)
        assert p['status']=='execution_complete'
        source,complementary=run.sources(seed,cfg,protocol);fleet=run.base.fleet_for(cfg,seed)
        for c in fleet['clients']:
            own=run.record_key(c,c['training_speed'])
            assert p['records'][own]['trace_sha256']==source['data'][c['client']]['trace_sha256']
            assert p['local'][own]['selected']==source['local'][c['client']]['selected']
            assert len({p['records'][run.record_key(c,v)]['paired_noise_sha256'] for v in protocol['same_speeds']})==1
            assert len({p['records'][run.record_key(c,v)]['paired_input_sha256'] for v in protocol['same_speeds']})==1
            for v in protocol['same_speeds']:
                key=run.record_key(c,v);entry=p['local'][key]
                assert p['records'][key]['operating_checks_passed'] and entry['selected']['available']
                assert p['records'][key]['source_trace_sha256']==source['data'][c['client']]['trace_sha256']
                assert entry['reused']==(v==c['training_speed'])
                if not entry['reused']:assert run.base.fl.select_local(entry['attempts'])==entry['selected']
        for v,condition in p['conditions'].items():
            state=condition['optimization'];h=state['history'];last=h[-1];objectives=np.array([r['objective'] for r in h])
            assert np.all(np.diff(objectives)<=1e-6*np.maximum(1.,abs(objectives[:-1])))
            expected=[p['local'][run.record_key(c,int(v))]['selected']['log_parameters'] for c in fleet['clients']]
            assert condition['initial_log_parameters']==expected
            z=np.array(state['log_parameters']);eta=np.array(state['centers']);w=np.array(cfg['sharing']['weights'])
            assert state['assignments']==[0]*len(z)
            np.testing.assert_allclose(eta[0],z.mean(axis=0),rtol=1e-12,atol=1e-12)
            np.testing.assert_allclose(last['penalty'],np.sum(((z-eta[0])*w)**2),rtol=1e-12)
            np.testing.assert_allclose(last['objective'],last['data_sse']+last['penalty'],rtol=1e-12)
            relative=abs(h[-2]['objective']-last['objective'])/max(1.,abs(h[-2]['objective']))
            converged=relative<cfg['sharing']['personal_relative_tolerance'] and last['max_log_step']<cfg['sharing']['personal_step_tolerance'] and last['all_local_optimizers_converged'] and not last['assignments_changed']
            assert (condition['status']=='converged')==converged
            if not converged:assert condition['status']=='budget_exhausted' and state['round']==30
            optimization.append(dict(seed=seed,training_speed=int(v),status=condition['status'],rounds=state['round'],
                final_max_log_step=last['max_log_step'],final_relative_objective_change=relative,
                all_final_local_solves_converged=last['all_local_optimizers_converged']))
        metadata={c['client']:c for c in fleet['clients']}
        for key,entry in p['evaluation'].items():
            old=source['evaluation'][key];c=metadata[old['client']]
            assert entry['source_trace_sha256']==old['trace_sha256']
            methods={m:old['methods'][m] for m in ['LocalOE','NominalPrior']}
            methods.update(Complementary=complementary['evaluation'][key]['metrics'],SameSpeed=entry['metrics'])
            for method,metrics in methods.items():
                assert metrics['status']=='complete'
                assert all(np.isfinite(metrics[k]) for k in ['yaw_rmse_deg_s','measured_yaw_rmse_deg_s','beta_rmse_deg'])
                rows.append(dict(seed=seed,client=c['client'],training_speed=c['training_speed'],speed=old['speed'],probe=old['probe'],
                    region=run.REGIONS[old['region']],method=method,**metrics))
        # Replay one changed-speed training record and one saved test per fleet.
        i=next(i for i,c in enumerate(fleet['clients']) if c['training_speed']==13);c=fleet['clients'][i]
        original,_=run.base.generate(c,seed,'training',13,cfg);a=copy.deepcopy(original)
        a['state']=run.base.batch_predict(c['physical_parameters'],[a['steering']],[29],c['geometry_m'],cfg['sample_time'],cfg['integration_max_step'])[0,0]
        a['measured_yaw']=a['state'][:,1]+a['noise'];saved=p['records'][run.record_key(c,29)]
        assert run.base.storage.trace_hash({'noise':a['noise']})==saved['paired_noise_sha256']
        input_exact=run.base.storage.trace_hash({'steering':a['steering']})==saved['paired_input_sha256']
        np.testing.assert_allclose(a['steering'][::10],source['data'][c['client']]['preview']['steering'],rtol=1e-10,atol=1e-12)
        metrics=dict(beta_peak_deg=float(np.rad2deg(max(abs(a['state'][:,0])))),ay_peak_mps2=float(max(abs(a['state'][:,2]+a['state'][:,3]))),
            steering_rate_peak_deg_s=float(np.rad2deg(max(abs(np.diff(np.r_[0,a['steering'],0]))))/cfg['sample_time']))
        np.testing.assert_allclose(list(metrics.values()),[saved['metrics'][k] for k in metrics],rtol=1e-8,atol=1e-10)
        replays.append(dict(seed=seed,stage='training',exact_input_match=input_exact,exact_trace_match=run.base.storage.trace_hash(a)==saved['trace_sha256'],
            max_metric_difference=max(abs(v-saved['metrics'][k]) for k,v in metrics.items())))
        a,_=run.base.generate(c,seed,'slow',21,cfg);theta=np.exp(p['conditions']['13']['optimization']['log_parameters'][i])
        y=run.base.batch_predict(theta,[a['steering']],[21],c['geometry_m'],cfg['sample_time'],cfg['integration_max_step'])[0,0]
        rms=lambda x:float(np.rad2deg(np.sqrt(np.mean(x*x))))
        metrics=dict(yaw_rmse_deg_s=rms(y[:,1]-a['state'][:,1]),measured_yaw_rmse_deg_s=rms(y[:,1]-a['measured_yaw']),beta_rmse_deg=rms(y[:,0]-a['state'][:,0]))
        saved=p['evaluation'][f'{c["client"]}_v21_slow'];a['predicted_SameSpeed']=y
        np.testing.assert_allclose(list(metrics.values()),[saved['metrics'][k] for k in metrics],rtol=1e-8,atol=1e-10)
        replays.append(dict(seed=seed,stage='evaluation',exact_trace_match=run.base.storage.trace_hash(a)==saved['trace_sha256'],
            max_metric_difference=max(abs(v-saved['metrics'][k]) for k,v in metrics.items())))
        print(f'Fleet {seed}: pairing, initialization, optimization and saved-case replays verified',flush=True)
    cases=pd.DataFrame(rows);keys=['seed','client','speed','probe','method']
    pd.testing.assert_frame_equal(cases.sort_values(keys).reset_index(drop=True),pd.read_csv(run.OUT/f'{run.PREFIX}_cases.csv').sort_values(keys).reset_index(drop=True),check_exact=False,rtol=1e-10,atol=1e-12)
    summary,paired=run.summaries(cases)
    for name,frame in [('summary',summary),('paired',paired)]:
        pd.testing.assert_frame_equal(frame,pd.read_csv(run.OUT/f'{run.PREFIX}_{name}.csv'),check_exact=False,rtol=1e-10,atol=1e-12)
    regions=cases.pivot_table(index='region',columns='method',values='yaw_rmse_deg_s')
    primary=cases[cases.region=='target_unseen_anchor']
    fleet=primary.pivot_table(index='seed',columns='method',values='yaw_rmse_deg_s')
    speed=primary.pivot_table(index='training_speed',columns='method',values='yaw_rmse_deg_s')
    strata=primary.pivot_table(index=['seed','training_speed'],columns='method',values='yaw_rmse_deg_s')
    for frame in [regions,fleet,speed,strata]:frame['complementary_reduction_vs_same_pct']=100*(1-frame.Complementary/frame.SameSpeed)
    pivot=primary.pivot(index=['seed','client','speed','probe'],columns='method',values='yaw_rmse_deg_s');vehicles=pivot.groupby(['seed','client']).mean()
    means=pivot.mean();info=dict(status='audit_passed',source_hashes=hashes,source_checkpoints=protocol['source_checkpoints'],
        new_fits=0,new_test_conditions=0,converged_same_speed=sum(r['status']=='converged' for r in optimization),same_speed_conditions=len(optimization),
        total_comparison_rows=len(cases),same_speed_reduction_vs_local_pct=100*(1-means.SameSpeed/means.LocalOE),
        complementary_reduction_vs_same_pct=100*(1-means.Complementary/means.SameSpeed),
        absolute_peer_speed_gain_deg_s=float(means.SameSpeed-means.Complementary),
        same_speed_fraction_of_complementary_absolute_gain_vs_local=float((means.LocalOE-means.SameSpeed)/(means.LocalOE-means.Complementary)),
        complementary_case_wins_vs_same=int((pivot.Complementary<pivot.SameSpeed).sum()),peer_cases=len(pivot),
        complementary_vehicle_wins_vs_same=int((vehicles.Complementary<vehicles.SameSpeed).sum()),vehicles=len(vehicles),
        replayed_saved_traces=len(replays),exact_replayed_traces=sum(r['exact_trace_match'] for r in replays),
        audit_runtime=run.base.identity(cfg)['runtime'])
    assert hashes=={name:run.base.canonical_hash(run.ROOT/name) for name in hashes}
    tables=dict(regions=regions.reset_index(),fleets=fleet.reset_index(),speeds=speed.reset_index(),strata=strata.reset_index(),optimization=pd.DataFrame(optimization),replays=pd.DataFrame(replays))
    render(tables,info)
    for name,frame in tables.items():frame.to_csv(run.OUT/f'{PREFIX}_{name}.csv',index=False)
    run.base.storage.atomic_json(run.OUT/f'{PREFIX}.json',info)
    print({k:v for k,v in info.items() if k not in ['source_hashes','source_checkpoints','audit_runtime']});print(REPORT)


def render(t,info):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib.rcParams['svg.hashsalt']=PREFIX
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for method,color in [('LocalOE','#68778a'),('NominalPrior','#ba842b'),('SameSpeed','#8b62ac'),('Complementary','#087d87')]:
        axes[0].plot(t['fleets'].seed.astype(str),t['fleets'][method],marker='o',color=color,label=method)
        axes[1].plot(t['speeds'].training_speed.astype(str),t['speeds'][method],marker='o',color=color,label=method)
    axes[0].set(xlabel='Fleet',title='Target-unseen anchor speeds')
    axes[1].set(xlabel='Target training speed [m/s]',title='Benefit depends on target speed')
    for ax in axes:ax.set_ylabel('Clean yaw prediction RMSE [deg/s]');ax.grid(alpha=.2)
    axes[0].legend(fontsize=8);b=io.StringIO();fig.savefig(b,format='svg',metadata={'Date':None});plt.close(fig);svg=b.getvalue();svg=svg[svg.index('<svg'):]
    tab=lambda frame:'<div class="scroll">'+frame.to_html(index=False,border=0,float_format=lambda v:f'{v:.6g}')+'</div>'
    text=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Matched peer-speed result review</title>
<style>body{{font:16px/1.65 system-ui;background:#edf2f5;color:#173348}}main{{max-width:1150px;margin:auto;padding:28px}}section{{background:white;padding:25px;border-radius:12px;margin:22px 0}}h1,h2{{line-height:1.25}}svg{{width:100%;height:auto}}.scroll{{overflow:auto}}table{{border-collapse:collapse;font-size:13px}}th,td{{padding:9px;border-bottom:1px solid #ddd;white-space:nowrap}}</style><main>
<h1>Complementary speeds add a small, mainly low-speed benefit</h1>
<p>At target-unseen anchor speeds, mean error is 0.004127 deg/s with SameSpeed peers and 0.003949 deg/s with Complementary peers: a 4.3% reduction. Most of the improvement over unregularized local OE is already available from same-speed sharing or a fixed nominal prior.</p>
<section><h2>1. What the matched comparison establishes</h2>{svg}{tab(t['regions'])}{tab(t['fleets'])}
<p>SameSpeed reduces the LocalOE mean error by 47.2%; Complementary reduces it by 49.5%. SameSpeed therefore accounts for approximately 95.4% of the absolute reduction achieved by Complementary versus LocalOE. This is an arithmetic comparison, not a causal decomposition. The extra reduction attributable to the tested peer-speed intervention is approximately 0.000177 deg/s in this pilot.</p>
<p>Complementary improves fleet-mean unseen-anchor error by 0.12%, 3.83% and 10.35% in fleets 1301, 1302 and 1303. It wins in {info['complementary_case_wins_vs_same']}/108 trajectory cases and {info['complementary_vehicle_wins_vs_same']}/27 vehicle-averaged comparisons. Thus the pooled mean benefit is not a uniform individual benefit.</p>
<p>The nominal prior remains close: its pooled error is 0.004246 deg/s. Complementary is 7.0% better, while SameSpeed is 2.8% better. These are clean-output prediction errors; no closed-loop benefit is established.</p></section>
<section><h2>2. Low-speed targets explain the direction of the effect</h2>{tab(t['speeds'])}{tab(t['strata'])}
<p>For targets trained at 13 m/s, Complementary improves mean unseen-anchor prediction by 7.1% versus SameSpeed. At 21 m/s it is 0.13% worse, and at 29 m/s it is 1.63% worse. The strongest low-speed result occurs in fleet 1303: a 19.1% reduction, with both conditions converged. This supports a conditional benefit for some low-speed targets, rather than a general need for complementary coverage.</p>
<p>At the target-observed speed, SameSpeed has a very small mean advantage: Complementary increases error by 0.085%. Complementary reduces intermediate-reference and outside-reference errors by 3.1% and 4.6%, respectively.</p>
<p>Changing peer speeds also changes output amplitude and parameter sensitivity for the same steering input. The test measures that combined operating-speed intervention; it cannot isolate a geometric coverage effect independent of excitation or output signal-to-noise ratio. Shared true relaxation lengths also favor transfer.</p></section>
<section><h2>3. Two low-speed fits remain at the round limit</h2>{tab(t['optimization'])}
<p>Seven of nine SameSpeed conditions converged. The 13 m/s conditions in fleets 1301 and 1302 reached 30 rounds. Their last maximum log-parameter changes are 0.00115 and 0.00349 versus the 0.001 criterion. Relative objective changes already pass; every final client solve converged, and objectives decrease smoothly. The remaining issue is the outer parameter-change criterion.</p>
<p>Those two conditions contribute directly to the main low-speed comparison, so the 4.3% pooled result is provisional as a comparison of converged estimates. Do not relabel the conditions as converged merely because their losses change little. This limitation does not explain away the fully converged fleet-1303 result.</p></section>
<section><h2>4. Integrity audit</h2><p>All 1512 method-case rows are present and finite. All 81 acquisition records pass their saved operating checks; 27 are reused originals and 54 change speed. Original target records and local fits match exactly. Each peer's input and noise hashes match across speed conditions. Source locks, local-start selection, condition-local initialization, server-center equations, penalties, objective monotonicity and recorded convergence statuses were verified. Case, summary and paired CSVs reproduce the checkpoint values.</p>
<p>Three saved changed-speed training records and three saved test predictions were replayed numerically. Metrics agree within tolerance; {info['exact_replayed_traces']}/6 full trace hashes match exactly. No claim of full byte-exact restoration is made. No new optimizer runs, test conditions or original-output changes occurred during this audit.</p>{tab(t['replays'])}</section>
<section><h2>5. Decision</h2><p><b>Keep personalized FL as a candidate, but do not claim that complementary coverage is the dominant source of its benefit.</b> The large gain versus LocalOE mostly survives when peers all operate at the target speed. The additional coverage benefit is small, concentrated at low target speed, and not yet fully converged in two relevant conditions.</p>
<p>For numerical closure, continue only the two unfinished SameSpeed conditions from their saved states under unchanged criteria and evaluate their affected targets again, preserving the original experiment. Then consolidate these results before expanding the plant or launching a large study. If the conclusion remains unchanged, a defensible statement is that fleet-learned regularization can modestly improve low-speed local identification, with limited additional benefit from complementary speeds in this model/data regime.</p>
<p>The existing three fleets are development populations. Independent confirmation and a meaningful closed-loop consequence would still be needed for a strong FL-and-control claim. No new experiment or manuscript change is included here.</p>
<p><a href="federated_relaxation_peers.html">Original generated report</a> · <a href="../docs/federated_relaxation_peers.html">Frozen protocol</a></p></section></main></html>'''
    REPORT.write_text('\n'.join(line.rstrip() for line in text.splitlines())+'\n',encoding='utf-8',newline='\n')


if __name__=='__main__':main()
