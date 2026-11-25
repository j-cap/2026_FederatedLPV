"""Read-only audit and interpretation of the frozen limited-speed FL pilot.

No fitting, controller tests, new conditions or changes to original outputs.
Replays saved preview cases numerically; distinguishes exact trace recovery.
"""
from __future__ import annotations
import io
import json
from pathlib import Path
import numpy as np
import pandas as pd
import paper_federated_relaxation_coverage as run

ROOT=run.ROOT
PREFIX=run.PREFIX+'_review'
REPORT=ROOT/'report/federated_relaxation_coverage_review.html'
METHODS=['LocalOE','NominalPrior','Global','FedPersonal','FedGroups']
REGIONS=['local_observed','peer_observed','fleet_interpolation','fleet_extrapolation']


def audit(cfg):
    rows=[];histories=[];clients=[];replays=[];selected=[];compositions=[]
    original_files=sorted((ROOT/'results/tables').glob(run.PREFIX+'*'))
    original_files=[p for p in original_files if '_review' not in p.name]+[run.REPORT]
    hashes={p.relative_to(ROOT).as_posix():run.canonical_hash(p) for p in original_files}
    current=run.identity(cfg);runtimes={}
    for fleet in cfg['fleets']:
        seed=fleet['seed'];p=run.storage.read_checkpoint(run.path(seed))
        assert p['status']=='execution_complete'
        # Runtime differences do not invalidate a read-only saved-result audit.
        # The scientific source/configuration locks must still match exactly.
        assert p['provenance']['files']==current['files']
        assert p['provenance']['source']==current['source']
        runtimes[str(seed)]=p['provenance']['runtime']
        expected={f'{c["client"]}_v{v}_{probe}' for c in fleet['clients']
                  for v in cfg['evaluation_speeds'] for probe in ['slow','broadband']}
        assert set(p['evaluation'])==expected
        for i,c in enumerate(fleet['clients']):
            cid=c['client'];local=p['local'][cid];chosen=run.fl.select_local(local['attempts'])
            assert chosen==local['selected'] and chosen['available']
            assert p['nominal_prior'][cid]['optimizer_converged']
            assert p['data'][cid]['operating_checks_passed']
            s=chosen['data_singular_values']
            clients.append(dict(seed=seed,client=cid,family=c['family'],training_speed=c['training_speed'],
                local_rank=chosen['data_relative_rank'],local_jacobian_condition=s[0]/s[-1],
                local_mu_at_bound=chosen['near_bound'][3]))
            for method in ['FedPersonal','FedGroups']:
                q=p['selected'][method]
                compositions.append(dict(seed=seed,client=cid,method=method,group=q['assignments'][i],
                    family=c['family'],training_speed=c['training_speed']))
            # Replay one previously saved evaluation preview per probe/fleet.
            if cid=='vehicle_00':
                for probe in ['slow','broadband']:
                    key=f'{cid}_v21_{probe}';old=p['evaluation'][key]
                    new,arrays=run.evaluation_case(p,c,i,21,probe,cfg)
                    difference=0.
                    for method in cfg['methods']:
                        for metric in ['yaw_rmse_deg_s','measured_yaw_rmse_deg_s','beta_rmse_deg']:
                            a=new['methods'][method][metric];b=old['methods'][method][metric]
                            np.testing.assert_allclose(a,b,rtol=1e-8,atol=1e-10)
                            difference=max(difference,abs(a-b))
                    preview_difference=0.
                    for name,values in old['preview'].items():
                        np.testing.assert_allclose(arrays[name][::10],values,rtol=1e-8,atol=1e-10)
                        preview_difference=max(preview_difference,float(np.max(abs(arrays[name][::10]-values))))
                    replays.append(dict(seed=seed,case=key,max_metric_difference=difference,
                        max_preview_difference=preview_difference,exact_trace_match=new['trace_sha256']==old['trace_sha256']))
        for method,attempts in [('Global',p['global_attempts']),*p['sharing_attempts'].items()]:
            choice=run.choose_sharing(attempts);assert choice==p['selected'][method]
            chosen=attempts[choice['selected_attempt']]
            selected.append(dict(seed=seed,method=method,status=chosen['status'],
                objective=choice['objective'],last_max_log_step=chosen['history'][-1]['max_log_step']))
            for init,state in attempts.items():
                objective=np.array([h['objective'] for h in state['history']])
                assert np.all(np.diff(objective)<=1e-6*np.maximum(1.,abs(objective[:-1])))
                for j,h in enumerate(state['history']):
                    histories.append(dict(seed=seed,method=method,initialization=init,
                        selected=init==choice['selected_attempt'],step=j+1,objective=h['objective'],
                        gap_to_final=h['objective']-objective[-1],max_log_step=h['max_log_step']))
        metadata={c['client']:c for c in fleet['clients']}
        for case in p['evaluation'].values():
            c=metadata[case['client']]
            assert set(case['methods'])==set(cfg['methods']) and case['operating_checks_passed']
            assert case['region']==run.fl.coverage_region(case['speed'],c['training_speed'],cfg['fleet_training_speeds'])
            for method,result in case['methods'].items():
                assert result['status']=='complete'
                assert all(np.isfinite(result[k]) for k in ['yaw_rmse_deg_s','measured_yaw_rmse_deg_s','beta_rmse_deg'])
                if method=='Exact':assert result['yaw_rmse_deg_s']==0
                rows.append(dict(seed=seed,client=case['client'],speed=case['speed'],probe=case['probe'],
                    region=case['region'],method=method,training_speed=c['training_speed'],family=c['family'],
                    operating_checks_passed=case['operating_checks_passed'],**result))
        print(json.dumps(dict(seed=seed,audit='checkpoint, provenance, selection, monotonicity and two saved previews passed')),flush=True)
    cases=pd.DataFrame(rows)
    saved=pd.read_csv(run.OUT/f'{run.PREFIX}_cases.csv');keys=['seed','client','speed','probe','method']
    pd.testing.assert_frame_equal(cases[saved.columns].sort_values(keys).reset_index(drop=True),
        saved.sort_values(keys).reset_index(drop=True),check_exact=False,rtol=1e-10,atol=1e-12)
    computed=cases.groupby(['seed','method','region']).yaw_rmse_deg_s.mean().sort_index()
    saved_summary=pd.read_csv(run.OUT/f'{run.PREFIX}_fleets.csv').set_index(['seed','method','region']).sort_index()
    np.testing.assert_allclose(computed,saved_summary.mean_yaw_rmse_deg_s,rtol=1e-10,atol=1e-12)
    pairs=[]
    for region,sub in cases.groupby('region'):
        for scope,group in [('all',sub),*[(str(s),g) for s,g in sub.groupby('seed')]]:
            pivot=group.pivot(index=['seed','client','speed','probe'],columns='method',values='yaw_rmse_deg_s')
            for method,baseline in [('NominalPrior','LocalOE'),('FedPersonal','LocalOE'),('FedPersonal','NominalPrior'),
                                    ('FedGroups','LocalOE'),('FedGroups','NominalPrior'),('FedGroups','FedPersonal')]:
                pairs.append(dict(fleet=scope,region=region,method=method,baseline=baseline,
                    method_mean=pivot[method].mean(),baseline_mean=pivot[baseline].mean(),
                    reduction_pct=100*(1-pivot[method].mean()/pivot[baseline].mean()),
                    absolute_reduction_deg_s=(pivot[baseline]-pivot[method]).mean(),
                    improved_cases=int((pivot[method]<pivot[baseline]).sum()),cases=len(pivot)))
    # Verify no original execution or analysis file was changed by this audit.
    assert hashes=={name:run.canonical_hash(ROOT/name) for name in hashes}
    info=dict(source_hashes=hashes,source_runtime=runtimes,audit_runtime=current['runtime'],
        comparisons=len(cases),clients=len(clients),new_fits=0,new_test_conditions=0,
        saved_preview_cases_replayed=len(replays),exact_replay_traces=sum(r['exact_trace_match'] for r in replays),
        replay_scope='vehicle_00, speed 21 m/s, both saved probes, each fleet; 36 model comparisons',
        original_outputs_unchanged=True)
    tables=dict(cases=cases,paired=pd.DataFrame(pairs),clients=pd.DataFrame(clients),
        histories=pd.DataFrame(histories),selected=pd.DataFrame(selected),groups=pd.DataFrame(compositions),replays=pd.DataFrame(replays))
    return tables,info


def render(t,info):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({'svg.hashsalt':PREFIX,'axes.spines.top':False,'axes.spines.right':False,'font.size':10})
    colors=dict(zip(METHODS,['#68778a','#bd8426','#a13e42','#087d87','#7352a1']))
    def svg(fig):
        output=io.StringIO();fig.savefig(output,format='svg',metadata={'Date':None});plt.close(fig)
        return output.getvalue()[output.getvalue().index('<svg'):]
    def table(data):
        return '<div class="scroll">'+data.to_html(index=False,border=0,float_format=lambda x:f'{x:.6g}')+'</div>'
    cases=t['cases'];summary=cases.pivot_table(index='method',columns='region',values='yaw_rmse_deg_s').reindex(METHODS)[REGIONS]
    peers=cases[cases.region=='peer_observed']
    byfleet=peers.pivot_table(index='seed',columns='method',values='yaw_rmse_deg_s')[METHODS]
    byspeed=peers.pivot_table(index='training_speed',columns='method',values='yaw_rmse_deg_s')[METHODS]
    fig,axes=plt.subplots(1,2,figsize=(12,4.1),layout='constrained')
    for m in METHODS:
        axes[0].plot(range(4),summary.loc[m],marker='o',label=m,color=colors[m])
        if m!='Global':axes[1].plot(range(4),summary.loc[m],marker='o',label=m,color=colors[m])
    for ax in axes:
        ax.set(xticks=range(4),xticklabels=['Local','Peer-observed','Intermediate','Outside fleet'],ylabel='Clean yaw RMSE [deg/s]');ax.grid(alpha=.2)
    axes[0].set_title('All methods');axes[1].set_title('Detail: local and personalized estimates')
    axes[0].legend(fontsize=8);overview=svg(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for m in [x for x in METHODS if x!='Global']:
        axes[0].plot(byfleet.index.astype(str),byfleet[m],marker='o',label=m,color=colors[m])
        axes[1].plot(byspeed.index,byspeed[m],marker='o',label=m,color=colors[m])
    axes[0].set(xlabel='Pilot fleet',title='Peer-speed prediction by fleet')
    axes[1].set(xlabel='Locally observed training speed [m/s]',xticks=[13,21,29],title='Where does sharing help?')
    for ax in axes:ax.set_ylabel('Clean yaw RMSE [deg/s]');ax.grid(alpha=.2)
    axes[0].legend(fontsize=8);strata=svg(fig)
    paired=peers.pivot(index=['seed','client','speed','probe'],columns='method',values='yaw_rmse_deg_s')
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for ax,baseline in zip(axes,['LocalOE','NominalPrior']):
        for seed in [1301,1302,1303]:
            g=paired.loc[seed];ax.scatter(g[baseline],g.FedPersonal,label=str(seed),s=20,alpha=.7)
        upper=max(paired[baseline].max(),paired.FedPersonal.max())*1.05
        ax.plot([0,upper],[0,upper],color='#999',linestyle='--');ax.set(xlim=(0,upper),ylim=(0,upper),
            xlabel=f'{baseline} yaw RMSE [deg/s]',ylabel='FedPersonal yaw RMSE [deg/s]',title='Each point: one vehicle, speed and probe')
        ax.grid(alpha=.2)
    axes[0].legend(title='Fleet');scatter=svg(fig)
    groups=t['groups'][t['groups'].method=='FedGroups']
    fig,axes=plt.subplots(1,3,figsize=(12,4.3),layout='constrained')
    fam={'nominal':0,'heavy':1,'handling':2}
    for ax,(seed,g) in zip(axes,groups.groupby('seed')):
        for r in g.itertuples():
            ax.scatter(r.training_speed,fam[r.family],s=240,color=['#087d87','#bd8426','#7352a1'][r.group])
            ax.text(r.training_speed,fam[r.family],str(r.group),ha='center',va='center',color='white')
        ax.set(xticks=[13,21,29],yticks=[0,1,2],yticklabels=['nominal','heavy','handling'],
            ylim=(-.5,2.5),xlim=(10,32),xlabel='Training speed [m/s]',title=f'Fleet {seed}')
        ax.grid(alpha=.15)
    groupplot=svg(fig)
    pairtable=t['paired'];mainpairs=pairtable[(pairtable.fleet=='all')&(pairtable.region=='peer_observed')]
    doc=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Limited-speed federated identification — result review</title><style>
body{{font:16px/1.65 system-ui;background:#edf2f5;color:#173348;margin:0}}main{{max-width:1180px;margin:auto;padding:28px}}
section{{background:white;padding:26px;border-radius:12px;margin:22px 0}}h1,h2{{line-height:1.25}}svg{{width:100%;height:auto}}
.scroll{{overflow:auto}}table{{border-collapse:collapse;font-size:13px;white-space:nowrap}}th,td{{padding:9px;border-bottom:1px solid #ddd;text-align:right}}a{{color:#087d87}}.lead{{font-size:20px}}code{{background:#edf2f5}}
</style><main><h1>Personalization helps; the additional value of fleet learning is modest</h1>
<p class="lead">At peer-observed speeds, FedPersonal reduces mean clean-yaw prediction error by 49.5% versus LocalOE, but only 7.0% versus a fixed nominal prior. Grouping worsens error by 55.1% versus FedPersonal. These are exploratory identification results, not closed-loop gains.</p>
<section><h2>1. What was actually tested?</h2><p>Three previously used pilot fleets, nine vehicles each. Every vehicle learns from one 12 s yaw-only record at 13, 21 or 29 m/s. Two held-out steering probes are evaluated at seven speeds. All methods use the same six-parameter, four-state physical model. No earlier broad-speed estimates initialize these fits.</p>
<p>Local = the vehicle's training speed on a fresh probe. Peer-observed = the two other fleet training speeds. Intermediate = 17/25 m/s. Outside fleet = 9/33 m/s. A single common parameter vector (Global) differs fundamentally from a common learned prior with individual models (FedPersonal).</p>
<p>All 27 local fits and 27 nominal-prior fits converged. All 2268 comparison rows are finite and pass the saved operating checks. Exact-model prediction errors are zero. Checkpoint digests, source/configuration locks, selected starts and all saved optimization histories were checked. Case and fleet-summary CSVs agree with the checkpoints.</p>
<p>Six previously saved preview trajectories (36 model predictions) were replayed, with no new fitting or test conditions. Their metrics and previews agree to the audit tolerances; {info['exact_replay_traces']}/6 match the saved full numerical trace hashes exactly. This is a subset replay, not full restoration of every ignored trajectory.</p>{table(t['replays'])}</section>
<section><h2>2. Prediction results</h2>{overview}{table(summary.reset_index())}
<p>FedPersonal also improves mean error at the locally observed speed by 6.6% versus LocalOE and 2.6% versus NominalPrior. Thus there is no average local-speed sacrifice, although this does not imply improvement for every vehicle. The clean-yaw error at peer speeds decreases from 0.007817 to 0.003950 deg/s. The additional decrease beyond NominalPrior is only 0.000296 deg/s.</p>
<p>The nominal prior reduces LocalOE's peer-speed error by 45.7%, accounting for about 92% of the absolute reduction achieved by FedPersonal. This is an arithmetic comparison, not a causal decomposition of the learning mechanism.</p>{table(mainpairs)}
<p>The reported percentage is one minus the ratio of mean paired errors, not the mean of per-case percentages. All comparisons use the same complete cases. The sensor noise standard deviation is 0.03 deg/s, while these errors compare predictions against clean simulated yaw; it is not a noise-floor test.</p></section>
<section><h2>3. Consistency and local speed coverage</h2>{strata}{table(byfleet.reset_index())}{table(byspeed.reset_index())}
<p>Relative to NominalPrior, FedPersonal is 8.9% worse in fleet 1301, 15.5% better in fleet 1302 and 17.0% better in fleet 1303. It improves 62/108 peer-speed cases and 15/27 vehicle-averaged peer-speed errors. Against LocalOE, the corresponding counts are 79/108 and 21/27. These repeated trajectories are not independent fleet replicates.</p>{scatter}
<p>The strongest effect is for vehicles trained at 13 m/s. Their peer-speed error is 0.018475 deg/s for LocalOE, 0.008176 for NominalPrior and 0.007405 for FedPersonal. For vehicles trained at 29 m/s, NominalPrior slightly outperforms FedPersonal. The evidence supports a particular low-speed identification limitation, rather than a universal benefit whenever a speed is absent locally.</p></section>
<section><h2>4. Why grouping is not supported by this pilot</h2>{groupplot}
<p>Group numbers are arbitrary within each fleet. Handling vehicles form one group in all three fleets. Nominal and heavy vehicles mostly share a group, while the third group contains one or two vehicles trained at 13 m/s. Fleets 1302 and 1303 contain a singleton group, which supplies that vehicle no peer information through its group center.</p>
<p>This pattern is consistent with clusters reflecting local estimation uncertainty as well as physical differences. It does not prove a causal explanation. The grouped regularized training objective is lower, but peer-speed prediction is worse than both the single learned prior and the nominal prior. The more flexible training objective is therefore insufficient evidence for better transfer.</p>
<p>FedGroups improves 57/108 individual peer-speed cases versus FedPersonal, yet increases their mean error by 55.1%. The larger deteriorations dominate; a case-win count alone would hide this outcome. K=3 was fixed before running, and three groups are not established as necessary.</p></section>
<section><h2>5. Convergence and physical interpretation</h2>{table(t['selected'])}
<p>Selected Global fits converged in all fleets. FedPersonal reached the eight-round budget in fleets 1301/1302; its last maximum log-parameter steps were 0.00110/0.00127 against a 0.001 stopping threshold. FedGroups reached the budget in fleets 1301/1303, with last steps 0.00109/0.00453. All local updates in these selected rounds converged, and all saved alternating objectives decreased. These finite-budget estimates must not be presented as converged solutions.</p>
<p>All local sensitivity matrices have numerical rank six under the configured relative threshold. This is not proof of practical or global parameter identifiability: friction reaches a bound in 14/27 selected local fits. The current comparison concerns estimation/generalization under finite noisy data, not demonstrated structural nonidentifiability.</p>
<p>True relaxation lengths are identical across all vehicles (0.4/0.7 m). The fixed nominal prior uses 0.6/0.6 m, while learned fleet centers approach the shared true lengths. This offers a plausible source of the incremental benefit, but the experiment does not isolate it from sharing the three CT ratios. Equal true relaxation lengths favor transfer and must remain explicit.</p>
<p>Global's poor result establishes that one identical model is unsuitable for this fleet under this protocol. It does not establish that grouping is required: personalized models around one shared center work much better.</p></section>
<section><h2>6. Decision and next step</h2><p><b>Keep FedPersonal as the candidate and NominalPrior as a mandatory baseline. Do not advance FedGroups to a larger study on this evidence.</b> The pilot establishes a useful effect of regularization; the incremental advantage of learning from peers is smaller and inconsistent across the three fleets.</p>
<p>First complete a separately versioned convergence continuation, with unchanged data, objective and evaluation, preserving these original checkpoints. Then, if the advantage persists, compare matched peer counts and family composition with complementary-speed versus same-speed peers. That would test whether other speeds supply the missing information, rather than merely supplying more noisy parameter estimates. A targeted sharing ablation can subsequently separate common relaxation-length learning from sharing CT ratios.</p>
<p>No new experiment was run during this review. No controller claim follows from these results, and the earlier prediction-to-control issue remains open. Three reused pilot fleets do not support a confirmatory population claim. The manuscript remains unchanged.</p>
<p><a href="federated_relaxation_coverage.html">Original generated result report</a> · <a href="../docs/federated_relaxation_coverage.html">Frozen protocol</a></p></section></main></html>'''
    REPORT.write_text('\n'.join(line.rstrip() for line in doc.splitlines())+'\n',encoding='utf-8',newline='\n')
    return summary,byfleet,byspeed


def main():
    cfg=run.preflight();tables,info=audit(cfg)
    summary,byfleet,byspeed=render(tables,info)
    for name in ['paired','clients','histories','selected','replays']:
        tables[name].to_csv(run.OUT/f'{PREFIX}_{name}.csv',index=False)
    for name,frame in [('regions',summary),('peer_fleets',byfleet),('peer_training_speeds',byspeed)]:
        frame.to_csv(run.OUT/f'{PREFIX}_{name}.csv')
    run.storage.atomic_json(run.OUT/f'{PREFIX}.json',info)
    print(REPORT)


if __name__=='__main__':main()
