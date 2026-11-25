"""Audit the frozen convergence continuation without fitting or modifying inputs."""
import io
from pathlib import Path
import numpy as np
import pandas as pd
import paper_federated_relaxation_continue as run

PREFIX=run.PREFIX+'_review'
REPORT=run.ROOT/'report/federated_relaxation_continue_review.html'


def main():
    cfg,protocol=run.preflight();rows=[];convergence=[];replays=[]
    files=list(run.OUT.glob(run.PREFIX+'*'))
    files=[p for p in files if '_review' not in p.name]+[run.REPORT]
    hashes={str(p.relative_to(run.ROOT)):run.base.canonical_hash(p) for p in files}
    for seed in protocol['seeds']:
        source=run.load_source(seed,cfg,protocol)
        p=run.base.storage.read_checkpoint(run.path(seed));run.validate(p,source,cfg,protocol,execution=False)
        assert p['status']=='execution_complete'
        state=p['optimization'];old=run.source_state(source);h=state['history'];last=h[-1]
        assert state['status']=='converged' and not state['pending']
        objectives=np.array([x['objective'] for x in h])
        assert np.all(np.diff(objectives)<=1e-6*np.maximum(1.,abs(objectives[:-1])))
        relative=abs(h[-2]['objective']-last['objective'])/max(1.,abs(h[-2]['objective']))
        assert relative<cfg['sharing']['personal_relative_tolerance']
        assert last['max_log_step']<cfg['sharing']['personal_step_tolerance']
        assert last['all_local_optimizers_converged'] and not last['assignments_changed']
        assert all(x['optimizer_converged'] for x in last['local_updates'])
        z=np.array(state['log_parameters']);eta=np.array(state['centers']);w=np.array(cfg['sharing']['weights'])
        assert state['assignments']==[0]*len(z)
        np.testing.assert_allclose(eta[0],z.mean(axis=0),rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(last['penalty'],np.sum(((z-eta[0])*w)**2),rtol=1e-12)
        np.testing.assert_allclose(last['objective'],last['data_sse']+last['penalty'],rtol=1e-12)
        if old['status']=='converged':assert state==old
        change=float(np.max(abs(np.expm1(z-np.array(old['log_parameters'])))))*100
        convergence.append(dict(seed=seed,original_rounds=old['round'],final_rounds=state['round'],
            added_rounds=state['round']-old['round'],status=state['status'],relative_objective_change=relative,
            max_log_step=last['max_log_step'],max_parameter_change_pct=change,
            extra_float64_bytes=8*(state['communication_float64']-old['communication_float64'])))
        for key,entry in p['evaluation'].items():
            previous=source['evaluation'][key]
            assert entry['source_trace_sha256']==previous['trace_sha256']
            assert entry['reused']==(old['status']=='converged')
            if entry['reused']:assert entry['metrics']==previous['methods']['FedPersonal']
            methods={m:previous['methods'][m] for m in ['LocalOE','NominalPrior']}
            methods.update(FedPersonalBefore=previous['methods']['FedPersonal'],FedPersonalAfter=entry['metrics'])
            for method,metrics in methods.items():
                assert metrics['status']=='complete'
                assert all(np.isfinite(metrics[k]) for k in ['yaw_rmse_deg_s','measured_yaw_rmse_deg_s','beta_rmse_deg'])
                rows.append(dict(seed=seed,client=previous['client'],speed=previous['speed'],probe=previous['probe'],
                    region=previous['region'],method=method,**metrics))
        # Numerical replay of four saved test cases; no new test conditions.
        if old['status']!='converged':
            clients=run.base.fleet_for(cfg,seed)['clients'];i=next(i for i,c in enumerate(clients) if c['client']=='vehicle_00');c=clients[i]
            for probe in ['slow','broadband']:
                key=f'vehicle_00_v21_{probe}';arrays,_=run.base.generate(c,seed,probe,21,cfg)
                pred=run.base.batch_predict(np.exp(z[i]),[arrays['steering']],[21],c['geometry_m'],
                    cfg['sample_time'],cfg['integration_max_step'])[0,0]
                rms=lambda x:float(np.rad2deg(np.sqrt(np.mean(x*x))))
                metrics=dict(yaw_rmse_deg_s=rms(pred[:,1]-arrays['state'][:,1]),
                    measured_yaw_rmse_deg_s=rms(pred[:,1]-arrays['measured_yaw']),beta_rmse_deg=rms(pred[:,0]-arrays['state'][:,0]))
                saved=p['evaluation'][key];difference=max(abs(v-saved['metrics'][k]) for k,v in metrics.items())
                np.testing.assert_allclose(list(metrics.values()),[saved['metrics'][k] for k in metrics],rtol=1e-8,atol=1e-10)
                arrays['predicted_FedPersonal']=pred
                replays.append(dict(seed=seed,case=key,max_metric_difference=difference,
                    exact_trace_match=run.base.storage.trace_hash(arrays)==saved['trace_sha256']))
    cases=pd.DataFrame(rows);saved=pd.read_csv(run.OUT/f'{run.PREFIX}_cases.csv');keys=['seed','client','speed','probe','method']
    pd.testing.assert_frame_equal(cases.sort_values(keys).reset_index(drop=True),saved.sort_values(keys).reset_index(drop=True),
        check_exact=False,rtol=1e-10,atol=1e-12)
    summary=cases.groupby(['seed','region','method'],as_index=False).yaw_rmse_deg_s.mean()
    pd.testing.assert_frame_equal(summary,pd.read_csv(run.OUT/f'{run.PREFIX}_summary.csv'),check_exact=False,rtol=1e-10,atol=1e-12)
    regions=cases.pivot_table(index='region',columns='method',values='yaw_rmse_deg_s')
    regions['gain_vs_local_pct']=100*(1-regions.FedPersonalAfter/regions.LocalOE)
    regions['gain_vs_nominal_pct']=100*(1-regions.FedPersonalAfter/regions.NominalPrior)
    regions['change_vs_before_pct']=100*(regions.FedPersonalAfter/regions.FedPersonalBefore-1)
    peers=cases[cases.region=='peer_observed'].pivot_table(index='seed',columns='method',values='yaw_rmse_deg_s')
    peers['gain_vs_nominal_pct']=100*(1-peers.FedPersonalAfter/peers.NominalPrior)
    pairs=cases[cases.region=='peer_observed'].pivot(index=['seed','client','speed','probe'],columns='method',values='yaw_rmse_deg_s')
    vehicle=pairs.groupby(['seed','client']).mean()
    info=dict(status='audit_passed',source_files=hashes,source_checkpoints=protocol['source_checkpoints'],
        cases=len(cases),personalized_cases=len(pairs),all_fleets_converged=True,new_fits=0,new_test_conditions=0,
        replayed_saved_cases=len(replays),exact_replayed_traces=sum(x['exact_trace_match'] for x in replays),
        peer_case_wins_vs_nominal=int((pairs.FedPersonalAfter<pairs.NominalPrior).sum()),
        peer_vehicle_wins_vs_nominal=int((vehicle.FedPersonalAfter<vehicle.NominalPrior).sum()),
        numerical_replay_max_metric_difference=max(x['max_metric_difference'] for x in replays),
        audit_runtime=run.base.identity(cfg)['runtime'])
    assert hashes=={name:run.base.canonical_hash(run.ROOT/name) for name in hashes}
    tables=dict(convergence=pd.DataFrame(convergence),regions=regions.reset_index(),peers=peers.reset_index(),replays=pd.DataFrame(replays))
    render(tables,info)
    for name,frame in tables.items():frame.to_csv(run.OUT/f'{PREFIX}_{name}.csv',index=False)
    run.base.storage.atomic_json(run.OUT/f'{PREFIX}.json',info)
    print(regions.to_string());print(peers.to_string());print(info['status']);print(REPORT)


def render(t,info):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib.rcParams['svg.hashsalt']=PREFIX
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    peers=t['peers']
    for method,color,style in [('LocalOE','#67778a','-'),('NominalPrior','#b57c1d','-'),
                               ('FedPersonalBefore','#9c74b6','--'),('FedPersonalAfter','#087d87','-')]:
        axes[0].plot(peers.seed.astype(str),peers[method],marker='o',color=color,linestyle=style,label=method)
    axes[0].set(title='Peer-speed prediction: before and after',ylabel='Clean yaw RMSE [deg/s]',xlabel='Fleet')
    axes[0].legend(fontsize=8);axes[0].grid(alpha=.2)
    axes[1].bar(peers.seed.astype(str),peers.gain_vs_nominal_pct,color=['#b75a55' if v<0 else '#087d87' for v in peers.gain_vs_nominal_pct])
    axes[1].axhline(0,color='#777',linewidth=1);axes[1].set(title='Final benefit beyond the nominal prior',ylabel='Reduction in mean yaw error [%]',xlabel='Fleet')
    buf=io.StringIO();fig.savefig(buf,format='svg',metadata={'Date':None});plt.close(fig)
    svg=buf.getvalue();svg=svg[svg.index('<svg'):]
    table=lambda frame:'<div class="scroll">'+frame.to_html(index=False,border=0,float_format=lambda v:f'{v:.7g}')+'</div>'
    doc=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Convergence check: personalized FL</title><style>body{{font:16px/1.65 system-ui;color:#173348;background:#edf2f5}}main{{max-width:1150px;margin:auto;padding:28px}}section{{background:white;border-radius:12px;padding:25px;margin:22px 0}}svg{{width:100%;height:auto}}h1,h2{{line-height:1.25}}table{{border-collapse:collapse;font-size:13px}}th,td{{padding:9px;border-bottom:1px solid #ddd;white-space:nowrap}}.scroll{{overflow:auto}}a{{color:#087d87}}</style><main>
<h1>Convergence is resolved; the scientific interpretation is unchanged</h1>
<p>Both unfinished fleets converged at round 9, after one extra round. The third fleet's converged round-5 solution was reused exactly. The mean peer-speed advantage over the nominal prior remains approximately 7%.</p>
<section><h2>1. The additional computation was sufficient</h2>{table(t['convergence'])}
<p>The saved final states satisfy the existing assignment, relative-objective, log-parameter-step and local-solver convergence checks. Each continued fleet added nine client updates. The largest individual parameter changes were below 0.073%. This is convergence under the prescribed numerical criteria, not a proof of a global optimum.</p></section>
<section><h2>2. Prediction errors barely changed</h2>{svg}{table(t['peers'])}{table(t['regions'])}
<p>At peer-observed speeds, the mean error decreases from 0.003950 to 0.003949 deg/s: approximately 0.027% relative improvement from continuing the optimizer. The final reduction is 49.5% versus LocalOE and 7.0% versus NominalPrior. These are ratios of mean errors on the same paired cases.</p>
<p>Compared with the nominal prior, the final personalized model is 9.0% worse in fleet 1301, 15.6% better in fleet 1302 and 17.0% better in fleet 1303. It improves {info['peer_case_wins_vs_nominal']}/108 peer-speed trajectory cases and {info['peer_vehicle_wins_vs_nominal']}/27 vehicle-averaged peer-speed errors. The incremental benefit is small and inconsistent across fleets. The fixed-prior baseline remains essential.</p>
<p>Local-speed mean error also remains slightly lower than the nominal-prior baseline. Better prediction does not establish better closed-loop tracking; no controller was tested in this continuation.</p></section>
<section><h2>3. What was checked?</h2><p>All three continuation checkpoints are complete. Their source digests and configuration/code locks match. Original optimization histories are preserved as exact prefixes; fleet 1303 and all reused baseline values are unchanged. All 1512 comparison rows are finite, and the case and summary CSVs match values reconstructed from checkpoints. The objective decreases through the recorded histories; the final center and penalty were independently recomputed from saved parameters.</p>
<p>Four existing test cases were replayed numerically and match their saved metrics within the audit tolerance. Exact trace hashes match for {info['exact_replayed_traces']}/4; numerical agreement and byte-exact recovery are distinct checks. No new fitting, new test conditions or original output modifications were performed.</p>{table(t['replays'])}</section>
<section><h2>4. The next scientific question</h2><p><b>Keep the converged personalized model as the candidate, and keep the nominal prior as the control.</b> The optimizer budget no longer explains the modest gain. Further round extensions are not the useful next step.</p>
<p>The next comparison should hold each target vehicle's local record fixed and match peer count, physical composition, record duration, input design, noise settings and regularization, while changing only whether peers cover complementary speeds or the target's own speed. Use matched realizations wherever possible, and report the paired target-level changes across independent fleets. Keep the target evaluation set unchanged; choose no hyperparameters from these test results.</p>
<p>This isolates information from complementary speed coverage from generic regularization and additional peer observations. Treat any new results on these same three fleets as development evidence. If complementary peers add no consistent benefit beyond same-speed peers and the nominal prior, the intended missing-speed-information claim is not supported by this plant and data regime.</p>
<p>The previously tested grouped method was not continued; this check settles only FedPersonal's convergence. No manuscript change or new experiment is included here.</p>
<p><a href="federated_relaxation_continue.html">Original continuation report</a> · <a href="federated_relaxation_coverage_review.html">Original pilot review</a></p></section></main></html>'''
    REPORT.write_text('\n'.join(line.rstrip() for line in doc.splitlines())+'\n',encoding='utf-8',newline='\n')


if __name__=='__main__':main()
