"""9E: causal audit of why per-client Local control is unusually strong."""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import argparse,gzip,hashlib,json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from federated_lpv.fleet import VehicleClient
from federated_lpv.output_error import fit,prepare,predict
import experiment_9d_adaptive_calibration as nined

ROOT=nined.ROOT;OUT=nined.OUT;base=nined.base;corrected=nined.corrected;nineb=nined.nineb;ninea=nined.ninea
CONFIG=ROOT/'code/config/experiment_9e.json'


def geometric_mean(values):return np.exp(np.mean(np.log(np.asarray(values)),axis=0))


def drift_client(client,scales):
    p=client.parameters
    return VehicleClient(client.client_id,client.family,replace(p,mass=p.mass*scales['mass'],yaw_inertia=p.yaw_inertia*scales['yaw_inertia'],front_stiffness=p.front_stiffness*scales['front_stiffness'],rear_stiffness=p.rear_stiffness*scales['rear_stiffness']))


def gain_error(control,reference):
    grid=np.linspace(10,30,161)
    def table(c):return np.column_stack([np.interp(grid,c.speeds,c.gains['global'][:,q]) for q in range(3)])
    a,b=table(control),table(reference)
    return float(np.linalg.norm(a-b)/np.linalg.norm(b))


def prediction_error(parameters,records):
    data=prepare(records);error=predict(parameters,data)-data['y'];scale=np.deg2rad([.05,.1])
    return float(np.sqrt(np.mean((error/scale)**2)))


def run_seed(seed):
    cfg=json.loads(CONFIG.read_text());clients=nineb.discrete_fleet(seed+10000,cfg['test_counts']);records,nominal=nined.collect_pool(clients,seed+20000,cfg);fitted={};fit_rows=[]
    for c in clients:
        (count,p,cov,condition,d),_=nined.fit_policy(records[c.client_id],cfg);fitted[c.client_id]=p
        fit_rows.append(dict(seed=seed,client=c.client_id,family=c.family,records=count,seconds=nined.calibration_seconds(count,cfg),success=d['success'],condition=condition,parameter_error=np.linalg.norm(np.log(p/nineb.ratios(c)))))
    families={f:[c for c in clients if c.family==f] for f in base.FAMILIES}
    family_models={f:geometric_mean([nineb.ratios(c) for c in cs]) for f,cs in families.items()};global_model=geometric_mean([nineb.ratios(c) for c in clients])
    model_maps={
        'Local':{c.client_id:fitted[c.client_id] for c in clients},
        'SwapWithinFamily':{},
        'ExactIndividual':{c.client_id:nineb.ratios(c) for c in clients},
        'ExactFamily':{c.client_id:family_models[c.family] for c in clients},
        'GlobalExact':{c.client_id:global_model for c in clients}}
    for cs in families.values():
        for j,c in enumerate(cs):model_maps['SwapWithinFamily'][c.client_id]=fitted[cs[(j+1)%len(cs)].client_id]
    controls={m:{c.client_id:ninea.eightg.eightf.eighte.eightd.prior.eight.seven.seven.controller(models[c.client_id]) for c in clients} for m,models in model_maps.items()}
    heldout={c.client_id:[] for c in clients};t=np.arange(0,24+base.DT,base.DT);scenarios=[]
    for name in cfg['test_scenarios']:
        v,ref=corrected.scenario(name,t/2);v=corrected.six.remap(v,(10,30));x,u,_=corrected.simulate(clients,nominal,v,ref,'vy');scenarios.append((name,v,ref))
        for j,c in enumerate(clients):heldout[c.client_id].append(dict(state=x[:,j],input=u[:,j],speed=v))
    diagnostics=[]
    for method,models in model_maps.items():
        for c in clients:
            exact=controls['ExactIndividual'][c.client_id]
            diagnostics.append(dict(seed=seed,method=method,client=c.client_id,family=c.family,parameter_error=float(np.linalg.norm(np.log(models[c.client_id]/nineb.ratios(c)))),gain_error=gain_error(controls[method][c.client_id],exact),prediction_error=prediction_error(models[c.client_id],heldout[c.client_id])))
    rows=[]
    for condition,scales in cfg['deployment_conditions'].items():
        deployed=[drift_client(c,scales) for c in clients];deployed_exact={c.client_id:ninea.eightg.eightf.eighte.eightd.prior.eight.seven.seven.controller(nineb.ratios(d)) for c,d in zip(clients,deployed)}
        condition_controls={**controls,'ExactDeployment':deployed_exact}
        for method,control_map in condition_controls.items():
            for scenario,v,ref in scenarios:
                _,_,metrics=corrected.simulate(deployed,nominal,v,ref,'vy',client_controllers=control_map)
                for j,c in enumerate(clients):rows.append(dict(seed=seed,condition=condition,method=method,scenario=scenario,client=c.client_id,family=c.family,**{q:float(x[j]) for q,x in metrics.items()}))
    for suffix,items in [('fits',fit_rows),('diagnostics',diagnostics),('clients',rows)]:
        (OUT/f'experiment_9e_seed{seed}_{suffix}.csv.gz').write_bytes(gzip.compress(pd.DataFrame(items).to_csv(index=False).encode(),mtime=0))
    print(seed,flush=True)


def summarize():
    cfg=json.loads(CONFIG.read_text());raw=pd.concat([pd.read_csv(OUT/f'experiment_9e_seed{s}_clients.csv.gz') for s in cfg['seeds']]);diag=pd.concat([pd.read_csv(OUT/f'experiment_9e_seed{s}_diagnostics.csv.gz') for s in cfg['seeds']]);fits=pd.concat([pd.read_csv(OUT/f'experiment_9e_seed{s}_fits.csv.gz') for s in cfg['seeds']])
    units=raw.groupby(['seed','condition','method']).agg(tracking=('tracking','mean'),steering=('steering_rms','mean'),beta=('beta_rms','mean'),feasible=('feasible','min')).reset_index();summary=units.groupby(['condition','method']).agg(tracking=('tracking','mean'),tracking_std=('tracking','std'),steering=('steering','mean'),beta=('beta','mean'),feasible_rate=('feasible','mean')).reset_index();dunits=diag.groupby(['seed','method']).agg(parameter_error=('parameter_error','mean'),gain_error=('gain_error','mean'),prediction_error=('prediction_error','mean')).reset_index();dsummary=dunits.groupby('method').agg(parameter_error=('parameter_error','mean'),gain_error=('gain_error','mean'),prediction_error=('prediction_error','mean')).reset_index()
    nominal_client=raw[raw.condition=='nominal'].groupby(['seed','method','client']).tracking.mean().reset_index().merge(diag,on=['seed','method','client']);cor=[]
    for method,g in nominal_client.groupby('method'):
        for metric in ('parameter_error','gain_error','prediction_error'):
            if g[metric].nunique()<2:rho,p=np.nan,np.nan
            else:rho,p=spearmanr(g[metric],g.tracking)
            cor.append(dict(method=method,metric=metric,spearman=float(rho) if np.isfinite(rho) else np.nan,p_value=float(p) if np.isfinite(p) else np.nan))
    comparisons=[]
    for condition,g in summary.groupby('condition'):
        s=g.set_index('method');local=s.loc['Local','tracking']
        for method in ('ExactIndividual','SwapWithinFamily','ExactFamily','GlobalExact','ExactDeployment'):
            comparisons.append(dict(condition=condition,method=method,tracking_vs_local_pct=100*(s.loc[method,'tracking']/local-1)))
    conclusions=dict(provenance={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [CONFIG,ROOT/'code/experiments/experiment_9e_local_advantage_audit.py']},fits=len(fits),all_fits_converged=bool(fits.success.all()),closed_loop_evaluations=len(raw),all_feasible=bool(raw.feasible.all()),purpose='diagnostic attribution; no positive-result gate')
    summary.to_csv(OUT/'experiment_9e_summary.csv',index=False);dsummary.to_csv(OUT/'experiment_9e_diagnostics.csv',index=False);pd.DataFrame(cor).to_csv(OUT/'experiment_9e_correlations.csv',index=False);pd.DataFrame(comparisons).to_csv(OUT/'experiment_9e_comparisons.csv',index=False);units.to_csv(OUT/'experiment_9e_seed_summary.csv',index=False);(OUT/'experiment_9e_conclusions.json').write_text(json.dumps(conclusions,indent=2)+'\n')
    view=summary[summary.method.isin(['Local','ExactIndividual','SwapWithinFamily','ExactFamily','GlobalExact'])];order=['Local','ExactIndividual','SwapWithinFamily','ExactFamily','GlobalExact'];conditions=list(cfg['deployment_conditions']);fig,ax=plt.subplots(figsize=(8.2,4.2));x=np.arange(len(conditions));w=.15
    for j,m in enumerate(order):ax.bar(x+(j-2)*w,view[view.method==m].set_index('condition').loc[conditions].tracking,w,label=m)
    ax.set(xticks=x,xticklabels=['Nominal','Tire fade','Load shift'],ylabel='Tracking RMSE [rad/s]');ax.grid(axis='y',alpha=.2);ax.legend(ncol=3,fontsize=8);fig.tight_layout();fig.savefig(ROOT/'results/figures/experiment_9e_local_advantage_audit.pdf');plt.close(fig)
    print(json.dumps(conclusions,indent=2));print(summary.to_string(index=False));print(dsummary.to_string(index=False));print(pd.DataFrame(comparisons).to_string(index=False));print(pd.DataFrame(cor).to_string(index=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=5);p.add_argument('--summarize-only',action='store_true');a=p.parse_args();cfg=json.loads(CONFIG.read_text())
    if not a.summarize_only:
        with ProcessPoolExecutor(max_workers=a.workers) as pool:list(pool.map(run_seed,cfg['seeds']))
    summarize()
