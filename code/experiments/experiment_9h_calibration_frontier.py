"""9H: matched-prefix calibration-efficiency frontier for personalization."""
from concurrent.futures import ProcessPoolExecutor
import argparse,gzip,hashlib,json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score
from federated_lpv.output_error import fit
from federated_lpv.privacy import normalize_log_parameters
from federated_lpv.uncertainty_clustering import log_parameter_covariance,normalized_covariance
import experiment_9g_finite_personalized as nineg
import experiment_9c_uncertainty_groups as ninec

ROOT=nineg.ROOT;OUT=nineg.OUT;base=nineg.base;corrected=nineg.corrected;nineb=nineg.nineb;ninea=nineg.ninea;nined=nineg.nined
CONFIG=ROOT/'code/config/experiment_9h.json'


def prefix_records(records,budget,dt=base.DT):
    remaining=int(round(budget/dt));selected=[]
    for record in records:
        if remaining<=0:break
        n=min(remaining,len(record['input']))
        if n: selected.append(dict(state=record['state'][:n+1],input=record['input'][:n],speed=record['speed'][:n+1]))
        remaining-=n
    if remaining:raise ValueError('requested budget exceeds available records')
    return selected


def select_personalized(records,backbones,rank,cfg):
    candidates=[]
    for group in sorted({g for g,r in backbones if r==rank}):
        center,directions=backbones[group,rank];p,a,d=nineg.coefficient_fit(records,center,directions,cfg);loss=nineg.validation_loss(p,records,cfg);candidates.append((loss,group,p,a,d))
    return min(candidates,key=lambda x:x[0])


def select_backbone(records,backbones,cfg):
    candidates=[]
    for group in sorted({g for g,r in backbones if r==1}):
        center,_=backbones[group,1];p=np.exp(center);candidates.append((nineg.validation_loss(p,records,cfg),group,p))
    return min(candidates,key=lambda x:x[0])


def run_seed(seed):
    cfg=json.loads(CONFIG.read_text());train=nineb.discrete_fleet(seed,cfg['train_counts']);test=nineb.discrete_fleet(seed+10000,cfg['test_counts']);train_records,_=nined.collect_pool(train,seed+50000,cfg);test_records,nominal=nined.collect_pool(test,seed+60000,cfg);bounds=np.asarray(cfg['bounds']);train_p=[];train_cov=[];fits=[]
    for c in train:
        p,d=fit(train_records[c.client_id]);cov,_=log_parameter_covariance(train_records[c.client_id],p,cfg['noise_deg']);train_p.append(p);train_cov.append(normalized_covariance(cov,bounds));fits.append(dict(seed=seed,scope='train',budget=max(cfg['calibration_budgets']),client=c.client_id,family=c.family,success=d['success']))
    train_p=np.asarray(train_p);train_cov=np.asarray(train_cov);z=normalize_log_parameters(train_p,bounds);mixture,candidates=ninec.select_mixture(z,train_cov,cfg,seed);labels=np.argmax(mixture['responsibility'],axis=1);truth=np.array([base.FAMILIES.index(c.family) for c in train]);learned=nineg.build_backbones(train_p,labels,mixture['k'],cfg);oracle=nineg.build_backbones(train_p,truth,3,cfg);assignments=[];diagnostics=[];rows=[];t=np.arange(0,24+base.DT,base.DT);scenarios=[]
    for scenario in cfg['test_scenarios']:
        v,ref=corrected.scenario(scenario,t/2);scenarios.append((scenario,corrected.six.remap(v,(10,30)),ref))
    for budget in cfg['calibration_budgets']:
        maps={m:{} for m in ('Local','LearnedBackbone','LearnedRank1','LearnedRank2','OracleGroupRank2','ExactIndividual')}
        for c in test:
            records=prefix_records(test_records[c.client_id],budget);p,d=fit(records);maps['Local'][c.client_id]=p;fits.append(dict(seed=seed,scope='test',budget=budget,client=c.client_id,family=c.family,success=d['success']))
            _,group,p0=select_backbone(records,learned,cfg);maps['LearnedBackbone'][c.client_id]=p0
            for rank in (1,2):
                loss,g,p,a,d=select_personalized(records,learned,rank,cfg);maps[f'LearnedRank{rank}'][c.client_id]=p;assignments.append(dict(seed=seed,budget=budget,method=f'LearnedRank{rank}',client=c.client_id,family=c.family,selected_group=g,success=d['success'],validation_loss=loss))
            family=base.FAMILIES.index(c.family);center,directions=oracle[family,2];p,a,d=nineg.coefficient_fit(records,center,directions,cfg);maps['OracleGroupRank2'][c.client_id]=p;maps['ExactIndividual'][c.client_id]=nineb.ratios(c)
        controls={m:{c.client_id:ninea.eightg.eightf.eighte.eightd.prior.eight.seven.seven.controller(models[c.client_id]) for c in test} for m,models in maps.items()}
        for method,models in maps.items():
            rank={'LearnedBackbone':0,'LearnedRank1':1,'LearnedRank2':2,'OracleGroupRank2':2}.get(method,3)
            for c in test:diagnostics.append(dict(seed=seed,budget=budget,method=method,client=c.client_id,family=c.family,local_coordinates=rank,parameter_error=float(np.linalg.norm(np.log(models[c.client_id]/nineb.ratios(c)))),gain_error=nineg.ninef.ninee.gain_error(controls[method][c.client_id],controls['ExactIndividual'][c.client_id])))
        for scenario,v,ref in scenarios:
            for method,control_map in controls.items():
                _,_,metrics=corrected.simulate(test,nominal,v,ref,'vy',client_controllers=control_map)
                for j,c in enumerate(test):rows.append(dict(seed=seed,budget=budget,method=method,scenario=scenario,client=c.client_id,family=c.family,**{q:float(x[j]) for q,x in metrics.items()}))
    clusters=[dict(seed=seed,selected_k=mixture['k'],train_ari=adjusted_rand_score(truth,labels),candidate_k=x['k'],bic=x['bic'],minimum_size=x['minimum_size'],admissible=x['admissible']) for x in candidates]
    for suffix,items in [('fits',fits),('assignments',assignments),('diagnostics',diagnostics),('clusters',clusters),('clients',rows)]:
        (OUT/f'experiment_9h_seed{seed}_{suffix}.csv.gz').write_bytes(gzip.compress(pd.DataFrame(items).to_csv(index=False).encode(),mtime=0))
    print(seed,flush=True)


def summarize():
    cfg=json.loads(CONFIG.read_text());raw=pd.concat([pd.read_csv(OUT/f'experiment_9h_seed{s}_clients.csv.gz') for s in cfg['seeds']]);diag=pd.concat([pd.read_csv(OUT/f'experiment_9h_seed{s}_diagnostics.csv.gz') for s in cfg['seeds']]);fits=pd.concat([pd.read_csv(OUT/f'experiment_9h_seed{s}_fits.csv.gz') for s in cfg['seeds']]);assign=pd.concat([pd.read_csv(OUT/f'experiment_9h_seed{s}_assignments.csv.gz') for s in cfg['seeds']]);clusters=pd.concat([pd.read_csv(OUT/f'experiment_9h_seed{s}_clusters.csv.gz') for s in cfg['seeds']])
    units=raw.groupby(['seed','budget','method']).agg(tracking=('tracking','mean'),feasible=('feasible','min')).reset_index();summary=units.groupby(['budget','method']).agg(tracking=('tracking','mean'),tracking_std=('tracking','std'),feasible_rate=('feasible','mean')).reset_index().merge(diag.groupby(['budget','method']).agg(local_coordinates=('local_coordinates','mean'),parameter_error=('parameter_error','mean'),gain_error=('gain_error','mean')).reset_index(),on=['budget','method']);pivot=summary.pivot(index='budget',columns='method',values='tracking');reference=float(pivot.loc[cfg['local_reference_budget'],'Local']);target=(1+cfg['quality_tolerance_pct']/100)*reference;efficiency=[]
    for method in pivot.columns:
        reached=pivot.index[pivot[method]<=target];efficiency.append(dict(method=method,target_tracking=target,time_to_target=np.nan if len(reached)==0 else float(reached.min()),tracking_at_075=float(pivot.loc[.75,method]),tracking_at_125=float(pivot.loc[1.25,method]),tracking_at_175=float(pivot.loc[1.75,method])))
    efficiency=pd.DataFrame(efficiency);equal=[];local_curve=pivot['Local']
    for budget in pivot.index:
        for method in ('LearnedBackbone','LearnedRank1','LearnedRank2'):
            equal.append(dict(budget=budget,method=method,improvement_vs_local_pct=100*(1-pivot.loc[budget,method]/local_curve.loc[budget])))
    equal=pd.DataFrame(equal);log_budget=np.log(np.asarray(pivot.index,float));auc=pd.DataFrame([dict(method=m,log_time_average_tracking=float(np.trapezoid(pivot[m],log_budget)/(log_budget[-1]-log_budget[0]))) for m in pivot.columns]);rank2=bool(pivot.loc[cfg['rank2_target_budget'],'LearnedRank2']<=target);rank1=bool(pivot.loc[cfg['rank2_target_budget'],'LearnedRank1']<=target);selected=clusters[clusters.candidate_k==clusters.selected_k].groupby('seed').first();conclusions=dict(provenance={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [CONFIG,ROOT/'code/experiments/experiment_9h_calibration_frontier.py']},fits=len(fits),all_fits_converged=bool(fits.success.all() and assign.success.all()),closed_loop_evaluations=len(raw),all_feasible=bool(raw.feasible.all()),selected_k_mean=float(selected.selected_k.mean()),train_ari=float(selected.train_ari.mean()),local_reference_tracking=reference,target_tracking=target,rank2_primary_gate=rank2,rank1_at_target_budget=rank1)
    summary.to_csv(OUT/'experiment_9h_summary.csv',index=False);efficiency.to_csv(OUT/'experiment_9h_efficiency.csv',index=False);equal.to_csv(OUT/'experiment_9h_equal_budget.csv',index=False);auc.to_csv(OUT/'experiment_9h_auc.csv',index=False);units.to_csv(OUT/'experiment_9h_seed_summary.csv',index=False);(OUT/'experiment_9h_conclusions.json').write_text(json.dumps(conclusions,indent=2)+'\n')
    fig,ax=plt.subplots(figsize=(7.8,4.4));order=['Local','LearnedBackbone','LearnedRank1','LearnedRank2','OracleGroupRank2'];colors=['#d9822b','.55','#2878b5','#16537e','#8ebad9']
    for method,color in zip(order,colors):g=summary[summary.method==method];ax.plot(g.budget,g.tracking,'o-',label=method,color=color)
    ax.axhline(target,color='.25',ls='--',lw=1,label='Local 1.75 s + 2%');ax.set(xlabel='Calibration time [s]',ylabel='Tracking RMSE [rad/s]',xscale='log');ax.set_xticks(cfg['calibration_budgets'],labels=[str(x) for x in cfg['calibration_budgets']]);ax.grid(alpha=.2);ax.legend(fontsize=8,ncol=2);fig.tight_layout();fig.savefig(ROOT/'results/figures/experiment_9h_calibration_frontier.pdf');plt.close(fig)
    print(json.dumps(conclusions,indent=2));print(summary.to_string(index=False));print(efficiency.to_string(index=False));print(equal.to_string(index=False));print(auc.to_string(index=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=5);p.add_argument('--summarize-only',action='store_true');a=p.parse_args();cfg=json.loads(CONFIG.read_text())
    if not a.summarize_only:
        with ProcessPoolExecutor(max_workers=a.workers) as pool:list(pool.map(run_seed,cfg['seeds']))
    summarize()
