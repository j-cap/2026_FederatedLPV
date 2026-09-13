"""9G: finite-data personalized federated LPV identification."""
from concurrent.futures import ProcessPoolExecutor
import argparse,gzip,hashlib,json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from sklearn.metrics import adjusted_rand_score
from federated_lpv.output_error import fit,prepare,predict
from federated_lpv.privacy import normalize_log_parameters
from federated_lpv.uncertainty_clustering import log_parameter_covariance,normalized_covariance
import experiment_9f_personalized_manifold as ninef
import experiment_9c_uncertainty_groups as ninec

ROOT=ninef.ROOT;OUT=ninef.OUT;base=ninef.base;corrected=ninef.corrected;nineb=ninef.nineb;ninea=ninef.ninea;nined=ninef.nined
CONFIG=ROOT/'code/config/experiment_9g.json'


def coefficient_fit(records,center,directions,cfg):
    data=prepare(records);scales=np.deg2rad(np.asarray(cfg['noise_deg']));rank=directions.shape[1]
    def residual(a):
        output=((predict(np.exp(center+directions@a),data)-data['y'])/scales).ravel()
        penalty=np.sqrt(output.size*cfg['coefficient_prior_strength'])*a/cfg['coefficient_prior_scale']
        return np.r_[output,penalty]
    starts=[np.zeros(rank)]+[np.eye(rank)[j]*s for j in range(rank) for s in (-.05,.05)]
    solutions=[least_squares(residual,a,ftol=1e-9,xtol=1e-9,gtol=1e-9,max_nfev=100) for a in starts];result=min(solutions,key=lambda x:x.cost)
    return np.exp(center+directions@result.x),result.x,dict(success=bool(result.success),cost=float(result.cost),nfev=result.nfev)


def validation_loss(parameters,records,cfg):
    data=prepare(records);error=(predict(parameters,data)-data['y'])/np.deg2rad(np.asarray(cfg['noise_deg']))
    return float(np.mean(error**2))


def build_backbones(parameters,labels,n_groups,cfg):
    backbones={}
    for group in range(n_groups):
        z=np.log(np.asarray(parameters)[labels==group]);center=z.mean(0);weight=ninef.gain_metric(center,cfg)
        for rank in cfg['ranks']:
            mu,directions,_=ninef.learn_basis(z,weight,rank);backbones[group,rank]=(mu,directions)
    return backbones


def personalize(records,backbones,rank,cfg):
    candidates=[]
    for group in sorted({g for g,r in backbones if r==rank}):
        center,directions=backbones[group,rank];p,_,d=coefficient_fit(records[:cfg['initial_records']],center,directions,cfg);loss=validation_loss(p,records[cfg['initial_records']:],cfg);candidates.append((loss,group,d))
    _,group,_=min(candidates,key=lambda x:x[0]);center,directions=backbones[group,rank];p,a,d=coefficient_fit(records,center,directions,cfg)
    return p,group,a,d,min(x[0] for x in candidates)


def run_seed(seed):
    cfg=json.loads(CONFIG.read_text());train=nineb.discrete_fleet(seed,cfg['train_counts']);test=nineb.discrete_fleet(seed+10000,cfg['test_counts']);train_records,_=nined.collect_pool(train,seed+30000,cfg);test_records,nominal=nined.collect_pool(test,seed+40000,cfg);bounds=np.asarray(cfg['bounds']);train_p=[];train_cov=[];fit_rows=[];used_test_records={}
    for scope,clients,records in [('train',train,train_records),('test',test,test_records)]:
        for c in clients:
            (count,p,cov,condition,d),_=nined.fit_policy(records[c.client_id],cfg)
            if scope=='train':train_p.append(p);train_cov.append(cov)
            else:used_test_records[c.client_id]=records[c.client_id][:count]
            fit_rows.append(dict(seed=seed,scope=scope,client=c.client_id,family=c.family,success=d['success'],records=count,seconds=nined.calibration_seconds(count,cfg),parameter_error=float(np.linalg.norm(np.log(p/nineb.ratios(c))))))
    train_p=np.asarray(train_p);train_cov=np.asarray(train_cov);z=normalize_log_parameters(train_p,bounds);mixture,candidates=ninec.select_mixture(z,train_cov,cfg,seed);learned_labels=np.argmax(mixture['responsibility'],axis=1);truth=np.array([base.FAMILIES.index(c.family) for c in train]);oracle_backbones=build_backbones(train_p,truth,len(base.FAMILIES),cfg);learned_backbones=build_backbones(train_p,learned_labels,mixture['k'],cfg)
    model_maps={};assignments=[]
    local={}
    for c in test:local[c.client_id]=fit(used_test_records[c.client_id])[0]
    model_maps['Local']=local;model_maps['ExactIndividual']={c.client_id:nineb.ratios(c) for c in test}
    for source,backbones in [('Learned',learned_backbones),('OracleGroup',oracle_backbones)]:
        for rank in cfg['ranks']:
            method=f'{source}Rank{rank}';model_maps[method]={}
            for c in test:
                p,group,a,d,loss=personalize(used_test_records[c.client_id],backbones,rank,cfg);model_maps[method][c.client_id]=p;assignments.append(dict(seed=seed,method=method,client=c.client_id,family=c.family,selected_group=group,coefficient_1=a[0],coefficient_2=np.nan if rank==1 else a[1],validation_loss=loss,success=d['success']))
    controls={m:{c.client_id:ninea.eightg.eightf.eighte.eightd.prior.eight.seven.seven.controller(models[c.client_id]) for c in test} for m,models in model_maps.items()};rows=[];diagnostics=[]
    for method,models in model_maps.items():
        rank=3 if method in ('Local','ExactIndividual') else int(method[-1]);groups=0 if method in ('Local','ExactIndividual') else (mixture['k'] if method.startswith('Learned') else 3)
        for c in test:diagnostics.append(dict(seed=seed,method=method,client=c.client_id,family=c.family,groups=groups,local_coordinates=rank,parameter_error=float(np.linalg.norm(np.log(models[c.client_id]/nineb.ratios(c)))),gain_error=ninef.ninee.gain_error(controls[method][c.client_id],controls['ExactIndividual'][c.client_id])))
    t=np.arange(0,24+base.DT,base.DT)
    for scenario in cfg['test_scenarios']:
        v,ref=corrected.scenario(scenario,t/2);v=corrected.six.remap(v,(10,30))
        for method,control_map in controls.items():
            _,_,metrics=corrected.simulate(test,nominal,v,ref,'vy',client_controllers=control_map)
            for j,c in enumerate(test):rows.append(dict(seed=seed,method=method,scenario=scenario,client=c.client_id,family=c.family,**{q:float(x[j]) for q,x in metrics.items()}))
    clusters=[dict(seed=seed,selected_k=mixture['k'],train_ari=adjusted_rand_score(truth,learned_labels),candidate_k=x['k'],bic=x['bic'],minimum_size=x['minimum_size'],admissible=x['admissible']) for x in candidates]
    for suffix,items in [('fits',fit_rows),('assignments',assignments),('diagnostics',diagnostics),('clusters',clusters),('clients',rows)]:
        (OUT/f'experiment_9g_seed{seed}_{suffix}.csv.gz').write_bytes(gzip.compress(pd.DataFrame(items).to_csv(index=False).encode(),mtime=0))
    print(seed,flush=True)


def summarize():
    cfg=json.loads(CONFIG.read_text());raw=pd.concat([pd.read_csv(OUT/f'experiment_9g_seed{s}_clients.csv.gz') for s in cfg['seeds']]);diag=pd.concat([pd.read_csv(OUT/f'experiment_9g_seed{s}_diagnostics.csv.gz') for s in cfg['seeds']]);fits=pd.concat([pd.read_csv(OUT/f'experiment_9g_seed{s}_fits.csv.gz') for s in cfg['seeds']]);clusters=pd.concat([pd.read_csv(OUT/f'experiment_9g_seed{s}_clusters.csv.gz') for s in cfg['seeds']]);assign=pd.concat([pd.read_csv(OUT/f'experiment_9g_seed{s}_assignments.csv.gz') for s in cfg['seeds']])
    units=raw.groupby(['seed','method']).agg(tracking=('tracking','mean'),feasible=('feasible','min')).reset_index();summary=units.groupby('method').agg(tracking=('tracking','mean'),tracking_std=('tracking','std'),feasible_rate=('feasible','mean')).reset_index().merge(diag.groupby('method').agg(groups=('groups','mean'),local_coordinates=('local_coordinates','mean'),parameter_error=('parameter_error','mean'),gain_error=('gain_error','mean')).reset_index(),on='method');s=summary.set_index('method');local=s.loc['Local','tracking'];comparisons=pd.DataFrame([dict(method=m,tracking_vs_local_pct=100*(r.tracking/local-1)) for m,r in s.iterrows()]);selected=clusters[clusters.candidate_k==clusters.selected_k].groupby('seed').first().reset_index();cluster_summary=pd.DataFrame([dict(k_mean=selected.selected_k.mean(),k_min=selected.selected_k.min(),k_max=selected.selected_k.max(),train_ari=selected.train_ari.mean())]);rank1=bool(s.loc['LearnedRank1','tracking']<=1.02*local);rank2=bool(s.loc['LearnedRank2','tracking']<=local);conclusions=dict(provenance={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [CONFIG,ROOT/'code/experiments/experiment_9g_finite_personalized.py']},fits=len(fits),all_fits_converged=bool(fits.success.all() and assign.success.all()),closed_loop_evaluations=len(raw),all_feasible=bool(raw.feasible.all()),rank1_gate=rank1,rank2_gate=rank2)
    summary.to_csv(OUT/'experiment_9g_summary.csv',index=False);comparisons.to_csv(OUT/'experiment_9g_comparisons.csv',index=False);cluster_summary.to_csv(OUT/'experiment_9g_cluster_summary.csv',index=False);units.to_csv(OUT/'experiment_9g_seed_summary.csv',index=False);(OUT/'experiment_9g_conclusions.json').write_text(json.dumps(conclusions,indent=2)+'\n')
    order=['Local','LearnedRank1','LearnedRank2','OracleGroupRank1','OracleGroupRank2','ExactIndividual'];view=s.loc[order];fig,ax=plt.subplots(figsize=(8.2,4.1));ax.bar(np.arange(len(order)),view.tracking,color=['#d9822b','#2878b5','#16537e','#8ebad9','#5b9bd5','#3c9d5d']);ax.set(xticks=np.arange(len(order)),xticklabels=['Local','Learned r1','Learned r2','Oracle-group r1','Oracle-group r2','Exact indiv.'],ylabel='Tracking RMSE [rad/s]');ax.tick_params(axis='x',rotation=20);ax.grid(axis='y',alpha=.2);fig.tight_layout();fig.savefig(ROOT/'results/figures/experiment_9g_finite_personalized.pdf');plt.close(fig)
    print(json.dumps(conclusions,indent=2));print(summary.to_string(index=False));print(comparisons.to_string(index=False));print(cluster_summary.to_string(index=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=5);p.add_argument('--summarize-only',action='store_true');a=p.parse_args();cfg=json.loads(CONFIG.read_text())
    if not a.summarize_only:
        with ProcessPoolExecutor(max_workers=a.workers) as pool:list(pool.map(run_seed,cfg['seeds']))
    summarize()
