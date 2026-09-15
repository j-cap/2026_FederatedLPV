"""9J: distributed sufficient-statistic learning of personalized LPV backbones."""
from concurrent.futures import ProcessPoolExecutor
import argparse,gzip,hashlib,json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score
from federated_lpv.output_error import fit
from federated_lpv.privacy import normalize_log_parameters
from federated_lpv.uncertainty_clustering import log_parameter_covariance,normalized_covariance
from federated_lpv.federated_manifold import select_federated_mixture,aggregate_group_moments
import experiment_9h_calibration_frontier as nineh
import experiment_9c_uncertainty_groups as ninec

ROOT=nineh.ROOT;OUT=nineh.OUT;base=nineh.base;corrected=nineh.corrected;nineb=nineh.nineb;ninea=nineh.ninea;nined=nineh.nined;nineg=nineh.nineg;ninef=nineg.ninef
CONFIG=ROOT/'code/config/experiment_9j.json'

def basis_from_covariance(center,covariance,weight,rank):
    ev,eq=np.linalg.eigh(weight);root=eq@np.diag(np.sqrt(ev))@eq.T;invroot=eq@np.diag(1/np.sqrt(ev))@eq.T;values,vectors=np.linalg.eigh(root@covariance@root);return invroot@vectors[:,np.argsort(values)[::-1][:rank]]

def distributed_backbones(parameters,mixture,cfg):
    backbones={}
    for g,(count,total,outer) in enumerate(aggregate_group_moments(np.log(parameters),mixture['responsibility'])):
        center=total/count;covariance=outer/count-np.outer(center,center);weight=ninef.gain_metric(center,cfg)
        for rank in (1,2):backbones[g,rank]=(center,basis_from_covariance(center,covariance,weight,rank))
    return backbones

def run_seed(seed):
    cfg=json.loads(CONFIG.read_text());train=nineb.discrete_fleet(seed,cfg['train_counts']);test=nineb.discrete_fleet(seed+10000,cfg['test_counts']);train_records,_=nined.collect_pool(train,seed+81000,cfg);test_records,nominal=nined.collect_pool(test,seed+82000,cfg);bounds=np.asarray(cfg['bounds']);parameters=[];covariances=[];fits=[]
    for c in train:
        p,d=fit(train_records[c.client_id]);cov,_=log_parameter_covariance(train_records[c.client_id],p,cfg['noise_deg']);parameters.append(p);covariances.append(normalized_covariance(cov,bounds));fits.append(dict(seed=seed,scope='train',client=c.client_id,success=d['success']))
    parameters=np.asarray(parameters);covariances=np.asarray(covariances);x=normalize_log_parameters(parameters,bounds);central,_=ninec.select_mixture(x,covariances,cfg,seed);central_backbones=nineg.build_backbones(parameters,np.argmax(central['responsibility'],axis=1),central['k'],{**cfg,'ranks':[1,2]});systems={'Central':(central,central_backbones)};protocol=[];truth=np.array([base.FAMILIES.index(c.family) for c in train])
    for rate in cfg['participation_rates']:
        model,_=select_federated_mixture(x,covariances,cfg['candidate_clusters'],cfg['federated_rounds'],rate,seed,cfg['server_damping'],cfg['intrinsic_floor'],cfg['minimum_cluster_size']);name=f'Fed{int(100*rate)}';systems[name]=(model,distributed_backbones(parameters,model,cfg));k=model['k'];protocol.append(dict(seed=seed,method=name,participation=rate,selected_k=k,train_ari=adjusted_rand_score(truth,np.argmax(model['responsibility'],axis=1)),coverage=model['coverage'],client_messages=model['messages'],upload_bytes=model['messages']*17*k*cfg['float_bytes']+len(train)*10*k*cfg['float_bytes'],download_bytes=model['messages']*10*k*cfg['float_bytes']))
    rows=[];diagnostics=[];assign=[];t=np.arange(0,24+base.DT,base.DT)
    for budget in cfg['calibration_budgets']:
        rank=cfg['rank_by_budget'][str(budget)];maps={m:{} for m in [*systems,'Local','ExactIndividual']}
        for c in test:
            records=nineh.prefix_records(test_records[c.client_id],budget);p,d=fit(records);maps['Local'][c.client_id]=p;maps['ExactIndividual'][c.client_id]=nineb.ratios(c);fits.append(dict(seed=seed,scope='test',client=c.client_id,success=d['success']))
            for name,(_,backbones) in systems.items():loss,g,p,a,d=nineh.select_personalized(records,backbones,rank,cfg);maps[name][c.client_id]=p;assign.append(dict(seed=seed,budget=budget,method=name,client=c.client_id,success=d['success']))
        controls={m:{c.client_id:ninea.eightg.eightf.eighte.eightd.prior.eight.seven.seven.controller(models[c.client_id]) for c in test} for m,models in maps.items()}
        for method,models in maps.items():
            for c in test:diagnostics.append(dict(seed=seed,budget=budget,method=method,client=c.client_id,parameter_error=float(np.linalg.norm(np.log(models[c.client_id]/nineb.ratios(c)))),gain_error=ninef.ninee.gain_error(controls[method][c.client_id],controls['ExactIndividual'][c.client_id])))
        for scenario in cfg['test_scenarios']:
            v,ref=corrected.scenario(scenario,t/2);v=corrected.six.remap(v,(10,30))
            for method,control_map in controls.items():
                _,_,metrics=corrected.simulate(test,nominal,v,ref,'vy',client_controllers=control_map)
                for j,c in enumerate(test):rows.append(dict(seed=seed,budget=budget,method=method,scenario=scenario,client=c.client_id,**{q:float(y[j]) for q,y in metrics.items()}))
    repeat,_=select_federated_mixture(x,covariances,cfg['candidate_clusters'],cfg['federated_rounds'],1.,seed,cfg['server_damping'],cfg['intrinsic_floor'],cfg['minimum_cluster_size']);fed=systems['Fed100'][0];equiv=[dict(seed=seed,k_equal=repeat['k']==fed['k'],mean_difference=np.linalg.norm(repeat['means']-fed['means']),covariance_difference=np.linalg.norm(repeat['intrinsic']-fed['intrinsic']))]
    for suffix,items in [('fits',fits),('assignments',assign),('protocol',protocol),('equivalence',equiv),('diagnostics',diagnostics),('clients',rows)]:(OUT/f'experiment_9j_seed{seed}_{suffix}.csv.gz').write_bytes(gzip.compress(pd.DataFrame(items).to_csv(index=False).encode(),mtime=0))
    print(seed,flush=True)

def summarize():
    cfg=json.loads(CONFIG.read_text());load=lambda s:pd.concat([pd.read_csv(OUT/f'experiment_9j_seed{x}_{s}.csv.gz') for x in cfg['development_seeds']]);raw=load('clients');diag=load('diagnostics');fits=load('fits');assign=load('assignments');protocol=load('protocol');equiv=load('equivalence');units=raw.groupby(['seed','budget','method']).agg(tracking=('tracking','mean'),q95=('tracking',lambda x:np.quantile(x,.95)),feasible=('feasible','min')).reset_index();summary=units.groupby(['budget','method']).agg(tracking=('tracking','mean'),tracking_std=('tracking','std'),q95=('q95','mean'),feasible_rate=('feasible','mean')).reset_index().merge(diag.groupby(['budget','method']).agg(parameter_error=('parameter_error','mean'),gain_error=('gain_error','mean')).reset_index(),on=['budget','method']);central=summary[summary.method=='Central'][['budget','tracking']].rename(columns={'tracking':'central'});comparison=summary.merge(central,on='budget');comparison['vs_central_pct']=100*(comparison.tracking/comparison.central-1);comm=protocol.groupby('method').agg(participation=('participation','first'),selected_k=('selected_k','mean'),ari=('train_ari','mean'),coverage=('coverage','mean'),messages=('client_messages','mean'),upload_bytes=('upload_bytes','mean'),download_bytes=('download_bytes','mean')).reset_index();s=summary.set_index(['budget','method']);gate_a=bool(equiv.k_equal.all() and equiv.mean_difference.max()<1e-12 and equiv.covariance_difference.max()<1e-12);gate_b=bool(comparison[comparison.method=='Fed50'].vs_central_pct.max()<1 and comparison[comparison.method=='Fed20'].vs_central_pct.max()<2);gate_c=bool(all(s.loc[(b,'Fed20'),'tracking']<s.loc[(b,'Local'),'tracking'] for b in cfg['calibration_budgets']));conclusions=dict(provenance={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [CONFIG,ROOT/'code/src/federated_lpv/federated_manifold.py',ROOT/'code/experiments/experiment_9j_distributed_backbone.py']},fits=len(fits),all_fits_converged=bool(fits.success.all() and assign.success.all()),evaluations=len(raw),all_feasible=bool(raw.feasible.all()),gate_a=gate_a,gate_b=gate_b,gate_c=gate_c,reserved_confirmation_seeds=cfg['reserved_confirmation_seeds']);summary.to_csv(OUT/'experiment_9j_summary.csv',index=False);comparison.to_csv(OUT/'experiment_9j_comparisons.csv',index=False);comm.to_csv(OUT/'experiment_9j_communication.csv',index=False);units.to_csv(OUT/'experiment_9j_seed_summary.csv',index=False);(OUT/'experiment_9j_conclusions.json').write_text(json.dumps(conclusions,indent=2)+'\n');fig,ax=plt.subplots(figsize=(7.8,4.2));order=['Local','Central','Fed100','Fed50','Fed20'];x=np.arange(2);w=.15
    for j,m in enumerate(order):g=summary[summary.method==m].set_index('budget').loc[cfg['calibration_budgets']];ax.bar(x+(j-2)*w,g.tracking,w,label=m)
    ax.set(xticks=x,xticklabels=['0.75','1.25'],xlabel='Calibration time [s]',ylabel='Tracking RMSE [rad/s]');ax.legend(ncol=3,fontsize=8);ax.grid(axis='y',alpha=.2);fig.tight_layout();fig.savefig(ROOT/'results/figures/experiment_9j_distributed_backbone.pdf');plt.close(fig);print(json.dumps(conclusions,indent=2));print(summary.to_string(index=False));print(comparison[['budget','method','vs_central_pct']].to_string(index=False));print(comm.to_string(index=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=5);p.add_argument('--summarize-only',action='store_true');a=p.parse_args();cfg=json.loads(CONFIG.read_text())
    if not a.summarize_only:
        with ProcessPoolExecutor(max_workers=a.workers) as pool:list(pool.map(run_seed,cfg['development_seeds']))
    summarize()
