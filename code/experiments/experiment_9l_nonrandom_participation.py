"""9L: personalized federated LPV learning under biased client availability."""
from concurrent.futures import ProcessPoolExecutor
import argparse,gzip,hashlib,json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score
from federated_lpv.output_error import fit
from federated_lpv.privacy import normalize_log_parameters
from federated_lpv.uncertainty_clustering import log_parameter_covariance,normalized_covariance
from federated_lpv.federated_manifold import select_federated_mixture_cohorts,aggregate_group_moments
import experiment_9j_distributed_backbone as ninej

ROOT=ninej.ROOT;OUT=ninej.OUT;base=ninej.base;corrected=ninej.corrected;nineb=ninej.nineb;ninea=ninej.ninea;nined=ninej.nined;nineg=ninej.nineg;ninef=ninej.ninef
CONFIG=ROOT/'code/config/experiment_9l.json'


def calibration_categories(clients,collection_seed,cfg):
    rng=np.random.default_rng(collection_seed+910000);order=rng.permutation(len(clients))
    return np.array([int(order[j]%len(cfg['coverage_speeds'])) for j in range(len(clients))])


def availability_trace(regime,n,truth,categories,cfg,seed):
    rng=np.random.default_rng(seed);dropout_rng=np.random.default_rng(seed+777777);size=int(np.ceil(cfg['participation_rate']*n));rounds=cfg['federated_rounds']+1;kind=regime['kind']
    if kind=='persistent':
        chosen=np.sort(rng.choice(n,size,replace=False));return [chosen.copy() for _ in range(rounds)]
    traces=[]
    for _ in range(rounds):
        if kind=='family_skew':weights=np.where(truth==2,regime['minority_weight'],1.).astype(float)
        elif kind=='operating_skew':weights=np.linspace(1.,regime['high_speed_weight'],len(cfg['coverage_speeds']))[categories]
        else:weights=np.ones(n)
        invited=np.sort(rng.choice(n,size,replace=False,p=weights/weights.sum()))
        if kind=='dropout':
            received=invited[dropout_rng.random(len(invited))>=regime['dropout_probability']];minimum=2*max(cfg['candidate_clusters'])
            if len(received)<minimum:received=np.sort(dropout_rng.choice(invited,minimum,replace=False))
            traces.append(np.sort(received))
        else:traces.append(invited)
    return traces


def subset_backbones(parameters,model,cfg):
    indices=model['observed_indices'];logp=np.log(parameters[indices]);responsibility=model['responsibility'][indices];backbones={}
    for group,(count,total,outer) in enumerate(aggregate_group_moments(logp,responsibility)):
        center=total/count;covariance=outer/count-np.outer(center,center);weight=ninef.gain_metric(center,cfg)
        for rank in (1,2):backbones[group,rank]=(center,ninej.basis_from_covariance(center,covariance,weight,rank))
    return backbones


def run_seed(seed):
    cfg=json.loads(CONFIG.read_text());train=nineb.discrete_fleet(seed,cfg['train_counts']);test=nineb.discrete_fleet(seed+10000,cfg['test_counts']);train_records,_=nined.collect_pool(train,seed+81000,cfg);test_records,nominal=nined.collect_pool(test,seed+82000,cfg);bounds=np.asarray(cfg['bounds']);parameters=[];covariances=[];fit_rows=[]
    for client in train:
        estimate,diagnostic=fit(train_records[client.client_id]);covariance,_=log_parameter_covariance(train_records[client.client_id],estimate,cfg['noise_deg']);parameters.append(estimate);covariances.append(normalized_covariance(covariance,bounds));fit_rows.append(dict(seed=seed,scope='train',client=client.client_id,success=diagnostic['success']))
    parameters=np.asarray(parameters);covariances=np.asarray(covariances);x=normalize_log_parameters(parameters,bounds);truth=np.array([base.FAMILIES.index(client.family) for client in train]);categories=calibration_categories(train,seed+81000,cfg);systems={};protocol=[]
    for index,regime in enumerate(cfg['availability_regimes']):
        trace_seed=seed+91000 if regime['kind'] in ('uniform','dropout') else seed+91000+1000*index;cohorts=availability_trace(regime,len(train),truth,categories,cfg,trace_seed);model,_=select_federated_mixture_cohorts(x,covariances,cfg['candidate_clusters'],cohorts,cfg['server_damping'],cfg['intrinsic_floor'],cfg['minimum_cluster_size']);systems[regime['name']]=(model,subset_backbones(parameters,model,cfg));k=model['k'];observed=model['observed_indices'];observed_truth=truth[observed];labels=np.argmax(model['responsibility'][observed],axis=1);family_coverage=[np.mean(np.isin(np.flatnonzero(truth==group),observed)) for group in range(3)];category_coverage=[np.mean(np.isin(np.flatnonzero(categories==group),observed)) for group in range(len(cfg['coverage_speeds']))];final_messages=len(observed);protocol.append(dict(seed=seed,method=regime['name'],kind=regime['kind'],selected_k=k,train_ari=adjusted_rand_score(observed_truth,labels),coverage=model['coverage'],minimum_family_coverage=min(family_coverage),minimum_category_coverage=min(category_coverage),client_messages=model['messages']+final_messages,upload_bytes=model['messages']*17*k*cfg['float_bytes']+final_messages*10*k*cfg['float_bytes'],download_bytes=model['messages']*10*k*cfg['float_bytes']))
    rows=[];diagnostics=[];assignment_rows=[];t=np.arange(0,24+base.DT,base.DT)
    for budget in cfg['calibration_budgets']:
        rank=cfg['rank_by_budget'][str(budget)];maps={name:{} for name in [*systems,'Local','ExactIndividual']}
        for client in test:
            records=ninej.nineh.prefix_records(test_records[client.client_id],budget);local,diagnostic=fit(records);maps['Local'][client.client_id]=local;maps['ExactIndividual'][client.client_id]=nineb.ratios(client);fit_rows.append(dict(seed=seed,scope='test',client=client.client_id,success=diagnostic['success']))
            for name,(_,backbones) in systems.items():
                _,group,estimate,_,diagnostic=ninej.nineh.select_personalized(records,backbones,rank,cfg);maps[name][client.client_id]=estimate;assignment_rows.append(dict(seed=seed,budget=budget,method=name,client=client.client_id,group=group,success=diagnostic['success']))
        controls={method:{client.client_id:ninea.eightg.eightf.eighte.eightd.prior.eight.seven.seven.controller(models[client.client_id]) for client in test} for method,models in maps.items()}
        for method,models in maps.items():
            for client in test:diagnostics.append(dict(seed=seed,budget=budget,method=method,client=client.client_id,parameter_error=float(np.linalg.norm(np.log(models[client.client_id]/nineb.ratios(client)))),gain_error=ninef.ninee.gain_error(controls[method][client.client_id],controls['ExactIndividual'][client.client_id])))
        for scenario in cfg['test_scenarios']:
            speed,reference=corrected.scenario(scenario,t/2);speed=corrected.six.remap(speed,(10,30))
            for method,control_map in controls.items():
                _,_,metrics=corrected.simulate(test,nominal,speed,reference,'vy',client_controllers=control_map)
                for j,client in enumerate(test):rows.append(dict(seed=seed,budget=budget,method=method,scenario=scenario,client=client.client_id,family=client.family,**{key:float(value[j]) for key,value in metrics.items()}))
    for suffix,items in [('fits',fit_rows),('assignments',assignment_rows),('protocol',protocol),('diagnostics',diagnostics),('clients',rows)]:
        (OUT/f'experiment_9l_seed{seed}_{suffix}.csv.gz').write_bytes(gzip.compress(pd.DataFrame(items).to_csv(index=False).encode(),mtime=0))
    print(seed,flush=True)


def summarize():
    cfg=json.loads(CONFIG.read_text());load=lambda suffix:pd.concat([pd.read_csv(OUT/f'experiment_9l_seed{seed}_{suffix}.csv.gz') for seed in cfg['development_seeds']]);raw=load('clients');diagnostics=load('diagnostics');fits=load('fits');assignments=load('assignments');protocol=load('protocol');units=raw.groupby(['seed','budget','method']).agg(tracking=('tracking','mean'),feasible=('feasible','min')).reset_index();summary=units.groupby(['budget','method']).agg(tracking=('tracking','mean'),tracking_std=('tracking','std'),feasible_rate=('feasible','mean')).reset_index().merge(diagnostics.groupby(['budget','method']).agg(parameter_error=('parameter_error','mean'),gain_error=('gain_error','mean')).reset_index(),on=['budget','method']);uniform=summary[summary.method=='Uniform20'][['budget','tracking']].rename(columns={'tracking':'uniform'});local=summary[summary.method=='Local'][['budget','tracking']].rename(columns={'tracking':'local'});comparison=summary.merge(uniform,on='budget').merge(local,on='budget');comparison['vs_uniform_pct']=100*(comparison.tracking/comparison.uniform-1);comparison['vs_local_pct']=100*(comparison.tracking/comparison.local-1);communication=protocol.groupby('method').agg(selected_k=('selected_k','mean'),ari=('train_ari','mean'),coverage=('coverage','mean'),minimum_family_coverage=('minimum_family_coverage','mean'),minimum_category_coverage=('minimum_category_coverage','mean'),messages=('client_messages','mean'),upload_bytes=('upload_bytes','mean'),download_bytes=('download_bytes','mean')).reset_index();cold=comparison[comparison.budget==.75].set_index('method');stress=[r['name'] for r in cfg['availability_regimes'] if r['name']!='Uniform20'];seed_cold=units[units.budget==.75].pivot(index='seed',columns='method',values='tracking');gates=[]
    for method in stress:
        paired=100*(seed_cold[method]/seed_cold.Local-1);gates.append(dict(method=method,degradation_vs_uniform_pct=float(cold.loc[method,'vs_uniform_pct']),improvement_vs_local_pct=float(-cold.loc[method,'vs_local_pct']),within_uniform_tolerance=bool(cold.loc[method,'vs_uniform_pct']<=cfg['cold_start_max_degradation_vs_uniform_pct']),beats_local=bool(cold.loc[method,'vs_local_pct']<0),fleetwise_win_rate=float(np.mean(paired<0)),worst_fleet_vs_local_pct=float(paired.max())))
    conclusions=dict(provenance={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in [CONFIG,ROOT/'code/src/federated_lpv/federated_manifold.py',ROOT/'code/experiments/experiment_9l_nonrandom_participation.py']},fits=len(fits),all_fits_converged=bool(fits.success.all() and assignments.success.all()),evaluations=len(raw),all_feasible=bool(raw.feasible.all()),stress_gates=gates,all_stress_regimes_within_tolerance=bool(all(row['within_uniform_tolerance'] for row in gates)),all_stress_regimes_beat_local_in_aggregate=bool(all(row['beats_local'] for row in gates)),all_stress_regimes_beat_local_every_fleet=bool(all(row['fleetwise_win_rate']==1 for row in gates)),reserved_confirmation_seeds=cfg['reserved_confirmation_seeds']);summary.to_csv(OUT/'experiment_9l_summary.csv',index=False);comparison.to_csv(OUT/'experiment_9l_comparisons.csv',index=False);communication.to_csv(OUT/'experiment_9l_communication.csv',index=False);units.to_csv(OUT/'experiment_9l_seed_summary.csv',index=False);pd.DataFrame(gates).to_csv(OUT/'experiment_9l_gates.csv',index=False);(OUT/'experiment_9l_conclusions.json').write_text(json.dumps(conclusions,indent=2)+'\n');order=['Local','Uniform20',*stress];fig,axes=plt.subplots(1,2,figsize=(10,4));x=np.arange(2);width=.12
    for j,method in enumerate(order):group=summary[summary.method==method].set_index('budget').loc[cfg['calibration_budgets']];axes[0].bar(x+(j-(len(order)-1)/2)*width,group.tracking,width,label=method)
    axes[0].set(xticks=x,xticklabels=['0.75','1.25'],xlabel='Calibration time [s]',ylabel='Tracking RMSE [rad/s]');axes[0].legend(fontsize=7,ncol=2);axes[0].grid(axis='y',alpha=.2);view=communication.set_index('method').loc[[r['name'] for r in cfg['availability_regimes']]];axes[1].bar(np.arange(len(view)),view.coverage);axes[1].set(xticks=np.arange(len(view)),xticklabels=['Uniform','Persistent','Family skew','Operating skew','Dropout'],ylabel='Distinct-client coverage',ylim=(0,1.05));axes[1].tick_params(axis='x',rotation=30);axes[1].grid(axis='y',alpha=.2);fig.tight_layout();fig.savefig(ROOT/'results/figures/experiment_9l_nonrandom_participation.pdf');plt.close(fig);print(json.dumps(conclusions,indent=2));print(summary.to_string(index=False));print(comparison[['budget','method','vs_uniform_pct','vs_local_pct']].to_string(index=False));print(communication.to_string(index=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=5);parser.add_argument('--summarize-only',action='store_true');args=parser.parse_args();cfg=json.loads(CONFIG.read_text())
    if not args.summarize_only:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:list(pool.map(run_seed,cfg['development_seeds']))
    summarize()
