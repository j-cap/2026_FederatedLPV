"""9M: unified confirmation-fleet ablation and paired statistical analysis."""
from concurrent.futures import ProcessPoolExecutor
import argparse,gzip,hashlib,json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score
from federated_lpv.output_error import fit
from federated_lpv.privacy import normalize_log_parameters
from federated_lpv.uncertainty_clustering import log_parameter_covariance,normalized_covariance
from federated_lpv.federated_manifold import select_federated_mixture
import experiment_9j_distributed_backbone as ninej
import experiment_9c_uncertainty_groups as ninec

ROOT=ninej.ROOT;OUT=ninej.OUT;base=ninej.base;corrected=ninej.corrected;nineb=ninej.nineb;ninea=ninej.ninea;nined=ninej.nined;nineg=ninej.nineg;ninef=ninej.ninef
CONFIG=ROOT/'code/config/experiment_9m.json';NINEK_CONFIG=ROOT/'code/config/experiment_9k.json'


def assert_frozen(cfg):
    reference=json.loads(NINEK_CONFIG.read_text());keys=['train_counts','test_counts','federated_rounds','server_damping','calibration_budgets','initial_records','initial_record_duration','maximum_records','additional_record_duration','coverage_speeds','additional_speed_offsets','noise_deg','bounds','candidate_clusters','minimum_cluster_size','mixture_restarts','mixture_iterations','intrinsic_floor','gain_speeds','gain_difference_step','weight_floor','coefficient_prior_scale','coefficient_prior_strength','test_scenarios']
    if any(cfg[key]!=reference[key] for key in keys):raise RuntimeError(f"9M differs from 9K: {[key for key in keys if cfg[key]!=reference[key]]}")
    if cfg['participation_rate']!=min(reference['participation_rates']):raise RuntimeError('9M must use the frozen 20% federated ablation')
    if cfg['seeds']!=reference['confirmation_seeds']:raise RuntimeError('9M must use exactly the 9K confirmation fleets')


def run_seed(seed):
    cfg=json.loads(CONFIG.read_text());assert_frozen(cfg);train=nineb.discrete_fleet(seed,cfg['train_counts']);test=nineb.discrete_fleet(seed+10000,cfg['test_counts']);train_records,_=nined.collect_pool(train,seed+81000,cfg);test_records,nominal=nined.collect_pool(test,seed+82000,cfg);bounds=np.asarray(cfg['bounds']);parameters=[];covariances=[];fit_rows=[]
    for client in train:
        estimate,diagnostic=fit(train_records[client.client_id]);covariance,_=log_parameter_covariance(train_records[client.client_id],estimate,cfg['noise_deg']);parameters.append(estimate);covariances.append(normalized_covariance(covariance,bounds));fit_rows.append(dict(seed=seed,scope='train',client=client.client_id,success=diagnostic['success']))
    parameters=np.asarray(parameters);covariances=np.asarray(covariances);x=normalize_log_parameters(parameters,bounds);truth=np.array([base.FAMILIES.index(client.family) for client in train]);central,candidates=ninec.select_mixture(x,covariances,cfg,seed);central_labels=np.argmax(central['responsibility'],axis=1);central_backbones=nineg.build_backbones(parameters,central_labels,central['k'],{**cfg,'ranks':[1,2]});oracle_backbones=nineg.build_backbones(parameters,truth,3,{**cfg,'ranks':[1,2]});fed,_=select_federated_mixture(x,covariances,cfg['candidate_clusters'],cfg['federated_rounds'],cfg['participation_rate'],seed,cfg['server_damping'],cfg['intrinsic_floor'],cfg['minimum_cluster_size']);fed_backbones=ninej.distributed_backbones(parameters,fed,cfg);global_parameter=np.exp(np.log(parameters).mean(0));rows=[];diagnostics=[];assignments=[];t=np.arange(0,24+base.DT,base.DT)
    for budget in cfg['calibration_budgets']:
        maps={method:{} for method in ('Local','GlobalK1','LearnedCenter','CentralRank1','CentralRank2','Fed20Rank1','Fed20Rank2','OracleFamilyRank2','ExactIndividual')}
        for client in test:
            records=ninej.nineh.prefix_records(test_records[client.client_id],budget);local,diagnostic=fit(records);maps['Local'][client.client_id]=local;maps['GlobalK1'][client.client_id]=global_parameter;maps['ExactIndividual'][client.client_id]=nineb.ratios(client);fit_rows.append(dict(seed=seed,scope='test',client=client.client_id,success=diagnostic['success']))
            _,group,center=ninej.nineh.select_backbone(records,central_backbones,cfg);maps['LearnedCenter'][client.client_id]=center;assignments.append(dict(seed=seed,budget=budget,method='LearnedCenter',client=client.client_id,group=group,success=True))
            for source,backbones in [('Central',central_backbones),('Fed20',fed_backbones)]:
                for rank in (1,2):
                    _,group,estimate,_,diagnostic=ninej.nineh.select_personalized(records,backbones,rank,cfg);method=f'{source}Rank{rank}';maps[method][client.client_id]=estimate;assignments.append(dict(seed=seed,budget=budget,method=method,client=client.client_id,group=group,success=diagnostic['success']))
            family=base.FAMILIES.index(client.family);center,directions=oracle_backbones[family,2];estimate,_,diagnostic=nineg.coefficient_fit(records,center,directions,cfg);maps['OracleFamilyRank2'][client.client_id]=estimate;assignments.append(dict(seed=seed,budget=budget,method='OracleFamilyRank2',client=client.client_id,group=family,success=diagnostic['success']))
        controls={method:{client.client_id:ninea.eightg.eightf.eighte.eightd.prior.eight.seven.seven.controller(models[client.client_id]) for client in test} for method,models in maps.items()}
        coordinates={'GlobalK1':0,'LearnedCenter':0,'CentralRank1':1,'Fed20Rank1':1,'CentralRank2':2,'Fed20Rank2':2,'OracleFamilyRank2':2,'Local':3,'ExactIndividual':3}
        for method,models in maps.items():
            for client in test:diagnostics.append(dict(seed=seed,budget=budget,method=method,client=client.client_id,local_coordinates=coordinates[method],parameter_error=float(np.linalg.norm(np.log(models[client.client_id]/nineb.ratios(client)))),gain_error=ninef.ninee.gain_error(controls[method][client.client_id],controls['ExactIndividual'][client.client_id])))
        for scenario in cfg['test_scenarios']:
            speed,reference=corrected.scenario(scenario,t/2);speed=corrected.six.remap(speed,(10,30))
            for method,control_map in controls.items():
                _,_,metrics=corrected.simulate(test,nominal,speed,reference,'vy',client_controllers=control_map)
                for j,client in enumerate(test):rows.append(dict(seed=seed,budget=budget,method=method,scenario=scenario,client=client.client_id,family=client.family,**{key:float(value[j]) for key,value in metrics.items()}))
    cluster_rows=[dict(seed=seed,source='Central',selected_k=central['k'],train_ari=adjusted_rand_score(truth,central_labels),coverage=1.) ,dict(seed=seed,source='Fed20',selected_k=fed['k'],train_ari=adjusted_rand_score(truth,np.argmax(fed['responsibility'],axis=1)),coverage=fed['coverage'])]
    for suffix,items in [('fits',fit_rows),('assignments',assignments),('clusters',cluster_rows),('diagnostics',diagnostics),('clients',rows)]:
        (OUT/f'experiment_9m_seed{seed}_{suffix}.csv.gz').write_bytes(gzip.compress(pd.DataFrame(items).to_csv(index=False).encode(),mtime=0))
    print(seed,flush=True)


def paired_bootstrap(units,cfg):
    pivot=units.pivot(index=['seed','budget'],columns='method',values='tracking');rng=np.random.default_rng(cfg['bootstrap_seed']);comparisons=[(.75,'Fed20Rank1','Local','cold_start_superiority'),(.75,'CentralRank1','Local','central_rank1'),(.75,'LearnedCenter','Local','clustering_only'),(.75,'GlobalK1','Local','global_pooling'),(.75,'LearnedCenter','GlobalK1','clustering_increment'),(.75,'Fed20Rank1','LearnedCenter','personalization_increment'),(1.25,'Fed20Rank2','Local','late_noninferiority'),(1.25,'Fed20Rank1','Local','late_rank1'),(1.25,'Fed20Rank1','LearnedCenter','late_personalization_increment'),(1.25,'Fed20Rank1','Fed20Rank2','rank1_vs_rank2'),(1.25,'CentralRank2','Local','central_rank2'),(.75,'Fed20Rank1','CentralRank1','federation_cost'),(1.25,'Fed20Rank2','CentralRank2','federation_cost')];rows=[]
    for budget,method,reference,label in comparisons:
        group=pivot.xs(budget,level='budget');relative=100*(group[reference].to_numpy()-group[method].to_numpy())/group[reference].to_numpy();indices=rng.integers(0,len(relative),size=(cfg['bootstrap_resamples'],len(relative)));boot=relative[indices].mean(1);rows.append(dict(budget=budget,method=method,reference=reference,label=label,mean_improvement_pct=float(relative.mean()),median_improvement_pct=float(np.median(relative)),ci95_low_pct=float(np.quantile(boot,.025)),ci95_high_pct=float(np.quantile(boot,.975)),wins=int(np.sum(relative>0)),fleets=len(relative),minimum_improvement_pct=float(relative.min()),maximum_improvement_pct=float(relative.max())))
    return pd.DataFrame(rows)


def summarize():
    cfg=json.loads(CONFIG.read_text());assert_frozen(cfg);load=lambda suffix:pd.concat([pd.read_csv(OUT/f'experiment_9m_seed{seed}_{suffix}.csv.gz') for seed in cfg['seeds']]);raw=load('clients');diagnostics=load('diagnostics');fits=load('fits');assignments=load('assignments');clusters=load('clusters');units=raw.groupby(['seed','budget','method']).agg(tracking=('tracking','mean'),feasible=('feasible','min')).reset_index();summary=units.groupby(['budget','method']).agg(tracking=('tracking','mean'),tracking_std=('tracking','std'),tracking_median=('tracking','median'),tracking_q25=('tracking',lambda x:np.quantile(x,.25)),tracking_q75=('tracking',lambda x:np.quantile(x,.75)),feasible_rate=('feasible','mean')).reset_index().merge(diagnostics.groupby(['budget','method']).agg(local_coordinates=('local_coordinates','mean'),parameter_error=('parameter_error','mean'),gain_error=('gain_error','mean')).reset_index(),on=['budget','method']);statistics=paired_bootstrap(units,cfg);cold=statistics[statistics.label=='cold_start_superiority'].iloc[0];late=statistics[statistics.label=='late_noninferiority'].iloc[0];gates=dict(cold_start_superiority=bool(cold.ci95_low_pct>0),late_noninferiority=bool(late.ci95_low_pct>-cfg['noninferiority_margin_pct']));cluster_summary=clusters.groupby('source').agg(selected_k=('selected_k','mean'),ari=('train_ari','mean'),coverage=('coverage','mean')).reset_index();conclusions=dict(retrospective_confirmation_fleet_analysis=True,provenance={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in [CONFIG,NINEK_CONFIG,ROOT/'code/experiments/experiment_9m_unified_ablation.py']},fits=len(fits),all_fits_converged=bool(fits.success.all() and assignments.success.all()),evaluations=len(raw),all_feasible=bool(raw.feasible.all()),statistical_gates=gates,bootstrap_unit='fleet',bootstrap_resamples=cfg['bootstrap_resamples']);summary.to_csv(OUT/'experiment_9m_summary.csv',index=False);units.to_csv(OUT/'experiment_9m_seed_summary.csv',index=False);statistics.to_csv(OUT/'experiment_9m_statistics.csv',index=False);cluster_summary.to_csv(OUT/'experiment_9m_clusters.csv',index=False);(OUT/'experiment_9m_conclusions.json').write_text(json.dumps(conclusions,indent=2)+'\n');order=['Local','GlobalK1','LearnedCenter','CentralRank1','CentralRank2','Fed20Rank1','Fed20Rank2','OracleFamilyRank2','ExactIndividual'];fig,axes=plt.subplots(1,2,figsize=(11,4.2));x=np.arange(len(order));colors=['#d9822b','.65','#7a9e9f','#4d91c6','#1f5f99','#66a9d8','#154c79','#77b7a5','#3c9d5d']
    for axis,budget in zip(axes,cfg['calibration_budgets']):view=summary[summary.budget==budget].set_index('method').loc[order];axis.bar(x,view.tracking,color=colors);axis.set(xticks=x,xticklabels=['Local','Global','Center','Central r1','Central r2','Fed20 r1','Fed20 r2','Oracle r2','Exact'],ylabel='Tracking RMSE [rad/s]',title=f'{budget:.2f} s calibration');axis.tick_params(axis='x',rotation=35);axis.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(ROOT/'results/figures/experiment_9m_unified_ablation.pdf');plt.close(fig);print(json.dumps(conclusions,indent=2));print(summary.to_string(index=False));print(statistics.to_string(index=False));print(cluster_summary.to_string(index=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=5);parser.add_argument('--summarize-only',action='store_true');args=parser.parse_args();cfg=json.loads(CONFIG.read_text());assert_frozen(cfg)
    if not args.summarize_only:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:list(pool.map(run_seed,cfg['seeds']))
    summarize()
