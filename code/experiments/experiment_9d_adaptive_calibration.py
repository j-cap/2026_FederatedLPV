"""9D: uncertainty-triggered complementary calibration before group sharing."""
from concurrent.futures import ProcessPoolExecutor
import argparse,gzip,hashlib,json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score
from federated_lpv import discrete_bicycle_matrices,augmented_tracking_matrices
from federated_lpv.output_error import fit
from federated_lpv.privacy import normalize_log_parameters,denormalize_log_parameters
from federated_lpv.uncertainty_clustering import log_parameter_covariance,normalized_covariance,predict_membership
import experiment_9c_uncertainty_groups as ninec

ROOT=ninec.ROOT;OUT=ninec.OUT;base=ninec.base;corrected=ninec.corrected;nineb=ninec.nineb;ninea=ninec.ninea
CONFIG=ROOT/'code/config/experiment_9d.json'


def speed_plan(category,cfg):
    speeds=cfg['coverage_speeds'];indices=[category]+[(category+o)%len(speeds) for o in cfg['additional_speed_offsets']]
    return [speeds[i] for i in indices[:cfg['maximum_records']-cfg['initial_records']+1]]


def collect_pool(clients,seed,cfg):
    nominal=ninea.eightg.eightf.eighte.eightd.prior.eight.seven.seven.controller(np.array([80000/1500,80000/1500,1500/2500]));rng=np.random.default_rng(seed+910000);order=rng.permutation(len(clients));categories={c.client_id:int(order[j]%len(cfg['coverage_speeds'])) for j,c in enumerate(clients)};controls={c.client_id:nominal for c in clients};records={c.client_id:[] for c in clients}
    for c in clients:
        plan=speed_plan(categories[c.client_id],cfg)
        sequence=[(plan[0],cfg['initial_record_duration'])]*cfg['initial_records']+[(v,cfg['additional_record_duration']) for v in plan[1:]]
        for episode,(speed_value,duration) in enumerate(sequence):
            n=round(duration/base.DT);t=np.arange(n+1)*base.DT
            ref=np.deg2rad(3)*(1-np.exp(-t/.3))*np.sin(2*np.pi*.35*t+.7*episode);speed=np.full_like(t,speed_value);x,u,m=corrected.simulate([c],nominal,speed,ref,'vy',client_controllers=controls)
            if not m['feasible'].all():raise RuntimeError('calibration collection infeasible')
            records[c.client_id].append(dict(state=x[:,0]+rng.normal(size=x[:,0].shape)*np.deg2rad(cfg['noise_deg']),input=u[:,0],speed=speed))
    return records,nominal


def fit_policy(records,cfg):
    bounds=np.asarray(cfg['bounds'],float);history=[]
    for count in range(cfg['initial_records'],cfg['maximum_records']+1):
        p,d=fit(records[:count]);cov,condition=log_parameter_covariance(records[:count],p,cfg['noise_deg']);cn=normalized_covariance(cov,bounds);history.append((count,p,cn,condition,d))
        if np.trace(cn)<=cfg['uncertainty_trace_threshold']:break
    return history[-1],history


def calibration_seconds(count,cfg):
    return cfg['initial_records']*cfg['initial_record_duration']+max(0,count-cfg['initial_records'])*cfg['additional_record_duration']


def mixture_models(z,covariances,cfg,seed):
    model,candidates=ninec.select_mixture(z,covariances,cfg,seed);bounds=np.asarray(cfg['bounds'],float);models={j:denormalize_log_parameters(model['means'][j],bounds) for j in range(model['k'])}
    return model,models,candidates


def run_seed(task):
    phase,seed=task;cfg=json.loads(CONFIG.read_text());train=nineb.discrete_fleet(seed,cfg['train_counts']);test=nineb.discrete_fleet(seed+10000,cfg['test_counts']);train_data,nominal=collect_pool(train,seed,cfg);test_data,_=collect_pool(test,seed+20000,cfg);bounds=np.asarray(cfg['bounds'],float);fit_rows=[];sets={}
    for scope,clients,data in [('train',train,train_data),('test',test,test_data)]:
        sets[scope]={}
        for c in clients:
            short_p,short_d=fit(data[c.client_id][:cfg['initial_records']]);short_cov,_=log_parameter_covariance(data[c.client_id][:cfg['initial_records']],short_p,cfg['noise_deg']);(count,p,cov,condition,d),history=fit_policy(data[c.client_id],cfg);full_p,full_d=fit(data[c.client_id]);full_cov,full_condition=log_parameter_covariance(data[c.client_id],full_p,cfg['noise_deg']);full_cov=normalized_covariance(full_cov,bounds);sets[scope][c.client_id]=dict(short=(short_p,normalized_covariance(short_cov,bounds)),adaptive=(p,cov,count),full=(full_p,full_cov))
            fit_rows.append(dict(phase=phase,seed=seed,scope=scope,client=c.client_id,family=c.family,adaptive_records=count,adaptive_seconds=calibration_seconds(count,cfg),adaptive_trace=float(np.trace(cov)),full_trace=float(np.trace(full_cov)),adaptive_parameter_error=float(np.linalg.norm(np.log(p/nineb.ratios(c)))),short_parameter_error=float(np.linalg.norm(np.log(short_p/nineb.ratios(c)))),full_parameter_error=float(np.linalg.norm(np.log(full_p/nineb.ratios(c)))),adaptive_condition=condition,full_condition=full_condition,adaptive_success=d['success'],short_success=short_d['success'],full_success=full_d['success']))
    z={};cov={}
    for scope,clients in [('train',train),('test',test)]:
        for version in ('short','adaptive','full'):
            z[scope,version]=normalize_log_parameters([sets[scope][c.client_id][version][0] for c in clients],bounds);cov[scope,version]=np.asarray([sets[scope][c.client_id][version][1] for c in clients])
    truth_train=np.array([base.FAMILIES.index(c.family) for c in train]);truth_test=np.array([base.FAMILIES.index(c.family) for c in test]);point_cfg={'candidate_clusters':cfg['candidate_clusters'][1:],'minimum_cluster_size':cfg['minimum_cluster_size'],'minimum_silhouette':cfg['point_minimum_silhouette'],'kmeans_restarts':cfg['point_restarts']};pl,pc,pk,_=ninea.fit_partition(z['train','short'],point_cfg,seed=seed);pt=np.argmin(np.linalg.norm(z['test','short'][:,None,:]-pc[None,:,:],axis=2),axis=1);point_models=ninea.cluster_models(z['train','short'],pl,pk,bounds)
    adaptive,adaptive_models,candidates=mixture_models(z['train','adaptive'],cov['train','adaptive'],cfg,seed+100);ap_train=adaptive['responsibility'];ap_test=predict_membership(z['test','adaptive'],cov['test','adaptive'],adaptive);full,full_models,_=mixture_models(z['train','full'],cov['train','full'],cfg,seed+200);fp_test=predict_membership(z['test','full'],cov['test','full'],full)
    diagnostics=[dict(phase=phase,seed=seed,selected_k=adaptive['k'],train_ari=adjusted_rand_score(truth_train,np.argmax(ap_train,axis=1)),test_ari=adjusted_rand_score(truth_test,np.argmax(ap_test,axis=1)),test_entropy=float(np.mean(-np.sum(ap_test*np.log(np.maximum(ap_test,1e-15)),axis=1))),candidate_k=r['k'],bic=r['bic'],minimum_size=r['minimum_size'],admissible=r['admissible']) for r in candidates]
    oracle_models={j:denormalize_log_parameters(z['train','adaptive'][truth_train==j].mean(0),bounds) for j in range(3)};global_model={0:denormalize_log_parameters(z['train','adaptive'].mean(0),bounds)};local_models={j:sets['test'][c.client_id]['adaptive'][0] for j,c in enumerate(test)}
    controls={};parameter_models={};counts={}
    hard={'ShortPoint':(pt,point_models),'OracleAdaptive':(truth_test,oracle_models),'GlobalAdaptive':(np.zeros(len(test),int),global_model),'LocalAdaptive':(np.arange(len(test)),local_models)}
    for method,(labels,models) in hard.items():controls[method]={c.client_id:ninea.eightg.eightf.eighte.eightd.prior.eight.seven.seven.controller(models[int(labels[j])]) for j,c in enumerate(test)};parameter_models[method]={j:models[int(labels[j])] for j in range(len(test))};counts[method]=len(models)
    for method,posterior,models in [('AdaptiveSoft',ap_test,adaptive_models),('FixedFullSoft',fp_test,full_models)]:controls[method]={c.client_id:ninec.blend_controller(posterior[j],models) for j,c in enumerate(test)};parameter_models[method]={j:np.exp(sum(posterior[j,q]*np.log(models[q]) for q in range(len(models)))) for j in range(len(test))};counts[method]=len(models)
    adaptive_seconds=float(np.mean([calibration_seconds(sets['test'][c.client_id]['adaptive'][2],cfg) for c in test]));full_seconds=calibration_seconds(cfg['maximum_records'],cfg);short_seconds=calibration_seconds(cfg['initial_records'],cfg);mean_seconds={m:(short_seconds if m=='ShortPoint' else full_seconds if m=='FixedFullSoft' else adaptive_seconds) for m in controls}
    rows=[];grid=np.linspace(10,30,161);t=np.arange(0,24+base.DT,base.DT);scenarios=[]
    for name in cfg['test_scenarios']:
        v,ref=corrected.scenario(name,t/2);scenarios.append((name,corrected.six.remap(v,(10,30)),ref))
    for method,control_map in controls.items():
        rhos=[]
        for c in test:
            pairs=[augmented_tracking_matrices(*discrete_bicycle_matrices(float(v),c.parameters,base.DT),base.DT) for v in grid];a=np.array([p[0] for p in pairs]);b=np.array([p[1] for p in pairs]);control=control_map[c.client_id];g=np.column_stack([np.interp(grid,control.speeds,control.gains['global'][:,q]) for q in range(3)]);rhos.append(float(np.max(abs(np.linalg.eigvals(a-b@g[:,None,:])))))
        for scenario,v,ref in scenarios:
            _,_,metrics=corrected.simulate(test,nominal,v,ref,'vy',client_controllers=control_map)
            for j,c in enumerate(test):rows.append(dict(phase=phase,seed=seed,method=method,models=counts[method],mean_calibration_seconds=mean_seconds[method],scenario=scenario,client=c.client_id,family=c.family,parameter_error=float(np.linalg.norm(np.log(parameter_models[method][j]/nineb.ratios(c)))),rho=rhos[j],**{q:float(x[j]) for q,x in metrics.items()}))
    for suffix,items in [('fits',fit_rows),('clusters',diagnostics),('clients',rows)]:(OUT/f'experiment_9d_{phase}_seed{seed}_{suffix}.csv.gz').write_bytes(gzip.compress(pd.DataFrame(items).to_csv(index=False).encode(),mtime=0))
    print(phase,seed,flush=True)


def summarize(phases):
    cfg=json.loads(CONFIG.read_text());frames=[];fit_frames=[];cluster_frames=[]
    for phase in phases:
        key='development_seeds' if phase=='development' else 'confirmation_seeds';frames.extend(pd.read_csv(OUT/f'experiment_9d_{phase}_seed{s}_clients.csv.gz') for s in cfg[key]);fit_frames.extend(pd.read_csv(OUT/f'experiment_9d_{phase}_seed{s}_fits.csv.gz') for s in cfg[key]);cluster_frames.extend(pd.read_csv(OUT/f'experiment_9d_{phase}_seed{s}_clusters.csv.gz') for s in cfg[key])
    df=pd.concat(frames);fits=pd.concat(fit_frames);clusters=pd.concat(cluster_frames);units=df.groupby(['phase','seed','method','models','mean_calibration_seconds']).agg(tracking=('tracking','mean'),parameter_error=('parameter_error','mean'),feasible=('feasible','min'),rho=('rho','max')).reset_index();summary=units.groupby(['phase','method']).agg(tracking=('tracking','mean'),tracking_std=('tracking','std'),parameter_error=('parameter_error','mean'),models=('models','mean'),models_min=('models','min'),models_max=('models','max'),calibration_seconds=('mean_calibration_seconds','mean'),feasible_rate=('feasible','mean'),rho=('rho','max')).reset_index();selected=clusters[clusters.candidate_k==clusters.selected_k].groupby('phase').agg(k_mean=('selected_k','mean'),k_min=('selected_k','min'),k_max=('selected_k','max'),train_ari=('train_ari','mean'),test_ari=('test_ari','mean'),entropy=('test_entropy','mean')).reset_index();gates=[]
    for phase in phases:
        s=summary[summary.phase==phase].set_index('method');a=s.loc['AdaptiveSoft'];o=s.loc['OracleAdaptive'];q=s.loc['ShortPoint'];f=s.loc['FixedFullSoft'];cost=100*(a.tracking/o.tracking-1);gain=100*(1-a.tracking/q.tracking);passed=bool(cost<=cfg['oracle_tracking_tolerance_pct'] and a.tracking<q.tracking and a.calibration_seconds<f.calibration_seconds and a.feasible_rate==1 and a.rho<1 and a.models<s.loc['LocalAdaptive','models']);gates.append(dict(phase=phase,tracking_cost_vs_oracle_pct=float(cost),improvement_vs_short_point_pct=float(gain),adaptive_seconds=float(a.calibration_seconds),fixed_full_seconds=float(f.calibration_seconds),passed=passed))
    gate=pd.DataFrame(gates);dev=gate[gate.phase=='development'];conf=gate[gate.phase=='confirmation'];conclusions=dict(provenance={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [CONFIG,ROOT/'code/experiments/experiment_9d_adaptive_calibration.py']},fits=len(fits),closed_loop_evaluations=len(df),all_fits_converged=bool(fits[['adaptive_success','short_success','full_success']].all().all()),development_gate=bool(len(dev) and dev.passed.all()),confirmation_run=bool(len(conf)),confirmation_gate=None if conf.empty else bool(conf.passed.all()),reserved_confirmation_seeds=cfg['confirmation_seeds'] if conf.empty else []);summary.to_csv(OUT/'experiment_9d_summary.csv',index=False);units.to_csv(OUT/'experiment_9d_seed_summary.csv',index=False);selected.to_csv(OUT/'experiment_9d_cluster_selection.csv',index=False);gate.to_csv(OUT/'experiment_9d_gates.csv',index=False);fits.groupby(['phase','scope']).agg(mean_records=('adaptive_records','mean'),q90_records=('adaptive_records',lambda x:np.quantile(x,.9)),threshold_rate=('adaptive_trace',lambda x:np.mean(x<=cfg['uncertainty_trace_threshold']))).reset_index().to_csv(OUT/'experiment_9d_budget.csv',index=False);(OUT/'experiment_9d_conclusions.json').write_text(json.dumps(conclusions,indent=2)+'\n');view=summary[summary.phase==phases[-1]].set_index('method').loc[['OracleAdaptive','ShortPoint','AdaptiveSoft','FixedFullSoft']];fig,ax=plt.subplots(figsize=(7.5,3.8));ax.bar(np.arange(len(view)),view.tracking);ax.set(xticks=np.arange(len(view)),xticklabels=['Oracle','Short point','Adaptive','Fixed full'],ylabel='Tracking RMSE [rad/s]');ax.grid(axis='y',alpha=.2);fig.tight_layout();fig.savefig(ROOT/'results/figures/experiment_9d_adaptive_calibration.pdf');plt.close(fig);print(json.dumps(conclusions,indent=2));print(summary.to_string(index=False));print(selected.to_string(index=False));print(gate.to_string(index=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['development','confirmation'],default='development');p.add_argument('--summarize-only',action='store_true');p.add_argument('--workers',type=int,default=5);a=p.parse_args();cfg=json.loads(CONFIG.read_text());phases=(a.phase,)
    if not a.summarize_only:
        key='development_seeds' if a.phase=='development' else 'confirmation_seeds'
        with ProcessPoolExecutor(max_workers=a.workers) as pool:list(pool.map(run_seed,[(a.phase,s) for s in cfg[key]]))
    summarize(phases)
