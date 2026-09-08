"""9C: uncertainty-aware discovery and soft use of LPV compatibility groups."""
from concurrent.futures import ProcessPoolExecutor
import argparse,gzip,hashlib,json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score
from federated_lpv import ScheduledController,discrete_bicycle_matrices,augmented_tracking_matrices
from federated_lpv.output_error import fit
from federated_lpv.privacy import normalize_log_parameters,denormalize_log_parameters
from federated_lpv.uncertainty_clustering import log_parameter_covariance,normalized_covariance,fit_mixture,predict_membership
import experiment_9b_group_robustness as nineb

ROOT=nineb.ROOT;OUT=nineb.OUT;base=nineb.base;corrected=nineb.corrected;ninea=nineb.ninea
CONFIG=ROOT/'code/config/experiment_9c.json'


def blend_controller(probabilities,models):
    controls=[ninea.eightg.eightf.eighte.eightd.prior.eight.seven.seven.controller(models[j]) for j in range(len(probabilities))]
    grid=controls[0].speeds;g=sum(probabilities[j]*controls[j].gains['global'] for j in range(len(controls)));p=sum(probabilities[j]*controls[j].prefilters['global'] for j in range(len(controls)))
    return ScheduledController('9C-soft','global','linear',grid,{'global':g},{'global':p})


def select_mixture(z,covariances,cfg,seed):
    records=[];models={}
    for k in cfg['candidate_clusters']:
        model=fit_mixture(z,covariances,k,seed=seed,restarts=cfg['mixture_restarts'],max_iter=cfg['mixture_iterations'],intrinsic_floor=cfg['intrinsic_floor']);labels=np.argmax(model['responsibility'],axis=1);minimum=int(np.bincount(labels,minlength=k).min());admissible=minimum>=cfg['minimum_cluster_size'];records.append(dict(k=k,bic=model['bic'],log_likelihood=model['log_likelihood'],minimum_size=minimum,admissible=admissible));models[k]=model
    valid=[r for r in records if r['admissible']];selected=min(valid,key=lambda r:(r['bic'],r['k'])) if valid else records[0]
    return models[selected['k']],records


def run_seed(task):
    phase,seed=task;cfg=json.loads(CONFIG.read_text());rows=[];diagnostics=[];fits_rows=[]
    for ri,regime in enumerate(cfg['regimes']):
        train=nineb.discrete_fleet(seed+ri*1000,regime['train_counts']);test=nineb.discrete_fleet(seed+10000+ri*1000,regime['test_counts']);local_cfg={**cfg,**regime};train_data,nominal=nineb.collect(train,seed+ri*1000,local_cfg);test_data,_=nineb.collect(test,seed+20000+ri*1000,local_cfg);fits={};covariances={}
        for scope,clients,data in [('train',train,train_data),('test',test,test_data)]:
            for c in clients:
                p,d=fit(data[c.client_id]);cov,info_condition=log_parameter_covariance(data[c.client_id],p,regime['noise_deg']);fits[c.client_id]=p;covariances[c.client_id]=normalized_covariance(cov,cfg['bounds']);fits_rows.append(dict(phase=phase,seed=seed,regime=regime['name'],scope=scope,client=c.client_id,family=c.family,parameter_error=float(np.linalg.norm(np.log(p/nineb.ratios(c)))),uncertainty_trace=float(np.trace(covariances[c.client_id])),information_condition=info_condition,**d))
        bounds=np.asarray(cfg['bounds'],float);ztrain=normalize_log_parameters([fits[c.client_id] for c in train],bounds);ztest=normalize_log_parameters([fits[c.client_id] for c in test],bounds);ctrain=np.asarray([covariances[c.client_id] for c in train]);ctest=np.asarray([covariances[c.client_id] for c in test]);family_to_int={f:j for j,f in enumerate(base.FAMILIES)};truth_train=np.array([family_to_int[c.family] for c in train]);truth_test=np.array([family_to_int[c.family] for c in test])
        point_cfg={'candidate_clusters':cfg['candidate_clusters'][1:],'minimum_cluster_size':cfg['minimum_cluster_size'],'minimum_silhouette':cfg['point_minimum_silhouette'],'kmeans_restarts':cfg['point_restarts']};point_labels,point_centers,point_k,_=ninea.fit_partition(ztrain,point_cfg,seed=seed+ri*100);point_test=np.argmin(np.linalg.norm(ztest[:,None,:]-point_centers[None,:,:],axis=2),axis=1);point_models=ninea.cluster_models(ztrain,point_labels,point_k,bounds)
        mixture,candidates=select_mixture(ztrain,ctrain,cfg,seed+ri*100);posterior_train=mixture['responsibility'];posterior_test=predict_membership(ztest,ctest,mixture);hard_train=np.argmax(posterior_train,axis=1);hard_test=np.argmax(posterior_test,axis=1);mixture_models={j:denormalize_log_parameters(mixture['means'][j],bounds) for j in range(mixture['k'])}
        diagnostics.extend(dict(phase=phase,seed=seed,regime=regime['name'],selected_k=mixture['k'],train_ari=adjusted_rand_score(truth_train,hard_train),test_ari=adjusted_rand_score(truth_test,hard_test),mean_test_entropy=float(np.mean(-np.sum(posterior_test*np.log(np.maximum(posterior_test,1e-15)),axis=1))),candidate_k=r['k'],bic=r['bic'],log_likelihood=r['log_likelihood'],minimum_size=r['minimum_size'],admissible=r['admissible']) for r in candidates)
        oracle_models={j:denormalize_log_parameters(ztrain[truth_train==j].mean(0),bounds) for j in range(3)};global_model={0:denormalize_log_parameters(ztrain.mean(0),bounds)};local_models={j:fits[c.client_id] for j,c in enumerate(test)}
        assignments={'Local':np.arange(len(test)),'Global':np.zeros(len(test),int),'OracleFamily':truth_test,'PointK':point_test,'UncertainHard':hard_test};models={'Local':local_models,'Global':global_model,'OracleFamily':oracle_models,'PointK':point_models,'UncertainHard':mixture_models}
        controllers={}
        for method in assignments:controllers[method]={c.client_id:ninea.eightg.eightf.eighte.eightd.prior.eight.seven.seven.controller(models[method][int(assignments[method][j])]) for j,c in enumerate(test)}
        controllers['UncertainSoft']={c.client_id:blend_controller(posterior_test[j],mixture_models) for j,c in enumerate(test)}
        parameter_models={**models,'UncertainSoft':{j:denormalize_log_parameters(posterior_test[j]@mixture['means'],bounds) for j in range(len(test))}};model_counts={m:len(v) for m,v in models.items()};model_counts['UncertainSoft']=mixture['k'];assignment_all={**assignments,'UncertainSoft':np.arange(len(test))}
        t=np.arange(0,24+base.DT,base.DT);tests=[]
        for name in cfg['test_scenarios']:
            v,ref=corrected.scenario(name,t/2);tests.append((name,corrected.six.remap(v,(10,30)),ref))
        grid=np.linspace(10,30,161)
        for method,control_map in controllers.items():
            rhos=[]
            for c in test:
                pairs=[augmented_tracking_matrices(*discrete_bicycle_matrices(float(v),c.parameters,base.DT),base.DT) for v in grid];a=np.array([p[0] for p in pairs]);b=np.array([p[1] for p in pairs]);control=control_map[c.client_id];g=np.column_stack([np.interp(grid,control.speeds,control.gains['global'][:,q]) for q in range(3)]);rhos.append(float(np.max(abs(np.linalg.eigvals(a-b@g[:,None,:])))))
            for scenario,v,ref in tests:
                _,_,metrics=corrected.simulate(test,nominal,v,ref,'vy',client_controllers=control_map)
                for j,c in enumerate(test):
                    model=parameter_models[method][int(assignment_all[method][j])];rows.append(dict(phase=phase,seed=seed,regime=regime['name'],method=method,models=model_counts[method],scenario=scenario,client=c.client_id,family=c.family,parameter_error=float(np.linalg.norm(np.log(model/nineb.ratios(c)))),rho=rhos[j],**{q:float(x[j]) for q,x in metrics.items()}))
        print(phase,seed,regime['name'],flush=True)
    for suffix,items in [('fits',fits_rows),('clusters',diagnostics),('clients',rows)]: (OUT/f'experiment_9c_{phase}_seed{seed}_{suffix}.csv.gz').write_bytes(gzip.compress(pd.DataFrame(items).to_csv(index=False).encode(),mtime=0))


def summarize(phases=('development','confirmation')):
    cfg=json.loads(CONFIG.read_text());parts=[]
    for phase,key in [('development','development_seeds'),('confirmation','confirmation_seeds')]:
        if phase not in phases:continue
        def read(suffix):return pd.concat([pd.read_csv(OUT/f'experiment_9c_{phase}_seed{s}_{suffix}.csv.gz') for s in cfg[key]],ignore_index=True)
        fits=read('fits');clusters=read('clusters');df=read('clients');keys=['phase','seed','regime','method','models'];units=df.groupby(keys).agg(tracking=('tracking','mean'),parameter_error=('parameter_error','mean'),feasible=('feasible','min'),rho=('rho','max')).reset_index();summary=units.groupby(['phase','regime','method']).agg(tracking=('tracking','mean'),tracking_std=('tracking','std'),parameter_error=('parameter_error','mean'),models=('models','mean'),models_min=('models','min'),models_max=('models','max'),feasible_rate=('feasible','mean'),rho=('rho','max')).reset_index();selected=clusters[clusters.candidate_k==clusters.selected_k].groupby(['phase','regime']).agg(k_mean=('selected_k','mean'),k_min=('selected_k','min'),k_max=('selected_k','max'),train_ari=('train_ari','mean'),test_ari=('test_ari','mean'),entropy=('mean_test_entropy','mean')).reset_index();parts.append((fits,units,summary,selected))
    fits=pd.concat([p[0] for p in parts]);units=pd.concat([p[1] for p in parts]);summary=pd.concat([p[2] for p in parts]);selected=pd.concat([p[3] for p in parts]);gates=[]
    for phase in phases:
        for regime in [r['name'] for r in cfg['regimes']]:
            s=summary[(summary.phase==phase)&(summary.regime==regime)].set_index('method');u=s.loc['UncertainSoft'];oracle=s.loc['OracleFamily'];point=s.loc['PointK'];cost=100*(u.tracking/oracle.tracking-1);improvement=100*(1-u.tracking/point.tracking);passed=bool(cost<=cfg['oracle_tracking_tolerance_pct'] and u.tracking<point.tracking and u.feasible_rate==1 and u.rho<1 and u.models<s.loc['Local','models']);gates.append(dict(phase=phase,regime=regime,tracking_cost_vs_oracle_pct=float(cost),improvement_vs_point_pct=float(improvement),passed=passed))
    gate_df=pd.DataFrame(gates);development=gate_df[gate_df.phase=='development'];confirmation=gate_df[gate_df.phase=='confirmation'];evaluations=int(sum(len(pd.read_csv(OUT/f'experiment_9c_{phase}_seed{s}_clients.csv.gz')) for phase,key in [('development','development_seeds'),('confirmation','confirmation_seeds')] if phase in phases for s in cfg[key]));conclusions=dict(provenance={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [CONFIG,ROOT/'code/src/federated_lpv/uncertainty_clustering.py',ROOT/'code/experiments/experiment_9c_uncertainty_groups.py']},local_fits=len(fits),closed_loop_evaluations=evaluations,all_local_fits_converged=bool(fits.success.all()),development_gates=development.to_dict('records'),development_gate=bool(len(development)>0 and development.passed.all()),confirmation_run=bool(len(confirmation)>0),confirmation_gates=confirmation.to_dict('records'),confirmation_gate=None if confirmation.empty else bool(confirmation.passed.all()),reserved_confirmation_seeds=cfg['confirmation_seeds'] if confirmation.empty else [],privacy_applied=False);units.to_csv(OUT/'experiment_9c_seed_summary.csv',index=False);summary.to_csv(OUT/'experiment_9c_summary.csv',index=False);selected.to_csv(OUT/'experiment_9c_cluster_selection.csv',index=False);gate_df.to_csv(OUT/'experiment_9c_gates.csv',index=False);(OUT/'experiment_9c_conclusions.json').write_text(json.dumps(conclusions,indent=2)+'\n')
    display_phase='confirmation' if 'confirmation' in phases else 'development';view=summary[(summary.phase==display_phase)&summary.method.isin(['OracleFamily','PointK','UncertainSoft'])];pivot=view.pivot(index='regime',columns='method',values='tracking').loc[[r['name'] for r in cfg['regimes']]];pivot.plot.bar(figsize=(7.5,3.8),rot=0);plt.ylabel('Tracking RMSE [rad/s]');plt.grid(axis='y',alpha=.2);plt.tight_layout();plt.savefig(ROOT/'results/figures/experiment_9c_uncertainty_groups.pdf');plt.close();print(json.dumps(conclusions,indent=2));print(summary.to_string(index=False));print(selected.to_string(index=False));print(gate_df.to_string(index=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['development','confirmation','all'],default='all');parser.add_argument('--summarize-only',action='store_true');parser.add_argument('--no-summarize',action='store_true');parser.add_argument('--workers',type=int,default=5);args=parser.parse_args();cfg=json.loads(CONFIG.read_text())
    if not args.summarize_only:
        phases=['development','confirmation'] if args.phase=='all' else [args.phase];tasks=[(phase,seed) for phase in phases for seed in cfg['development_seeds' if phase=='development' else 'confirmation_seeds']]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:list(pool.map(run_seed,tasks))
    if not args.no_summarize:summarize(('development','confirmation') if args.phase=='all' else (args.phase,))
