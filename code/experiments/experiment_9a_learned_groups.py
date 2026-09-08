"""9A: learn the number and membership of compatible LPV client groups."""
from concurrent.futures import ProcessPoolExecutor
import argparse,gzip,hashlib,json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score,silhouette_score
from federated_lpv.output_error import fit
from federated_lpv.privacy import normalize_log_parameters,denormalize_log_parameters
from federated_lpv import discrete_bicycle_matrices,augmented_tracking_matrices
import experiment_8g_blind_confirmation as eightg

ROOT=eightg.ROOT;OUT=eightg.OUT;base=eightg.base;corrected=eightg.corrected
CONFIG=ROOT/'code/config/experiment_9a.json'
METHODS=('Local','Global','OracleFamily','FixedK3','LearnedKParameter','LearnedKControl')


def fit_partition(z,cfg,weights=None,fixed_k=None,seed=0):
    """Fit a label-free K-means partition and select K by silhouette."""
    x=z if weights is None else z*np.asarray(weights,float)
    candidates=[fixed_k] if fixed_k is not None else cfg['candidate_clusters']
    records=[];models={}
    for k in candidates:
        model=KMeans(n_clusters=k,n_init=cfg['kmeans_restarts'],random_state=seed+k).fit(x)
        sizes=np.bincount(model.labels_,minlength=k);admissible=bool(sizes.min()>=cfg['minimum_cluster_size'])
        score=float(silhouette_score(x,model.labels_)) if admissible else -np.inf
        records.append(dict(k=k,silhouette=score,min_cluster_size=int(sizes.min()),admissible=admissible));models[k]=model
    valid=[r for r in records if r['admissible'] and r['silhouette']>=cfg['minimum_silhouette']]
    if not valid:return np.zeros(len(z),int),np.mean(x,axis=0,keepdims=True),1,records
    best=max(valid,key=lambda r:(r['silhouette'],-r['k']));model=models[best['k']]
    return model.labels_,model.cluster_centers_,best['k'],records


def cluster_models(z,labels,k,bounds):
    return {j:denormalize_log_parameters(z[labels==j].mean(axis=0),bounds) for j in range(k)}


def run_seed(seed):
    cfg=json.loads(CONFIG.read_text());train=corrected.six.fleet(seed,1,cfg['training_clients_per_family']);train_data,_,nominal=eightg.eightf.collect(train,seed,cfg)
    test=corrected.six.fleet(seed+10000,1,cfg['test_clients_per_family']);test_data,_,_=eightg.eightf.collect(test,seed+20000,cfg)
    fits={};fit_rows=[]
    for scope,clients,data in [('train',train,train_data),('test',test,test_data)]:
        for c in clients:
            p,d=fit(data[c.client_id]);fits[c.client_id]=p
            fit_rows.append(dict(seed=seed,scope=scope,client=c.client_id,family=c.family,cf_over_m=p[0],cr_over_m=p[1],m_over_iz=p[2],parameter_error=float(np.linalg.norm(np.log(p/eightg.true_ratios(c)))),**d))
    bounds=np.asarray(cfg['bounds'],float);ztrain=normalize_log_parameters([fits[c.client_id] for c in train],bounds);ztest=normalize_log_parameters([fits[c.client_id] for c in test],bounds)
    family_names=list(base.FAMILIES);family_to_int={f:j for j,f in enumerate(family_names)};train_truth=np.array([family_to_int[c.family] for c in train]);test_truth=np.array([family_to_int[c.family] for c in test])
    assignments={};models={};diagnostics=[]
    assignments['Local']=np.arange(len(test));models['Local']={j:fits[c.client_id] for j,c in enumerate(test)}
    assignments['Global']=np.zeros(len(test),int);models['Global']={0:denormalize_log_parameters(ztrain.mean(axis=0),bounds)}
    assignments['OracleFamily']=test_truth;models['OracleFamily']={j:denormalize_log_parameters(ztrain[train_truth==j].mean(axis=0),bounds) for j in range(len(family_names))}
    for method,weights,fixed in [('FixedK3',None,3),('LearnedKParameter',None,None),('LearnedKControl',cfg['global_control_weights'],None)]:
        labels,centers,k,records=fit_partition(ztrain,cfg,weights,fixed,seed)
        x_test=ztest if weights is None else ztest*np.asarray(weights);test_labels=np.argmin(np.linalg.norm(x_test[:,None,:]-centers[None,:,:],axis=2),axis=1)
        assignments[method]=test_labels;models[method]=cluster_models(ztrain,labels,k,bounds)
        diagnostics.extend(dict(seed=seed,method=method,selected_k=k,train_ari=adjusted_rand_score(train_truth,labels),test_ari=adjusted_rand_score(test_truth,test_labels),candidate_k=r['k'],silhouette=r['silhouette'],minimum_size=r['min_cluster_size'],admissible=r['admissible']) for r in records)
    t=np.arange(0,24+base.DT,base.DT);tests=[]
    for name in cfg['test_scenarios']:
        v,ref=corrected.scenario(name,t/2);tests.append((name,corrected.six.remap(v,(10,30)),ref))
    grid=np.linspace(10,30,161);audit={}
    for c in test:
        pairs=[augmented_tracking_matrices(*discrete_bicycle_matrices(float(v),c.parameters,base.DT),base.DT) for v in grid];audit[c.client_id]=(np.array([p[0] for p in pairs]),np.array([p[1] for p in pairs]))
    rows=[]
    for method in METHODS:
        labels=assignments[method];controls={c.client_id:eightg.eightf.eighte.eightd.prior.eight.seven.seven.controller(models[method][int(labels[j])]) for j,c in enumerate(test)};rhos=[]
        for c in test:
            a,b=audit[c.client_id];control=controls[c.client_id];g=np.column_stack([np.interp(grid,control.speeds,control.gains['global'][:,q]) for q in range(3)]);rhos.append(float(np.max(abs(np.linalg.eigvals(a-b@g[:,None,:])))))
        for scenario,v,ref in tests:
            _,_,m=corrected.simulate(test,nominal,v,ref,'vy',client_controllers=controls)
            for j,c in enumerate(test):rows.append(dict(seed=seed,method=method,models=len(models[method]),scenario=scenario,client=c.client_id,family=c.family,assigned_cluster=int(labels[j]),parameter_error=float(np.linalg.norm(np.log(models[method][int(labels[j])]/eightg.true_ratios(c)))),rho=rhos[j],**{q:float(x[j]) for q,x in m.items()}))
        print(seed,method,flush=True)
    for suffix,items in [('fits',fit_rows),('clusters',diagnostics),('clients',rows)]:
        (OUT/f'experiment_9a_seed{seed}_{suffix}.csv.gz').write_bytes(gzip.compress(pd.DataFrame(items).to_csv(index=False).encode(),mtime=0))


def summarize():
    cfg=json.loads(CONFIG.read_text());read=lambda x:pd.concat([pd.read_csv(OUT/f'experiment_9a_seed{s}_{x}.csv.gz') for s in cfg['seeds']],ignore_index=True)
    fits=read('fits');clusters=read('clusters');df=read('clients');keys=['seed','method','models']
    units=df.groupby(keys).agg(tracking=('tracking','mean'),worst_client=('tracking',lambda x:x.groupby(df.loc[x.index,'client']).mean().max()),parameter_error=('parameter_error','mean'),feasible=('feasible','min'),rho=('rho','max'),steering_rms=('steering_rms','mean')).reset_index()
    summary=units.groupby('method').agg(tracking=('tracking','mean'),tracking_std=('tracking','std'),worst_client=('worst_client','mean'),parameter_error=('parameter_error','mean'),models=('models','mean'),models_min=('models','min'),models_max=('models','max'),feasible_rate=('feasible','mean'),rho=('rho','max'),steering_rms=('steering_rms','mean')).reset_index()
    selected=clusters[clusters.candidate_k==clusters.selected_k].groupby('method').agg(k_mean=('selected_k','mean'),k_min=('selected_k','min'),k_max=('selected_k','max'),train_ari=('train_ari','mean'),test_ari=('test_ari','mean'),silhouette=('silhouette','mean'),minimum_size=('minimum_size','min')).reset_index()
    s=summary.set_index('method');proposed=s.loc['LearnedKControl'];oracle=s.loc['OracleFamily'];gate=bool(proposed.tracking<=oracle.tracking and proposed.feasible_rate==1 and proposed.rho<1 and proposed.models<s.loc['Local','models'])
    conclusions=dict(provenance={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [CONFIG,ROOT/'code/experiments/experiment_9a_learned_groups.py']},local_fits=len(fits),closed_loop_evaluations=len(df),all_local_fits_converged=bool(fits.success.all()),learned_control_tracking=float(proposed.tracking),oracle_family_tracking=float(oracle.tracking),learned_control_models=float(proposed.models),local_models=float(s.loc['Local','models']),learned_grouping_gate=gate,privacy_applied=False)
    units.to_csv(OUT/'experiment_9a_seed_summary.csv',index=False);summary.to_csv(OUT/'experiment_9a_summary.csv',index=False);selected.to_csv(OUT/'experiment_9a_cluster_selection.csv',index=False);clusters.to_csv(OUT/'experiment_9a_candidate_scores.csv',index=False);(OUT/'experiment_9a_conclusions.json').write_text(json.dumps(conclusions,indent=2)+'\n')
    order=list(METHODS);view=summary.set_index('method').loc[order];fig,axes=plt.subplots(1,2,figsize=(10,3.6),constrained_layout=True);axes[0].bar(np.arange(len(order)),view.tracking);axes[0].set(xticks=np.arange(len(order)),xticklabels=['Local','Global','Oracle','Fixed K=3','Learned K\nparameter','Learned K\ncontrol'],ylabel='Tracking RMSE [rad/s]',title='Unseen-client control');axes[0].tick_params(axis='x',rotation=25);axes[0].grid(axis='y',alpha=.2)
    learn=selected.set_index('method').loc[['FixedK3','LearnedKParameter','LearnedKControl']];axes[1].bar(np.arange(3)-.18,learn.k_mean,.36,label='Models');axes[1].bar(np.arange(3)+.18,learn.test_ari,.36,label='Test ARI');axes[1].set(xticks=np.arange(3),xticklabels=['Fixed K=3','Learned K\nparameter','Learned K\ncontrol'],title='Learned structure');axes[1].legend();axes[1].grid(axis='y',alpha=.2);fig.savefig(ROOT/'results/figures/experiment_9a_learned_groups.pdf');plt.close(fig);print(json.dumps(conclusions,indent=2));print(summary.to_string(index=False));print(selected.to_string(index=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--summarize-only',action='store_true');p.add_argument('--workers',type=int,default=5);a=p.parse_args()
    if not a.summarize_only:
        with ProcessPoolExecutor(max_workers=a.workers) as pool:list(pool.map(run_seed,json.loads(CONFIG.read_text())['seeds']))
    summarize()
