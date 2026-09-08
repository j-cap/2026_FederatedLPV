"""9B: robustness of label-free LPV compatibility discovery."""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import argparse,gzip,hashlib,json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score
from federated_lpv import VehicleClient,VehicleParameters,discrete_bicycle_matrices,augmented_tracking_matrices
from federated_lpv.output_error import fit
from federated_lpv.privacy import normalize_log_parameters,denormalize_log_parameters
from federated_lpv.vehicle import family_centers
import experiment_9a_learned_groups as ninea

ROOT=ninea.ROOT;OUT=ninea.OUT;base=ninea.base;corrected=ninea.corrected
CONFIG=ROOT/'code/config/experiment_9b.json'


def discrete_fleet(seed,counts):
    raw=corrected.six.fleet(seed,1,max(counts));result=[]
    for family,count in zip(base.FAMILIES,counts):result.extend([c for c in raw if c.family==family][:count])
    return result


def continuous_fleet(seed,count):
    rng=np.random.default_rng(seed);centers=family_centers();a=centers['handling'];b=centers['heavy'];result=[]
    locations=(np.arange(count)+rng.uniform(size=count))/count;rng.shuffle(locations)
    for j,t in enumerate(locations):
        vals={name:(1-t)*getattr(a,name)+t*getattr(b,name) for name in VehicleParameters.__dataclass_fields__}
        vals['mass']*=rng.uniform(.99,1.01);vals['yaw_inertia']*=rng.uniform(.99,1.01);vals['front_stiffness']*=rng.uniform(.98,1.02);vals['rear_stiffness']*=rng.uniform(.98,1.02)
        result.append(VehicleClient(f'continuum_{j:03d}','continuum',VehicleParameters(**vals)))
    return result


def collect(clients,seed,cfg):
    nominal=ninea.eightg.eightf.eighte.eightd.prior.eight.seven.seven.controller(np.array([80000/1500,80000/1500,1500/2500]));n=round(cfg['episode_duration']/base.DT);t=np.arange(n+1)*base.DT;rng=np.random.default_rng(seed+900000);speeds=cfg['coverage_speeds'];order=rng.permutation(len(clients));categories={c.client_id:int(order[j]%len(speeds)) for j,c in enumerate(clients)};data={c.client_id:[] for c in clients}
    collection_controllers={c.client_id:nominal for c in clients}
    for episode in range(cfg['local_episodes']):
        ref=np.deg2rad(3)*(1-np.exp(-t/.3))*np.sin(2*np.pi*.35*t+.7*episode)
        for c in clients:
            speed=np.full_like(t,speeds[categories[c.client_id]]);x,u,m=corrected.simulate([c],nominal,speed,ref,'vy',client_controllers=collection_controllers)
            if not m['feasible'].all():raise RuntimeError('collection infeasible')
            data[c.client_id].append(dict(state=x[:,0]+rng.normal(size=x[:,0].shape)*np.deg2rad(cfg['noise_deg']),input=u[:,0],speed=speed))
    return data,nominal


def ratios(c):return ninea.eightg.true_ratios(c)


def run_seed(seed):
    cfg=json.loads(CONFIG.read_text());all_rows=[];all_diag=[];all_fits=[]
    for ri,regime in enumerate(cfg['regimes']):
        maker=continuous_fleet if regime['kind']=='continuous' else discrete_fleet;train=maker(seed+ri*1000,regime['train_counts'][0] if regime['kind']=='continuous' else regime['train_counts']);test=maker(seed+10000+ri*1000,regime['test_counts'][0] if regime['kind']=='continuous' else regime['test_counts']);local_cfg={**cfg,**regime};train_data,nominal=collect(train,seed+ri*1000,local_cfg);test_data,_=collect(test,seed+20000+ri*1000,local_cfg);fits={}
        for scope,clients,data in [('train',train,train_data),('test',test,test_data)]:
            for c in clients:
                p,d=fit(data[c.client_id]);fits[c.client_id]=p;all_fits.append(dict(seed=seed,regime=regime['name'],scope=scope,client=c.client_id,family=c.family,parameter_error=float(np.linalg.norm(np.log(p/ratios(c)))),**d))
        bounds=np.asarray(cfg['bounds'],float);ztrain=normalize_log_parameters([fits[c.client_id] for c in train],bounds);ztest=normalize_log_parameters([fits[c.client_id] for c in test],bounds);zt_train=normalize_log_parameters([ratios(c) for c in train],bounds);zt_test=normalize_log_parameters([ratios(c) for c in test],bounds)
        learned_labels,learned_centers,k,records=ninea.fit_partition(ztrain,cfg,seed=seed+ri*100);learned_test=np.argmin(np.linalg.norm(ztest[:,None,:]-learned_centers[None,:,:],axis=2),axis=1)
        true_labels,true_centers,kt,true_records=ninea.fit_partition(zt_train,cfg,seed=seed+ri*100+50);true_test=np.argmin(np.linalg.norm(zt_test[:,None,:]-true_centers[None,:,:],axis=2),axis=1)
        assignments={'Local':np.arange(len(test)),'Global':np.zeros(len(test),int),'LearnedK':learned_test,'OraclePartition':true_test};models={'Local':{j:fits[c.client_id] for j,c in enumerate(test)},'Global':{0:denormalize_log_parameters(ztrain.mean(0),bounds)},'LearnedK':ninea.cluster_models(ztrain,learned_labels,k,bounds),'OraclePartition':ninea.cluster_models(ztrain,true_labels,kt,bounds)}
        if regime['kind']=='discrete':
            fam={f:j for j,f in enumerate(base.FAMILIES)};tr=np.array([fam[c.family] for c in train]);te=np.array([fam[c.family] for c in test]);assignments['OracleFamily']=te;models['OracleFamily']={j:denormalize_log_parameters(ztrain[tr==j].mean(0),bounds) for j in range(3)};train_ari=adjusted_rand_score(tr,learned_labels);test_ari=adjusted_rand_score(te,learned_test)
        else:train_ari=np.nan;test_ari=np.nan
        for source,recs,selected_k in [('LearnedK',records,k),('OraclePartition',true_records,kt)]:all_diag.extend(dict(seed=seed,regime=regime['name'],source=source,selected_k=selected_k,train_ari=train_ari if source=='LearnedK' else np.nan,test_ari=test_ari if source=='LearnedK' else np.nan,candidate_k=r['k'],silhouette=r['silhouette'],minimum_size=r['min_cluster_size'],admissible=r['admissible']) for r in recs)
        t=np.arange(0,24+base.DT,base.DT);tests=[]
        for name in cfg['test_scenarios']:
            v,ref=corrected.scenario(name,t/2);tests.append((name,corrected.six.remap(v,(10,30)),ref))
        grid=np.linspace(10,30,161);audit={}
        for c in test:
            pairs=[augmented_tracking_matrices(*discrete_bicycle_matrices(float(v),c.parameters,base.DT),base.DT) for v in grid];audit[c.client_id]=(np.array([p[0] for p in pairs]),np.array([p[1] for p in pairs]))
        for method in assignments:
            labels=assignments[method];controls={c.client_id:ninea.eightg.eightf.eighte.eightd.prior.eight.seven.seven.controller(models[method][int(labels[j])]) for j,c in enumerate(test)};rhos=[]
            for c in test:
                a,b=audit[c.client_id];control=controls[c.client_id];g=np.column_stack([np.interp(grid,control.speeds,control.gains['global'][:,q]) for q in range(3)]);rhos.append(float(np.max(abs(np.linalg.eigvals(a-b@g[:,None,:])))))
            for scenario,v,ref in tests:
                _,_,m=corrected.simulate(test,nominal,v,ref,'vy',client_controllers=controls)
                for j,c in enumerate(test):all_rows.append(dict(seed=seed,regime=regime['name'],kind=regime['kind'],method=method,models=len(models[method]),scenario=scenario,client=c.client_id,family=c.family,assigned_cluster=int(labels[j]),parameter_error=float(np.linalg.norm(np.log(models[method][int(labels[j])]/ratios(c)))),rho=rhos[j],**{q:float(x[j]) for q,x in m.items()}))
        print(seed,regime['name'],flush=True)
    for suffix,items in [('fits',all_fits),('clusters',all_diag),('clients',all_rows)]: (OUT/f'experiment_9b_seed{seed}_{suffix}.csv.gz').write_bytes(gzip.compress(pd.DataFrame(items).to_csv(index=False).encode(),mtime=0))


def summarize():
    cfg=json.loads(CONFIG.read_text());read=lambda x:pd.concat([pd.read_csv(OUT/f'experiment_9b_seed{s}_{x}.csv.gz') for s in cfg['seeds']],ignore_index=True);fits=read('fits');clusters=read('clusters');df=read('clients');keys=['seed','regime','kind','method','models'];units=df.groupby(keys).agg(tracking=('tracking','mean'),worst_client=('tracking',lambda x:x.groupby(df.loc[x.index,'client']).mean().max()),parameter_error=('parameter_error','mean'),feasible=('feasible','min'),rho=('rho','max')).reset_index();summary=units.groupby(['regime','kind','method']).agg(tracking=('tracking','mean'),tracking_std=('tracking','std'),worst_client=('worst_client','mean'),parameter_error=('parameter_error','mean'),models=('models','mean'),models_min=('models','min'),models_max=('models','max'),feasible_rate=('feasible','mean'),rho=('rho','max')).reset_index();selected=clusters[clusters.candidate_k==clusters.selected_k].groupby(['regime','source']).agg(k_mean=('selected_k','mean'),k_min=('selected_k','min'),k_max=('selected_k','max'),test_ari=('test_ari','mean'),silhouette=('silhouette','mean'),minimum_size=('minimum_size','min')).reset_index()
    gates=[]
    for regime in cfg['regimes']:
        s=summary[summary.regime==regime['name']].set_index('method');learn=s.loc['LearnedK']
        if regime['kind']=='discrete':
            reference=s.loc['OracleFamily'];ari=selected[(selected.regime==regime['name'])&(selected.source=='LearnedK')].test_ari.iloc[0];cost=100*(learn.tracking/reference.tracking-1);passed=cost<=cfg['discrete_tracking_tolerance_pct'] and ari>=cfg['discrete_minimum_test_ari']
        else:
            reference=s.loc['OraclePartition'];ari=np.nan;cost=100*(learn.tracking/reference.tracking-1);gain=100*(1-learn.tracking/s.loc['Global','tracking']);passed=cost<=cfg['continuous_oracle_tolerance_pct'] and gain>=cfg['continuous_global_improvement_pct']
        passed=bool(passed and learn.feasible_rate==1 and learn.rho<1 and learn.models<s.loc['Local','models']);gates.append(dict(regime=regime['name'],kind=regime['kind'],tracking_cost_vs_reference_pct=float(cost),global_improvement_pct=float(100*(1-learn.tracking/s.loc['Global','tracking'])),test_ari=None if np.isnan(ari) else float(ari),passed=passed))
    gate_df=pd.DataFrame(gates);conclusions=dict(provenance={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [CONFIG,ROOT/'code/experiments/experiment_9b_group_robustness.py']},local_fits=len(fits),closed_loop_evaluations=len(df),all_local_fits_converged=bool(fits.success.all()),regime_gates=gates,robustness_gate=bool(gate_df.passed.all()),privacy_applied=False);units.to_csv(OUT/'experiment_9b_seed_summary.csv',index=False);summary.to_csv(OUT/'experiment_9b_summary.csv',index=False);selected.to_csv(OUT/'experiment_9b_cluster_selection.csv',index=False);gate_df.to_csv(OUT/'experiment_9b_gates.csv',index=False);(OUT/'experiment_9b_conclusions.json').write_text(json.dumps(conclusions,indent=2)+'\n')
    learned=summary[summary.method=='LearnedK'].set_index('regime').loc[[r['name'] for r in cfg['regimes']]];fig,axes=plt.subplots(1,2,figsize=(10,3.6),constrained_layout=True);x=np.arange(len(learned));axes[0].bar(x,learned.tracking);axes[0].set(xticks=x,xticklabels=learned.index,ylabel='Tracking RMSE [rad/s]',title='Learned-group control');axes[0].tick_params(axis='x',rotation=25);axes[0].grid(axis='y',alpha=.2);sel=selected[selected.source=='LearnedK'].set_index('regime').loc[learned.index];axes[1].bar(x,sel.k_mean);axes[1].set(xticks=x,xticklabels=learned.index,ylabel='Selected clusters',title='Learned model count');axes[1].tick_params(axis='x',rotation=25);axes[1].grid(axis='y',alpha=.2);fig.savefig(ROOT/'results/figures/experiment_9b_group_robustness.pdf');plt.close(fig);print(json.dumps(conclusions,indent=2));print(summary.to_string(index=False));print(selected.to_string(index=False));print(gate_df.to_string(index=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--summarize-only',action='store_true');p.add_argument('--workers',type=int,default=5);a=p.parse_args()
    if not a.summarize_only:
        with ProcessPoolExecutor(max_workers=a.workers) as pool:list(pool.map(run_seed,json.loads(CONFIG.read_text())['seeds']))
    summarize()
