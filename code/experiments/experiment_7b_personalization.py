"""7B: validation-selected family-informed local output-error identification."""
from concurrent.futures import ProcessPoolExecutor
import argparse
import gzip
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from federated_lpv.output_error import fit,prepare,predict
from federated_lpv.personalization import select_strength
from federated_lpv import discrete_bicycle_matrices,augmented_tracking_matrices
import experiment_7a_local_identification as seven

ROOT=seven.ROOT;OUT=seven.OUT;base=seven.base;corrected=seven.corrected
CONFIG=ROOT/'code/config/experiment_7b.json'
METHODS=('Local','Family','Personalized')


def run_seed(seed):
    cfg=json.loads(CONFIG.read_text());clients=corrected.six.fleet(seed,1,cfg['clients_per_family'])
    records,categories,nominal=seven.collection_data(clients,seed,cfg)
    sigma=np.deg2rad(cfg['noise']['noisy'])
    data={c.client_id:[dict(state=r['state']+r['noise']*sigma,input=r['input'],speed=r['speed'])
                       for r in records['restricted'][c.client_id]] for c in clients}
    fitted={};diagnostics=[];cv=[];selection=[];manifest=[]

    def estimate(key,group,prior=None,strength=0):
        p,d=fit(group,prior,strength,cfg['prior_log_scale'])
        if not d['success']:raise RuntimeError(f'unconverged fit {seed} {key}')
        fitted[key]=p
        diagnostics.append(dict(seed=seed,group=key,strength=strength,
                                cf_over_m=p[0],cr_over_m=p[1],m_over_iz=p[2],**d))
        return p

    for family in base.FAMILIES:
        estimate(family,[r for c in clients if c.family==family for r in data[c.client_id]])
    for c in clients:
        own=data[c.client_id]
        local=estimate(c.client_id,own)
        donors=[d.client_id for d in clients if d.family==c.family and d.client_id!=c.client_id]
        prior=estimate('prior_'+c.client_id,[r for donor in donors for r in data[donor]])
        strength,folds=select_strength(own,prior,cfg['strengths'],cfg['prior_log_scale'])
        cv.extend(dict(seed=seed,client=c.client_id,family=c.family,**row) for row in folds)
        key='personal_'+c.client_id
        if strength==0:fitted[key]=local.copy()
        else:estimate(key,own,prior,strength)
        losses={s:np.mean([r['validation_loss'] for r in folds if r['strength']==s]) for s in cfg['strengths']}
        selection.append(dict(seed=seed,client=c.client_id,family=c.family,strength=strength,
                              local_validation_loss=losses[0],selected_validation_loss=losses[strength]))
        digest=hashlib.sha256()
        for r in own:
            for field in ('state','input','speed'):digest.update(np.ascontiguousarray(r[field]).tobytes())
        manifest.append(dict(seed=seed,client=c.client_id,family=c.family,category=categories[c.client_id],
                             donors='|'.join(donors),local_transitions=sum(len(r['input']) for r in own),
                             data_sha256=digest.hexdigest()))
    print(f'completed identification seed {seed}',flush=True)

    # No test trajectory is generated or inspected until selection is complete.
    t=np.arange(0,24+base.DT,base.DT);tests={};grid=np.linspace(10,30,161);audit={}
    for name in cfg['test_scenarios']:
        v,ref=corrected.scenario(name,t/2);v=corrected.six.remap(v,(10,30))
        x,u,_=corrected.simulate(clients,nominal,v,ref,'vy')
        tests[name]=(v,ref,[dict(state=x[:,j],input=u[:,j],speed=v) for j in range(len(clients))])
    for c in clients:
        pairs=[augmented_tracking_matrices(*discrete_bicycle_matrices(float(v),c.parameters,base.DT),base.DT) for v in grid]
        audit[c.client_id]=(np.array([p[0] for p in pairs]),np.array([p[1] for p in pairs]))
    designs={key:seven.controller(p) for key,p in fitted.items() if not key.startswith('prior_')}
    rows=[];errors=[]
    for method in METHODS:
        def key(c):return c.family if method=='Family' else ('personal_'+c.client_id if method=='Personalized' else c.client_id)
        selected={c.client_id:designs[key(c)] for c in clients};rhos={}
        for c in clients:
            a,b=audit[c.client_id];control=selected[c.client_id]
            gains=np.column_stack([np.interp(grid,control.speeds,control.gains['global'][:,j]) for j in range(3)])
            rhos[c.client_id]=float(np.max(abs(np.linalg.eigvals(a-b@gains[:,None,:]))))
        for name,(v,ref,common) in tests.items():
            _,_,metrics=corrected.simulate(clients,nominal,v,ref,'vy',client_controllers=selected)
            for j,c in enumerate(clients):
                p=fitted[key(c)];truth=np.array([c.parameters.front_stiffness/c.parameters.mass,
                    c.parameters.rear_stiffness/c.parameters.mass,c.parameters.mass/c.parameters.yaw_inertia])
                err=np.rad2deg(np.sqrt(np.mean((predict(p,prepare([common[j]]))-common[j]['state'][1:])**2,axis=0)))
                meta=dict(seed=seed,scenario=name,method=method,client=c.client_id,family=c.family)
                rows.append(dict(**meta,rho=rhos[c.client_id],**{k:float(value[j]) for k,value in metrics.items()}))
                errors.append(dict(**meta,beta_prediction=err[0],yaw_prediction=err[1],
                    parameter_relative_error=float(np.linalg.norm((p-truth)/truth)/np.sqrt(3))))
    for suffix,items in [('clients',rows),('parameters',diagnostics),('prediction',errors),('validation',cv),
                         ('selection',selection),('manifest',manifest)]:
        (OUT/f'experiment_7b_seed{seed}_{suffix}.csv.gz').write_bytes(gzip.compress(pd.DataFrame(items).to_csv(index=False).encode(),mtime=0))
    print(f'completed evaluation seed {seed}',flush=True)


def summarize():
    cfg=json.loads(CONFIG.read_text())
    def read(suffix):return pd.concat([pd.read_csv(OUT/f'experiment_7b_seed{s}_{suffix}.csv.gz') for s in cfg['seeds']],ignore_index=True)
    df=read('clients');pr=read('prediction');pa=read('parameters');cv=read('validation');sel=read('selection');manifest=read('manifest')
    if len(df)!=2700 or df.duplicated(['seed','scenario','method','client']).any():raise RuntimeError('incomplete evaluation')
    if len(cv)!=5400 or len(sel)!=300:raise RuntimeError('incomplete validation')
    if not all(row.client not in row.donors.split('|') and len(row.donors.split('|'))==9 for row in manifest.itertuples()):raise RuntimeError('prior leakage')
    keys=['seed','method']
    sr=df.groupby(keys).agg(tracking=('tracking','mean'),feasible=('feasible','min'),rho=('rho','max'),
                           steering_rms=('steering_rms','mean'),peak_rate=('peak_steering_rate','max')).reset_index()
    worst=df.groupby(keys+['family']).tracking.mean().groupby(keys).max().rename('worst_family').reset_index()
    sr=sr.merge(worst,on=keys).merge(pr.groupby(keys)[['beta_prediction','yaw_prediction','parameter_relative_error']].mean().reset_index(),on=keys)
    sr.to_csv(OUT/'experiment_7b_seed_summary.csv',index=False)
    summary=sr.groupby('method').agg(mean=('tracking','mean'),std=('tracking','std'),worst_family=('worst_family','mean'),
        feasible=('feasible','min'),rho=('rho','max'),steering_rms=('steering_rms','mean'),peak_rate=('peak_rate','max'),
        beta_prediction=('beta_prediction','mean'),yaw_prediction=('yaw_prediction','mean'),parameter_relative_error=('parameter_relative_error','mean')).reset_index()
    summary.to_csv(OUT/'experiment_7b_summary.csv',index=False)
    rng=np.random.default_rng(cfg['bootstrap_seed']);idx=rng.integers(0,len(cfg['seeds']),(cfg['bootstrap_repetitions'],len(cfg['seeds'])))
    comparisons=[]
    for baseline in ('Local','Family'):
        for metric in ('tracking','worst_family','yaw_prediction','parameter_relative_error'):
            a=sr[sr.method==baseline].sort_values('seed')[metric].to_numpy();b=sr[sr.method=='Personalized'].sort_values('seed')[metric].to_numpy()
            delta=a-b;ci=np.quantile(delta[idx].mean(axis=1),[.025,.975])
            comparisons.append(dict(baseline=baseline,metric=metric,reduction_pct=100*delta.mean()/a.mean(),
                                    mean_difference=delta.mean(),ci_low=ci[0],ci_high=ci[1],positive_pairs=int(sum(delta>0))))
    pd.DataFrame(comparisons).to_csv(OUT/'experiment_7b_comparisons.csv',index=False)
    counts=sel.strength.value_counts().reindex(cfg['strengths'],fill_value=0).rename_axis('strength').reset_index(name='clients')
    counts.to_csv(OUT/'experiment_7b_selection_counts.csv',index=False)
    primary=comparisons[0];p=summary[summary.method=='Personalized'].iloc[0]
    provenance={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in
                [CONFIG,ROOT/'code/experiments/experiment_7b_personalization.py',ROOT/'code/src/federated_lpv/output_error.py',ROOT/'code/src/federated_lpv/personalization.py']}
    all_diagnostics=pd.concat([pa,cv],ignore_index=True)
    conclusions=dict(provenance=provenance,evaluations=len(df),fits=len(all_diagnostics),
        all_fits_converged=bool(all_diagnostics.success.all()),all_starts_converged=bool(all_diagnostics.both_starts_success.all()),
        bound_hits=int(all_diagnostics.bound_hit.sum()),max_multistart_difference=float(all_diagnostics.multistart_difference.max()),
        all_amplitude_feasible=bool(df.feasible.min()==1),max_frozen_rho=float(df.rho.max()),
        selected_zero=int((sel.strength==0).sum()),selected_max=int((sel.strength==max(cfg['strengths'])).sum()),
        primary=primary,personalization_gate=bool(primary['reduction_pct']>=cfg['primary_min_reduction_pct'] and primary['ci_low']>0 and p.feasible==1 and p.rho<1))
    (OUT/'experiment_7b_conclusions.json').write_text(json.dumps(conclusions,indent=2)+'\n')
    fig,axes=plt.subplots(1,3,figsize=(11,3.4),constrained_layout=True)
    colors=['#2878b5','#34934b','#9867aa']
    for ax,metric,title in zip(axes[:2],['mean','yaw_prediction'],['Held-out tracking','Common-input prediction']):
        sub=summary.set_index('method').loc[list(METHODS)]
        ax.bar(METHODS,sub[metric],color=colors)
        if metric=='mean':ax.errorbar(METHODS,sub[metric],yerr=sub['std'],fmt='none',ecolor='black',capsize=3)
        ax.set(title=title,ylabel='Yaw RMSE [deg/s]');ax.tick_params(axis='x',labelrotation=20);ax.grid(axis='y',alpha=.2)
    axes[2].bar([str(s) for s in cfg['strengths']],counts.clients,color=colors[2])
    axes[2].set(title='Validation-selected strength',xlabel='Regularization strength',ylabel='Clients (out of 300)')
    fig.savefig(ROOT/'results/figures/experiment_7b_personalization.pdf');plt.close(fig)
    print(json.dumps(conclusions,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--summarize-only',action='store_true');parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    if not args.summarize_only:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:list(pool.map(run_seed,json.loads(CONFIG.read_text())['seeds']))
    summarize()
