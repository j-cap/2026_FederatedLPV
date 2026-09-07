"""7A: finite-data output-error fitting before any federated implementation."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import argparse
import gzip
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from federated_lpv.output_error import fit,prepare,predict
from federated_lpv import ScheduledController, design_lqi_gain, discrete_bicycle_matrices, augmented_tracking_matrices
from experiment_4f_structured_identification import StructuredModel
import experiment_6b_acceleration_validation as corrected

ROOT=corrected.ROOT;OUT=ROOT/'results/tables';base=corrected.base
CONFIG=ROOT/'code/config/experiment_7a.json'
METHODS=('Local','Global','Family')

def controller(p):
    model=StructuredModel('global','lpv',{'global':p});grid=np.linspace(10,30,5)
    designs=[design_lqi_gain(*model.matrix('global',v),base.DT,base.Q,base.R) for v in grid]
    return ScheduledController('7A','global','linear',grid,{'global':np.array([r[0] for r in designs])},
        {'global':np.array([r[1] for r in designs])})

def collection_data(clients,seed,cfg):
    nominal=controller(np.array([80000/1500,80000/1500,1500/2500]))
    t=np.arange(0,max(cfg['episode_durations'])+base.DT,base.DT)
    rng=np.random.default_rng(seed+710000)
    order=rng.permutation(cfg['clients_per_family'])
    categories={c.client_id:int(order[int(c.client_id.rsplit('_',1)[1])]%3) for c in clients}
    records={coverage:{c.client_id:[] for c in clients} for coverage in cfg['coverage']}
    noise={c.client_id:rng.normal(size=(3,len(t),2)) for c in clients}
    for coverage in cfg['coverage']:
        for episode in range(3):
            phase=.7*episode
            reference=np.deg2rad(3)*(1-np.exp(-t/.3))*np.sin(2*np.pi*.35*t+phase)
            for category in range(3):
                selected=[c for c in clients if categories[c.client_id]==category]
                v=cfg['broad_speeds'][episode] if coverage=='broad' else cfg['restricted_speeds'][category][episode]
                speed=np.full_like(t,v)
                x,u,metrics=corrected.simulate(selected,nominal,speed,reference,'vy')
                if not metrics['feasible'].all():raise RuntimeError('infeasible collection')
                for j,c in enumerate(selected):
                    records[coverage][c.client_id].append(dict(state=x[:,j],input=u[:,j],speed=speed,
                        noise=noise[c.client_id][episode],family=c.family,client=c.client_id))
    return records,categories,nominal

def run_seed(seed):
    cfg=json.loads(CONFIG.read_text());clients=corrected.six.fleet(seed,1,cfg['clients_per_family'])
    records,categories,nominal=collection_data(clients,seed,cfg)
    t=np.arange(0,24+base.DT,base.DT);tests={}
    for name in cfg['test_scenarios']:
        v,ref=corrected.scenario(name,t/2);v=corrected.six.remap(v,(10,30))
        x,u,_=corrected.simulate(clients,nominal,v,ref,'vy')
        tests[name]=(v,ref,[dict(state=x[:,j],input=u[:,j],speed=v) for j in range(len(clients))])
    grid=np.linspace(10,30,161);audit={}
    for c in clients:
        pairs=[augmented_tracking_matrices(*discrete_bicycle_matrices(float(v),c.parameters,base.DT),base.DT) for v in grid]
        audit[c.client_id]=(np.array([p[0] for p in pairs]),np.array([p[1] for p in pairs]))
    rows=[];parameters=[];errors=[];manifest=[]
    for coverage in cfg['coverage']:
        for duration in cfg['episode_durations']:
            n=round(duration/base.DT)
            for noise_name,sigma in cfg['noise'].items():
                meta=dict(seed=seed,coverage=coverage,episode_duration=duration,noise=noise_name)
                data={}
                for c in clients:
                    data[c.client_id]=[dict(state=r['state'][:n+1]+r['noise'][:n+1]*np.deg2rad(sigma),
                        input=r['input'][:n],speed=r['speed'][:n+1],family=c.family) for r in records[coverage][c.client_id]]
                    manifest.append(dict(**meta,client=c.client_id,family=c.family,category=categories[c.client_id],
                        transitions=3*n,min_speed=min(r['speed'][0] for r in data[c.client_id]),
                        max_speed=max(r['speed'][0] for r in data[c.client_id])))
                groups={c.client_id:data[c.client_id] for c in clients}
                groups['global']=[r for rs in data.values() for r in rs]
                for f in base.FAMILIES:groups[f]=[r for c in clients if c.family==f for r in data[c.client_id]]
                fitted={};designs={}
                for key,group in groups.items():
                    p,diagnostics=fit(group)
                    if not diagnostics['success']:raise RuntimeError(f'unconverged fit {meta} {key}')
                    fitted[key]=p;designs[key]=controller(p)
                    parameters.append(dict(**meta,group=key,cf_over_m=p[0],cr_over_m=p[1],m_over_iz=p[2],**diagnostics))
                for method in METHODS:
                    key=lambda c:c.client_id if method=='Local' else ('global' if method=='Global' else c.family)
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
                            prediction=predict(p,prepare([common[j]]));err=np.rad2deg(np.sqrt(np.mean((prediction-common[j]['state'][1:])**2,axis=0)))
                            rows.append(dict(**meta,scenario=name,method=method,client=c.client_id,family=c.family,
                                rho=rhos[c.client_id],**{k:float(value[j]) for k,value in metrics.items()}))
                            errors.append(dict(**meta,scenario=name,method=method,client=c.client_id,family=c.family,
                                beta_prediction=err[0],yaw_prediction=err[1],parameter_relative_error=float(np.linalg.norm((p-truth)/truth)/np.sqrt(3))))
                print('completed',meta,flush=True)
    for suffix,items in [('clients',rows),('parameters',parameters),('prediction',errors),('manifest',manifest)]:
        (OUT/f'experiment_7a_seed{seed}_{suffix}.csv.gz').write_bytes(gzip.compress(pd.DataFrame(items).to_csv(index=False).encode(),mtime=0))

def summarize():
    cfg=json.loads(CONFIG.read_text())
    def read(suffix):return pd.concat([pd.read_csv(OUT/f'experiment_7a_seed{s}_{suffix}.csv.gz') for s in cfg['seeds']],ignore_index=True)
    df=read('clients');pr=read('prediction');pa=read('parameters')
    unique=['seed','coverage','episode_duration','noise','scenario','method','client']
    if len(df)!=21600 or df.duplicated(unique).any():raise RuntimeError('incomplete/duplicate outputs')
    keys=['seed','coverage','episode_duration','noise','method']
    sr=df.groupby(keys).agg(tracking=('tracking','mean'),feasible=('feasible','min'),rho=('rho','max'),
        steering_rms=('steering_rms','mean'),peak_rate=('peak_steering_rate','max')).reset_index()
    worst=df.groupby(keys+['family']).tracking.mean().groupby(keys).max().rename('worst_family').reset_index()
    sr=sr.merge(worst,on=keys).merge(pr.groupby(keys)[['beta_prediction','yaw_prediction','parameter_relative_error']].mean().reset_index(),on=keys)
    sr.to_csv(OUT/'experiment_7a_seed_summary.csv',index=False)
    summary=sr.groupby(keys[1:]).agg(mean=('tracking','mean'),std=('tracking','std'),worst_family=('worst_family','mean'),
        feasible=('feasible','min'),rho=('rho','max'),steering_rms=('steering_rms','mean'),peak_rate=('peak_rate','max'),
        beta_prediction=('beta_prediction','mean'),yaw_prediction=('yaw_prediction','mean'),parameter_relative_error=('parameter_relative_error','mean')).reset_index()
    summary.to_csv(OUT/'experiment_7a_summary.csv',index=False)
    n=len(cfg['seeds']);rng=np.random.default_rng(cfg['bootstrap_seed']);idx=rng.integers(0,n,(cfg['bootstrap_repetitions'],n));comparisons=[]
    for (coverage,duration,noise),cell in sr.groupby(['coverage','episode_duration','noise']):
        for baseline in ('Local','Global'):
            for metric in ('tracking','worst_family','yaw_prediction'):
                a=cell[cell.method==baseline].sort_values('seed')[metric].to_numpy();b=cell[cell.method=='Family'].sort_values('seed')[metric].to_numpy()
                delta=a-b;ci=np.quantile(delta[idx].mean(axis=1),[.025,.975])
                comparisons.append(dict(coverage=coverage,episode_duration=duration,noise=noise,baseline=baseline,metric=metric,
                    reduction_pct=100*delta.mean()/a.mean(),mean_difference=delta.mean(),ci_low=ci[0],ci_high=ci[1],positive_pairs=int(sum(delta>0))))
    pd.DataFrame(comparisons).to_csv(OUT/'experiment_7a_comparisons.csv',index=False)
    primary=next(r for r in comparisons if r['coverage']=='restricted' and r['episode_duration']==2 and r['noise']=='noisy' and r['baseline']=='Local' and r['metric']=='tracking')
    f=summary[(summary.coverage=='restricted')&(summary.episode_duration==2)&(summary.noise=='noisy')&(summary.method=='Family')].iloc[0]
    conclusions=dict(protocol_sha256=hashlib.sha256(CONFIG.read_bytes()).hexdigest(),evaluations=len(df),fits=len(pa),
        all_fits_converged=bool(pa.success.all()),all_starts_converged=bool(pa.both_starts_success.all()),bound_hits=int(pa.bound_hit.sum()),
        max_condition=float(pa.condition.max()),max_multistart_difference=float(pa.multistart_difference.max()),
        all_amplitude_feasible=bool(df.feasible.min()==1),max_frozen_rho=float(df.rho.max()),primary=primary,
        collaboration_gate=bool(primary['reduction_pct']>=5 and primary['ci_low']>0 and f.feasible==1 and f.rho<1))
    (OUT/'experiment_7a_conclusions.json').write_text(json.dumps(conclusions,indent=2,default=lambda v:v.item())+'\n')
    fig,axes=plt.subplots(1,2,figsize=(9,3.6),constrained_layout=True)
    for ax,metric,title in zip(axes,['mean','yaw_prediction'],['Closed-loop tracking','Common-input free-run prediction']):
        for method in METHODS:
            sub=summary[(summary.coverage=='restricted')&(summary.noise=='noisy')&(summary.method==method)].sort_values('episode_duration')
            ax.plot(3*sub.episode_duration,sub[metric],'o-',label=method)
        ax.set(xlabel='Local data duration [s]',ylabel='Yaw RMSE [deg/s]',title=title,xticks=[6,24]);ax.grid(alpha=.2);ax.legend()
    fig.savefig(ROOT/'results/figures/experiment_7a_local_identification.pdf');plt.close(fig)
    print(json.dumps(conclusions,indent=2,default=lambda v:v.item()))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--summarize-only',action='store_true');parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    if not args.summarize_only:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:list(pool.map(run_seed,json.loads(CONFIG.read_text())['seeds']))
    summarize()
