"""7C: oracle-family sharing under complementary single-speed local coverage."""
from concurrent.futures import ProcessPoolExecutor
import argparse
import gzip
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from federated_lpv.output_error import fit, prepare, predict
from federated_lpv import discrete_bicycle_matrices, augmented_tracking_matrices
import experiment_7a_local_identification as seven

ROOT=seven.ROOT; OUT=seven.OUT; base=seven.base; corrected=seven.corrected
CONFIG=ROOT/'code/config/experiment_7c.json'
METHODS=('Local','Global','Family','LocalExtra')


def collect(clients, seed, cfg):
    nominal=seven.controller(np.array([80000/1500,80000/1500,1500/2500]))
    n=round(cfg['episode_duration']/base.DT); t=np.arange(n+1)*base.DT
    rng=np.random.default_rng(seed+730000)
    order=rng.permutation(cfg['clients_per_family'])
    categories={c.client_id:int(order[int(c.client_id.rsplit('_',1)[1])]) for c in clients}
    records={c.client_id:[] for c in clients}; extra={c.client_id:[] for c in clients}
    for episode in range(cfg['local_episodes']):
        phase=.7*episode
        reference=np.deg2rad(3)*(1-np.exp(-t/.3))*np.sin(2*np.pi*.35*t+phase)
        for c in clients:
            speed_value=cfg['coverage_speeds'][categories[c.client_id]]
            speed=np.full_like(t,speed_value)
            x,u,m=corrected.simulate([c],nominal,speed,reference,'vy')
            if not m['feasible'].all(): raise RuntimeError('infeasible local collection')
            noise=rng.normal(size=x[:,0].shape)*np.deg2rad(cfg['noise_deg'])
            records[c.client_id].append(dict(state=x[:,0]+noise,input=u[:,0],speed=speed))
    # Same transition budget as the ten-client family pool, but collected locally
    # across the complete speed set. Independent phases/noise prevent duplication.
    for c in clients:
        for k,speed_value in enumerate(cfg['coverage_speeds']):
            for episode in range(cfg['local_episodes']):
                phase=.7*episode+.11*k
                reference=np.deg2rad(3)*(1-np.exp(-t/.3))*np.sin(2*np.pi*.35*t+phase)
                speed=np.full_like(t,speed_value)
                x,u,m=corrected.simulate([c],nominal,speed,reference,'vy')
                if not m['feasible'].all(): raise RuntimeError('infeasible extra-local collection')
                noise=rng.normal(size=x[:,0].shape)*np.deg2rad(cfg['noise_deg'])
                extra[c.client_id].append(dict(state=x[:,0]+noise,input=u[:,0],speed=speed))
    return records,extra,categories,nominal


def run_seed(seed):
    cfg=json.loads(CONFIG.read_text()); clients=corrected.six.fleet(seed,1,cfg['clients_per_family'])
    data,extra,categories,nominal=collect(clients,seed,cfg)
    groups={c.client_id:data[c.client_id] for c in clients}
    groups.update({'extra_'+c.client_id:extra[c.client_id] for c in clients})
    groups['global']=[r for rs in data.values() for r in rs]
    for family in base.FAMILIES:
        groups[family]=[r for c in clients if c.family==family for r in data[c.client_id]]
    fitted={}; diagnostics=[]
    for key,group in groups.items():
        p,d=fit(group)
        if not d['success']: raise RuntimeError(f'unconverged fit {seed} {key}')
        fitted[key]=p; diagnostics.append(dict(seed=seed,group=key,cf_over_m=p[0],cr_over_m=p[1],m_over_iz=p[2],**d))
    print(f'completed identification seed {seed}',flush=True)

    designs={key:seven.controller(p) for key,p in fitted.items()}
    t=np.arange(0,24+base.DT,base.DT); tests={}
    for name in cfg['test_scenarios']:
        v,ref=corrected.scenario(name,t/2); v=corrected.six.remap(v,(10,30))
        x,u,_=corrected.simulate(clients,nominal,v,ref,'vy')
        tests[name]=(v,ref,[dict(state=x[:,j],input=u[:,j],speed=v) for j in range(len(clients))])
    grid=np.linspace(10,30,161); audit={}
    for c in clients:
        pairs=[augmented_tracking_matrices(*discrete_bicycle_matrices(float(v),c.parameters,base.DT),base.DT) for v in grid]
        audit[c.client_id]=(np.array([p[0] for p in pairs]),np.array([p[1] for p in pairs]))
    rows=[]; errors=[]; manifest=[]
    for c in clients:
        manifest.append(dict(seed=seed,client=c.client_id,family=c.family,category=categories[c.client_id],
            local_speed=cfg['coverage_speeds'][categories[c.client_id]],local_transitions=sum(len(r['input']) for r in data[c.client_id]),
            family_transitions=sum(len(r['input']) for d in clients if d.family==c.family for r in data[d.client_id]),
            extra_local_transitions=sum(len(r['input']) for r in extra[c.client_id])))
    for method in METHODS:
        def key(c):
            return c.client_id if method=='Local' else ('global' if method=='Global' else (c.family if method=='Family' else 'extra_'+c.client_id))
        selected={c.client_id:designs[key(c)] for c in clients}; rhos={}
        for c in clients:
            a,b=audit[c.client_id]; control=selected[c.client_id]
            gains=np.column_stack([np.interp(grid,control.speeds,control.gains['global'][:,j]) for j in range(3)])
            rhos[c.client_id]=float(np.max(abs(np.linalg.eigvals(a-b@gains[:,None,:]))))
        for name,(v,ref,common) in tests.items():
            _,_,metrics=corrected.simulate(clients,nominal,v,ref,'vy',client_controllers=selected)
            for j,c in enumerate(clients):
                p=fitted[key(c)]; truth=np.array([c.parameters.front_stiffness/c.parameters.mass,
                    c.parameters.rear_stiffness/c.parameters.mass,c.parameters.mass/c.parameters.yaw_inertia])
                err=np.rad2deg(np.sqrt(np.mean((predict(p,prepare([common[j]]))-common[j]['state'][1:])**2,axis=0)))
                meta=dict(seed=seed,scenario=name,method=method,client=c.client_id,family=c.family,local_speed=cfg['coverage_speeds'][categories[c.client_id]])
                rows.append(dict(**meta,rho=rhos[c.client_id],**{k:float(value[j]) for k,value in metrics.items()}))
                errors.append(dict(**meta,beta_prediction=err[0],yaw_prediction=err[1],parameter_relative_error=float(np.linalg.norm((p-truth)/truth)/np.sqrt(3))))
    for suffix,items in [('clients',rows),('parameters',diagnostics),('prediction',errors),('manifest',manifest)]:
        (OUT/f'experiment_7c_seed{seed}_{suffix}.csv.gz').write_bytes(gzip.compress(pd.DataFrame(items).to_csv(index=False).encode(),mtime=0))
    print(f'completed evaluation seed {seed}',flush=True)


def summarize():
    cfg=json.loads(CONFIG.read_text())
    def read(suffix): return pd.concat([pd.read_csv(OUT/f'experiment_7c_seed{s}_{suffix}.csv.gz') for s in cfg['seeds']],ignore_index=True)
    df=read('clients'); pr=read('prediction'); pa=read('parameters'); manifest=read('manifest')
    if len(df)!=3600 or df.duplicated(['seed','scenario','method','client']).any(): raise RuntimeError('incomplete evaluation')
    if not (manifest.local_transitions*10==manifest.family_transitions).all() or not (manifest.family_transitions==manifest.extra_local_transitions).all(): raise RuntimeError('budget mismatch')
    keys=['seed','method']
    sr=df.groupby(keys).agg(tracking=('tracking','mean'),feasible=('feasible','min'),rho=('rho','max'),steering_rms=('steering_rms','mean'),peak_rate=('peak_steering_rate','max')).reset_index()
    worst=df.groupby(keys+['family']).tracking.mean().groupby(keys).max().rename('worst_family').reset_index()
    sr=sr.merge(worst,on=keys).merge(pr.groupby(keys)[['beta_prediction','yaw_prediction','parameter_relative_error']].mean().reset_index(),on=keys)
    sr.to_csv(OUT/'experiment_7c_seed_summary.csv',index=False)
    summary=sr.groupby('method').agg(mean=('tracking','mean'),std=('tracking','std'),worst_family=('worst_family','mean'),feasible=('feasible','min'),rho=('rho','max'),steering_rms=('steering_rms','mean'),peak_rate=('peak_rate','max'),beta_prediction=('beta_prediction','mean'),yaw_prediction=('yaw_prediction','mean'),parameter_relative_error=('parameter_relative_error','mean')).reset_index()
    summary.to_csv(OUT/'experiment_7c_summary.csv',index=False)
    rng=np.random.default_rng(cfg['bootstrap_seed']); idx=rng.integers(0,len(cfg['seeds']),(cfg['bootstrap_repetitions'],len(cfg['seeds']))); comparisons=[]
    for baseline in ('Local','Global','LocalExtra'):
        for metric in ('tracking','worst_family','yaw_prediction','parameter_relative_error'):
            a=sr[sr.method==baseline].sort_values('seed')[metric].to_numpy(); b=sr[sr.method=='Family'].sort_values('seed')[metric].to_numpy(); delta=a-b
            ci=np.quantile(delta[idx].mean(axis=1),[.025,.975])
            comparisons.append(dict(baseline=baseline,metric=metric,reduction_pct=100*delta.mean()/a.mean(),mean_difference=delta.mean(),ci_low=ci[0],ci_high=ci[1],positive_pairs=int(sum(delta>0))))
    pd.DataFrame(comparisons).to_csv(OUT/'experiment_7c_comparisons.csv',index=False)
    primary=comparisons[0]; f=summary[summary.method=='Family'].iloc[0]
    provenance={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in [CONFIG,ROOT/'code/experiments/experiment_7c_complementary_coverage.py',ROOT/'code/src/federated_lpv/output_error.py']}
    conclusions=dict(provenance=provenance,evaluations=len(df),fits=len(pa),all_fits_converged=bool(pa.success.all()),all_starts_converged=bool(pa.both_starts_success.all()),bound_hits=int(pa.bound_hit.sum()),max_multistart_difference=float(pa.multistart_difference.max()),all_amplitude_feasible=bool(df.feasible.min()==1),max_frozen_rho=float(df.rho.max()),primary=primary,collaboration_gate=bool(primary['reduction_pct']>=cfg['primary_min_reduction_pct'] and primary['ci_low']>0 and f.feasible==1 and f.rho<1))
    (OUT/'experiment_7c_conclusions.json').write_text(json.dumps(conclusions,indent=2,default=lambda v:v.item())+'\n')
    fig,axes=plt.subplots(1,3,figsize=(11,3.4),constrained_layout=True); colors=['#2878b5','#d65f5f','#34934b','#9867aa']; sub=summary.set_index('method').loc[list(METHODS)]
    for ax,metric,title in zip(axes[:2],['mean','yaw_prediction'],['Held-out tracking','Full-envelope prediction']):
        ax.bar(METHODS,sub[metric],color=colors); ax.set(title=title,ylabel='Yaw RMSE [deg/s]'); ax.tick_params(axis='x',labelrotation=20); ax.grid(axis='y',alpha=.2)
    per_speed=df.groupby(['local_speed','method']).tracking.mean().unstack()
    for method,color in zip(METHODS,colors): axes[2].plot(per_speed.index,per_speed[method],'o-',label=method,color=color)
    axes[2].set(title='Tracking by local speed',xlabel='Locally observed speed [m/s]',ylabel='Yaw RMSE [deg/s]'); axes[2].grid(alpha=.2); axes[2].legend(fontsize=7)
    fig.savefig(ROOT/'results/figures/experiment_7c_complementary_coverage.pdf'); plt.close(fig)
    print(json.dumps(conclusions,indent=2,default=lambda v:v.item()))


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--summarize-only',action='store_true'); parser.add_argument('--workers',type=int,default=4); args=parser.parse_args()
    if not args.summarize_only:
        with ProcessPoolExecutor(max_workers=args.workers) as pool: list(pool.map(run_seed,json.loads(CONFIG.read_text())['seeds']))
    summarize()
