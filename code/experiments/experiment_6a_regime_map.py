"""6A: independent-fleet scheduling/heterogeneity factorial study."""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
import argparse
import csv
import hashlib
import gzip
import json
import numpy as np
import matplotlib.pyplot as plt
from experiment_4g_independent_validation import simulate_fleet, scenario
from experiment_4f_structured_identification import fit_structured, StructuredModel, step
import experiment_4c_nonlinear_oracle_models as base
from federated_lpv import (sample_fleet, ScheduledController, design_lqi_gain,
    discrete_bicycle_matrices, augmented_tracking_matrices)
from federated_lpv.vehicle import family_centers

ROOT=Path(__file__).resolve().parents[2]
CONFIG=ROOT/'code/config/experiment_6a.json'
OUT=ROOT/'results/tables'
METHODS=('M1','M2','M3','M4')

def fleet(seed,gamma,count=10):
    centers=family_centers(); nominal=centers['nominal']
    names=('mass','yaw_inertia','front_stiffness','rear_stiffness')
    result=[]
    for c in sample_fleet(seed,count):
        center=centers[c.family]
        p=replace(c.parameters,**{k:(getattr(nominal,k)+gamma*(getattr(center,k)-getattr(nominal,k)))
            *getattr(c.parameters,k)/getattr(center,k) for k in names})
        result.append(replace(c,parameters=p))
    return result

def remap(speed,envelope):
    lo,hi=envelope
    return lo+(np.asarray(speed)-12.)*(hi-lo)/16.

def design(models,envelope):
    result={}
    for m in METHODS:
        model=models[m]; grid=np.array([20.]) if m in ('M1','M2') else np.linspace(*envelope,5)
        keys=('global',) if model.specialization=='global' else base.FAMILIES
        gains={}; pre={}
        for key in keys:
            pairs=[design_lqi_gain(*model.matrix(key,float(v)),base.DT,base.Q,base.R) for v in grid]
            gains[key]=np.array([p[0] for p in pairs]); pre[key]=np.array([p[1] for p in pairs])
        result[m]=ScheduledController(m,model.specialization,'constant' if len(grid)==1 else 'linear',grid,gains,pre)
    return result

def audit(clients,controllers,envelope):
    rows=[]
    for c in clients:
        maxima={m:0. for m in METHODS}
        for v in np.linspace(*envelope,161):
            a,b=discrete_bicycle_matrices(float(v),c.parameters,base.DT)
            aa,bb=augmented_tracking_matrices(a,b,base.DT)
            for m,controller in controllers.items():
                gain,_=controller.evaluate(c.family,float(v))
                maxima[m]=max(maxima[m],float(max(abs(np.linalg.eigvals(aa-bb@gain[None,:])))))
        rows.extend(dict(client=c.client_id,family=c.family,method=m,rho=r) for m,r in maxima.items())
    return rows

def run_seed(seed):
    cfg=json.loads(CONFIG.read_text()); time=np.arange(0,base.DURATION+base.DT,base.DT)
    rows=[]; audits=[]; parameters=[]; prediction=[]
    for gamma in cfg['heterogeneity']:
        train=fleet(seed,gamma,cfg['clients_per_family'])
        test=fleet(seed+cfg['test_seed_offset'],gamma,cfg['clients_per_family'])
        for envelope in cfg['envelopes']:
            meta=dict(seed=seed,gamma=gamma,low=envelope[0],high=envelope[1])
            # Oracle family-center controller collects common identification/evaluation data.
            p={}
            for f in base.FAMILIES:
                group=[c.parameters for c in train if c.family==f]
                p[f]=np.mean([[c.front_stiffness/c.mass,c.rear_stiffness/c.mass,c.mass/c.yaw_inertia] for c in group],axis=0)
            p['global']=np.mean(list(p.values()),axis=0)
            physical={m:StructuredModel('global' if m in ('M1','M3') else 'family',
                'constant' if m in ('M1','M2') else 'lpv',p) for m in METHODS}
            collection=design(physical,envelope)['M4']; records=[]
            for profile,maneuver in zip(base.TRAIN_PROFILES,('lane','sine')):
                v=remap(base.smooth_profile(time,profile),envelope)
                x,u,metrics=simulate_fleet(train,collection,v,.5*base.reference_signal(time,maneuver))
                if not np.all(metrics['feasible']): raise RuntimeError('infeasible identification data')
                records.extend(dict(family=c.family,state=x[:,j],input=u[:,j],speed=v) for j,c in enumerate(train))
            models,diagnostics=fit_structured(records); models={m:models[m] for m in METHODS}
            controllers=design(models,envelope)
            parameters.extend(dict(**meta,**r) for r in diagnostics)
            audits.extend(dict(**meta,**r) for r in audit(test,controllers,envelope))
            for name in cfg['scenarios']:
                original,ref=scenario(name,time); v=remap(original,envelope)
                for severity,scale in cfg['severities'].items():
                    for m,controller in controllers.items():
                        _,_,metrics=simulate_fleet(test,controller,v,scale*ref)
                        rows.extend(dict(**meta,scenario=name,severity=severity,method=m,client=c.client_id,
                            family=c.family,**{k:float(val[j]) for k,val in metrics.items()}) for j,c in enumerate(test))
                # Common moderate held-out records avoid method-dependent prediction inputs.
                x,u,_=simulate_fleet(test,collection,v,ref)
                for m,model in models.items():
                    for j,c in enumerate(test):
                        key='global' if model.specialization=='global' else c.family
                        speeds=np.full_like(v[:-1],20.) if m in ('M1','M2') else v[:-1]
                        error=step(model.parameters[key],x[:-1,j],u[:,j],speeds)-x[1:,j]
                        rmse=np.rad2deg(np.sqrt(np.mean(error**2,axis=0)))
                        prediction.append(dict(**meta,scenario=name,method=m,client=c.client_id,family=c.family,
                            beta_rmse=rmse[0],yaw_rmse=rmse[1]))
            print('completed cell',meta,flush=True)
    for suffix,data in [('clients',rows),('stability',audits),('parameters',parameters),('prediction',prediction)]:
        base.write_csv(OUT/f'experiment_6a_seed{seed}_{suffix}.csv',data)
    return seed

def summarize():
    cfg=json.loads(CONFIG.read_text())
    def read(suffix):
        result=[]
        for seed in cfg['training_seeds']:
            path=OUT/f'experiment_6a_seed{seed}_{suffix}.csv'
            if path.exists():
                path.with_suffix('.csv.gz').write_bytes(gzip.compress(path.read_bytes(),mtime=0))
            with gzip.open(path.with_suffix('.csv.gz'),'rt') as f: result.extend(csv.DictReader(f))
        return result
    import pandas as pd
    df=pd.DataFrame(read('clients')); au=pd.DataFrame(read('stability')); pr=pd.DataFrame(read('prediction'))
    read('parameters')  # Preserve fit diagnostics alongside the compressed raw results.
    for frame in (df,au,pr):
        for c in frame:
            if c not in ('scenario','severity','method','client','family'):frame[c]=pd.to_numeric(frame[c])
    keys=['seed','gamma','low','high','severity','method']
    sr=df.groupby(keys).agg(tracking=('tracking','mean'),steering_rms=('steering_rms','mean'),
        peak_steering_rate=('peak_steering_rate','max'),feasible=('feasible','min')).reset_index()
    worst=df.groupby(keys+['family']).tracking.mean().groupby(keys).max().rename('worst_family').reset_index()
    sr=sr.merge(worst,on=keys).merge(au.groupby(['seed','gamma','low','high','method']).rho.max().reset_index())
    sr.to_csv(OUT/'experiment_6a_seed_summary.csv',index=False)
    summary=sr.groupby(['gamma','low','high','severity','method']).agg(mean=('tracking','mean'),
        std=('tracking','std'),worst_family=('worst_family','mean'),steering_rms=('steering_rms','mean'),
        peak_rate=('peak_steering_rate','max'),feasible=('feasible','min'),rho=('rho','max')).reset_index()
    summary.to_csv(OUT/'experiment_6a_summary.csv',index=False)
    pr.groupby(['gamma','low','high','method'])[['beta_rmse','yaw_rmse']].mean().to_csv(OUT/'experiment_6a_prediction_summary.csv')
    n=len(cfg['training_seeds'])
    rng=np.random.default_rng(cfg['bootstrap_seed']); idx=rng.integers(0,n,(cfg['bootstrap_repetitions'],n))
    comparisons=[]; cells=[]
    for gamma in cfg['heterogeneity']:
        for lo,hi in cfg['envelopes']:
            cell=sr[(sr.gamma==gamma)&(sr.low==lo)&(sr.severity=='moderate')]
            values={m:cell[cell.method==m].sort_values('seed').tracking.to_numpy() for m in METHODS}
            best=min(METHODS,key=lambda m:values[m].mean()); eligible=[]
            for m in METHODS:
                delta=values[m]-(1+cfg['practical_tolerance'])*values[best]
                ci=np.quantile(delta[idx].mean(axis=1),[.025,.975])
                allsev=sr[(sr.gamma==gamma)&(sr.low==lo)&(sr.method==m)]
                ok=bool(ci[1]<=0 and allsev.feasible.min()==1 and allsev.rho.max()<1)
                if ok:eligible.append(m)
                comparisons.append(dict(gamma=gamma,low=lo,high=hi,method=m,best=best,
                    tolerance_ci_low=ci[0],tolerance_ci_high=ci[1],eligible=ok))
            # Gain storage order: 1,3,5,15 vectors. Each includes its prefilter.
            chosen=next((m for m in METHODS if m in eligible),'none')
            item=dict(gamma=gamma,low=lo,high=hi,best=best,simplest_within_5pct=chosen)
            for baseline,improved,label in [('M1','M3','global_scheduling'),('M3','M4','lpv_specialization'),('M2','M4','family_scheduling')]:
                delta=values[baseline]-values[improved]; ci=np.quantile(delta[idx].mean(axis=1),[.025,.975])
                item[label+'_pct']=100*delta.mean()/values[baseline].mean()
                item[label+'_ci_low']=ci[0];item[label+'_ci_high']=ci[1]
            cells.append(item)
    base.write_csv(OUT/'experiment_6a_comparisons.csv',comparisons)
    base.write_csv(OUT/'experiment_6a_regimes.csv',cells)
    (OUT/'experiment_6a_conclusions.json').write_text(json.dumps(dict(protocol_sha256=hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
        evaluations=len(df),all_feasible=bool(df.feasible.min()==1),max_rho=float(au.rho.max()),regimes=cells),indent=2)+'\n')
    fig,axes=plt.subplots(1,3,figsize=(11,4),constrained_layout=True)
    for ax,key,title in zip(axes,['global_scheduling_pct','lpv_specialization_pct','family_scheduling_pct'],
                            ['Scheduling: M1 to M3','Specialization: M3 to M4','Scheduling: M2 to M4']):
        data=np.array([r[key] for r in cells]).reshape(4,3)
        im=ax.imshow(data,origin='lower',aspect='auto',cmap='RdBu',vmin=-max(5,abs(data).max()),vmax=max(5,abs(data).max()))
        for i in range(4):
            for j in range(3):ax.text(j,i,f'{data[i,j]:.1f}%',ha='center',va='center',color='black',bbox=dict(facecolor='white',alpha=.75,edgecolor='none'))
        ax.set(xticks=range(3),xticklabels=['18–22','15–25','10–30'],yticks=range(4),yticklabels=cfg['heterogeneity'],
               xlabel='Speed envelope [m/s]',ylabel='Family separation scale',title=title)
        fig.colorbar(im,ax=ax,shrink=.65,label='Tracking reduction [%]')
    fig.savefig(ROOT/'results/figures/experiment_6a_regime_map.pdf');plt.close(fig)
    print(json.dumps(cells,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--summarize-only',action='store_true');parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    if not args.summarize_only:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:list(pool.map(run_seed,json.loads(CONFIG.read_text())['training_seeds']))
    summarize()
