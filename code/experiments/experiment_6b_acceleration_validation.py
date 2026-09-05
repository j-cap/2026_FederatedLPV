"""6B: test frozen 6A controllers under coordinate-consistent varying speed."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import argparse
import gzip
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import experiment_6a_regime_map as six
from experiment_4g_independent_validation import simulate_fleet as legacy, scenario
from experiment_4f_structured_identification import StructuredModel

ROOT=six.ROOT
OUT=ROOT/'results/tables'
CONFIG=ROOT/'code/config/experiment_6b.json'
base=six.base
METHODS=six.METHODS

def vehicle_arrays(clients):
    return np.array([[c.parameters.mass,c.parameters.yaw_inertia,c.parameters.front_length,
        c.parameters.rear_length,c.parameters.front_stiffness,c.parameters.rear_stiffness] for c in clients]).T

def rhs(z,u,v,acc,p,mode):
    """Small-angle beta proxy is vy/v. Force laws unchanged from 6A."""
    mass,iz,lf,lr,cf,cr=p
    beta=z[:,0]/v if mode=='vy' else z[:,0]
    ff=base.MU*mass*9.81*lr/(lf+lr); fr=base.MU*mass*9.81*lf/(lf+lr)
    front=ff*np.tanh(cf*(u-beta-lf*z[:,1]/v)/ff)
    rear=fr*np.tanh(cr*(-beta+lr*z[:,1]/v)/fr)
    ay=(front+rear)/mass
    lateral=ay-v*z[:,1] if mode=='vy' else ay/v-z[:,1]
    if mode=='beta_corrected':lateral-=acc*beta/v
    return np.column_stack((lateral,(lf*front-lr*rear)/iz))

def advance(z,u,v0,v1,p,mode,dt=base.DT):
    acc=(v1-v0)/dt; vm=(v0+v1)/2
    k1=rhs(z,u,v0,acc,p,mode)
    k2=rhs(z+dt*k1/2,u,vm,acc,p,mode)
    k3=rhs(z+dt*k2/2,u,vm,acc,p,mode)
    k4=rhs(z+dt*k3,u,v1,acc,p,mode)
    return z+dt*(k1+2*k2+2*k3+k4)/6

def simulate(clients,controller,speed,reference,mode,dt=base.DT):
    if mode=='legacy':
        if dt!=base.DT:raise ValueError('legacy regression uses original dt')
        return legacy(clients,controller,speed,reference)
    if mode not in ('vy','beta_noacc','beta_corrected'):raise ValueError(mode)
    n=len(clients); nt=len(speed); p=vehicle_arrays(clients)
    z=np.zeros((n,2)); x=np.zeros((nt,n,2)); u=np.zeros((nt-1,n)); integral=np.zeros(n)
    gains=np.empty((nt-1,n,3)); pre=np.empty((nt-1,n))
    for f in base.FAMILIES:
        mask=np.array([c.family==f for c in clients])
        values=[controller.evaluate(f,float(v)) for v in speed[:-1]]
        gains[:,mask,:]=np.array([q[0] for q in values])[:,None,:]
        pre[:,mask]=np.array([q[1] for q in values])[:,None]
    for k in range(nt-1):
        u[k]=-np.sum(gains[k]*np.column_stack((x[k],integral)),axis=1)+pre[k]*reference[k]
        z=advance(z,u[k],speed[k],speed[k+1],p,mode,dt)
        x[k+1]=z
        if mode=='vy':x[k+1,:,0]/=speed[k+1]
        integral+=dt*(x[k,:,1]-reference[k])
    mass,iz,lf,lr,cf,cr=p
    ff=base.MU*mass*9.81*lr/(lf+lr);fr=base.MU*mass*9.81*lf/(lf+lr)
    front=ff*np.tanh(cf*(u-x[:-1,:,0]-lf*x[:-1,:,1]/speed[:-1,None])/ff)
    rear=fr*np.tanh(cr*(-x[:-1,:,0]+lr*x[:-1,:,1]/speed[:-1,None])/fr)
    peak_u=np.rad2deg(np.max(abs(u),axis=0));peak_acc=np.max(abs((front+rear)/mass),axis=0)
    finite=np.all(np.isfinite(x),axis=(0,2)) & np.all(np.isfinite(u),axis=0)
    metrics=dict(tracking=np.rad2deg(np.sqrt(np.mean((x[:,:,1]-reference[:,None])**2,axis=0))),
        peak_steering=peak_u,peak_acceleration=peak_acc,
        peak_steering_rate=np.rad2deg(np.max(abs(np.diff(u,axis=0)/dt),axis=0)),
        beta_rms=np.rad2deg(np.sqrt(np.mean(x[:,:,0]**2,axis=0))),
        steering_rms=np.rad2deg(np.sqrt(np.mean(u**2,axis=0))),
        feasible=finite & (peak_u<=12.) & (peak_acc<=base.MU*9.81))
    return x,u,metrics

def load_controllers(seed,gamma,envelope):
    path=OUT/f'experiment_6a_seed{seed}_parameters.csv.gz'
    table=pd.read_csv(path); selected=table[(table.gamma==gamma)&(table.low==envelope[0])]
    if len(selected)!=4:raise ValueError('missing frozen 6A parameters')
    params={r.group:np.array([r.cf_over_m,r.cr_over_m,r.m_over_iz]) for r in selected.itertuples()}
    models={m:StructuredModel('global' if m in ('M1','M3') else 'family',
        'constant' if m in ('M1','M2') else 'lpv',params) for m in METHODS}
    return six.design(models,envelope)

def run_seed(seed):
    cfg=json.loads(CONFIG.read_text()); rows=[]; profiles=[]
    for gamma in cfg['heterogeneity']:
        clients=six.fleet(seed+cfg['test_seed_offset'],gamma,cfg['clients_per_family'])
        for envelope in cfg['envelopes']:
            controllers=load_controllers(seed,gamma,envelope)
            for scale in cfg['time_scales']:
                time=np.arange(0,base.DURATION*scale+base.DT,base.DT)
                for name in cfg['scenarios']:
                    v,ref=scenario(name,time/scale);v=six.remap(v,envelope)
                    meta=dict(seed=seed,gamma=gamma,low=envelope[0],high=envelope[1],time_scale=scale,scenario=name)
                    profiles.append(dict(**meta,duration=time[-1],max_speed_rate=float(max(abs(np.diff(v)/base.DT)))))
                    for severity,factor in cfg['severities'].items():
                        for mode in cfg['plant_modes']:
                            for m,controller in controllers.items():
                                _,_,metrics=simulate(clients,controller,v,factor*ref,mode)
                                rows.extend(dict(**meta,severity=severity,plant=mode,method=m,client=c.client_id,family=c.family,
                                    **{k:float(value[j]) for k,value in metrics.items()}) for j,c in enumerate(clients))
                print('completed',seed,gamma,envelope,scale,flush=True)
    for suffix,data in [('clients',rows),('profiles',profiles)]:
        content=pd.DataFrame(data).to_csv(index=False).encode()
        (OUT/f'experiment_6b_seed{seed}_{suffix}.csv.gz').write_bytes(gzip.compress(content,mtime=0))

def summarize():
    cfg=json.loads(CONFIG.read_text())
    df=pd.concat([pd.read_csv(OUT/f'experiment_6b_seed{s}_clients.csv.gz') for s in cfg['seeds']],ignore_index=True)
    expected=len(cfg['seeds'])*len(cfg['heterogeneity'])*len(cfg['envelopes'])*len(cfg['time_scales'])*len(cfg['scenarios'])*len(cfg['severities'])*len(cfg['plant_modes'])*4*3*cfg['clients_per_family']
    frozen_audit=pd.concat([pd.read_csv(OUT/f'experiment_6a_seed{s}_stability.csv.gz') for s in cfg['seeds']])
    frozen_audit=frozen_audit[frozen_audit.gamma.isin(cfg['heterogeneity'])&frozen_audit.low.isin([e[0] for e in cfg['envelopes']])]
    frozen_rho=float(frozen_audit.rho.max())
    if frozen_rho>=1:raise ValueError('unchanged frozen audit no longer passes')
    unique=['seed','gamma','low','time_scale','scenario','severity','plant','method','client']
    if len(df)!=expected or df.duplicated(unique).any():raise ValueError('incomplete/duplicate results')
    keys=['seed','gamma','low','high','time_scale','plant','severity','method']
    sr=df.groupby(keys).agg(tracking=('tracking','mean'),steering_rms=('steering_rms','mean'),
        peak_rate=('peak_steering_rate','max'),feasible=('feasible','min')).reset_index()
    worst=df.groupby(keys+['family']).tracking.mean().groupby(keys).max().rename('worst_family').reset_index()
    sr=sr.merge(worst,on=keys)
    sr.to_csv(OUT/'experiment_6b_seed_summary.csv',index=False)
    group=['gamma','low','high','time_scale','plant','severity','method']
    agg=sr.groupby(group).agg(mean=('tracking','mean'),std=('tracking','std'),worst_family=('worst_family','mean'),
        steering_rms=('steering_rms','mean'),peak_rate=('peak_rate','max'),feasible=('feasible','min')).reset_index()
    agg.to_csv(OUT/'experiment_6b_summary.csv',index=False)
    df.groupby(['gamma','low','time_scale','plant','severity','scenario','method']).tracking.mean().to_csv(OUT/'experiment_6b_scenario_summary.csv')
    n=len(cfg['seeds']);rng=np.random.default_rng(cfg['bootstrap_seed']);idx=rng.integers(0,n,(cfg['bootstrap_repetitions'],n))
    comparisons=[];regimes=[];effects=[]
    for (gamma,lo,hi,scale,mode),cell in sr[sr.severity=='moderate'].groupby(['gamma','low','high','time_scale','plant']):
        meta=dict(gamma=gamma,low=lo,high=hi,time_scale=scale,plant=mode)
        vals={m:cell[cell.method==m].sort_values('seed').tracking.to_numpy() for m in METHODS}
        best=min(vals,key=lambda m:vals[m].mean());eligible=[]
        for m in METHODS:
            d=vals[m]-(1+cfg['tolerance'])*vals[best];ci=np.quantile(d[idx].mean(axis=1),[.025,.975])
            feasible=df[(df.gamma==gamma)&(df.low==lo)&(df.time_scale==scale)&(df.plant==mode)&(df.method==m)].feasible.min()==1
            if ci[1]<=0 and feasible:eligible.append(m)
        regimes.append(dict(**meta,best=best,simplest_within_5pct=next((m for m in METHODS if m in eligible),'none')))
        for baseline,improved in [('M1','M3'),('M3','M4'),('M2','M4')]:
            d=vals[baseline]-vals[improved];ci=np.quantile(d[idx].mean(axis=1),[.025,.975])
            comparisons.append(dict(**meta,baseline=baseline,improved=improved,reduction_pct=100*d.mean()/vals[baseline].mean(),
                ci_low=ci[0],ci_high=ci[1],positive_pairs=int(sum(d>0))))
        if mode!='legacy':
            for m in METHODS:
                old=sr[(sr.gamma==gamma)&(sr.low==lo)&(sr.time_scale==scale)&(sr.plant=='legacy')&(sr.severity=='moderate')&(sr.method==m)].sort_values('seed').tracking.to_numpy()
                d=vals[m]-old;ci=np.quantile(d[idx].mean(axis=1),[.025,.975])
                effects.append(dict(**meta,method=m,change_pct=100*d.mean()/old.mean(),ci_low=ci[0],ci_high=ci[1]))
    for suffix,data in [('comparisons',comparisons),('regimes',regimes),('plant_effects',effects)]:
        pd.DataFrame(data).to_csv(OUT/f'experiment_6b_{suffix}.csv',index=False)
    # Match the legacy replay against committed 6A on every retained raw metric.
    old=pd.concat([pd.read_csv(OUT/f'experiment_6a_seed{s}_clients.csv.gz') for s in cfg['seeds']])
    old=old[old.gamma.isin(cfg['heterogeneity'])&old.low.isin([e[0] for e in cfg['envelopes']])]
    join=['seed','gamma','low','high','scenario','severity','method','client','family']
    replay=df[(df.plant=='legacy')&(df.time_scale==1)].merge(old,on=join,suffixes=('_new','_old'),validate='one_to_one')
    metrics=['tracking','peak_steering','peak_acceleration','peak_steering_rate','beta_rms','steering_rms','feasible']
    error=max(float(max(abs(replay[k+'_new']-replay[k+'_old']))) for k in metrics)
    if len(replay)!=len(old) or error>1e-9:raise ValueError(f'6A replay mismatch: {error}')
    primary=[r for r in comparisons if r['gamma']==1 and r['low']==10 and r['plant']=='vy' and r['improved']=='M4']
    gate=all(r['reduction_pct']>=5 and r['ci_low']>0 for r in primary) and df.feasible.min()==1
    conclusion=dict(protocol_sha256=hashlib.sha256(CONFIG.read_bytes()).hexdigest(),evaluations=len(df),
        frozen_fit_sha256={str(s):hashlib.sha256((OUT/f'experiment_6a_seed{s}_parameters.csv.gz').read_bytes()).hexdigest() for s in cfg['seeds']},
        legacy_replay_max_error=error,unchanged_frozen_rho_max=frozen_rho,
        all_amplitude_feasible=bool(df.feasible.min()==1),robustness_gate=bool(gate),primary_comparisons=primary)
    (OUT/'experiment_6b_conclusions.json').write_text(json.dumps(conclusion,indent=2,default=lambda v:v.item())+'\n')
    fig,axes=plt.subplots(1,2,figsize=(8.8,3.5),constrained_layout=True)
    subset=agg[(agg.gamma==1)&(agg.low==10)&(agg.severity=='moderate')]
    for ax,scale in zip(axes,cfg['time_scales']):
        for mode,marker in [('legacy','o'),('beta_noacc','s'),('vy','^')]:
            data=subset[(subset.time_scale==scale)&(subset.plant==mode)].set_index('method').loc[list(METHODS)]
            ax.errorbar(METHODS,data['mean'],yerr=data['std'],marker=marker,capsize=3,label=mode)
        ax.set(title=f'Time scale {scale}: {12*scale} s',ylabel='Yaw tracking RMSE [deg/s]')
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.savefig(ROOT/'results/figures/experiment_6b_acceleration.pdf');plt.close(fig)
    print(json.dumps(conclusion,indent=2,default=lambda v:v.item()))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--summarize-only',action='store_true');parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    if not args.summarize_only:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:list(pool.map(run_seed,json.loads(CONFIG.read_text())['seeds']))
    summarize()
