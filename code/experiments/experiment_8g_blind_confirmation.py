"""8G: blind confirmation of the private FL-LPV deployment claim."""
from concurrent.futures import ProcessPoolExecutor
import argparse,gzip,hashlib,json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from federated_lpv.output_error import fit
from federated_lpv.privacy import private_mean,shaped_private_mean
from federated_lpv import discrete_bicycle_matrices,augmented_tracking_matrices
import experiment_8f_deployment_robustness as eightf

ROOT=eightf.ROOT;OUT=eightf.OUT;base=eightf.base;corrected=eightf.corrected
CONFIG=ROOT/'code/config/experiment_8g.json'


def true_ratios(c):
    p=c.parameters
    return np.array([p.front_stiffness/p.mass,p.rear_stiffness/p.mass,p.mass/p.yaw_inertia])


def run_seed(seed):
    cfg=json.loads(CONFIG.read_text());ntrain=cfg['training_clients_per_family']
    train=corrected.six.fleet(seed,1,ntrain);train_data,train_cat,nominal=eightf.collect(train,seed,cfg)
    test=corrected.six.fleet(seed+10000,1,cfg['test_clients_per_family']);test_data,_,_=eightf.collect(test,seed+20000,cfg)
    train_fit={};test_fit={};fits=[]
    for scope,clients,data,target in [('train',train,train_data,train_fit),('test',test,test_data,test_fit)]:
        for c in clients:
            p,d=fit(data[c.client_id]);target[c.client_id]=p
            truth=true_ratios(c)
            fits.append(dict(seed=seed,scope=scope,client=c.client_id,family=c.family,parameter_error=float(np.linalg.norm(np.log(p/truth))),cf_over_m=p[0],cr_over_m=p[1],m_over_iz=p[2],**d))
    t=np.arange(0,24+base.DT,base.DT);tests=[]
    for name in cfg['test_scenarios']:
        v,ref=corrected.scenario(name,t/2);tests.append((name,corrected.six.remap(v,(10,30)),ref))
    grid=np.linspace(10,30,161);audit={}
    for c in test:
        pairs=[augmented_tracking_matrices(*discrete_bicycle_matrices(float(v),c.parameters,base.DT),base.DT) for v in grid]
        audit[c.client_id]=(np.array([p[0] for p in pairs]),np.array([p[1] for p in pairs]))
    rows=[];releases=[]

    def evaluate(point,method,draw,params):
        selected={c.client_id:eightf.eighte.eightd.prior.eight.seven.seven.controller(params[c.client_id] if method=='Local' else params[c.family]) for c in test}
        rhos=[]
        for c in test:
            a,b=audit[c.client_id];control=selected[c.client_id]
            gains=np.column_stack([np.interp(grid,control.speeds,control.gains['global'][:,j]) for j in range(3)])
            rhos.append(float(np.max(abs(np.linalg.eigvals(a-b@gains[:,None,:])))))
        for name,v,ref in tests:
            _,_,m=corrected.simulate(test,nominal,v,ref,'vy',client_controllers=selected)
            for j,c in enumerate(test):
                truth=true_ratios(c)
                estimate=params[c.client_id] if method=='Local' else params[c.family]
                rows.append(dict(seed=seed,point=point['name'],cohort=point['cohort'],epsilon=str(point['epsilon']),method=method,draw=draw,scenario=name,client=c.client_id,family=c.family,parameter_error=float(np.linalg.norm(np.log(estimate/truth))),rho=rhos[j],**{k:float(x[j]) for k,x in m.items()}))

    for point in cfg['operating_points']:
        family_values={};family_mean={}
        for fi,family in enumerate(base.FAMILIES):
            members=[c for c in train if c.family==family]
            chosen=eightf.select_members(members,train_cat,point['cohort'],'balanced',seed*1000+point['cohort']*10+fi)
            values=np.array([train_fit[c.client_id] for c in chosen]);family_values[family]=values;family_mean[family]=private_mean(values,np.inf,cfg['delta'],np.random.default_rng(0),bounds=cfg['bounds'])[0]
        evaluate(point,'Local',0,test_fit);evaluate(point,'FamilyNonPrivate',0,family_mean)
        for draw in range(cfg['privacy_draws']):
            for method in ('IsotropicDP','ControlAwareDP'):
                rng=np.random.default_rng(cfg['dp_seed']+seed*100000+point['cohort']*1000+draw)
                params={}
                for family in base.FAMILIES:
                    if method=='IsotropicDP':p,d=private_mean(family_values[family],point['epsilon'],cfg['delta'],rng,bounds=cfg['bounds'])
                    else:p,d=shaped_private_mean(family_values[family],point['epsilon'],cfg['delta'],rng,cfg['control_weights'][family],cfg['bounds'])
                    params[family]=p;releases.append(dict(seed=seed,point=point['name'],cohort=point['cohort'],epsilon=str(point['epsilon']),method=method,draw=draw,family=family,cf_over_m=p[0],cr_over_m=p[1],m_over_iz=p[2],**d))
                evaluate(point,method,draw,params)
        print(seed,point['name'],flush=True)
    for suffix,items in [('fits',fits),('releases',releases),('clients',rows)]:
        (OUT/f'experiment_8g_seed{seed}_{suffix}.csv.gz').write_bytes(gzip.compress(pd.DataFrame(items).to_csv(index=False).encode(),mtime=0))


def summarize():
    cfg=json.loads(CONFIG.read_text());read=lambda x:pd.concat([pd.read_csv(OUT/f'experiment_8g_seed{s}_{x}.csv.gz') for s in cfg['seeds']],ignore_index=True)
    fits=read('fits');rel=read('releases');df=read('clients');keys=['seed','point','cohort','epsilon','method','draw']
    units=df.groupby(keys).agg(tracking=('tracking','mean'),parameter_error=('parameter_error','mean'),feasible=('feasible','min'),rho=('rho','max')).reset_index()
    summary=units.groupby(['point','cohort','epsilon','method']).agg(tracking=('tracking','mean'),tracking_std=('tracking','std'),parameter_error=('parameter_error','mean'),feasible_rate=('feasible','mean'),rho=('rho','max')).reset_index()
    baseline=summary[summary.method=='FamilyNonPrivate'][['point','tracking']].rename(columns={'tracking':'family_baseline'});summary=summary.merge(baseline,on='point');summary['tracking_increase_pct']=100*(summary.tracking/summary.family_baseline-1)
    summary['viable']=(summary.tracking_increase_pct<=cfg['acceptable_tracking_increase_pct'])&(summary.feasible_rate==1)&(summary.rho<1)
    paired=[]
    for point,cell in units[units.method.isin(['IsotropicDP','ControlAwareDP'])].groupby('point'):
        p=cell.pivot(index=['seed','draw'],columns='method',values='tracking');seed_delta=(p.ControlAwareDP-p.IsotropicDP).groupby(level='seed').mean();stat,pval=wilcoxon(seed_delta,alternative='less')
        paired.append(dict(point=point,relative_improvement_pct=float(100*(1-p.ControlAwareDP.mean()/p.IsotropicDP.mean())),improved_seeds=int((seed_delta<0).sum()),wilcoxon_statistic=float(stat),wilcoxon_p_less=float(pval)))
    paired=pd.DataFrame(paired);primary=summary[summary.point=='discriminating'].set_index('method');pp=paired.set_index('point').loc['discriminating']
    gate=bool(primary.loc['ControlAwareDP','viable'] and not primary.loc['IsotropicDP','viable'] and pp.improved_seeds==len(cfg['seeds']))
    conclusions=dict(provenance={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [CONFIG,ROOT/'code/src/federated_lpv/privacy.py',ROOT/'code/experiments/experiment_8g_blind_confirmation.py']},local_fits=len(fits),privacy_releases=len(rel),closed_loop_evaluations=len(df),all_local_fits_converged=bool(fits.success.all()),primary_prediction=dict(control_aware_viable=bool(primary.loc['ControlAwareDP','viable']),isotropic_viable=bool(primary.loc['IsotropicDP','viable']),improved_seeds=int(pp.improved_seeds),relative_improvement_pct=float(pp.relative_improvement_pct),wilcoxon_p_less=float(pp.wilcoxon_p_less)),confirmation_gate=gate,local_superiority_claimed=False)
    summary.to_csv(OUT/'experiment_8g_summary.csv',index=False);units.to_csv(OUT/'experiment_8g_seed_draw_summary.csv',index=False);paired.to_csv(OUT/'experiment_8g_paired_comparison.csv',index=False);(OUT/'experiment_8g_conclusions.json').write_text(json.dumps(conclusions,indent=2)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(9.5,3.5),constrained_layout=True);order=['Local','FamilyNonPrivate','IsotropicDP','ControlAwareDP'];colors=['.35','.55','#d95f02','#1b9e77']
    for ax,(point,cell) in zip(axes,summary.groupby('point',sort=False)):
        cell=cell.set_index('method').loc[order];ax.bar(np.arange(4),cell.tracking_increase_pct,color=colors);ax.axhline(10,color='k',ls=':');ax.set(xticks=np.arange(4),xticklabels=['Local','Family\nnon-private','Isotropic\nDP','Control-aware\nDP'],ylabel='Tracking cost vs family non-private [%]',title=point);ax.grid(axis='y',alpha=.2)
    fig.savefig(ROOT/'results/figures/experiment_8g_blind_confirmation.pdf');plt.close(fig);print(json.dumps(conclusions,indent=2));print(summary.to_string(index=False));print(paired.to_string(index=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--summarize-only',action='store_true');p.add_argument('--workers',type=int,default=5);a=p.parse_args()
    if not a.summarize_only:
        with ProcessPoolExecutor(max_workers=a.workers) as pool:list(pool.map(run_seed,json.loads(CONFIG.read_text())['seeds']))
    summarize()
