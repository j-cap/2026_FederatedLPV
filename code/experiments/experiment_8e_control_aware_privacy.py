"""8E: control-aware shaped client-level DP aggregation."""
from concurrent.futures import ProcessPoolExecutor
import argparse,gzip,hashlib,json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from federated_lpv.output_error import fit
from federated_lpv.privacy import private_mean,shaped_private_mean
from federated_lpv import discrete_bicycle_matrices,augmented_tracking_matrices
import experiment_8d_privacy_recovery as eightd

ROOT=eightd.ROOT;OUT=eightd.OUT;base=eightd.base;corrected=eightd.corrected
CONFIG=ROOT/'code/config/experiment_8e.json';METHODS=('Isotropic','ControlAware')


def run_seed(seed):
    cfg=json.loads(CONFIG.read_text());max_cohort=max(cfg['cohorts'])
    train=corrected.six.fleet(seed,1,max_cohort);data,nominal=eightd.collect(train,seed,cfg);local={};fits=[]
    for c in train:
        p,d=fit(data[c.client_id]);local[c.client_id]=p
        fits.append(dict(seed=seed,client=c.client_id,family=c.family,cf_over_m=p[0],cr_over_m=p[1],m_over_iz=p[2],**d))
    test=corrected.six.fleet(seed+10000,1,cfg['test_clients_per_family']);t=np.arange(0,24+base.DT,base.DT);tests=[]
    for name in cfg['test_scenarios']:
        v,ref=corrected.scenario(name,t/2);tests.append((name,corrected.six.remap(v,(10,30)),ref))
    grid=np.linspace(10,30,161);audit={}
    for c in test:
        pairs=[augmented_tracking_matrices(*discrete_bicycle_matrices(float(v),c.parameters,base.DT),base.DT) for v in grid]
        audit[c.client_id]=(np.array([p[0] for p in pairs]),np.array([p[1] for p in pairs]))
    rows=[];releases=[]
    for cohort in cfg['cohorts']:
        for raw_eps in cfg['epsilons']:
            eps=np.inf if raw_eps=='inf' else float(raw_eps);draws=1 if np.isinf(eps) else cfg['privacy_draws']
            for method in METHODS:
                for draw in range(draws):
                    # Resetting this seed across methods gives common Gaussian draws.
                    rng=np.random.default_rng(cfg['dp_seed']+seed*100000+draw+cohort*1000+int(100*(0 if np.isinf(eps) else eps)))
                    params={}
                    for family in base.FAMILIES:
                        members=[c for c in train if c.family==family][:cohort];values=[local[c.client_id] for c in members]
                        if method=='Isotropic':p,d=private_mean(values,eps,cfg['delta'],rng,bounds=cfg['bounds'])
                        else:p,d=shaped_private_mean(values,eps,cfg['delta'],rng,cfg['control_weights'][family],cfg['bounds'])
                        params[family]=p;releases.append(dict(seed=seed,method=method,cohort=cohort,epsilon=eightd.prior.label(eps),draw=draw,family=family,cf_over_m=p[0],cr_over_m=p[1],m_over_iz=p[2],**d))
                    selected={c.client_id:eightd.prior.eight.seven.seven.controller(params[c.family]) for c in test};rhos=[]
                    for c in test:
                        a,b=audit[c.client_id];control=selected[c.client_id]
                        gains=np.column_stack([np.interp(grid,control.speeds,control.gains['global'][:,j]) for j in range(3)])
                        rhos.append(float(np.max(abs(np.linalg.eigvals(a-b@gains[:,None,:])))))
                    for name,v,ref in tests:
                        _,_,metrics=corrected.simulate(test,nominal,v,ref,'vy',client_controllers=selected)
                        for j,c in enumerate(test):rows.append(dict(seed=seed,method=method,cohort=cohort,epsilon=eightd.prior.label(eps),draw=draw,scenario=name,client=c.client_id,family=c.family,rho=rhos[j],**{k:float(x[j]) for k,x in metrics.items()}))
                print(seed,cohort,eightd.prior.label(eps),method,flush=True)
    for suffix,items in [('fits',fits),('releases',releases),('clients',rows)]:
        (OUT/f'experiment_8e_seed{seed}_{suffix}.csv.gz').write_bytes(gzip.compress(pd.DataFrame(items).to_csv(index=False).encode(),mtime=0))


def summarize():
    cfg=json.loads(CONFIG.read_text())
    read=lambda x:pd.concat([pd.read_csv(OUT/f'experiment_8e_seed{s}_{x}.csv.gz') for s in cfg['seeds']],ignore_index=True)
    fits=read('fits');releases=read('releases');df=read('clients');keys=['seed','method','cohort','epsilon','draw']
    units=df.groupby(keys).agg(tracking=('tracking','mean'),feasible=('feasible','min'),rho=('rho','max')).reset_index()
    units['epsilon']=units.epsilon.map(lambda v:eightd.prior.label(float(v)))
    summary=units.groupby(['method','cohort','epsilon']).agg(tracking=('tracking','mean'),std=('tracking','std'),q95=('tracking',lambda x:np.quantile(x,.95)),feasible_rate=('feasible','mean'),rho=('rho','max')).reset_index()
    baselines=summary[summary.epsilon=='inf'][['method','cohort','tracking']].rename(columns={'tracking':'baseline'})
    summary=summary.merge(baselines,on=['method','cohort']);summary['tracking_increase_pct']=100*(summary.tracking/summary.baseline-1)
    comparisons=[]
    finite=units[units.epsilon!='inf']
    for (cohort,epsilon),cell in finite.groupby(['cohort','epsilon']):
        pivot=cell.pivot(index=['seed','draw'],columns='method',values='tracking');seed_means=pivot.groupby(level='seed').mean();delta=seed_means.ControlAware-seed_means.Isotropic
        stat,pvalue=wilcoxon(delta,alternative='less') if np.any(delta) else (0.,1.)
        comparisons.append(dict(cohort=cohort,epsilon=epsilon,isotropic=float(pivot.Isotropic.mean()),control_aware=float(pivot.ControlAware.mean()),relative_improvement_pct=float(100*(1-pivot.ControlAware.mean()/pivot.Isotropic.mean())),improved_seeds=int((delta<0).sum()),wilcoxon_statistic=float(stat),wilcoxon_p_less=float(pvalue)))
    comp=pd.DataFrame(comparisons);merged=comp.merge(summary[summary.method=='ControlAware'][['cohort','epsilon','tracking_increase_pct','feasible_rate','rho']],on=['cohort','epsilon'])
    passing=merged[(merged.relative_improvement_pct>=cfg['minimum_relative_improvement_pct'])&(merged.tracking_increase_pct<=cfg['acceptable_tracking_increase_pct'])&(merged.feasible_rate==1)&(merged.rho<1)]
    summary.to_csv(OUT/'experiment_8e_summary.csv',index=False);units.to_csv(OUT/'experiment_8e_seed_draw_summary.csv',index=False);comp.to_csv(OUT/'experiment_8e_paired_comparison.csv',index=False)
    conclusions=dict(provenance={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [CONFIG,ROOT/'code/src/federated_lpv/privacy.py',ROOT/'code/experiments/experiment_8e_control_aware_privacy.py']},local_fits=len(fits),privacy_releases=len(releases),closed_loop_evaluations=len(df),all_local_fits_converged=bool(fits.success.all()),common_random_numbers=True,passing_cells=passing[['cohort','epsilon','relative_improvement_pct','tracking_increase_pct','rho']].to_dict('records'),control_aware_gate=bool(len(passing)>0),box_clipped_coordinates=int(releases.box_clipped_coordinates.sum()),ellipsoid_clipped_clients=int(releases.clipped_clients.sum()))
    (OUT/'experiment_8e_conclusions.json').write_text(json.dumps(conclusions,indent=2)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(9.5,3.5),constrained_layout=True)
    for method,style in [('Isotropic','o--'),('ControlAware','s-')]:
        for cohort in cfg['cohorts']:
            cell=summary[(summary.method==method)&(summary.cohort==cohort)&(summary.epsilon!='inf')].copy();cell['x']=cell.epsilon.astype(float);cell=cell.sort_values('x')
            axes[0].plot(cell.x,cell.tracking_increase_pct,style,label=f'{method}-{cohort}')
    for cohort,cell in comp.groupby('cohort'):
        cell=cell.copy();cell['x']=cell.epsilon.astype(float);cell=cell.sort_values('x');axes[1].plot(cell.x,cell.relative_improvement_pct,'o-',label=f'n={cohort}')
    axes[0].axhline(cfg['acceptable_tracking_increase_pct'],color='k',ls=':');axes[1].axhline(cfg['minimum_relative_improvement_pct'],color='k',ls=':')
    axes[0].set(xscale='log',xlabel=r'$\epsilon$',ylabel='Tracking increase [%]',title='Privacy cost');axes[1].set(xscale='log',xlabel=r'$\epsilon$',ylabel='Improvement over isotropic [%]',title='Control-aware gain')
    for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=6)
    fig.savefig(ROOT/'results/figures/experiment_8e_control_aware_privacy.pdf');plt.close(fig)
    print(json.dumps(conclusions,indent=2));print(summary.to_string(index=False));print(comp.to_string(index=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--summarize-only',action='store_true');parser.add_argument('--workers',type=int,default=5);args=parser.parse_args()
    if not args.summarize_only:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:list(pool.map(run_seed,json.loads(CONFIG.read_text())['seeds']))
    summarize()
