"""8C: client-level privacy--estimation--control frontier."""
from concurrent.futures import ProcessPoolExecutor
import argparse,gzip,hashlib,json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from federated_lpv.output_error import fit
from federated_lpv.privacy import private_mean
from federated_lpv import discrete_bicycle_matrices,augmented_tracking_matrices
import experiment_8b_federated_equivalence as eight

ROOT=eight.ROOT;OUT=eight.OUT;base=eight.base;corrected=eight.corrected;CONFIG=ROOT/'code/config/experiment_8c.json'


def label(e):return 'inf' if np.isinf(e) else str(e).rstrip('0').rstrip('.')


def run_seed(seed):
    cfg=json.loads(CONFIG.read_text());clients=corrected.six.fleet(seed,1,cfg['clients_per_family']);data,nominal=eight.collect(clients,seed,cfg);local={};fits=[]
    for c in clients:
        p,d=fit(data[c.client_id]);local[c.client_id]=p;fits.append(dict(seed=seed,client=c.client_id,family=c.family,cf_over_m=p[0],cr_over_m=p[1],m_over_iz=p[2],**d))
    tests=[];t=np.arange(0,24+base.DT,base.DT)
    for name in cfg['test_scenarios']:
        v,ref=corrected.scenario(name,t/2);tests.append((name,corrected.six.remap(v,(10,30)),ref))
    grid=np.linspace(10,30,161);audit={}
    for c in clients:
        pairs=[augmented_tracking_matrices(*discrete_bicycle_matrices(float(v),c.parameters,base.DT),base.DT) for v in grid];audit[c.client_id]=(np.array([p[0] for p in pairs]),np.array([p[1] for p in pairs]))
    rows=[];privacy=[]
    eps_values=[np.inf if e=='inf' else float(e) for e in cfg['epsilons']]
    for eps in eps_values:
        draws=1 if np.isinf(eps) else cfg['privacy_draws']
        for draw in range(draws):
            params={};rng=np.random.default_rng(cfg['dp_seed']+10000*seed+draw+int(100*(0 if np.isinf(eps) else eps)))
            for family in base.FAMILIES:
                members=[c for c in clients if c.family==family];p,d=private_mean([local[c.client_id] for c in members],eps,cfg['delta'],rng);params[family]=p;privacy.append(dict(seed=seed,epsilon=label(eps),draw=draw,family=family,cf_over_m=p[0],cr_over_m=p[1],m_over_iz=p[2],**d))
            selected={c.client_id:eight.seven.seven.controller(params[c.family]) for c in clients};rhos=[]
            for c in clients:
                a,b=audit[c.client_id];control=selected[c.client_id];g=np.column_stack([np.interp(grid,control.speeds,control.gains['global'][:,j]) for j in range(3)]);rhos.append(float(np.max(abs(np.linalg.eigvals(a-b@g[:,None,:])))))
            for name,v,ref in tests:
                _,_,m=corrected.simulate(clients,nominal,v,ref,'vy',client_controllers=selected)
                for j,c in enumerate(clients):rows.append(dict(seed=seed,epsilon=label(eps),draw=draw,scenario=name,client=c.client_id,family=c.family,rho=rhos[j],**{k:float(x[j]) for k,x in m.items()}))
        print(f'seed {seed} epsilon {label(eps)}',flush=True)
    for suffix,items in [('fits',fits),('privacy',privacy),('clients',rows)]:
        (OUT/f'experiment_8c_seed{seed}_{suffix}.csv.gz').write_bytes(gzip.compress(pd.DataFrame(items).to_csv(index=False).encode(),mtime=0))


def summarize():
    cfg=json.loads(CONFIG.read_text());read=lambda x:pd.concat([pd.read_csv(OUT/f'experiment_8c_seed{s}_{x}.csv.gz') for s in cfg['seeds']],ignore_index=True);fits=read('fits');pa=read('privacy');df=read('clients')
    keys=['seed','epsilon','draw'];sr=df.groupby(keys).agg(tracking=('tracking','mean'),feasible=('feasible','min'),rho=('rho','max'),steering_rms=('steering_rms','mean'),peak_rate=('peak_steering_rate','max')).reset_index();order=['0.5','1','2','4','8','inf'];sr['epsilon']=pd.Categorical(sr.epsilon.map(lambda v:label(float(v))),order,ordered=True);sr.to_csv(OUT/'experiment_8c_seed_draw_summary.csv',index=False)
    summary=sr.groupby('epsilon',observed=True).agg(tracking=('tracking','mean'),std=('tracking','std'),q95=('tracking',lambda x:np.quantile(x,.95)),feasible_rate=('feasible','mean'),rho=('rho','max'),steering_rms=('steering_rms','mean'),peak_rate=('peak_rate','max')).reset_index();base_row=summary[summary.epsilon=='inf'].iloc[0];summary['tracking_increase_pct']=100*(summary.tracking/base_row.tracking-1);summary.to_csv(OUT/'experiment_8c_summary.csv',index=False)
    acceptable=summary[(summary.epsilon!='inf')&(summary.tracking_increase_pct<=cfg['acceptable_tracking_increase_pct'])&(summary.feasible_rate==1)&(summary.rho<1)]
    conclusions=dict(provenance={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [CONFIG,ROOT/'code/src/federated_lpv/privacy.py',ROOT/'code/experiments/experiment_8c_private_frontier.py']},local_fits=len(fits),privacy_releases=len(pa),closed_loop_evaluations=len(df),all_local_fits_converged=bool(fits.success.all()),delta=cfg['delta'],replacement_adjacency=True,acceptable_epsilons=[str(x) for x in acceptable.epsilon.tolist()],strongest_acceptable_epsilon=None if acceptable.empty else str(acceptable.iloc[0].epsilon),all_nonprivate_feasible=bool(base_row.feasible_rate==1),max_nonprivate_rho=float(base_row.rho))
    (OUT/'experiment_8c_conclusions.json').write_text(json.dumps(conclusions,indent=2)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(9,3.5),constrained_layout=True);finite=summary[summary.epsilon!='inf'].copy();x=finite.epsilon.astype(float)
    axes[0].plot(x,finite.tracking_increase_pct,'o-');axes[0].axhline(cfg['acceptable_tracking_increase_pct'],color='k',ls='--',lw=1);axes[0].set(xscale='log',xlabel=r'Privacy budget $\epsilon$',ylabel='Tracking increase [%]',title='Privacy--control frontier');axes[0].grid(alpha=.2)
    axes[1].plot(x,finite.feasible_rate,'o-',label='Feasible fraction');axes[1].plot(x,finite.rho,'s-',label='Max frozen radius');axes[1].axhline(1,color='k',ls='--',lw=1);axes[1].set(xscale='log',xlabel=r'Privacy budget $\epsilon$',title='Safety diagnostics');axes[1].grid(alpha=.2);axes[1].legend()
    fig.savefig(ROOT/'results/figures/experiment_8c_private_frontier.pdf');plt.close(fig);print(json.dumps(conclusions,indent=2));print(summary.to_string(index=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--summarize-only',action='store_true');p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    if not a.summarize_only:
        with ProcessPoolExecutor(max_workers=a.workers) as pool:list(pool.map(run_seed,json.loads(CONFIG.read_text())['seeds']))
    summarize()
