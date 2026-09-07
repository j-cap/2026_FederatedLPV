"""8D: recover client-level DP utility via public bounds and cohort scaling."""
from concurrent.futures import ProcessPoolExecutor
import argparse,gzip,hashlib,json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from federated_lpv.output_error import fit
from federated_lpv.privacy import private_mean
from federated_lpv import discrete_bicycle_matrices,augmented_tracking_matrices
import experiment_8c_private_frontier as prior

ROOT=prior.ROOT;OUT=prior.OUT;base=prior.base;corrected=prior.corrected;CONFIG=ROOT/'code/config/experiment_8d.json'


def collect(clients,seed,cfg):
    """Collect balanced complementary records for cohorts larger than the speed grid."""
    nominal=prior.eight.seven.seven.controller(np.array([80000/1500,80000/1500,1500/2500]));n=round(cfg['episode_duration']/base.DT);t=np.arange(n+1)*base.DT
    rng=np.random.default_rng(seed+840000);data={c.client_id:[] for c in clients};speeds=cfg['coverage_speeds']
    for family in base.FAMILIES:
        members=[c for c in clients if c.family==family];order=rng.permutation(len(members))
        categories={c.client_id:int(order[j]%len(speeds)) for j,c in enumerate(members)}
        for episode in range(cfg['local_episodes']):
            ref=np.deg2rad(3)*(1-np.exp(-t/.3))*np.sin(2*np.pi*.35*t+.7*episode)
            for c in members:
                speed=np.full_like(t,speeds[categories[c.client_id]]);x,u,m=corrected.simulate([c],nominal,speed,ref,'vy')
                if not m['feasible'].all():raise RuntimeError('collection infeasible')
                data[c.client_id].append(dict(state=x[:,0]+rng.normal(size=x[:,0].shape)*np.deg2rad(cfg['noise_deg']),input=u[:,0],speed=speed))
    return data,nominal


def run_seed(seed):
    cfg=json.loads(CONFIG.read_text());max_cohort=max(m['cohort'] for m in cfg['mechanisms']);train=corrected.six.fleet(seed,1,max_cohort);data,nominal=collect(train,seed,cfg);local={};fits=[]
    for c in train:
        p,d=fit(data[c.client_id]);local[c.client_id]=p;fits.append(dict(seed=seed,client=c.client_id,family=c.family,cf_over_m=p[0],cr_over_m=p[1],m_over_iz=p[2],**d))
    test=corrected.six.fleet(seed+10000,1,cfg['test_clients_per_family']);t=np.arange(0,24+base.DT,base.DT);tests=[]
    for name in cfg['test_scenarios']:
        v,ref=corrected.scenario(name,t/2);tests.append((name,corrected.six.remap(v,(10,30)),ref))
    nominal_test=prior.eight.seven.seven.controller(np.array([80000/1500,80000/1500,1500/2500]));grid=np.linspace(10,30,161);audit={}
    for c in test:
        pairs=[augmented_tracking_matrices(*discrete_bicycle_matrices(float(v),c.parameters,base.DT),base.DT) for v in grid];audit[c.client_id]=(np.array([p[0] for p in pairs]),np.array([p[1] for p in pairs]))
    rows=[];releases=[]
    for mech in cfg['mechanisms']:
        for raw_eps in cfg['epsilons']:
            eps=np.inf if raw_eps=='inf' else float(raw_eps);draws=1 if np.isinf(eps) else cfg['privacy_draws']
            for draw in range(draws):
                rng=np.random.default_rng(cfg['dp_seed']+seed*100000+draw+int(100*(0 if np.isinf(eps) else eps))+mech['cohort']);params={}
                for family in base.FAMILIES:
                    members=[c for c in train if c.family==family][:mech['cohort']];p,d=private_mean([local[c.client_id] for c in members],eps,cfg['delta'],rng,bounds=mech['bounds']);params[family]=p;releases.append(dict(seed=seed,mechanism=mech['name'],cohort=mech['cohort'],epsilon=prior.label(eps),draw=draw,family=family,cf_over_m=p[0],cr_over_m=p[1],m_over_iz=p[2],**d))
                selected={c.client_id:prior.eight.seven.seven.controller(params[c.family]) for c in test};rhos=[]
                for c in test:
                    a,b=audit[c.client_id];control=selected[c.client_id];g=np.column_stack([np.interp(grid,control.speeds,control.gains['global'][:,j]) for j in range(3)]);rhos.append(float(np.max(abs(np.linalg.eigvals(a-b@g[:,None,:])))))
                for name,v,ref in tests:
                    _,_,m=corrected.simulate(test,nominal_test,v,ref,'vy',client_controllers=selected)
                    for j,c in enumerate(test):rows.append(dict(seed=seed,mechanism=mech['name'],cohort=mech['cohort'],epsilon=prior.label(eps),draw=draw,scenario=name,client=c.client_id,family=c.family,rho=rhos[j],**{k:float(x[j]) for k,x in m.items()}))
            print(seed,mech['name'],prior.label(eps),flush=True)
    for suffix,items in [('fits',fits),('releases',releases),('clients',rows)]:
        (OUT/f'experiment_8d_seed{seed}_{suffix}.csv.gz').write_bytes(gzip.compress(pd.DataFrame(items).to_csv(index=False).encode(),mtime=0))


def summarize():
    cfg=json.loads(CONFIG.read_text());read=lambda x:pd.concat([pd.read_csv(OUT/f'experiment_8d_seed{s}_{x}.csv.gz') for s in cfg['seeds']],ignore_index=True);fits=read('fits');rel=read('releases');df=read('clients');keys=['seed','mechanism','cohort','epsilon','draw'];sr=df.groupby(keys).agg(tracking=('tracking','mean'),feasible=('feasible','min'),rho=('rho','max')).reset_index();sr['epsilon']=sr.epsilon.map(lambda v:prior.label(float(v)));summary=sr.groupby(['mechanism','cohort','epsilon']).agg(tracking=('tracking','mean'),std=('tracking','std'),q95=('tracking',lambda x:np.quantile(x,.95)),feasible_rate=('feasible','mean'),rho=('rho','max')).reset_index();base_rows=summary[summary.epsilon=='inf'][['mechanism','tracking']].rename(columns={'tracking':'baseline'});summary=summary.merge(base_rows,on='mechanism');summary['tracking_increase_pct']=100*(summary.tracking/summary.baseline-1);summary.to_csv(OUT/'experiment_8d_summary.csv',index=False);sr.to_csv(OUT/'experiment_8d_seed_draw_summary.csv',index=False)
    finite=summary[summary.epsilon!='inf'];acceptable=finite[(finite.tracking_increase_pct<=cfg['acceptable_tracking_increase_pct'])&(finite.feasible_rate==1)&(finite.rho<1)];conclusions=dict(provenance={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [CONFIG,ROOT/'code/src/federated_lpv/privacy.py',ROOT/'code/experiments/experiment_8d_privacy_recovery.py']},local_fits=len(fits),privacy_releases=len(rel),closed_loop_evaluations=len(df),all_local_fits_converged=bool(fits.success.all()),acceptable_cells=acceptable[['mechanism','epsilon','tracking_increase_pct']].to_dict('records'),recovery_gate=bool(len(acceptable)>0),box_clipped_coordinates=int(rel.box_clipped_coordinates.sum()))
    (OUT/'experiment_8d_conclusions.json').write_text(json.dumps(conclusions,indent=2)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(10,3.6),constrained_layout=True)
    for mech,cell in finite.groupby('mechanism'):
        cell=cell.copy();cell['x']=cell.epsilon.astype(float);cell=cell.sort_values('x');axes[0].plot(cell.x,cell.tracking_increase_pct,'o-',label=mech);axes[1].plot(cell.x,cell.rho,'o-',label=mech)
    axes[0].axhline(cfg['acceptable_tracking_increase_pct'],color='k',ls='--');axes[1].axhline(1,color='k',ls='--');axes[0].set(xscale='log',xlabel=r'$\epsilon$',ylabel='Tracking increase [%]',title='Privacy utility recovery');axes[1].set(xscale='log',xlabel=r'$\epsilon$',ylabel='Max frozen radius',title='Stability diagnostic');[a.grid(alpha=.2) for a in axes];axes[0].legend(fontsize=7);fig.savefig(ROOT/'results/figures/experiment_8d_privacy_recovery.pdf');plt.close(fig);print(json.dumps(conclusions,indent=2));print(summary.to_string(index=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--summarize-only',action='store_true');p.add_argument('--workers',type=int,default=3);a=p.parse_args()
    if not a.summarize_only:
        with ProcessPoolExecutor(max_workers=a.workers) as pool:list(pool.map(run_seed,json.loads(CONFIG.read_text())['seeds']))
    summarize()
