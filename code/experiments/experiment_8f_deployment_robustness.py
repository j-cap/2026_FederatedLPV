"""8F: partial-participation and deployment robustness for private FL-LPV."""
from concurrent.futures import ProcessPoolExecutor
import argparse,gzip,hashlib,json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from federated_lpv.output_error import fit
from federated_lpv.privacy import private_mean,shaped_private_mean
from federated_lpv import discrete_bicycle_matrices,augmented_tracking_matrices
import experiment_8e_control_aware_privacy as eighte

ROOT=eighte.ROOT;OUT=eighte.OUT;base=eighte.base;corrected=eighte.corrected
CONFIG=ROOT/'code/config/experiment_8f.json';METHODS=('Isotropic','ControlAware')


def collect(clients,seed,cfg):
    nominal=eighte.eightd.prior.eight.seven.seven.controller(np.array([80000/1500,80000/1500,1500/2500]));n=round(cfg['episode_duration']/base.DT);t=np.arange(n+1)*base.DT
    rng=np.random.default_rng(seed+860000);data={c.client_id:[] for c in clients};categories={};speeds=cfg['coverage_speeds']
    for family in base.FAMILIES:
        members=[c for c in clients if c.family==family];order=rng.permutation(len(members))
        categories.update({c.client_id:int(order[j]%len(speeds)) for j,c in enumerate(members)})
        for episode in range(cfg['local_episodes']):
            ref=np.deg2rad(3)*(1-np.exp(-t/.3))*np.sin(2*np.pi*.35*t+.7*episode)
            for c in members:
                speed=np.full_like(t,speeds[categories[c.client_id]]);x,u,m=corrected.simulate([c],nominal,speed,ref,'vy')
                if not m['feasible'].all():raise RuntimeError('collection infeasible')
                data[c.client_id].append(dict(state=x[:,0]+rng.normal(size=x[:,0].shape)*np.deg2rad(cfg['noise_deg']),input=u[:,0],speed=speed))
    return data,categories,nominal


def select_members(members,categories,cohort,policy,seed):
    if policy=='random':
        order=np.random.default_rng(seed).permutation(len(members));return [members[j] for j in order[:cohort]]
    buckets={k:[] for k in range(10)}
    for c in members:buckets[categories[c.client_id]].append(c)
    rng=np.random.default_rng(seed)
    for k in buckets:rng.shuffle(buckets[k])
    if policy=='biased-low-speed':return [c for k in range(10) for c in buckets[k]][:cohort]
    if policy!='balanced':raise ValueError(policy)
    selected=[]
    while len(selected)<cohort:
        for k in range(10):
            if buckets[k]:selected.append(buckets[k].pop())
            if len(selected)==cohort:break
    return selected


def stress_bounds(bounds,condition):
    bounds=np.asarray(bounds,float)
    if condition in ('correct','outliers-5pct','outliers-10pct'):return bounds
    if condition=='shift-plus-5pct':return bounds*1.05
    if condition=='narrow-10pct':
        q=np.log(bounds);center=q.mean(axis=1);half=np.diff(q,axis=1).ravel()*.45
        return np.exp(np.column_stack((center-half,center+half)))
    raise ValueError(condition)


def contaminated(values,condition,seed):
    values=np.asarray(values,float).copy()
    frac={'outliers-5pct':.05,'outliers-10pct':.10}.get(condition,0)
    if frac:
        n=max(1,round(frac*len(values)));idx=np.random.default_rng(seed).choice(len(values),n,replace=False)
        values[idx]*=np.array([1.35,.70,1.30])
    return values


def release(values,method,eps,cfg,rng,family,bounds):
    if method=='Isotropic':return private_mean(values,eps,cfg['delta'],rng,bounds=bounds)
    return shaped_private_mean(values,eps,cfg['delta'],rng,cfg['control_weights'][family],bounds)


def run_seed(seed):
    cfg=json.loads(CONFIG.read_text());train=corrected.six.fleet(seed,1,max(cfg['cohorts']));data,categories,nominal=collect(train,seed,cfg);local={};fits=[]
    for c in train:
        p,d=fit(data[c.client_id]);local[c.client_id]=p;fits.append(dict(seed=seed,client=c.client_id,family=c.family,category=categories[c.client_id],cf_over_m=p[0],cr_over_m=p[1],m_over_iz=p[2],**d))
    test=corrected.six.fleet(seed+10000,1,cfg['test_clients_per_family']);t=np.arange(0,24+base.DT,base.DT);tests=[]
    for name in cfg['test_scenarios']:
        v,ref=corrected.scenario(name,t/2);tests.append((name,corrected.six.remap(v,(10,30)),ref))
    grid=np.linspace(10,30,161);audit={}
    for c in test:
        pairs=[augmented_tracking_matrices(*discrete_bicycle_matrices(float(v),c.parameters,base.DT),base.DT) for v in grid];audit[c.client_id]=(np.array([p[0] for p in pairs]),np.array([p[1] for p in pairs]))
    rows=[];releases=[]

    def evaluate(phase,point,policy,cohort,condition,mu,eps,method,draw):
        rng=np.random.default_rng(cfg['dp_seed']+seed*1000000+cohort*10000+draw+int(100*eps) if np.isfinite(eps) else cfg['dp_seed']+seed*1000000+cohort*10000+draw)
        params={};bounds=stress_bounds(cfg['bounds'],condition)
        for fi,family in enumerate(base.FAMILIES):
            members=[c for c in train if c.family==family];chosen=select_members(members,categories,cohort,policy,seed*1000+cohort*10+fi)
            values=contaminated([local[c.client_id] for c in chosen],condition,seed*10000+cohort*10+fi)
            p,d=release(values,method,eps,cfg,rng,family,bounds);params[family]=p
            releases.append(dict(seed=seed,phase=phase,point=point,policy=policy,cohort=cohort,condition=condition,mu=mu,epsilon=eighte.eightd.prior.label(eps),method=method,draw=draw,family=family,cf_over_m=p[0],cr_over_m=p[1],m_over_iz=p[2],**d))
        selected={c.client_id:eighte.eightd.prior.eight.seven.seven.controller(params[c.family]) for c in test};rhos=[]
        for c in test:
            a,b=audit[c.client_id];control=selected[c.client_id];g=np.column_stack([np.interp(grid,control.speeds,control.gains['global'][:,j]) for j in range(3)])
            rhos.append(float(np.max(abs(np.linalg.eigvals(a-b@g[:,None,:])))))
        for name,v,ref in tests:
            _,_,m=corrected.simulate(test,nominal,v,ref,'vy',client_controllers=selected,mu=mu)
            for j,c in enumerate(test):rows.append(dict(seed=seed,phase=phase,point=point,policy=policy,cohort=cohort,condition=condition,mu=mu,epsilon=eighte.eightd.prior.label(eps),method=method,draw=draw,scenario=name,client=c.client_id,family=c.family,rho=rhos[j],**{k:float(x[j]) for k,x in m.items()}))

    for policy in cfg['policies']:
        for cohort in cfg['cohorts']:
            for raw in cfg['epsilons']:
                eps=np.inf if raw=='inf' else float(raw);draws=1 if np.isinf(eps) else cfg['privacy_draws']
                for method in METHODS:
                    for draw in range(draws):evaluate('A','map',policy,cohort,'correct',.9,eps,method,draw)
            print(seed,'A',policy,cohort,flush=True)
    for point in cfg['stress_points']:
        for condition in cfg['stress_conditions']:
            for mu in cfg['frictions']:
                for eps in (float(point['epsilon']),np.inf):
                    draws=1 if np.isinf(eps) else cfg['stress_privacy_draws']
                    for method in METHODS:
                        for draw in range(draws):evaluate('B',point['name'],'balanced',point['cohort'],condition,mu,eps,method,draw)
        print(seed,'B',point['name'],flush=True)
    for suffix,items in [('fits',fits),('releases',releases),('clients',rows)]:
        (OUT/f'experiment_8f_seed{seed}_{suffix}.csv.gz').write_bytes(gzip.compress(pd.DataFrame(items).to_csv(index=False).encode(),mtime=0))


def summarize():
    cfg=json.loads(CONFIG.read_text());read=lambda x:pd.concat([pd.read_csv(OUT/f'experiment_8f_seed{s}_{x}.csv.gz') for s in cfg['seeds']],ignore_index=True)
    fits=read('fits');releases=read('releases');df=read('clients');keys=['seed','phase','point','policy','cohort','condition','mu','epsilon','method','draw']
    units=df.groupby(keys).agg(tracking=('tracking','mean'),feasible=('feasible','min'),rho=('rho','max')).reset_index();units['epsilon']=units.epsilon.map(lambda v:eighte.eightd.prior.label(float(v)))
    group=['phase','point','policy','cohort','condition','mu','epsilon','method'];summary=units.groupby(group).agg(tracking=('tracking','mean'),std=('tracking','std'),q95=('tracking',lambda x:np.quantile(x,.95)),feasible_rate=('feasible','mean'),rho=('rho','max')).reset_index()
    base_rows=summary[summary.epsilon=='inf'][['phase','point','policy','cohort','condition','mu','method','tracking']].rename(columns={'tracking':'baseline'})
    summary=summary.merge(base_rows,on=['phase','point','policy','cohort','condition','mu','method']);summary['tracking_increase_pct']=100*(summary.tracking/summary.baseline-1);summary['viable']=(summary.epsilon!='inf')&(summary.tracking_increase_pct<=cfg['acceptable_tracking_increase_pct'])&(summary.feasible_rate==1)&(summary.rho<1)
    finite=summary[summary.epsilon!='inf'];paired=[]
    for key,cell in units[units.epsilon!='inf'].groupby(['phase','point','policy','cohort','condition','mu','epsilon']):
        p=cell.pivot(index=['seed','draw'],columns='method',values='tracking');delta=(p.ControlAware-p.Isotropic).groupby(level='seed').mean()
        paired.append(dict(zip(['phase','point','policy','cohort','condition','mu','epsilon'],key),relative_improvement_pct=float(100*(1-p.ControlAware.mean()/p.Isotropic.mean())),improved_seeds=int((delta<0).sum())))
    paired=pd.DataFrame(paired);summary.to_csv(OUT/'experiment_8f_summary.csv',index=False);units.to_csv(OUT/'experiment_8f_seed_draw_summary.csv',index=False);paired.to_csv(OUT/'experiment_8f_paired_comparison.csv',index=False)
    amap=finite[finite.phase=='A'];minimum=[]
    for (method,policy,epsilon),cell in amap.groupby(['method','policy','epsilon']):
        good=cell[cell.viable];minimum.append(dict(method=method,policy=policy,epsilon=epsilon,minimum_viable_cohort=None if good.empty else int(good.cohort.min())))
    minimum=pd.DataFrame(minimum);minimum.to_csv(OUT/'experiment_8f_minimum_cohort.csv',index=False)
    b=finite[finite.phase=='B'];robust=b.groupby(['point','method']).viable.all().reset_index(name='viable_all_stresses')
    conclusions=dict(provenance={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [CONFIG,ROOT/'code/src/federated_lpv/privacy.py',ROOT/'code/experiments/experiment_6b_acceleration_validation.py',ROOT/'code/experiments/experiment_8f_deployment_robustness.py']},local_fits=len(fits),privacy_releases=len(releases),closed_loop_evaluations=len(df),all_local_fits_converged=bool(fits.success.all()),minimum_viable_cohorts=minimum.to_dict('records'),stress_robustness=robust.to_dict('records'),all_nonprivate_feasible=bool(summary[summary.epsilon=='inf'].feasible_rate.min()==1),deployment_gate=bool(robust.viable_all_stresses.any()),box_clipped_coordinates=int(releases.box_clipped_coordinates.sum()))
    (OUT/'experiment_8f_conclusions.json').write_text(json.dumps(conclusions,indent=2)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(10,3.6),constrained_layout=True)
    for (method,policy,epsilon),cell in amap.groupby(['method','policy','epsilon']):
        axes[0].plot(cell.cohort,cell.tracking_increase_pct,marker='o',label=f'{method[:3]}-{policy[:3]}-e{epsilon}')
    axes[0].axhline(10,color='k',ls=':');axes[0].set(xlabel='Active clients per family',ylabel='Tracking increase [%]',yscale='symlog',title='Partial-participation map');axes[0].grid(alpha=.2);axes[0].legend(fontsize=5,ncol=2)
    view=b[(b.epsilon=='4')&(b.point=='boundary')]
    labels=[f'{r.condition}\nmu={r.mu:g}' for r in view[view.method=='Isotropic'].itertuples()];x=np.arange(len(labels));width=.38
    for offset,method in [(-width/2,'Isotropic'),(width/2,'ControlAware')]:
        cell=view[view.method==method];axes[1].bar(x+offset,cell.tracking_increase_pct,width,label=method)
    axes[1].axhline(10,color='k',ls=':');axes[1].set(xticks=x,xticklabels=labels,ylabel='Tracking increase [%]',title='Boundary stress, n=30, epsilon=4');axes[1].tick_params(axis='x',rotation=45,labelsize=6);axes[1].legend();axes[1].grid(axis='y',alpha=.2)
    fig.savefig(ROOT/'results/figures/experiment_8f_deployment_robustness.pdf');plt.close(fig);print(json.dumps(conclusions,indent=2));print(minimum.to_string(index=False));print(robust.to_string(index=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--summarize-only',action='store_true');parser.add_argument('--workers',type=int,default=5);args=parser.parse_args()
    if not args.summarize_only:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:list(pool.map(run_seed,json.loads(CONFIG.read_text())['seeds']))
    summarize()
