"""8B: federated and secure-aggregation equivalence gate."""
from concurrent.futures import ProcessPoolExecutor
import argparse,gzip,hashlib,json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from federated_lpv.output_error import fit
from federated_lpv.federated import fit_federated
from federated_lpv import discrete_bicycle_matrices,augmented_tracking_matrices
import experiment_7c_complementary_coverage as seven

ROOT=seven.ROOT;OUT=seven.OUT;base=seven.base;corrected=seven.corrected;CONFIG=ROOT/'code/config/experiment_8b.json'
METHODS=('Central','Federated','Secure')


def collect(clients,seed,cfg):
    nominal=seven.seven.controller(np.array([80000/1500,80000/1500,1500/2500]));n=round(cfg['episode_duration']/base.DT);t=np.arange(n+1)*base.DT
    rng=np.random.default_rng(seed+820000);order=rng.permutation(cfg['clients_per_family']);categories={c.client_id:int(order[int(c.client_id.rsplit('_',1)[1])]) for c in clients};data={c.client_id:[] for c in clients}
    for episode in range(cfg['local_episodes']):
        ref=np.deg2rad(3)*(1-np.exp(-t/.3))*np.sin(2*np.pi*.35*t+.7*episode)
        for c in clients:
            speed=np.full_like(t,cfg['coverage_speeds'][categories[c.client_id]]);x,u,m=corrected.simulate([c],nominal,speed,ref,'vy')
            if not m['feasible'].all():raise RuntimeError('collection infeasible')
            data[c.client_id].append(dict(state=x[:,0]+rng.normal(size=x[:,0].shape)*np.deg2rad(cfg['noise_deg']),input=u[:,0],speed=speed))
    return data,nominal


def run_seed(seed):
    cfg=json.loads(CONFIG.read_text());clients=corrected.six.fleet(seed,1,cfg['clients_per_family']);data,nominal=collect(clients,seed,cfg)
    groups={'global':[c.client_id for c in clients]};groups.update({f:[c.client_id for c in clients if c.family==f] for f in base.FAMILIES});params={};diag=[]
    for group,ids in groups.items():
        local=[data[i] for i in ids];flat=[r for rs in local for r in rs];pc,dc=fit(flat);params[(group,'Central')]=pc;diag.append(dict(seed=seed,group=group,method='Central',cf_over_m=pc[0],cr_over_m=pc[1],m_over_iz=pc[2],**dc))
        for method,secure in [('Federated',False),('Secure',True)]:
            p,d=fit_federated(local,secure,seed);params[(group,method)]=p;diag.append(dict(seed=seed,group=group,method=method,cf_over_m=p[0],cr_over_m=p[1],m_over_iz=p[2],**d))
    print(f'identified seed {seed}',flush=True)
    t=np.arange(0,24+base.DT,base.DT);tests=[]
    for name in cfg['test_scenarios']:
        v,ref=corrected.scenario(name,t/2);v=corrected.six.remap(v,(10,30));tests.append((name,v,ref))
    grid=np.linspace(10,30,161);rows=[]
    for scope in ('global','family'):
        for method in METHODS:
            selected={}
            for c in clients:selected[c.client_id]=seven.seven.controller(params[(scope if scope=='global' else c.family,method)])
            rhos=[]
            for c in clients:
                pairs=[augmented_tracking_matrices(*discrete_bicycle_matrices(float(v),c.parameters,base.DT),base.DT) for v in grid];a=np.array([p[0] for p in pairs]);b=np.array([p[1] for p in pairs]);control=selected[c.client_id];g=np.column_stack([np.interp(grid,control.speeds,control.gains['global'][:,j]) for j in range(3)]);rhos.append(float(np.max(abs(np.linalg.eigvals(a-b@g[:,None,:])))))
            for name,v,ref in tests:
                _,_,m=corrected.simulate(clients,nominal,v,ref,'vy',client_controllers=selected)
                for j,c in enumerate(clients):rows.append(dict(seed=seed,scope=scope,method=method,scenario=name,client=c.client_id,family=c.family,rho=rhos[j],**{k:float(x[j]) for k,x in m.items()}))
    for suffix,items in [('parameters',diag),('clients',rows)]:
        (OUT/f'experiment_8b_seed{seed}_{suffix}.csv.gz').write_bytes(gzip.compress(pd.DataFrame(items).to_csv(index=False).encode(),mtime=0))


def summarize():
    cfg=json.loads(CONFIG.read_text());read=lambda x:pd.concat([pd.read_csv(OUT/f'experiment_8b_seed{s}_{x}.csv.gz') for s in cfg['seeds']],ignore_index=True);pa=read('parameters');df=read('clients')
    if len(pa)!=120 or len(df)!=5400:raise RuntimeError('incomplete outputs')
    comparisons=[]
    for seed in cfg['seeds']:
        for group in ['global',*base.FAMILIES]:
            cell=pa[(pa.seed==seed)&(pa.group==group)].set_index('method');central=cell.loc['Central',['cf_over_m','cr_over_m','m_over_iz']].to_numpy(float)
            for method in ('Federated','Secure'):
                p=cell.loc[method,['cf_over_m','cr_over_m','m_over_iz']].to_numpy(float);comparisons.append(dict(seed=seed,group=group,method=method,max_parameter_relative_error=np.max(abs(p-central)/central)))
    comp=pd.DataFrame(comparisons);comp.to_csv(OUT/'experiment_8b_parameter_equivalence.csv',index=False)
    sr=df.groupby(['seed','scope','method']).agg(tracking=('tracking','mean'),feasible=('feasible','min'),rho=('rho','max')).reset_index();sr.to_csv(OUT/'experiment_8b_seed_summary.csv',index=False)
    summary=sr.groupby(['scope','method']).agg(tracking=('tracking','mean'),std=('tracking','std'),feasible=('feasible','min'),rho=('rho','max')).reset_index();summary.to_csv(OUT/'experiment_8b_summary.csv',index=False)
    track=[]
    for (seed,scope),cell in sr.groupby(['seed','scope']):
        c=float(cell[cell.method=='Central'].tracking.iloc[0])
        for method in ('Federated','Secure'):track.append(dict(seed=seed,scope=scope,method=method,absolute_tracking_difference=abs(float(cell[cell.method==method].tracking.iloc[0])-c)))
    track=pd.DataFrame(track);track.to_csv(OUT/'experiment_8b_tracking_equivalence.csv',index=False)
    plain=pa[pa.method=='Federated'].sort_values(['seed','group']);secure=pa[pa.method=='Secure'].sort_values(['seed','group']);cols=['cf_over_m','cr_over_m','m_over_iz'];secure_difference=float(np.max(abs(plain[cols].to_numpy()-secure[cols].to_numpy())))
    conclusions=dict(provenance={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [CONFIG,ROOT/'code/src/federated_lpv/federated.py',ROOT/'code/experiments/experiment_8b_federated_equivalence.py']},fits=len(pa),evaluations=len(df),all_fits_converged=bool(pa.success.all()),max_parameter_relative_error=float(comp.max_parameter_relative_error.max()),max_tracking_difference=float(track.absolute_tracking_difference.max()),secure_plain_parameter_difference=secure_difference,all_amplitude_feasible=bool(df.feasible.min()==1),max_frozen_rho=float(df.rho.max()),total_uplink_floats=int(pa[pa.method!='Central'].uplink_floats.sum()),total_downlink_floats=int(pa[pa.method!='Central'].downlink_floats.sum()))
    conclusions['equivalence_gate']=bool(conclusions['all_fits_converged'] and conclusions['max_parameter_relative_error']<=cfg['parameter_tolerance'] and conclusions['max_tracking_difference']<=cfg['tracking_tolerance_deg_s'] and secure_difference<=cfg['secure_sum_tolerance'] and conclusions['all_amplitude_feasible'] and conclusions['max_frozen_rho']<1)
    (OUT/'experiment_8b_conclusions.json').write_text(json.dumps(conclusions,indent=2)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(8,3.4),constrained_layout=True)
    for ax,scope in zip(axes,('global','family')):
        sub=summary[summary.scope==scope].set_index('method').loc[list(METHODS)];ax.bar(METHODS,sub.tracking,color=['#555555','#2878b5','#34934b']);ax.set(title=scope.capitalize(),ylabel='Yaw tracking RMSE [deg/s]');ax.grid(axis='y',alpha=.2)
    fig.savefig(ROOT/'results/figures/experiment_8b_federated_equivalence.pdf');plt.close(fig);print(json.dumps(conclusions,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--summarize-only',action='store_true');p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    if not a.summarize_only:
        with ProcessPoolExecutor(max_workers=a.workers) as pool:list(pool.map(run_seed,json.loads(CONFIG.read_text())['seeds']))
    summarize()
