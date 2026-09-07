"""Independent-fleet validation of frozen 4F settings and historical LTI fits."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import argparse
import csv
import hashlib
import json
import numpy as np
import matplotlib.pyplot as plt
import experiment_4c_nonlinear_oracle_models as base
from experiment_4d_nonlinear_controllers import redesign_controllers, stability_audit
from experiment_4f_structured_identification import fit_structured
from federated_lpv import sample_fleet, fit_oracle_architectures, design_oracle_controllers

ROOT=Path(__file__).resolve().parents[2]
CONFIG=ROOT/'code/config/experiment_4g.json'
METHODS=(*base.METHODS,'LTI_all','LTI_near','physics_M4')

def simulate_fleet(clients,controller,speed,reference):
    """Vectorize clients only; retain historical frozen-speed RK4 and integrator."""
    n=len(clients); nt=len(speed)
    x=np.zeros((nt,n,2)); u=np.zeros((nt-1,n)); integral=np.zeros(n)
    p=np.array([[c.parameters.mass,c.parameters.yaw_inertia,c.parameters.front_length,
                 c.parameters.rear_length,c.parameters.front_stiffness,c.parameters.rear_stiffness] for c in clients])
    mass,iz,lf,lr,cf,cr=p.T
    front_limit=base.MU*mass*9.81*lr/(lf+lr); rear_limit=base.MU*mass*9.81*lf/(lf+lr)
    gains=np.empty((nt-1,n,3)); pre=np.empty((nt-1,n))
    for family in base.FAMILIES:
        mask=np.array([c.family==family for c in clients])
        values=[controller.evaluate(family,float(v)) for v in speed[:-1]]
        gains[:,mask,:]=np.array([v[0] for v in values])[:,None,:]
        pre[:,mask]=np.array([v[1] for v in values])[:,None]
    for k in range(nt-1):
        u[k]=-np.sum(gains[k]*np.column_stack((x[k],integral)),axis=1)+pre[k]*reference[k]
        def rhs(z):
            f=front_limit*np.tanh(cf*(u[k]-z[:,0]-lf*z[:,1]/speed[k])/front_limit)
            r=rear_limit*np.tanh(cr*(-z[:,0]+lr*z[:,1]/speed[k])/rear_limit)
            return np.column_stack(((f+r)/(mass*speed[k])-z[:,1],(lf*f-lr*r)/iz))
        k1=rhs(x[k]); k2=rhs(x[k]+base.DT*k1/2); k3=rhs(x[k]+base.DT*k2/2); k4=rhs(x[k]+base.DT*k3)
        x[k+1]=x[k]+base.DT*(k1+2*k2+2*k3+k4)/6
        integral+=base.DT*(x[k,:,1]-reference[k])
    f=front_limit*np.tanh(cf*(u-x[:-1,:,0]-lf*x[:-1,:,1]/speed[:-1,None])/front_limit)
    r=rear_limit*np.tanh(cr*(-x[:-1,:,0]+lr*x[:-1,:,1]/speed[:-1,None])/rear_limit)
    peak_u=np.rad2deg(np.max(abs(u),axis=0)); peak_acc=np.max(abs((f+r)/mass),axis=0)
    finite=np.all(np.isfinite(x),axis=(0,2)) & np.all(np.isfinite(u),axis=0)
    metrics=dict(tracking=np.rad2deg(np.sqrt(np.mean((x[:,:,1]-reference[:,None])**2,axis=0))),
        peak_steering=peak_u,peak_acceleration=peak_acc,
        peak_steering_rate=np.rad2deg(np.max(abs(np.diff(u,axis=0)/base.DT),axis=0)),
        beta_rms=np.rad2deg(np.sqrt(np.mean(x[:,:,0]**2,axis=0))),
        steering_rms=np.rad2deg(np.sqrt(np.mean(u**2,axis=0))),
        feasible=finite & (peak_u<=12.) & (peak_acc<=base.MU*9.81))
    return x,u,metrics

def scenario(name,time):
    if name=='previous_s_curve':
        return base.smooth_profile(time,base.TEST_PROFILE),base.reference_signal(time,'unseen_s_curve')
    if name=='shifted_double_lane':
        v=base.smooth_profile(time,(16.,27.,19.))
        r=8.5*(np.exp(-.5*((time-3.8)/.85)**2)-np.exp(-.5*((time-7.4)/.65)**2))
        return v,np.deg2rad(r)
    if name=='constant_22':
        r=7.5*np.sin(np.pi*time/base.DURATION)**2*np.sin(2*np.pi*.18*time+.35)
        return np.full_like(time,22.),np.deg2rad(r)
    raise ValueError(name)

def run_seed(seed):
    cfg=json.loads(CONFIG.read_text()); test_seed=seed+cfg['test_seed_offset']
    train=sample_fleet(seed,cfg['clients_per_family']); test=sample_fleet(test_seed,cfg['clients_per_family'])
    time=np.arange(0,base.DURATION+base.DT,base.DT)
    physics=fit_oracle_architectures(train,base.FIT_SPEEDS,base.DT)
    transfer=design_oracle_controllers(physics,base.DT,base.Q,base.R,base.CONTROL_SPEEDS,base.FIT_SPEEDS)
    all_rows=[]; near=[]
    for profile,maneuver in zip(base.TRAIN_PROFILES,('lane','sine')):
        v=base.smooth_profile(time,profile)
        for severity,scale in cfg['severities'].items():
            x,u,_=simulate_fleet(train,transfer['M4'],v,scale*base.reference_signal(time,maneuver))
            for j,c in enumerate(train):
                row=dict(family=c.family,state=x[:,j],input=u[:,j],speed=v)
                all_rows.append(row)
                if severity=='near_linear': near.append(row)
    models,parameters=fit_structured(near)
    controllers=redesign_controllers(models)
    for name,data in [('LTI_all',all_rows),('LTI_near',near)]:
        coefficients={f:base.fit_constant([r for r in data if r['family']==f]) for f in base.FAMILIES}
        model=base.DataModel('constant','family',coefficients)
        controllers[name]=redesign_controllers({'M2':model})['M2']
    controllers['physics_M4']=transfer['M4']
    audit=stability_audit(test,{'held_out':controllers})
    rows=[]
    for name in cfg['scenarios']:
        v,reference=scenario(name,time)
        for severity,scale in cfg['severities'].items():
            for method,controller in controllers.items():
                _,_,metrics=simulate_fleet(test,controller,v,scale*reference)
                for j,c in enumerate(test):
                    rows.append(dict(seed=seed,test_seed=test_seed,scenario=name,severity=severity,
                        method=method,client=c.client_id,family=c.family,
                        **{key:float(value[j]) for key,value in metrics.items()}))
    for row in audit: row.update(seed=seed,test_seed=test_seed)
    for row in parameters: row.update(seed=seed)
    output=ROOT/'results/tables'
    base.write_csv(output/f'experiment_4g_seed{seed}_clients.csv',rows)
    base.write_csv(output/f'experiment_4g_seed{seed}_stability.csv',audit)
    base.write_csv(output/f'experiment_4g_seed{seed}_parameters.csv',parameters)
    print('completed seed',seed,flush=True)
    return seed

def summarize():
    cfg=json.loads(CONFIG.read_text()); rows=[]; audits=[]
    out=ROOT/'results/tables'
    for seed in cfg['training_seeds']:
        with (out/f'experiment_4g_seed{seed}_clients.csv').open() as f: rows.extend(list(csv.DictReader(f)))
        with (out/f'experiment_4g_seed{seed}_stability.csv').open() as f: audits.extend(list(csv.DictReader(f)))
    seed_rows=[]
    for seed in cfg['training_seeds']:
        for severity in cfg['severities']:
            for name in [*cfg['scenarios'],'all']:
                for method in METHODS:
                    selected=[r for r in rows if int(r['seed'])==seed and r['severity']==severity and r['method']==method
                              and (name=='all' or r['scenario']==name)]
                    mean=lambda key:float(np.mean([float(r[key]) for r in selected]))
                    seed_rows.append(dict(seed=seed,severity=severity,scenario=name,method=method,tracking=mean('tracking'),
                        worst_family=max(np.mean([float(r['tracking']) for r in selected if r['family']==f]) for f in base.FAMILIES),
                        feasible=mean('feasible'),peak_steering_rate=max(float(r['peak_steering_rate']) for r in selected),
                        rho=max(float(r['max_small_signal_spectral_radius']) for r in audits if int(r['seed'])==seed and r['method']==method)))
    comparisons=[]; rng=np.random.default_rng(cfg['bootstrap_seed'])
    indices=rng.integers(0,len(cfg['training_seeds']),(cfg['bootstrap_repetitions'],len(cfg['training_seeds'])))
    for name in [*cfg['scenarios'],'all']:
        for method in METHODS:
            if method=='M4':continue
            select=lambda m:np.array([r['tracking'] for r in seed_rows if r['severity']=='moderate' and r['scenario']==name and r['method']==m])
            baseline=select(method); m4=select('M4'); delta=baseline-m4
            ci=np.quantile(delta[indices].mean(axis=1),[.025,.975])
            comparisons.append(dict(scenario=name,baseline=method,mean_delta=float(delta.mean()),ci_low=ci[0],ci_high=ci[1],
                reduction_pct=float(100*(baseline.mean()-m4.mean())/baseline.mean()),positive_seed_pairs=int(np.sum(delta>0))))
    aggregate=[]
    for severity in cfg['severities']:
        for method in METHODS:
            selected=[r for r in seed_rows if r['severity']==severity and r['scenario']=='all' and r['method']==method]
            aggregate.append(dict(severity=severity,method=method,mean=float(np.mean([r['tracking'] for r in selected])),
                std=float(np.std([r['tracking'] for r in selected],ddof=1)),worst_family_mean=float(np.mean([r['worst_family'] for r in selected])),
                min_feasible=min(r['feasible'] for r in selected),max_rho=max(r['rho'] for r in selected)))
    primary=[r for r in comparisons if r['scenario']=='all' and r['baseline'] in ('M2','M3','LTI_all','LTI_near')]
    m4=[r for r in aggregate if r['method']=='M4']
    conclusions=dict(protocol_sha256=hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
        recovery_gate=all(r['reduction_pct']>=cfg['minimum_mean_reduction_pct'] and r['ci_low']>0 for r in primary)
            and all(r['min_feasible']==1 and r['max_rho']<1 for r in m4),
        primary_comparisons=primary,replicates=len(cfg['training_seeds']),test_clients=30*len(cfg['training_seeds']))
    base.write_csv(out/'experiment_4g_seed_summary.csv',seed_rows)
    base.write_csv(out/'experiment_4g_comparisons.csv',comparisons)
    base.write_csv(out/'experiment_4g_summary.csv',aggregate)
    (out/'experiment_4g_conclusions.json').write_text(json.dumps(conclusions,indent=2)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(9,3.6),constrained_layout=True)
    moderate=[r for r in aggregate if r['severity']=='moderate']
    axes[0].bar(range(len(METHODS)),[r['mean'] for r in moderate],yerr=[r['std'] for r in moderate],capsize=3)
    axes[0].set(xticks=range(len(METHODS)),xticklabels=METHODS,ylabel='Yaw RMSE [deg/s]',title='Independent fleets: mean and seed SD')
    axes[0].tick_params(axis='x',rotation=55)
    for method in ('M2','M3','LTI_all','LTI_near'):
        vals=[next(r['reduction_pct'] for r in comparisons if r['scenario']==name and r['baseline']==method) for name in cfg['scenarios']]
        axes[1].plot(range(3),vals,'o-',label=method)
    axes[1].axhline(0,color='k',linestyle=':');axes[1].axhline(5,color='gray',linestyle='--')
    axes[1].set(xticks=range(3),xticklabels=['Original S','Shifted lane','Constant 22'],ylabel='M4 reduction [%]',title='Moderate tracking by scenario')
    axes[1].legend(fontsize=8);axes[1].grid(alpha=.2)
    fig.savefig(ROOT/'results/figures/experiment_4g_validation.pdf');plt.close(fig)
    print(json.dumps(conclusions,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--summarize-only',action='store_true');parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    if not args.summarize_only:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:list(pool.map(run_seed,json.loads(CONFIG.read_text())['training_seeds']))
    summarize()
