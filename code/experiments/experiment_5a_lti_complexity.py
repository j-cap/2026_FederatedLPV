"""5A: training-selected frozen-gain grids versus interpolated M4."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import argparse
import csv
import hashlib
import json
import numpy as np
import matplotlib.pyplot as plt
import experiment_4c_nonlinear_oracle_models as base
from experiment_4f_structured_identification import StructuredModel
from experiment_4g_independent_validation import simulate_fleet,scenario
from experiment_4d_nonlinear_controllers import redesign_controllers,stability_audit
from federated_lpv import sample_fleet,ScheduledController,design_lqi_gain

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/tables'
CONFIG=ROOT/'code/config/experiment_5a.json'

def candidates(n,speeds):
    if n==1:return [(f'single_{v}',np.array([float(v)])) for v in (16,20,24)]
    return [('endpoints',np.linspace(12,28,n)),
            ('cell_centers',12+(np.arange(n)+.5)*16/n),
            ('training_quantiles',np.quantile(speeds,(np.arange(n)+.5)/n))]

def controller(model,anchors):
    gains={};prefilters={}
    for family in base.FAMILIES:
        designs=[design_lqi_gain(*model.matrix(family,float(v)),base.DT,base.Q,base.R) for v in anchors]
        gains[family]=np.array([d[0] for d in designs]);prefilters[family]=np.array([d[1] for d in designs])
    return ScheduledController('frozen grid','family','nearest',anchors,gains,prefilters)

def run_seed(seed):
    cfg=json.loads(CONFIG.read_text());train=sample_fleet(seed);test=sample_fleet(seed+100)
    with (OUT/f'experiment_4g_seed{seed}_parameters.csv').open() as f:
        params={r['group']:np.array([float(r[k]) for k in ('cf_over_m','cr_over_m','m_over_iz')]) for r in csv.DictReader(f)}
    model=StructuredModel('family','lpv',params)
    controllers={'M4':redesign_controllers({'M4':model})['M4']}
    time=np.arange(0,base.DURATION+base.DT,base.DT)
    training=[(base.smooth_profile(time,p),base.reference_signal(time,m)) for p,m in zip(base.TRAIN_PROFILES,('lane','sine'))]
    selection=[];selected=[]
    for n in cfg['anchors_per_family']:
        options=[]
        for name,grid in candidates(n,np.concatenate([v for v,r in training])):
            c=controller(model,grid);metrics=[simulate_fleet(train,c,v,r)[2] for v,r in training]
            score=float(np.mean([m['tracking'].mean() for m in metrics]));feasible=all(np.all(m['feasible']) for m in metrics)
            selection.append(dict(seed=seed,n=n,layout=name,anchors=json.dumps(grid.tolist()),training_tracking=score,feasible=feasible))
            options.append((score if feasible else float('inf'),name,grid,c))
        score,name,grid,c=min(options,key=lambda o:o[0])
        if not np.isfinite(score):raise RuntimeError('no training-feasible grid')
        selected.append(dict(seed=seed,n=n,layout=name,anchors=json.dumps(grid.tolist())))
        controllers[f'LTI_{n}']=c
    audits=stability_audit(test,{'test':controllers});rows=[]
    for name in ('previous_s_curve','shifted_double_lane','constant_22'):
        v,ref=scenario(name,time)
        for severity,scale in base.SEVERITIES.items():
            for method,c in controllers.items():
                x,u,metrics=simulate_fleet(test,c,v,scale*ref)
                switches=0 if method=='M4' else int(np.count_nonzero(np.diff(np.argmin(abs(v[:-1,None]-c.speeds),axis=1))))
                for j,client in enumerate(test):
                    rows.append(dict(seed=seed,test_seed=seed+100,scenario=name,severity=severity,method=method,
                        client=client.client_id,family=client.family,switches=switches,**{k:float(val[j]) for k,val in metrics.items()}))
    for suffix,data in [('clients',rows),('selection',selection),('selected',selected),('stability',audits)]:
        base.write_csv(OUT/f'experiment_5a_seed{seed}_{suffix}.csv',data)
    print('completed',seed,flush=True)

def summarize():
    cfg=json.loads(CONFIG.read_text());seed_rows=[]
    methods=['M4']+[f'LTI_{n}' for n in cfg['anchors_per_family']]
    for seed in cfg['training_seeds']:
        with (OUT/f'experiment_5a_seed{seed}_clients.csv').open() as f:rows=list(csv.DictReader(f))
        with (OUT/f'experiment_5a_seed{seed}_stability.csv').open() as f:audit=list(csv.DictReader(f))
        for severity in base.SEVERITIES:
            for scenario_name in ('all','previous_s_curve','shifted_double_lane','constant_22'):
                for method in methods:
                    r=[r for r in rows if r['method']==method and r['severity']==severity and (scenario_name=='all' or r['scenario']==scenario_name)]
                    seed_rows.append(dict(seed=seed,severity=severity,scenario=scenario_name,method=method,
                        tracking=float(np.mean([float(x['tracking']) for x in r])),
                        effort=float(np.mean([float(x['steering_rms']) for x in r])),
                        peak_rate=max(float(x['peak_steering_rate']) for x in r),
                        feasible=min(float(x['feasible']) for x in r),
                        switches=float(np.mean([float(x['switches']) for x in r])),
                        rho=max(float(x['max_small_signal_spectral_radius']) for x in audit if x['method']==method)))
    summary=[];comparisons=[]
    rng=np.random.default_rng(cfg['bootstrap_seed']);idx=rng.integers(0,10,(cfg['bootstrap_repetitions'],10))
    for severity in base.SEVERITIES:
        for method in methods:
            r=[r for r in seed_rows if r['method']==method and r['severity']==severity and r['scenario']=='all']
            summary.append(dict(severity=severity,method=method,tracking=np.mean([x['tracking'] for x in r]),
                std=np.std([x['tracking'] for x in r],ddof=1),effort=np.mean([x['effort'] for x in r]),
                max_rate=max(x['peak_rate'] for x in r),feasible=min(x['feasible'] for x in r),rho=max(x['rho'] for x in r)))
    for name in ('all','previous_s_curve','shifted_double_lane','constant_22'):
        get=lambda m:np.array([r['tracking'] for r in seed_rows if r['method']==m and r['severity']=='moderate' and r['scenario']==name])
        m4=get('M4')
        for n in cfg['anchors_per_family']:
            values=get(f'LTI_{n}');gap=values-(1+cfg['matching_tolerance_fraction'])*m4
            lo,hi=np.quantile(gap[idx].mean(axis=1),[.025,.975])
            safe=all(r['feasible']==1 and r['rho']<1 for r in summary if r['method']==f'LTI_{n}')
            comparisons.append(dict(scenario=name,n=n,relative_gap_pct=100*(values.mean()/m4.mean()-1),
                tolerance_gap_low=lo,tolerance_gap_high=hi,matches=bool(hi<=0 and safe)))
    costs=[dict(method='M4',structural_families=3,gain_vectors=15,controller_scalars=65,
                compact_identified_scalars=9,explicit_frozen_matrix_scalars=0)]
    costs.extend(dict(method=f'LTI_{n}',structural_families=3,gain_vectors=3*n,controller_scalars=13*n,
                compact_identified_scalars=9,explicit_frozen_matrix_scalars=18*n) for n in cfg['anchors_per_family'])
    for suffix,data in [('summary',summary),('comparisons',comparisons),('seed_summary',seed_rows),('complexity',costs)]:
        base.write_csv(OUT/f'experiment_5a_{suffix}.csv',data)
    matching=[r['n'] for r in comparisons if r['scenario']=='all' and r['matches']]
    result=dict(protocol_sha256=hashlib.sha256(CONFIG.read_bytes()).hexdigest(),minimum_matching_anchors_per_family=min(matching) if matching else None,
        note='Finite candidate layouts and nearest switching only; no global optimality or certification claim.')
    (OUT/'experiment_5a_conclusions.json').write_text(json.dumps(result,indent=2)+'\n')
    fig,axes=plt.subplots(1,3,figsize=(10,3.5),constrained_layout=True)
    r=[r for r in summary if r['severity']=='moderate'];m4=r[0];x=np.array(cfg['anchors_per_family'])*3
    axes[0].errorbar(x,[z['tracking'] for z in r[1:]],yerr=[z['std'] for z in r[1:]],fmt='o-',capsize=3)
    axes[0].axhline(m4['tracking'],label='M4',color='k');axes[0].axhline(1.05*m4['tracking'],linestyle='--',color='gray',label='+5%')
    axes[0].set(xlabel='Frozen gain vectors',ylabel='Yaw RMSE [deg/s]');axes[0].legend(fontsize=8)
    axes[1].plot([13*n for n in cfg['anchors_per_family']],[z['tracking'] for z in r[1:]],'o-')
    axes[1].plot(65,m4['tracking'],'*',markersize=12,label='M4');axes[1].legend(fontsize=8)
    axes[1].set(xlabel='Controller scalars (gains + grid)',ylabel='Yaw RMSE [deg/s]')
    axes[2].plot(x,[z['max_rate'] for z in r[1:]],'o-');axes[2].axhline(m4['max_rate'],color='k',label='M4');axes[2].legend(fontsize=8)
    axes[2].set(xlabel='Frozen gain vectors',ylabel='Worst steering rate [deg/s]')
    for ax in axes:ax.grid(alpha=.2)
    fig.savefig(ROOT/'results/figures/experiment_5a_complexity.pdf');plt.close(fig)
    print(json.dumps(result),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--summarize-only',action='store_true');args=parser.parse_args()
    if not args.summarize_only:
        with ProcessPoolExecutor(max_workers=4) as pool:list(pool.map(run_seed,json.loads(CONFIG.read_text())['training_seeds']))
    summarize()
