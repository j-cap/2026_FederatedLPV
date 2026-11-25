r"""Limited-speed physical OE: local, global and personalized federated models.

python code\experiments\paper_federated_relaxation_coverage.py preflight
python code\experiments\paper_federated_relaxation_coverage.py smoke
python code\experiments\paper_federated_relaxation_coverage.py run --workers 3
"""
from __future__ import annotations
import os
for variable in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[variable]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import copy
import importlib.metadata
import io
import json
import multiprocessing
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'code/src'),str(ROOT/'code/experiments')]
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import paper_tire_relaxation_gate as storage
from paper_tire_relaxation_pilot import canonical_hash
from federated_lpv import federated_relaxation as fl
from federated_lpv.tire_relaxation_pilot import batch_predict

PREFIX='paper_federated_relaxation_coverage'
CONFIG=ROOT/'code/config'/f'{PREFIX}.json'
OUT=ROOT/'results/tables'
RAW=ROOT/'results/data'/PREFIX
REPORT=ROOT/'report/federated_relaxation_coverage.html'


def settings(cfg):
    return dict(cfg['fit'],**{k:cfg[k] for k in ['sample_time','integration_max_step','yaw_noise_std_deg_s']})


def identity(cfg):
    names=[CONFIG,Path(__file__),Path(fl.__file__),ROOT/'code/src/federated_lpv/tire_relaxation_pilot.py',
        ROOT/'code/src/federated_lpv/tire_relaxation.py',ROOT/'code/src/federated_lpv/physical_ct_oe.py',
        ROOT/'code/src/federated_lpv/structured_oe.py',Path(storage.__file__),
        ROOT/'code/experiments/paper_tire_relaxation_pilot.py']
    return dict(files={p.relative_to(ROOT).as_posix():canonical_hash(p) for p in names},source=cfg['source_sha256'],
        runtime=dict(python=f'{sys.version_info.major}.{sys.version_info.minor}',packages={n:importlib.metadata.version(n)
        for n in ['numpy','scipy','pandas','threadpoolctl','scikit-learn']}))


def preflight():
    cfg=json.loads(CONFIG.read_text(encoding='utf-8'))
    for name,sha in cfg['source_sha256'].items():
        if canonical_hash(ROOT/name)!=sha:raise ValueError(f'Frozen population source changed: {name}')
    assert len(cfg['fleets'])==3 and len(cfg['methods'])==6
    for fleet in cfg['fleets']:
        assert len(fleet['clients'])==9 and len({c['client'] for c in fleet['clients']})==9
        source=storage.read_checkpoint(ROOT/f'results/tables/paper_oe_lpv_integration_seed{fleet["seed"]}.json')
        for c in fleet['clients']:
            original=next(v for v in source['clients'] if v['client']==c['source_client'])
            physical=original['parameters']
            expected=[physical['front_stiffness']/physical['mass'],physical['rear_stiffness']/physical['mass'],
                physical['mass']/physical['yaw_inertia'],.9,.4,.7]
            np.testing.assert_allclose(c['physical_parameters'],expected,rtol=1e-14,atol=0)
            assert c['geometry_m']==[physical['front_length'],physical['rear_length']]==[1.2,1.6]

        for family in ['nominal','heavy','handling']:
            assert sorted(c['training_speed'] for c in fleet['clients'] if c['family']==family)==cfg['fleet_training_speeds']
    bounds=np.array(cfg['fit']['bounds'])
    assert all(np.all((np.array(s)>bounds[:,0])&(np.array(s)<bounds[:,1])) for s in cfg['starts'])
    assert cfg['integration_max_step'] <= min(bounds[4:,0])/max(cfg['evaluation_speeds'])/2
    assert cfg['sharing']['weights'][3]==0 and cfg['sharing']['lambda_value']==1.
    return cfg


def fleet_for(cfg,seed):return next(f for f in cfg['fleets'] if f['seed']==seed)
def path(seed):return OUT/f'{PREFIX}_seed{seed}.json'


def validate(p,cfg,seed):
    if p['seed']!=seed or p['provenance']!=identity(cfg):raise ValueError('Protocol/source/runtime changed; checkpoint preserved')
    if p['status'] not in ['running','execution_complete']:raise ValueError('Invalid status')
    if p['status']=='execution_complete':
        ids={c['client'] for c in fleet_for(cfg,seed)['clients']}
        assert set(p['data'])==ids and set(p['local'])==ids and set(p['nominal_prior'])==ids
        assert set(p['selected'])=={'Global','FedPersonal','FedGroups'}
        expected={f'{cid}_v{v}_{probe}' for cid in ids for v in cfg['evaluation_speeds'] for probe in ['slow','broadband']}
        assert set(p['evaluation'])==expected


def save(p,stage):
    storage.atomic_json(path(p['seed']),p)
    print(json.dumps(dict(seed=p['seed'],saved=stage)),flush=True)


def generate(c,seed,probe,speed,cfg):
    t,u=storage.steering(cfg,probe)
    x=batch_predict(c['physical_parameters'],[u],[speed],c['geometry_m'],cfg['sample_time'],cfg['integration_max_step'])[0,0]
    role={'training':0,'slow':1,'broadband':2}[probe]
    index=int(c['client'].split('_')[-1])
    eps=np.random.default_rng(np.random.SeedSequence([cfg['noise_namespace'],seed,index,role,int(speed)])).normal(
        0,np.deg2rad(cfg['yaw_noise_std_deg_s']),len(t))
    arrays=dict(time=t,steering=u,state=x,noise=eps,measured_yaw=x[:,1]+eps)
    metrics=dict(beta_peak_deg=float(np.rad2deg(max(abs(x[:,0])))),ay_peak_mps2=float(max(abs(x[:,2]+x[:,3]))),
        steering_rate_peak_deg_s=float(np.rad2deg(max(abs(np.diff(np.r_[0,u,0]))))/cfg['sample_time']))
    return arrays,dict(trace_sha256=storage.trace_hash(arrays),metrics=metrics,
        operating_checks_passed=all(metrics[k]<=bound for k,bound in cfg['operating_bounds'].items()),
        preview={k:v[::10].tolist() for k,v in arrays.items()})


def acquisition(p,c,cfg):
    name=c['client'];v=c['training_speed'];target=RAW/str(p['seed'])/f'{name}_training.npz'
    old=p['data'].get(name)
    if old is not None and target.exists():
        with np.load(target,allow_pickle=False) as loaded:arrays=dict(loaded)
    else:arrays,record=generate(c,p['seed'],'training',v,cfg)
    if old is not None:
        if storage.trace_hash(arrays)!=old['trace_sha256']:raise ValueError('Exact training hash mismatch; restore on original platform')
    else:
        p['data'][name]=record
        storage.save_arrays(target,arrays);save(p,f'acquired_{name}')
    if not p['data'][name]['operating_checks_passed']:raise ValueError('Training operating check failed; case preserved, no silent replacement')
    if not target.exists():storage.save_arrays(target,arrays)
    return dict(speed=v,applied_input=arrays['steering'],measured_yaw=arrays['measured_yaw'])


def choose_sharing(attempts):
    valid=[]
    for key,state in attempts.items():
        if state['status'] in ['converged','budget_exhausted']:
            loss=state['objective'] if 'objective' in state else state['history'][-1]['objective']
            valid.append((loss,key))
    if not valid:return dict(available=False,reason='No available sharing attempt')
    objective,key=min(valid);state=attempts[key]
    return dict(available=True,selected_attempt=key,optimization_status=state['status'],
        log_parameters=state['log_parameters'],objective=float(objective),
        assignments=state.get('assignments'),centers=state.get('centers'))


def fitted_parameters(p,c,index,cfg):
    available={'Exact':c['physical_parameters']}
    local=p['local'][c['client']]['selected'];prior=p['nominal_prior'][c['client']]
    if local['available']:available['LocalOE']=local['parameters']
    if prior['available'] and prior['optimizer_converged']:available['NominalPrior']=prior['parameters']
    for method in ['Global','FedPersonal','FedGroups']:
        entry=p['selected'][method]
        if entry['available']:
            logs=entry['log_parameters'] if method=='Global' else entry['log_parameters'][index]
            available[method]=np.exp(logs).tolist()
    return available


def evaluation_case(p,c,index,v,probe,cfg):
    # Test records and generating parameters are accessed only after all fits are frozen.
    arrays,plant=generate(c,p['seed'],probe,v,cfg)
    pars=fitted_parameters(p,c,index,cfg);names=list(pars)
    predictions=batch_predict([pars[n] for n in names],[arrays['steering']],[v],c['geometry_m'],
        cfg['sample_time'],cfg['integration_max_step'])[:,0]
    rows={};rms=lambda x:float(np.sqrt(np.mean(x*x)))
    for j,name in enumerate(names):
        prediction=predictions[j];arrays[f'predicted_{name}']=prediction
        rows[name]=dict(status='complete',yaw_rmse_deg_s=float(np.rad2deg(rms(prediction[:,1]-arrays['state'][:,1]))),
            measured_yaw_rmse_deg_s=float(np.rad2deg(rms(prediction[:,1]-arrays['measured_yaw']))),
            beta_rmse_deg=float(np.rad2deg(rms(prediction[:,0]-arrays['state'][:,0]))))
    for name in cfg['methods']:
        if name not in rows:rows[name]=dict(status='model_unavailable',yaw_rmse_deg_s=None,measured_yaw_rmse_deg_s=None,beta_rmse_deg=None)
    entry=dict(client=c['client'],speed=v,probe=probe,region=fl.coverage_region(v,c['training_speed'],cfg['fleet_training_speeds']),
        methods=rows,plant_metrics=plant['metrics'],operating_checks_passed=plant['operating_checks_passed'],trace_sha256=storage.trace_hash(arrays))
    if c['client']=='vehicle_00' and v==21:entry['preview']={k:val[::10].tolist() for k,val in arrays.items()}
    return entry,arrays


def worker(seed):
    cfg=preflight();fleet=fleet_for(cfg,seed);weights=np.array(cfg['sharing']['weights']);sh=dict(cfg['sharing'],bounds=cfg['fit']['bounds'])
    if path(seed).exists():p=storage.read_checkpoint(path(seed));validate(p,cfg,seed)
    else:p=dict(seed=seed,status='running',provenance=identity(cfg),data={},local={},nominal_prior={},global_attempts={},
        sharing_attempts={},selected={},evaluation={},implementation_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    if p['status']=='execution_complete':return dict(seed=seed,status='already_complete')
    with threadpool_limits(limits=1):
        endpoints=[fl.Client([acquisition(p,c,cfg)],c['geometry_m'],settings(cfg)) for c in fleet['clients']]
        for i,c in enumerate(fleet['clients']):
            entry=p['local'].setdefault(c['client'],dict(attempts=[]))
            for j in range(len(entry['attempts']),len(cfg['starts'])):
                entry['attempts'].append(endpoints[i].fit(np.log(cfg['starts'][j])))
                save(p,f'local_{c["client"]}_start{j}')
            entry['selected']=fl.select_local(entry['attempts'])
            if c['client'] not in p['nominal_prior']:
                start=entry['selected'].get('log_parameters',np.log(cfg['nominal_parameters']).tolist())
                p['nominal_prior'][c['client']]=endpoints[i].fit(start,np.log(cfg['nominal_parameters']),weights)
                save(p,f'nominal_prior_{c["client"]}')
        public_cache=fl.PublicPredictionCache(endpoints[0].u[0],cfg['fleet_training_speeds'],fleet['clients'][0]['geometry_m'],settings(cfg))
        for endpoint in endpoints:
            endpoint.public_predictor=public_cache;endpoint.cache={}
        for j,start in enumerate(cfg['starts']):
            key=str(j);state=p['global_attempts'].setdefault(key,fl.initialize_global(np.log(start).tolist()))
            while state['status']=='running':
                state=fl.global_step(endpoints,state,sh);p['global_attempts'][key]=state
                save(p,f'global_start{j}_step{state["iteration"]}')
        p['selected']['Global']=choose_sharing(p['global_attempts']);save(p,'global_selected')
        for endpoint in endpoints:
            endpoint.public_predictor=None;endpoint.cache={}
        local_ready=all(p['local'][c['client']]['selected']['available'] for c in fleet['clients'])
        for method,groups,seeds in [('FedPersonal',1,[0]),('FedGroups',sh['groups'],sh['group_initialization_seeds'])]:
            attempts=p['sharing_attempts'].setdefault(method,{})
            if not local_ready:
                p['selected'][method]=dict(available=False,reason='Local initialization unavailable for at least one client');save(p,method+'_unavailable');continue
            local_logs=[p['local'][c['client']]['selected']['log_parameters'] for c in fleet['clients']]
            for init_seed in seeds:
                key=str(init_seed);state=attempts.setdefault(key,fl.initialize_personal(local_logs,groups,init_seed,weights))
                while state['status']=='running':
                    for i,c in enumerate(fleet['clients']):
                        if str(i) in state['pending']:continue
                        eta=state['centers'][state['assignments'][i]]
                        fit=endpoints[i].fit(state['log_parameters'][i],eta,weights,max_nfev=sh['local_round_max_nfev'])
                        state['pending'][str(i)]=fit;save(p,f'{method}_init{key}_round{state["round"]+1}_{c["client"]}')
                    if not all(f['available'] for f in state['pending'].values()):
                        state['status']='local_update_failed';save(p,method+'_failed');break
                    fits=[state['pending'][str(i)] for i in range(len(endpoints))]
                    messages=[fl.client_round_message(f) for f in fits]
                    updated=fl.finish_personal_round(state,messages,weights,sh,all(f['optimizer_converged'] for f in fits))
                    # Keep local optimizer diagnostics after clearing the resumable partial round.
                    updated['history'][-1]['local_updates']=[{k:f[k] for k in ['optimizer_converged','nfev','optimality','near_bound']} for f in fits]
                    state=updated;attempts[key]=state;save(p,f'{method}_init{key}_round{state["round"]}')
            p['selected'][method]=choose_sharing(attempts);save(p,method+'_selected')
        for i,c in enumerate(fleet['clients']):
            for v in cfg['evaluation_speeds']:
                for probe in ['slow','broadband']:
                    key=f'{c["client"]}_v{v}_{probe}'
                    if key in p['evaluation']:continue
                    result,arrays=evaluation_case(p,c,i,v,probe,cfg)
                    storage.save_arrays(RAW/str(seed)/f'{key}.npz',arrays)
                    p['evaluation'][key]=result;save(p,'evaluation_'+key)
        p['status']='execution_complete';save(p,'execution_complete')
    return dict(seed=seed,status=p['status'])


def restore(cfg,seeds):
    for seed in seeds:
        p=storage.read_checkpoint(path(seed));validate(p,cfg,seed)
        count=0
        for i,c in enumerate(fleet_for(cfg,seed)['clients']):
            arrays,_=generate(c,seed,'training',c['training_speed'],cfg)
            pairs=[(f'{c["client"]}_training',arrays,p['data'][c['client']]['trace_sha256'])]
            for v in cfg['evaluation_speeds']:
                for probe in ['slow','broadband']:
                    key=f'{c["client"]}_v{v}_{probe}'
                    if key not in p['evaluation']:continue
                    _,a=evaluation_case(p,c,i,v,probe,cfg);pairs.append((key,a,p['evaluation'][key]['trace_sha256']))
            for key,a,sha in pairs:
                if storage.trace_hash(a)!=sha:raise ValueError(f'Exact restore mismatch: {seed}/{key}; use original numerical platform')
                target=RAW/str(seed)/f'{key}.npz'
                if target.exists():
                    with np.load(target,allow_pickle=False) as loaded:
                        if storage.trace_hash(dict(loaded))!=sha:raise ValueError('Existing raw array hash mismatch; original preserved')
                else:storage.save_arrays(target,a)
                count+=1
        print(json.dumps(dict(seed=seed,status='exact_arrays_verified',traces=count)))


def analyze(cfg):
    checkpoints=[];rows=[];parameters=[];optimization=[];groups=[]
    for fleet in cfg['fleets']:
        seed=fleet['seed'];p=storage.read_checkpoint(path(seed));validate(p,cfg,seed)
        if p['status']!='execution_complete':raise ValueError('Complete all three fleets before analysis')
        checkpoints.append(p)
        for key,case in p['evaluation'].items():
            for method,result in case['methods'].items():
                rows.append(dict(seed=seed,client=case['client'],speed=case['speed'],probe=case['probe'],region=case['region'],method=method,
                    operating_checks_passed=case['operating_checks_passed'],**result))
        for i,c in enumerate(fleet['clients']):
            for method,par in fitted_parameters(p,c,i,cfg).items():
                parameters.append(dict(seed=seed,client=c['client'],method=method,**{f'parameter{j}_error_pct':100*(value/c['physical_parameters'][j]-1) for j,value in enumerate(par)}))
            for method in ['FedPersonal','FedGroups']:
                selected=p['selected'][method]
                groups.append(dict(seed=seed,client=c['client'],family=c['family'],training_speed=c['training_speed'],method=method,
                    group=selected['assignments'][i] if selected['available'] else None))
        for name,attempts in [('Global',p['global_attempts']),*list(p['sharing_attempts'].items())]:
            for key,state in attempts.items():
                optimization.append(dict(seed=seed,method=name,attempt=key,status=state['status'],
                    selected=p['selected'][name].get('selected_attempt')==key,
                    steps=state.get('iteration',state.get('round')),objective=state.get('objective',state['history'][-1]['objective'] if state['history'] else None),
                    logical_float64_bytes=8*state['communication_float64']))
        for c in fleet['clients']:
            selected=p['local'][c['client']]['selected']
            optimization.append(dict(seed=seed,method='LocalOE',attempt=c['client'],status='converged' if selected['available'] else 'failed',selected=True,
                steps=selected.get('nfev'),objective=selected.get('objective'),logical_float64_bytes=0))
            a=p['nominal_prior'][c['client']]
            optimization.append(dict(seed=seed,method='NominalPrior',attempt=c['client'],status='converged' if a['available'] and a['optimizer_converged'] else 'failed',selected=True,
                steps=a.get('nfev'),objective=a.get('objective'),logical_float64_bytes=0))
    cases=pd.DataFrame(rows);summary=[];paired=[]
    for (seed,method,region),g in cases.groupby(['seed','method','region']):
        complete=int((g.status=='complete').sum())
        summary.append(dict(seed=seed,method=method,region=region,complete=complete,expected=len(g),
            mean_yaw_rmse_deg_s=float(g.yaw_rmse_deg_s.mean()) if complete==len(g) else None,
            finite_case_mean_yaw_rmse_deg_s=float(g.yaw_rmse_deg_s.mean()) if complete else None))
    for (seed,region),g in cases.groupby(['seed','region']):
        pivot=g.pivot(index=['client','speed','probe'],columns='method',values='yaw_rmse_deg_s')
        for method in ['Global','FedPersonal','FedGroups']:
            for baseline in ['LocalOE','NominalPrior']:
                valid=pivot[[method,baseline]].dropna();expected=len(pivot)
                paired.append(dict(seed=seed,region=region,method=method,baseline=baseline,paired=len(valid),expected=expected,
                    mean_difference_deg_s=float((valid[method]-valid[baseline]).mean()) if len(valid)==expected else None,
                    improved_cases=int((valid[method]<valid[baseline]).sum())))
    tables=dict(cases=cases,fleets=pd.DataFrame(summary),paired=pd.DataFrame(paired),parameters=pd.DataFrame(parameters),optimization=pd.DataFrame(optimization),groups=pd.DataFrame(groups))
    for name,table in tables.items():table.to_csv(OUT/f'{PREFIX}_{name}.csv',index=False)
    render(tables,checkpoints,cfg)
    storage.atomic_json(OUT/f'{PREFIX}_analysis.json',dict(status='analysis_complete',provenance=identity(cfg),
        checkpoints={path(f['seed']).name:canonical_hash(path(f['seed'])) for f in cfg['fleets']},cases=len(cases),scientific_pass=None))
    print(REPORT)


def render(tables,checkpoints,cfg):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({'svg.hashsalt':PREFIX,'axes.spines.top':False,'axes.spines.right':False})
    def svg(fig):
        s=io.StringIO();fig.savefig(s,format='svg',metadata={'Date':None});plt.close(fig);v=s.getvalue();return v[v.index('<svg'):]
    def table(df):return '<div class="scroll">'+df.to_html(index=False,border=0,float_format=lambda x:f'{x:.5g}')+'</div>'
    regions=['local_observed','peer_observed','fleet_interpolation','fleet_extrapolation']
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for method,g in tables['fleets'].groupby('method'):
        if method=='Exact':continue
        mean=g.groupby('region').mean_yaw_rmse_deg_s.mean();count=g.groupby('region').mean_yaw_rmse_deg_s.count()
        vals=[mean.get(r,np.nan) if count.get(r,0)==len(cfg['fleets']) else np.nan for r in regions]
        axes[0].plot(range(4),vals,marker='o',label=method)
    axes[0].set(xticks=range(4),xticklabels=['Local','Peers','Between','Outside fleet'],ylabel='Mean yaw RMSE [deg/s]',title='Prediction by speed coverage')
    for method,g in tables['cases'].groupby('method'):
        if method=='Exact':continue
        avg=g.groupby('speed').yaw_rmse_deg_s.mean();complete=g.groupby('speed').status.apply(lambda s:(s=='complete').all());avg[~complete]=np.nan
        axes[1].plot(avg.index,avg,marker='o',label=method)
    axes[1].set(xlabel='Speed [m/s]',ylabel='Mean yaw RMSE [deg/s]',title='Prediction across speeds')
    for ax in axes:ax.grid(alpha=.2)
    axes[0].legend(fontsize=8);plot=svg(fig)
    grouped=tables['groups'].groupby(['seed','method','group','training_speed'],dropna=False).size().reset_index(name='clients')
    families=tables['groups'].groupby(['seed','method','group','family'],dropna=False).size().reset_index(name='clients')
    html=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Federated relaxation models under limited local speed coverage</title>
<style>body{{font:16px/1.6 system-ui;background:#edf2f5;color:#173348}}main{{max-width:1150px;margin:auto;padding:26px}}section{{padding:24px;margin:22px 0;background:white;border-radius:10px}}h1,h2{{line-height:1.25}}svg{{width:100%;height:auto}}.scroll{{overflow:auto}}table{{border-collapse:collapse;font-size:13px;white-space:nowrap}}th,td{{padding:8px;border-bottom:1px solid #ddd}}</style><main>
<h1>Can peers improve prediction outside a vehicle's observed speed?</h1><p>Exploratory three-fleet pilot. Nine vehicles per fleet, one 12 s record per vehicle, identical four-state nonlinear model class. No control experiment or automatic scientific pass.</p>
<section><h2>Methods and access to data</h2><p>LocalOE fits each vehicle independently. NominalPrior applies the same fixed shrinkage to a common engineering nominal value. Global learns one common parameter vector from federated Gauss–Newton messages. FedPersonal uses one learned center and personalized parameters. FedGroups uses three learned centers and personalized parameters. Exact uses the generating parameters only as an evaluation reference.</p><p>All local fits use the same three predefined starts, independent of previous identified models and vehicle truth. Only the five dynamic/relaxation parameters enter the sharing penalty; friction remains locally free. The simulated population has common generating relaxation lengths, which is favorable for transfer and must be stated when interpreting results.</p></section>
<section><h2>Completeness and convergence first</h2>{table(tables['optimization'])}<p>Budget-exhausted sharing estimates are evaluated and explicitly flagged; they are not claimed to have converged. Unavailable models have null full-condition means. Logical payload counts include attempted starts/rounds, exclude transport overhead and integer/status fields, and do not establish privacy. This is an emulation of client-local computation and server messages on one machine.</p></section>
<section><h2>Primary result: peer-observed speeds</h2>{plot}{table(tables['fleets'])}<p>Local: fresh trajectory at the vehicle's training speed. Peers: other vehicles observed that exact speed. Between: 17 or 25 m/s, inside the fleet range but not sampled in training. Outside fleet: 9 or 33 m/s. These categories must not be combined into a claim of universal extrapolation. Plots show means only when every expected case is available.</p>{table(tables['paired'])}<p>Negative paired differences favor sharing. Compare against both LocalOE and NominalPrior before crediting fleet information. Compare FedGroups against FedPersonal to judge whether grouping adds value beyond a common learned prior. Three reused pilot populations do not support a confirmatory significance claim.</p></section>
<section><h2>What did the groups capture?</h2>{table(grouped)}{table(families)}<p>Family labels are revealed only for this post-fit diagnostic. Groups need not recover those labels. Check whether groups mostly separate the training speeds, which could indicate acquisition/identification effects rather than compatible dynamics. K=3 and regularization strength were fixed in advance; two group initializations are compared by the regularized training objective only.</p></section>
<section><h2>Interpretation</h2><p>A useful outcome is lower peer-speed error without unacceptable loss at the observed speed, consistent across fleets and beyond the fixed nominal-prior baseline. If Global is already competitive, grouping may be unnecessary. If local models already generalize well, these data do not demonstrate a need for sharing. Poor local estimates alone do not show that peers supply the missing information.</p><p>Friction errors, active bounds, optimizer status and local sensitivity are saved in checkpoints. Parameters and data are never repaired using evaluation trajectories. Test frequencies/phases and sensor-noise streams are separate from training. The four-state controller issue remains outside this experiment.</p><p><a href="../docs/federated_relaxation_coverage.html">Frozen protocol and commands</a></p></section></main></html>'''
    REPORT.write_text('\n'.join(line.rstrip() for line in html.splitlines())+'\n',encoding='utf-8',newline='\n')


def smoke():
    """Synthetic endpoint/message checks; no pilot acquisition or fitting."""
    cfg=preflight();s=settings(cfg);s['max_nfev']=25
    p=np.array([50.,55.,.6,.9,.4,.7]);t=np.arange(80)*.01;u=.006*np.sin(2*np.pi*.8*t)
    clients=[]
    for v in [13,21,29]:
        x=batch_predict(p,[u],[v],[1.2,1.6],.01,.001)[0,0]
        clients.append(fl.Client([dict(speed=v,applied_input=u,measured_yaw=x[:,1])],[1.2,1.6],s))
    q=np.log(p*1.01);messages=[c.information_message(q) for c in clients]
    loss,g,h=fl.aggregate(messages)
    residual,jac=zip(*[c.residual(q,True) for c in clients]);r=np.concatenate(residual);j=np.vstack(jac)
    np.testing.assert_allclose(loss,r@r);np.testing.assert_allclose(g,j.T@r);np.testing.assert_allclose(h,j.T@j)
    gs=fl.initialize_global(q.tolist());gs=fl.global_step(clients,gs,dict(cfg['sharing'],bounds=cfg['fit']['bounds']))
    assert gs['objective']<loss
    local=clients[0].fit(q,np.log(cfg['nominal_parameters']),np.array(cfg['sharing']['weights']))
    assert local['available'] and local['objective']<=local['initial_objective']
    with tempfile.TemporaryDirectory() as tmp:
        dest=Path(tmp)/'utf8.json';storage.atomic_json(dest,dict(message='β → μ; Prüfung',global_state=gs));assert storage.read_checkpoint(dest)['global_state']==gs
    return dict(status='smoke_passed',server_aggregation_matches_centralized=True,
        initial_objective=loss,after_global_step=gs['objective'],scientific_cases_run=0)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['preflight','smoke','run','status','analyze','restore'])
    parser.add_argument('--workers',type=int,default=3);parser.add_argument('--seed',type=int,choices=[1301,1302,1303])
    args=parser.parse_args();cfg=preflight();seeds=[f['seed'] for f in cfg['fleets'] if args.seed is None or f['seed']==args.seed]
    if args.command=='preflight':
        print(json.dumps(dict(status='preflight_passed',fleets=3,clients_per_fleet=9,local_records=27,local_multistart_fits=81,
            evaluation_cases=2268,controllers=0,groups=3,personalized_mu_sharing=False,raw_arrays_required=False)));return
    if args.command=='smoke':
        with ProcessPoolExecutor(max_workers=1,mp_context=multiprocessing.get_context('spawn')) as pool:print(json.dumps(pool.submit(smoke).result()))
        return
    if args.command=='status':
        for seed in seeds:
            if path(seed).exists():
                p=storage.read_checkpoint(path(seed));validate(p,cfg,seed)
                print(json.dumps(dict(seed=seed,status=p['status'],acquired=len(p['data']),local=len(p['local']),evaluation=len(p['evaluation']))))
            else:print(json.dumps(dict(seed=seed,status='missing')))
        return
    if not 1<=args.workers<=3:raise ValueError('Use 1–3 workers')
    RAW.mkdir(parents=True,exist_ok=True);lock=RAW/'run.lock'
    try:fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError:raise RuntimeError(f'Run lock exists: {lock}; remove only after confirming no active coverage process')
    os.write(fd,str(os.getpid()).encode());os.close(fd)
    try:
        if args.command=='analyze':analyze(cfg);return
        if args.command=='restore':restore(cfg,seeds);return
        with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            for result in as_completed([pool.submit(worker,seed) for seed in seeds]):print(json.dumps(result.result()),flush=True)
    finally:lock.unlink()
    print('Execution complete. Commit/push seed checkpoints before analyze; see docs/federated_relaxation_coverage.html.')


if __name__=='__main__':main()
