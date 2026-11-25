"""Continue only FedPersonal, preserving the original limited-speed pilot."""
from __future__ import annotations
import os
for variable in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[variable]='1'
import argparse
import copy
from concurrent.futures import ProcessPoolExecutor,as_completed
import io
import json
import multiprocessing
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import paper_federated_relaxation_coverage as base

ROOT=base.ROOT
PREFIX='paper_federated_relaxation_continue'
CONFIG=ROOT/'code/config'/f'{PREFIX}.json'
OUT=ROOT/'results/tables'
RAW=ROOT/'results/data'/PREFIX
REPORT=ROOT/'report/federated_relaxation_continue.html'


def path(seed):return OUT/f'{PREFIX}_seed{seed}.json'


def identity(cfg,protocol):
    return dict(base=base.identity(cfg),sources=protocol['source_checkpoints'],
        files={p.relative_to(ROOT).as_posix():base.canonical_hash(p) for p in [CONFIG,Path(__file__)]})


def source_state(source):
    chosen=source['selected']['FedPersonal']
    return source['sharing_attempts']['FedPersonal'][chosen['selected_attempt']]


def load_source(seed,cfg,protocol,execution=False):
    target=base.path(seed)
    if base.canonical_hash(target)!=protocol['source_checkpoints'][target.relative_to(ROOT).as_posix()]:
        raise ValueError('Original checkpoint changed; continuation stopped')
    p=base.storage.read_checkpoint(target)
    current=base.identity(cfg)
    if p['provenance']['files']!=current['files'] or p['provenance']['source']!=current['source']:
        raise ValueError('Original scientific source/configuration changed')
    if execution:
        base.validate(p,cfg,seed)
    if p['status']!='execution_complete':raise ValueError('Original execution incomplete')
    state=source_state(p)
    if state['status'] not in ['converged','budget_exhausted'] or state['pending']:
        raise ValueError('Expected a completed original FedPersonal attempt')
    return p


def preflight(execution=False):
    cfg=base.preflight();protocol=json.loads(CONFIG.read_text(encoding='utf-8'))
    assert protocol['method']=='FedPersonal' and protocol['total_round_cap']==30
    assert protocol['seeds']==[f['seed'] for f in cfg['fleets']]
    for seed in protocol['seeds']:load_source(seed,cfg,protocol,execution)
    return cfg,protocol


def initialize(source,cfg,protocol):
    state=copy.deepcopy(source_state(source))
    if state['status']=='budget_exhausted':state['status']='running'
    return dict(seed=source['seed'],status='running',provenance=identity(cfg,protocol),
        source_round=state['round'],source_communication_float64=state['communication_float64'],
        optimization=state,evaluation={})


def validate(p,source,cfg,protocol,execution=True):
    expected=identity(cfg,protocol);actual=copy.deepcopy(p['provenance'])
    if not execution:actual['base']['runtime']=expected['base']['runtime']
    if actual!=expected:raise ValueError('Continuation provenance changed; checkpoint preserved')
    old=source_state(source);state=p['optimization']
    assert p['seed']==source['seed'] and p['source_round']==old['round']
    assert state['history'][:len(old['history'])]==old['history']
    assert old['round']<=state['round']<=protocol['total_round_cap']
    assert len(state['history'])==state['round']
    assert set(p['evaluation']).issubset(source['evaluation'])
    if p['status']=='execution_complete':
        assert set(p['evaluation'])==set(source['evaluation'])
        assert state['status'] in ['converged','budget_exhausted'] and not state['pending']
    if old['status']=='converged':assert state==old


def save(p,stage):
    base.storage.atomic_json(path(p['seed']),p)
    print(json.dumps(dict(seed=p['seed'],saved=stage,total_round=p['optimization']['round'])),flush=True)


def training_record(source,c,cfg):
    """Read or exactly reconstruct original data without writing source paths."""
    file=base.RAW/str(source['seed'])/f'{c["client"]}_training.npz'
    if file.exists():
        with np.load(file,allow_pickle=False) as loaded:arrays=dict(loaded)
    else:arrays,_=base.generate(c,source['seed'],'training',c['training_speed'],cfg)
    if base.storage.trace_hash(arrays)!=source['data'][c['client']]['trace_sha256']:
        raise ValueError('Original training trace hash mismatch; use the original machine or original raw arrays')
    return dict(speed=c['training_speed'],applied_input=arrays['steering'],measured_yaw=arrays['measured_yaw'])


def continue_rounds(p,endpoints,cfg,protocol,checkpoint):
    weights=np.array(cfg['sharing']['weights'])
    settings=dict(cfg['sharing'],personal_max_rounds=protocol['total_round_cap'])
    state=p['optimization']
    while state['status']=='running':
        if state['round']>=protocol['total_round_cap']:raise ValueError('Invalid running round budget')
        for i,client in enumerate(endpoints):
            if str(i) in state['pending']:continue
            eta=state['centers'][state['assignments'][i]]
            state['pending'][str(i)]=client.fit(state['log_parameters'][i],eta,weights,
                max_nfev=settings['local_round_max_nfev'])
            checkpoint(p,f'round{state["round"]+1}_client{i}')
        fits=[state['pending'][str(i)] for i in range(len(endpoints))]
        if not all(f['available'] for f in fits):
            state['status']='local_update_failed';checkpoint(p,'local_update_failed')
            raise RuntimeError('Local update failed; checkpoint preserved, no fallback model')
        updated=base.fl.finish_personal_round(state,[base.fl.client_round_message(f) for f in fits],
            weights,settings,all(f['optimizer_converged'] for f in fits))
        updated['history'][-1]['local_updates']=[{k:f[k] for k in
            ['optimizer_converged','nfev','optimality','near_bound']} for f in fits]
        p['optimization']=updated;state=updated;checkpoint(p,f'round{state["round"]}')


def evaluate(p,source,c,index,key,cfg):
    old=source['evaluation'][key];target=base.RAW/str(p['seed'])/f'{key}.npz'
    if target.exists():
        with np.load(target,allow_pickle=False) as loaded:original=dict(loaded)
    else:
        # Recover the existing test trace, including old model predictions, to
        # verify its original hash before computing the continued prediction.
        _,original=base.evaluation_case(source,c,index,old['speed'],old['probe'],cfg)
    if base.storage.trace_hash(original)!=old['trace_sha256']:
        raise ValueError('Original evaluation trace hash mismatch; use original machine or original raw arrays')
    arrays={k:original[k] for k in ['time','steering','state','noise','measured_yaw']}
    theta=np.exp(p['optimization']['log_parameters'][index])
    predicted=base.batch_predict(theta,[arrays['steering']],[old['speed']],c['geometry_m'],
        cfg['sample_time'],cfg['integration_max_step'])[0,0]
    if not np.isfinite(predicted).all():raise ValueError('Nonfinite continued prediction; checkpoint preserved')
    arrays['predicted_FedPersonal']=predicted
    rms=lambda x:float(np.rad2deg(np.sqrt(np.mean(x*x))))
    metrics=dict(status='complete',yaw_rmse_deg_s=rms(predicted[:,1]-arrays['state'][:,1]),
        measured_yaw_rmse_deg_s=rms(predicted[:,1]-arrays['measured_yaw']),
        beta_rmse_deg=rms(predicted[:,0]-arrays['state'][:,0]))
    return dict(metrics=metrics,reused=False,trace_sha256=base.storage.trace_hash(arrays),
        source_trace_sha256=old['trace_sha256']),arrays


def worker(seed):
    cfg,protocol=preflight(execution=True);source=load_source(seed,cfg,protocol,execution=True)
    p=base.storage.read_checkpoint(path(seed)) if path(seed).exists() else initialize(source,cfg,protocol)
    validate(p,source,cfg,protocol)
    if p['status']=='execution_complete':return dict(seed=seed,status='already_complete')
    if p['optimization']['status']=='local_update_failed':raise RuntimeError('Saved local update failed; inspect checkpoint')
    save(p,'initialized_or_resumed')
    fleet=base.fleet_for(cfg,seed)
    with threadpool_limits(limits=1):
        if p['optimization']['status']=='running':
            clients=[base.fl.Client([training_record(source,c,cfg)],c['geometry_m'],base.settings(cfg)) for c in fleet['clients']]
            continue_rounds(p,clients,cfg,protocol,save)
        for i,c in enumerate(fleet['clients']):
            for v in cfg['evaluation_speeds']:
                for probe in ['slow','broadband']:
                    key=f'{c["client"]}_v{v}_{probe}'
                    if key in p['evaluation']:continue
                    if source_state(source)['status']=='converged':
                        p['evaluation'][key]=dict(metrics=copy.deepcopy(source['evaluation'][key]['methods']['FedPersonal']),
                            reused=True,source_trace_sha256=source['evaluation'][key]['trace_sha256'])
                    else:
                        entry,arrays=evaluate(p,source,c,i,key,cfg)
                        base.storage.save_arrays(RAW/str(seed)/f'{key}.npz',arrays)
                        p['evaluation'][key]=entry
                    save(p,'evaluated_'+key)
    p['status']='execution_complete';validate(p,source,cfg,protocol);save(p,'execution_complete')
    return dict(seed=seed,status=p['optimization']['status'],total_rounds=p['optimization']['round'],
        additional_rounds=p['optimization']['round']-p['source_round'])


def restore(cfg,protocol,seeds):
    for seed in seeds:
        source=load_source(seed,cfg,protocol,execution=True)
        p=base.storage.read_checkpoint(path(seed));validate(p,source,cfg,protocol)
        metadata={c['client']:(i,c) for i,c in enumerate(base.fleet_for(cfg,seed)['clients'])}
        restored=0
        for key,entry in p['evaluation'].items():
            if entry['reused']:continue
            i,c=metadata[source['evaluation'][key]['client']]
            _,arrays=evaluate(p,source,c,i,key,cfg)
            if base.storage.trace_hash(arrays)!=entry['trace_sha256']:raise ValueError('Continued trace reconstruction mismatch')
            target=RAW/str(seed)/f'{key}.npz'
            if target.exists():
                with np.load(target,allow_pickle=False) as loaded:
                    if base.storage.trace_hash(dict(loaded))!=entry['trace_sha256']:raise ValueError('Existing array mismatch; original preserved')
            else:base.storage.save_arrays(target,arrays)
            restored+=1
        print(json.dumps(dict(seed=seed,exact_continued_traces_verified=restored)))


def analyze(cfg,protocol):
    rows=[];optimization=[]
    for seed in protocol['seeds']:
        source=load_source(seed,cfg,protocol);p=base.storage.read_checkpoint(path(seed))
        validate(p,source,cfg,protocol,execution=False)
        if p['status']!='execution_complete':raise ValueError('Complete all three fleet checkpoints before analysis')
        state=p['optimization'];old=source_state(source)
        optimization.append(dict(seed=seed,before_status=old['status'],after_status=state['status'],
            original_rounds=old['round'],total_rounds=state['round'],additional_rounds=state['round']-old['round'],
            before_objective=old['history'][-1]['objective'],after_objective=state['history'][-1]['objective'],
            max_log_parameter_change=float(np.max(abs(np.array(state['log_parameters'])-old['log_parameters']))),
            additional_logical_bytes=8*(state['communication_float64']-old['communication_float64'])))
        for key,entry in p['evaluation'].items():
            oldcase=source['evaluation'][key]
            methods={m:oldcase['methods'][m] for m in ['LocalOE','NominalPrior']}
            methods.update(FedPersonalBefore=oldcase['methods']['FedPersonal'],FedPersonalAfter=entry['metrics'])
            for method,metrics in methods.items():
                rows.append(dict(seed=seed,client=oldcase['client'],speed=oldcase['speed'],probe=oldcase['probe'],
                    region=oldcase['region'],method=method,**metrics))
    cases=pd.DataFrame(rows)
    summary=cases.groupby(['seed','region','method'],as_index=False).yaw_rmse_deg_s.mean()
    paired=[]
    for (seed,region),g in cases.groupby(['seed','region']):
        pivot=g.pivot(index=['client','speed','probe'],columns='method',values='yaw_rmse_deg_s')
        for baseline in ['LocalOE','NominalPrior','FedPersonalBefore']:
            paired.append(dict(seed=seed,region=region,baseline=baseline,cases=len(pivot),
                after_mean=pivot.FedPersonalAfter.mean(),baseline_mean=pivot[baseline].mean(),
                reduction_pct=100*(1-pivot.FedPersonalAfter.mean()/pivot[baseline].mean()),
                improved_cases=int((pivot.FedPersonalAfter<pivot[baseline]).sum())))
    tables=dict(cases=cases,summary=summary,paired=pd.DataFrame(paired),optimization=pd.DataFrame(optimization))
    for name,frame in tables.items():frame.to_csv(OUT/f'{PREFIX}_{name}.csv',index=False)
    render(tables)
    base.storage.atomic_json(OUT/f'{PREFIX}_analysis.json',dict(status='analysis_complete',
        source_checkpoints=protocol['source_checkpoints'],continuation_checkpoints={str(s):base.canonical_hash(path(s)) for s in protocol['seeds']},
        scientific_pass=None,all_converged=all(r['after_status']=='converged' for r in optimization)))
    print(REPORT)


def render(tables):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib.rcParams['svg.hashsalt']=PREFIX
    c=tables['cases'];fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    peers=c[c.region=='peer_observed'].pivot_table(index='seed',columns='method',values='yaw_rmse_deg_s')
    regions=c.pivot_table(index='region',columns='method',values='yaw_rmse_deg_s').reindex(
        ['local_observed','peer_observed','fleet_interpolation','fleet_extrapolation'])
    for method in ['LocalOE','NominalPrior','FedPersonalBefore','FedPersonalAfter']:
        axes[0].plot(peers.index.astype(str),peers[method],marker='o',label=method)
        axes[1].plot(range(4),regions[method],marker='o',label=method)
    axes[0].set(title='Peer-observed speeds',xlabel='Fleet',ylabel='Clean yaw RMSE [deg/s]')
    axes[1].set(title='All speed regions',xticks=range(4),xticklabels=['Local','Peers','Intermediate','Outside fleet'],ylabel='Clean yaw RMSE [deg/s]')
    for ax in axes:ax.grid(alpha=.2)
    axes[0].legend(fontsize=8);buffer=io.StringIO();fig.savefig(buffer,format='svg',metadata={'Date':None});plt.close(fig)
    svg=buffer.getvalue();svg=svg[svg.index('<svg'):]
    sections=''.join('<h2>'+name.title()+'</h2><div class="scroll">'+tables[name].to_html(index=False,border=0,float_format=lambda v:f'{v:.6g}')+'</div>' for name in ['optimization','summary','paired'])
    text=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Personalized FL convergence continuation</title><style>body{{font:16px/1.6 system-ui;color:#173348;background:#eef3f5}}main{{max-width:1150px;margin:auto;background:white;padding:30px}}svg{{width:100%;height:auto}}.scroll{{overflow:auto}}td,th{{padding:8px;border-bottom:1px solid #ddd;white-space:nowrap}}table{{border-collapse:collapse;font-size:13px}}</style><main>
<h1>Does the personalized-FL result survive convergence?</h1><p>Only the total round cap changes, from 8 to 30. Existing converged solutions are reused. Local OE and nominal-prior metrics, data, objectives, stopping tolerances and evaluation conditions are unchanged. Original results remain preserved.</p>
<p>Check optimizer status first: reaching 30 rounds is not convergence. No test error selects a round or stopping point. Compare the final training-stopped estimate against the original estimate and both local baselines, separately for each fleet.</p>{svg}{sections}
<p>Positive reduction percentages favor the continued estimate. Counts are paired trajectory comparisons, not independent population replicates. This is a convergence diagnostic on the same three pilot fleets; no new evidence about complementary-speed versus same-speed peers or closed-loop control is supplied.</p>
<p><a href="../docs/federated_relaxation_continue.html">Protocol and commands</a> · <a href="federated_relaxation_coverage_review.html">Original pilot review</a></p></main></html>'''
    REPORT.parent.mkdir(parents=True,exist_ok=True)
    REPORT.write_text('\n'.join(line.rstrip() for line in text.splitlines())+'\n',encoding='utf-8',newline='\n')


def smoke():
    """Check the real local solver/round update on a short synthetic record."""
    cfg,_=preflight();u=.006*np.sin(np.arange(60)*.09);theta=np.array([50.,55.,.6,.9,.4,.7])
    clients=[]
    for v in [13,21,29]:
        y=base.batch_predict(theta,[u],[v],[1.2,1.6],.01,.001)[0,0,:,1]
        clients.append(base.fl.Client([dict(speed=v,applied_input=u,measured_yaw=y)],[1.2,1.6],base.settings(cfg)))
    state=base.fl.initialize_personal([np.log(theta*1.01).tolist()]*3,1,0,cfg['sharing']['weights'])
    p=dict(optimization=state)
    with tempfile.TemporaryDirectory() as folder:
        target=Path(folder)/'checkpoint.json'
        continue_rounds(p,clients,cfg,dict(total_round_cap=2),lambda p,stage:base.storage.atomic_json(target,p))
        assert base.storage.read_checkpoint(target)==p
    assert p['optimization']['round']<=2 and not p['optimization']['pending']
    return dict(status='smoke_passed',synthetic_rounds=p['optimization']['round'],scientific_fits_run=0)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['preflight','smoke','run','status','analyze','restore'])
    parser.add_argument('--workers',type=int,choices=[1,2,3],default=2)
    parser.add_argument('--seed',type=int,choices=[1301,1302,1303]);args=parser.parse_args()
    cfg,protocol=preflight(execution=args.command in ['run','restore'])
    seeds=[s for s in protocol['seeds'] if args.seed is None or s==args.seed]
    if args.command=='preflight':
        status={s:source_state(load_source(s,cfg,protocol))['status'] for s in protocol['seeds']}
        print(json.dumps(dict(status='preflight_passed',total_round_cap=30,source_status=status,
            continued_fleets=[s for s in status if status[s]=='budget_exhausted'],new_baseline_fits=0,
            source_runtime_matches=all(load_source(s,cfg,protocol)['provenance']['runtime']==base.identity(cfg)['runtime'] for s in seeds),
            exact_original_arrays_required_or_reconstructed=True)));return
    if args.command=='smoke':
        with ProcessPoolExecutor(max_workers=1,mp_context=multiprocessing.get_context('spawn')) as pool:print(json.dumps(pool.submit(smoke).result()))
        return
    if args.command=='status':
        for s in seeds:
            if path(s).exists():
                p=base.storage.read_checkpoint(path(s));validate(p,load_source(s,cfg,protocol),cfg,protocol,execution=False)
                print(json.dumps(dict(seed=s,status=p['status'],optimizer_status=p['optimization']['status'],round=p['optimization']['round'],evaluated=len(p['evaluation']))))
            else:print(json.dumps(dict(seed=s,status='not_started')))
        return
    RAW.mkdir(parents=True,exist_ok=True);lock=RAW/'run.lock'
    try:fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError:raise RuntimeError(f'Lock exists: {lock}; remove only after confirming no continuation process is active')
    os.write(fd,str(os.getpid()).encode());os.close(fd)
    try:
        if args.command=='analyze':analyze(cfg,protocol)
        elif args.command=='restore':restore(cfg,protocol,seeds)
        else:
            with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
                for result in as_completed([pool.submit(worker,s) for s in seeds]):print(json.dumps(result.result()),flush=True)
            print('Commit/push continuation seed checkpoints before running analyze. See docs/federated_relaxation_continue.html.')
    finally:lock.unlink()


if __name__=='__main__':main()
