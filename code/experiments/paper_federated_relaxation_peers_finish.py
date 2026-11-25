"""Finish only the two outstanding 13 m/s SameSpeed fits; preserve all inputs."""
from __future__ import annotations
import os
for name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[name]='1'
import argparse
import copy
from concurrent.futures import ProcessPoolExecutor,as_completed
import io
import json
import multiprocessing
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import paper_federated_relaxation_peers as peers

base=peers.base
ROOT=peers.ROOT
PREFIX='paper_federated_relaxation_peers_finish'
CONFIG=ROOT/'code/config'/f'{PREFIX}.json'
OUT=ROOT/'results/tables'
RAW=ROOT/'results/data'/PREFIX
REPORT=ROOT/'report/federated_relaxation_peers_finish.html'


def path(seed):return OUT/f'{PREFIX}_seed{seed}.json'


def identity(cfg,protocol):
    old=json.loads(peers.CONFIG.read_text(encoding='utf-8'))
    return dict(peer=peers.identity(cfg,old),sources=protocol['source_checkpoints'],
        files={p.relative_to(ROOT).as_posix():base.canonical_hash(p) for p in [CONFIG,Path(__file__)]})


def sources(seed,cfg,protocol,execution=False):
    target=peers.path(seed)
    if base.canonical_hash(target)!=protocol['source_checkpoints'][target.relative_to(ROOT).as_posix()]:
        raise ValueError('Frozen peer checkpoint changed')
    old_protocol=json.loads(peers.CONFIG.read_text(encoding='utf-8'))
    original,complementary=peers.sources(seed,cfg,old_protocol,execution)
    previous=base.storage.read_checkpoint(target);peers.validate(previous,cfg,old_protocol,execution)
    if previous['status']!='execution_complete':raise ValueError('Original peer experiment incomplete')
    return original,complementary,previous


def preflight(execution=False):
    cfg,_=peers.preflight(execution);protocol=json.loads(CONFIG.read_text(encoding='utf-8'))
    assert protocol['seeds']==[1301,1302] and protocol['training_speed']==13 and protocol['total_round_cap']==100
    unfinished=[]
    for seed in [1301,1302,1303]:
        _,_,old=sources(seed,cfg,protocol,execution)
        for speed,condition in old['conditions'].items():
            if condition['status']!='converged':
                assert condition['status']=='budget_exhausted' and condition['optimization']['round']==30
                assert not condition['optimization']['pending']
                unfinished.append((seed,int(speed)))
    assert unfinished==[(1301,13),(1302,13)]
    return cfg,protocol


def affected_keys(original,cfg,seed):
    targets={c['client'] for c in base.fleet_for(cfg,seed)['clients'] if c['training_speed']==13}
    return {key for key,c in original['evaluation'].items() if c['client'] in targets}


def initialize(seed,previous,cfg,protocol):
    if seed not in protocol['seeds']:raise ValueError('Only fleets 1301 and 1302 are authorized for continuation')
    state=copy.deepcopy(previous['conditions']['13']['optimization'])
    if state['status']!='budget_exhausted' or state['round']!=30 or state['pending']:
        raise ValueError('Expected saved complete round-30 state')
    state['status']='running'
    return dict(seed=seed,status='running',provenance=identity(cfg,protocol),optimization=state,evaluation={})


def validate(p,original,previous,cfg,protocol,execution=True):
    expected=identity(cfg,protocol);actual=copy.deepcopy(p['provenance'])
    if not execution:actual['peer']['base']['runtime']=expected['peer']['base']['runtime']
    if actual!=expected:raise ValueError('Continuation provenance changed; checkpoint preserved')
    assert p['seed'] in protocol['seeds'] and p['status'] in ['running','execution_complete']
    state=p['optimization'];old=previous['conditions']['13']['optimization']
    assert state['history'][:len(old['history'])]==old['history']
    assert old['round']<=state['round']<=protocol['total_round_cap']
    assert len(state['history'])==state['round']
    expected_keys=affected_keys(original,cfg,p['seed'])
    assert set(p['evaluation']).issubset(expected_keys)
    if p['status']=='execution_complete':
        assert set(p['evaluation'])==expected_keys and not state['pending']
        assert state['status'] in ['converged','budget_exhausted']


def save(p,stage):
    base.storage.atomic_json(path(p['seed']),p)
    print(json.dumps(dict(seed=p['seed'],saved=stage,total_round=p['optimization']['round'])),flush=True)


def record(original,previous,c,cfg):
    key=peers.record_key(c,13);target=peers.RAW/str(previous['seed'])/f'{key}.npz'
    if target.exists():
        with np.load(target,allow_pickle=False) as loaded:arrays=dict(loaded)
    else:arrays,_=peers.make_record(original,c,13,cfg)
    if base.storage.trace_hash(arrays)!=previous['records'][key]['trace_sha256']:
        raise ValueError('Original SameSpeed training hash mismatch; use original machine or raw arrays')
    if not previous['records'][key]['operating_checks_passed']:raise ValueError('Original operating check failed')
    return dict(speed=13,applied_input=arrays['steering'],measured_yaw=arrays['measured_yaw'])


def evaluate(p,original,c,index,key,cfg):
    wrapper=dict(seed=p['seed'],conditions={'13':dict(status=p['optimization']['status'],optimization=p['optimization'])})
    entry,arrays=peers.evaluate(wrapper,original,c,index,key,cfg)
    if arrays is None:raise RuntimeError('Unavailable continued model; checkpoint preserved')
    return entry,arrays


def worker(seed):
    cfg,protocol=preflight(execution=True)
    original,_,previous=sources(seed,cfg,protocol,execution=True)
    p=base.storage.read_checkpoint(path(seed)) if path(seed).exists() else initialize(seed,previous,cfg,protocol)
    validate(p,original,previous,cfg,protocol)
    if p['status']=='execution_complete':return dict(seed=seed,status='already_complete')
    if p['optimization']['status']=='local_update_failed':raise RuntimeError('Saved local update failed; inspect checkpoint')
    save(p,'initialized_or_resumed');fleet=base.fleet_for(cfg,seed)
    with threadpool_limits(limits=1):
        if p['optimization']['status']=='running':
            clients=[base.fl.Client([record(original,previous,c,cfg)],c['geometry_m'],base.settings(cfg)) for c in fleet['clients']]
            peers.continuation.continue_rounds(p,clients,cfg,protocol,save)
        metadata={c['client']:(i,c) for i,c in enumerate(fleet['clients'])}
        for key in sorted(affected_keys(original,cfg,seed)):
            if key in p['evaluation']:continue
            i,c=metadata[original['evaluation'][key]['client']]
            entry,arrays=evaluate(p,original,c,i,key,cfg)
            base.storage.save_arrays(RAW/str(seed)/f'{key}.npz',arrays)
            p['evaluation'][key]=entry;save(p,'evaluation_'+key)
    p['status']='execution_complete';validate(p,original,previous,cfg,protocol);save(p,'execution_complete')
    return dict(seed=seed,status=p['optimization']['status'],total_rounds=p['optimization']['round'],additional_rounds=p['optimization']['round']-30)


def restore(cfg,protocol,seeds):
    for seed in seeds:
        original,_,previous=sources(seed,cfg,protocol,execution=True)
        p=base.storage.read_checkpoint(path(seed));validate(p,original,previous,cfg,protocol)
        metadata={c['client']:(i,c) for i,c in enumerate(base.fleet_for(cfg,seed)['clients'])}
        for key,entry in p['evaluation'].items():
            i,c=metadata[original['evaluation'][key]['client']];_,arrays=evaluate(p,original,c,i,key,cfg)
            if base.storage.trace_hash(arrays)!=entry['trace_sha256']:raise ValueError('Exact reconstruction mismatch')
            target=RAW/str(seed)/f'{key}.npz'
            if target.exists():
                with np.load(target,allow_pickle=False) as loaded:
                    if base.storage.trace_hash(dict(loaded))!=entry['trace_sha256']:raise ValueError('Existing trace mismatch; original preserved')
            else:base.storage.save_arrays(target,arrays)
        print(json.dumps(dict(seed=seed,exact_traces_verified=len(p['evaluation']))))


def analyze(cfg,protocol):
    rows=[];optimization=[];locked={}
    for seed in [1301,1302,1303]:
        original,complementary,previous=sources(seed,cfg,protocol)
        p=None
        if seed in protocol['seeds']:
            p=base.storage.read_checkpoint(path(seed));validate(p,original,previous,cfg,protocol,execution=False)
            if p['status']!='execution_complete':raise ValueError('Complete both continuations before analysis')
            locked[str(seed)]=base.canonical_hash(path(seed))
        for speed,condition in previous['conditions'].items():
            old=condition['optimization'];new=p['optimization'] if p and speed=='13' else old
            optimization.append(dict(seed=seed,training_speed=int(speed),reused=new is old,original_rounds=old['round'],
                total_rounds=new['round'],added_rounds=new['round']-old['round'],status=new['status'],
                before_objective=old['history'][-1]['objective'],after_objective=new['history'][-1]['objective'],
                max_parameter_change_pct=float(np.max(abs(np.expm1(np.array(new['log_parameters'])-old['log_parameters']))))*100))
        metadata={c['client']:c for c in base.fleet_for(cfg,seed)['clients']}
        for key,oldcase in original['evaluation'].items():
            before=previous['evaluation'][key];after=p['evaluation'][key] if p and key in p['evaluation'] else before
            assert after['source_trace_sha256']==oldcase['trace_sha256']
            methods={m:oldcase['methods'][m] for m in ['LocalOE','NominalPrior']}
            methods.update(Complementary=complementary['evaluation'][key]['metrics'],SameSpeedBefore=before['metrics'],SameSpeedAfter=after['metrics'])
            for method,result in methods.items():
                rows.append(dict(seed=seed,client=oldcase['client'],training_speed=metadata[oldcase['client']]['training_speed'],
                    speed=oldcase['speed'],probe=oldcase['probe'],region=peers.REGIONS[oldcase['region']],method=method,**result))
    cases=pd.DataFrame(rows)
    summary=cases.groupby(['seed','training_speed','region','method'],as_index=False).yaw_rmse_deg_s.mean()
    paired=[]
    for (seed,speed,region),g in cases.groupby(['seed','training_speed','region']):
        pivot=g.pivot(index=['client','speed','probe'],columns='method',values='yaw_rmse_deg_s')
        for method,baseline in [('Complementary','SameSpeedAfter'),('Complementary','NominalPrior'),
                                ('SameSpeedAfter','SameSpeedBefore'),('SameSpeedAfter','NominalPrior'),('Complementary','LocalOE')]:
            a=pivot[method].mean();b=pivot[baseline].mean()
            paired.append(dict(seed=seed,training_speed=speed,region=region,method=method,baseline=baseline,cases=len(pivot),
                method_mean=a,baseline_mean=b,reduction_pct=100*(1-a/b) if b else None,
                improved_cases=int((pivot[method]<pivot[baseline]).sum())))
    tables=dict(cases=cases,summary=summary,paired=pd.DataFrame(paired),optimization=pd.DataFrame(optimization))
    for name,frame in tables.items():frame.to_csv(OUT/f'{PREFIX}_{name}.csv',index=False)
    render(tables)
    base.storage.atomic_json(OUT/f'{PREFIX}_analysis.json',dict(status='analysis_complete',sources=protocol['source_checkpoints'],
        finish_checkpoints=locked,all_same_speed_converged=all(r['status']=='converged' for r in optimization),scientific_pass=None))
    print(REPORT)


def render(t):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib.rcParams['svg.hashsalt']=PREFIX
    primary=t['cases'][t['cases'].region=='target_unseen_anchor']
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for method in ['LocalOE','NominalPrior','SameSpeedBefore','SameSpeedAfter','Complementary']:
        for ax,key in zip(axes,['seed','training_speed']):
            values=primary[primary.method==method].groupby(key).yaw_rmse_deg_s.mean()
            ax.plot(values.index.astype(str),values,marker='o',label=method)
    axes[0].set(xlabel='Fleet',title='Target-unseen speeds: by fleet');axes[1].set(xlabel='Target training speed [m/s]',title='Target-unseen speeds: by training speed')
    for ax in axes:ax.set_ylabel('Clean yaw RMSE [deg/s]');ax.grid(alpha=.2)
    axes[0].legend(fontsize=8);buf=io.StringIO();fig.savefig(buf,format='svg',metadata={'Date':None});plt.close(fig)
    svg=buf.getvalue();svg=svg[svg.index('<svg'):]
    tab=lambda d:'<div class="scroll">'+d.to_html(index=False,border=0,float_format=lambda x:f'{x:.6g}')+'</div>'
    text=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Final matched peer-speed convergence check</title>
<style>body{{font:16px/1.65 system-ui;color:#173348;background:#edf2f5}}main{{max-width:1150px;margin:auto;padding:30px;background:white}}svg{{width:100%;height:auto}}.scroll{{overflow:auto}}table{{border-collapse:collapse;font-size:13px}}th,td{{padding:8px;border-bottom:1px solid #ddd;white-space:nowrap}}</style><main>
<h1>Finish the two remaining fits, then consolidate</h1><p>Only the 13 m/s SameSpeed conditions in fleets 1301 and 1302 continue from their saved round-30 states. Seven completed conditions and all complementary/local-prior results are reused. Data, objectives, solver settings and stopping tolerances remain unchanged; the emergency cap is 100 total rounds.</p>
<h2>Convergence first</h2>{tab(t['optimization'])}<p>Any budget-exhausted condition remains explicitly unconverged. Test errors do not choose the stopping round. A continuation requires no new initialization or data acquisition.</p>
<h2>Full matched comparison</h2>{svg}{tab(t['paired'][t['paired'].region=='target_unseen_anchor'])}
<h2>All speed regions</h2>{tab(t['summary'])}
<p>Positive reductions favor the named method. Compare the final complementary-versus-same-speed effect against its previous value and the nominal-prior baseline, separately by fleet and target speed. Unseen anchors are the other two speeds among 13/21/29 m/s. Intermediate and outside-reference categories use the fixed original evaluation grid.</p>
<p>This closes a numerical issue on existing development fleets. It is not an independent confirmation or closed-loop experiment. No automatic scientific pass is assigned. Consolidate the physical-model, local-regularization, personalized-sharing and complementary-coverage findings before choosing further experiments.</p>
<p><a href="../docs/federated_relaxation_peers_finish.html">Protocol and commands</a> · <a href="federated_relaxation_peers_review.html">Original matched-peer review</a></p></main></html>'''
    REPORT.parent.mkdir(parents=True,exist_ok=True)
    REPORT.write_text('\n'.join(line.rstrip() for line in text.splitlines())+'\n',encoding='utf-8',newline='\n')


def smoke():
    preflight()
    result=peers.continuation.smoke()
    return dict(status='smoke_passed',optimizer=result,scientific_fits_run=0)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['preflight','smoke','run','status','analyze','restore'])
    parser.add_argument('--workers',type=int,choices=[1,2],default=2);parser.add_argument('--seed',type=int,choices=[1301,1302]);args=parser.parse_args()
    cfg,protocol=preflight(execution=args.command in ['run','restore'])
    seeds=[s for s in protocol['seeds'] if args.seed is None or s==args.seed]
    if args.command=='preflight':
        print(json.dumps(dict(status='preflight_passed',continued_conditions=[[1301,13],[1302,13]],start_round=30,
            total_round_cap=100,unchanged_conditions=7,new_training_records=0,new_initial_fits=0,reevaluated_targets=6,prediction_cases=84)));return
    if args.command=='smoke':
        with ProcessPoolExecutor(max_workers=1,mp_context=multiprocessing.get_context('spawn')) as pool:print(json.dumps(pool.submit(smoke).result()))
        return
    if args.command=='status':
        for seed in seeds:
            if path(seed).exists():
                original,_,old=sources(seed,cfg,protocol);p=base.storage.read_checkpoint(path(seed));validate(p,original,old,cfg,protocol,execution=False)
                print(json.dumps(dict(seed=seed,status=p['status'],optimizer_status=p['optimization']['status'],round=p['optimization']['round'],evaluated=len(p['evaluation']))))
            else:print(json.dumps(dict(seed=seed,status='not_started')))
        return
    RAW.mkdir(parents=True,exist_ok=True);lock=RAW/'run.lock'
    try:fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError:raise RuntimeError(f'Lock exists: {lock}; remove only after confirming no finish process is active')
    os.write(fd,str(os.getpid()).encode());os.close(fd)
    try:
        if args.command=='analyze':analyze(cfg,protocol)
        elif args.command=='restore':restore(cfg,protocol,seeds)
        else:
            with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
                for result in as_completed([pool.submit(worker,s) for s in seeds]):print(json.dumps(result.result()),flush=True)
            print('Commit/push both seed checkpoints before analyze; see docs/federated_relaxation_peers_finish.html.')
    finally:lock.unlink()


if __name__=='__main__':main()
