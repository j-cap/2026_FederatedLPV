"""Matched peer-speed diagnostic: identical targets/peers, changed peer speeds."""
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
import paper_federated_relaxation_continue as continuation

base=continuation.base
ROOT=base.ROOT
PREFIX='paper_federated_relaxation_peers'
CONFIG=ROOT/'code/config'/f'{PREFIX}.json'
OUT=ROOT/'results/tables'
RAW=ROOT/'results/data'/PREFIX
REPORT=ROOT/'report/federated_relaxation_peers.html'
REGIONS={'local_observed':'target_observed','peer_observed':'target_unseen_anchor',
         'fleet_interpolation':'intermediate_reference_speed','fleet_extrapolation':'outside_reference_range'}


def path(seed):return OUT/f'{PREFIX}_seed{seed}.json'
def record_key(c,speed):return f'{c["client"]}_train{speed}'


def identity(cfg,protocol):
    return dict(base=base.identity(cfg),source_checkpoints=protocol['source_checkpoints'],
        files={p.relative_to(ROOT).as_posix():base.canonical_hash(p) for p in
               [CONFIG,Path(__file__),Path(continuation.__file__),continuation.CONFIG]})


def sources(seed,cfg,protocol,execution=False):
    for file in [base.path(seed),continuation.path(seed)]:
        if base.canonical_hash(file)!=protocol['source_checkpoints'][file.relative_to(ROOT).as_posix()]:
            raise ValueError('Frozen source checkpoint changed')
    original_protocol=json.loads(continuation.CONFIG.read_text(encoding='utf-8'))
    original=continuation.load_source(seed,cfg,original_protocol,execution)
    final=base.storage.read_checkpoint(continuation.path(seed))
    continuation.validate(final,original,cfg,original_protocol,execution=execution)
    if final['status']!='execution_complete' or final['optimization']['status']!='converged':
        raise ValueError('Complementary condition must be complete and converged')
    return original,final


def preflight(execution=False):
    cfg=base.preflight();protocol=json.loads(CONFIG.read_text(encoding='utf-8'))
    assert protocol['total_round_cap']==30 and protocol['seeds']==[1301,1302,1303]
    assert protocol['same_speeds']==cfg['fleet_training_speeds']==[13,21,29]
    assert protocol['noise_pairing']=='original_client_training_noise_across_speeds'
    for seed in protocol['seeds']:
        original,final=sources(seed,cfg,protocol,execution)
        fleet=base.fleet_for(cfg,seed)
        assert len(fleet['clients'])==9
        for speed in protocol['same_speeds']:
            assert sum(c['training_speed']==speed for c in fleet['clients'])==3
        assert all(original['local'][c['client']]['selected']['available'] for c in fleet['clients'])
    return cfg,protocol


def original_arrays(source,c,cfg):
    file=base.RAW/str(source['seed'])/f'{c["client"]}_training.npz'
    if file.exists():
        with np.load(file,allow_pickle=False) as loaded:arrays=dict(loaded)
    else:arrays,_=base.generate(c,source['seed'],'training',c['training_speed'],cfg)
    if base.storage.trace_hash(arrays)!=source['data'][c['client']]['trace_sha256']:
        raise ValueError('Original training hash mismatch; use original machine or raw arrays')
    return arrays


def make_record(source,c,speed,cfg):
    """Simulation-only record construction; learning sees only yaw/u/speed."""
    original=original_arrays(source,c,cfg)
    arrays={k:v.copy() for k,v in original.items()}
    if speed!=c['training_speed']:
        arrays['state']=base.batch_predict(c['physical_parameters'],[arrays['steering']],[speed],c['geometry_m'],
            cfg['sample_time'],cfg['integration_max_step'])[0,0]
        arrays['measured_yaw']=arrays['state'][:,1]+arrays['noise']
    # A common random-number realization for each peer across the two conditions.
    assert np.array_equal(arrays['noise'],original['noise']) and np.array_equal(arrays['steering'],original['steering'])
    x=arrays['state'];u=arrays['steering']
    metrics=dict(beta_peak_deg=float(np.rad2deg(max(abs(x[:,0])))),ay_peak_mps2=float(max(abs(x[:,2]+x[:,3]))),
        steering_rate_peak_deg_s=float(np.rad2deg(max(abs(np.diff(np.r_[0,u,0]))))/cfg['sample_time']))
    record=dict(trace_sha256=base.storage.trace_hash(arrays),source_trace_sha256=source['data'][c['client']]['trace_sha256'],
        speed=speed,original_speed=c['training_speed'],reused=speed==c['training_speed'],metrics=metrics,
        operating_checks_passed=all(metrics[k]<=bound for k,bound in cfg['operating_bounds'].items()),
        paired_noise_sha256=base.storage.trace_hash({'noise':arrays['noise']}),
        paired_input_sha256=base.storage.trace_hash({'steering':arrays['steering']}))
    if record['reused']:assert record['trace_sha256']==record['source_trace_sha256']
    return arrays,record


def save(p,stage):
    base.storage.atomic_json(path(p['seed']),p)
    print(json.dumps(dict(seed=p['seed'],saved=stage)),flush=True)


def acquire(p,source,c,speed,cfg):
    key=record_key(c,speed);target=RAW/str(p['seed'])/f'{key}.npz'
    if key in p['records'] and target.exists():
        with np.load(target,allow_pickle=False) as loaded:arrays=dict(loaded)
        record=p['records'][key]
    else:arrays,record=make_record(source,c,speed,cfg)
    if key in p['records'] and record!=p['records'][key]:raise ValueError('Training metadata reconstruction mismatch')
    if base.storage.trace_hash(arrays)!=record['trace_sha256']:raise ValueError('Saved training array hash mismatch')
    if target.exists():
        with np.load(target,allow_pickle=False) as loaded:
            if base.storage.trace_hash(dict(loaded))!=record['trace_sha256']:
                raise ValueError('Existing training array mismatch; original preserved')
    else:base.storage.save_arrays(target,arrays)
    if key not in p['records']:p['records'][key]=record;save(p,'acquired_'+key)
    if not record['operating_checks_passed']:raise ValueError('Training operating check failed; case preserved')
    return dict(speed=speed,applied_input=arrays['steering'],measured_yaw=arrays['measured_yaw'])


def validate(p,cfg,protocol,execution=True):
    expected=identity(cfg,protocol);actual=copy.deepcopy(p['provenance'])
    if not execution:actual['base']['runtime']=expected['base']['runtime']
    if actual!=expected:raise ValueError('Protocol/source/runtime changed; checkpoint preserved')
    assert p['seed'] in protocol['seeds'] and p['status'] in ['running','execution_complete']
    for condition in p['conditions'].values():
        if 'optimization' in condition:
            state=condition['optimization'];assert state['round']<=protocol['total_round_cap']
            assert len(state['history'])==state['round']
    if p['status']=='execution_complete':
        fleet=base.fleet_for(cfg,p['seed'])
        assert set(p['conditions'])=={str(v) for v in protocol['same_speeds']}
        assert set(p['records'])=={record_key(c,v) for c in fleet['clients'] for v in protocol['same_speeds']}
        assert set(p['local'])==set(p['records'])
        expected_cases={f'{c["client"]}_v{v}_{probe}' for c in fleet['clients'] for v in cfg['evaluation_speeds'] for probe in ['slow','broadband']}
        assert set(p['evaluation'])==expected_cases
        assert all(c['status'] in ['converged','budget_exhausted','initialization_failed','local_update_failed'] for c in p['conditions'].values())


def evaluate(p,source,c,index,key,cfg):
    condition=p['conditions'][str(c['training_speed'])]
    if condition['status'] not in ['converged','budget_exhausted']:
        return dict(metrics=dict(status='model_unavailable',yaw_rmse_deg_s=None,measured_yaw_rmse_deg_s=None,beta_rmse_deg=None),
                    source_trace_sha256=source['evaluation'][key]['trace_sha256']),None
    entry,arrays=continuation.evaluate(dict(seed=p['seed'],optimization=condition['optimization']),source,c,index,key,cfg)
    arrays['predicted_SameSpeed']=arrays.pop('predicted_FedPersonal')
    entry['trace_sha256']=base.storage.trace_hash(arrays)
    return entry,arrays


def worker(seed):
    cfg,protocol=preflight(execution=True);source,complementary=sources(seed,cfg,protocol,execution=True)
    p=base.storage.read_checkpoint(path(seed)) if path(seed).exists() else dict(seed=seed,status='running',
        provenance=identity(cfg,protocol),records={},local={},conditions={},evaluation={})
    validate(p,cfg,protocol)
    if p['status']=='execution_complete':return dict(seed=seed,status='already_complete')
    fleet=base.fleet_for(cfg,seed)
    with threadpool_limits(limits=1):
        for speed in protocol['same_speeds']:
            key=str(speed)
            if key in p['conditions'] and p['conditions'][key]['status']!='running':continue
            endpoints=[base.fl.Client([acquire(p,source,c,speed,cfg)],c['geometry_m'],base.settings(cfg)) for c in fleet['clients']]
            for i,c in enumerate(fleet['clients']):
                rid=record_key(c,speed)
                if speed==c['training_speed']:
                    if rid not in p['local']:
                        p['local'][rid]=dict(selected=copy.deepcopy(source['local'][c['client']]['selected']),reused=True)
                        save(p,'reused_local_'+rid)
                    continue
                entry=p['local'].setdefault(rid,dict(attempts=[],reused=False))
                for j in range(len(entry['attempts']),len(cfg['starts'])):
                    # Only the same common guesses; no other-speed fitted parameters.
                    entry['attempts'].append(endpoints[i].fit(np.log(cfg['starts'][j])))
                    save(p,f'local_{rid}_start{j}')
                entry['selected']=base.fl.select_local(entry['attempts']);save(p,'selected_'+rid)
            selected=[p['local'][record_key(c,speed)]['selected'] for c in fleet['clients']]
            if not all(s['available'] for s in selected):
                p['conditions'][key]=dict(status='initialization_failed',reason='At least one local OE fit unavailable')
                save(p,'initialization_failed_speed'+key);continue
            if key not in p['conditions']:
                logs=[s['log_parameters'] for s in selected]
                p['conditions'][key]=dict(status='running',initial_log_parameters=copy.deepcopy(logs),
                    optimization=base.fl.initialize_personal(logs,1,0,cfg['sharing']['weights']))
                save(p,'initialized_speed'+key)
            condition=p['conditions'][key]
            try:
                continuation.continue_rounds(condition,endpoints,cfg,protocol,
                    lambda _,stage:save(p,f'speed{speed}_{stage}'))
            except RuntimeError:
                if condition['optimization']['status']!='local_update_failed':raise
            condition['status']=condition['optimization']['status'];save(p,'finished_speed'+key)
        for i,c in enumerate(fleet['clients']):
            for v in cfg['evaluation_speeds']:
                for probe in ['slow','broadband']:
                    key=f'{c["client"]}_v{v}_{probe}'
                    if key in p['evaluation']:continue
                    entry,arrays=evaluate(p,source,c,i,key,cfg)
                    if arrays is not None:base.storage.save_arrays(RAW/str(seed)/f'{key}.npz',arrays)
                    p['evaluation'][key]=entry;save(p,'evaluation_'+key)
        p['status']='execution_complete';validate(p,cfg,protocol);save(p,'execution_complete')
    return dict(seed=seed,status=p['status'],conditions={v:s['status'] for v,s in p['conditions'].items()})


def restore(cfg,protocol,seeds):
    for seed in seeds:
        p=base.storage.read_checkpoint(path(seed));validate(p,cfg,protocol)
        source,_=sources(seed,cfg,protocol,execution=True);fleet=base.fleet_for(cfg,seed);count=0
        def verify(name,arrays,sha):
            nonlocal count
            if base.storage.trace_hash(arrays)!=sha:raise ValueError('Exact reconstruction mismatch')
            target=RAW/str(seed)/f'{name}.npz'
            if target.exists():
                with np.load(target,allow_pickle=False) as loaded:
                    if base.storage.trace_hash(dict(loaded))!=sha:raise ValueError('Existing array mismatch; original preserved')
            else:base.storage.save_arrays(target,arrays)
            count+=1
        for i,c in enumerate(fleet['clients']):
            for speed in protocol['same_speeds']:
                key=record_key(c,speed)
                if key in p['records']:
                    arrays,_=make_record(source,c,speed,cfg);verify(key,arrays,p['records'][key]['trace_sha256'])
            for key,entry in p['evaluation'].items():
                if source['evaluation'][key]['client']!=c['client'] or entry['metrics']['status']!='complete':continue
                _,arrays=evaluate(p,source,c,i,key,cfg);verify(key,arrays,entry['trace_sha256'])
        print(json.dumps(dict(seed=seed,exact_traces_verified=count)))


def summaries(cases):
    rows=[];paired=[]
    for (seed,speed,region,method),g in cases.groupby(['seed','training_speed','region','method']):
        valid=g.yaw_rmse_deg_s.notna() & np.isfinite(g.yaw_rmse_deg_s)
        rows.append(dict(seed=seed,training_speed=speed,region=region,method=method,complete=int(valid.sum()),expected=len(g),
            mean_yaw_rmse_deg_s=float(g.yaw_rmse_deg_s.mean()) if valid.all() else None))
    for (seed,speed,region),g in cases.groupby(['seed','training_speed','region']):
        pivot=g.pivot(index=['client','speed','probe'],columns='method',values='yaw_rmse_deg_s')
        for baseline in ['SameSpeed','NominalPrior','LocalOE']:
            pair=pivot[['Complementary',baseline]].dropna();complete=len(pair)==len(pivot)
            bm=float(pair[baseline].mean()) if len(pair) else 0.
            paired.append(dict(seed=seed,training_speed=speed,region=region,baseline=baseline,paired=len(pair),expected=len(pivot),
                complementary_mean=float(pair.Complementary.mean()) if complete else None,baseline_mean=bm if complete else None,
                reduction_pct=100*(1-pair.Complementary.mean()/bm) if complete and bm>0 else None,
                mean_difference_deg_s=float((pair.Complementary-pair[baseline]).mean()) if complete else None,
                improved_cases=int((pair.Complementary<pair[baseline]).sum())))
    return pd.DataFrame(rows),pd.DataFrame(paired)


def analyze(cfg,protocol):
    rows=[];optimization=[];acquisition=[]
    for seed in protocol['seeds']:
        p=base.storage.read_checkpoint(path(seed));validate(p,cfg,protocol,execution=False)
        if p['status']!='execution_complete':raise ValueError('Complete all three fleets before analysis')
        source,complementary=sources(seed,cfg,protocol)
        metadata={c['client']:c for c in base.fleet_for(cfg,seed)['clients']}
        for c in metadata.values():
            key=record_key(c,c['training_speed'])
            assert p['records'][key]['trace_sha256']==source['data'][c['client']]['trace_sha256']
            assert p['local'][key]['selected']==source['local'][c['client']]['selected']
            noise={p['records'][record_key(c,v)]['paired_noise_sha256'] for v in protocol['same_speeds']}
            inputs={p['records'][record_key(c,v)]['paired_input_sha256'] for v in protocol['same_speeds']}
            assert len(noise)==len(inputs)==1
        for key,entry in p['records'].items():acquisition.append(dict(seed=seed,record=key,**entry['metrics'],reused=entry['reused'],operating_checks_passed=entry['operating_checks_passed']))
        for speed,condition in p['conditions'].items():
            state=condition.get('optimization',{})
            optimization.append(dict(seed=seed,condition='SameSpeed',training_speed=int(speed),status=condition['status'],
                rounds=state.get('round'),objective=state['history'][-1]['objective'] if state.get('history') else None,
                logical_float64_bytes=8*state.get('communication_float64',0)))
        state=complementary['optimization']
        optimization.append(dict(seed=seed,condition='Complementary',training_speed=None,status=state['status'],
            rounds=state['round'],objective=state['history'][-1]['objective'],logical_float64_bytes=8*state['communication_float64']))
        for key,entry in p['evaluation'].items():
            old=source['evaluation'][key];c=metadata[old['client']]
            assert entry['source_trace_sha256']==old['trace_sha256']
            methods={m:old['methods'][m] for m in ['LocalOE','NominalPrior']}
            methods.update(Complementary=complementary['evaluation'][key]['metrics'],SameSpeed=entry['metrics'])
            for name,result in methods.items():
                rows.append(dict(seed=seed,client=c['client'],training_speed=c['training_speed'],speed=old['speed'],probe=old['probe'],
                    region=REGIONS[old['region']],method=name,**result))
    cases=pd.DataFrame(rows);summary,paired=summaries(cases)
    tables=dict(cases=cases,summary=summary,paired=paired,optimization=pd.DataFrame(optimization),acquisition=pd.DataFrame(acquisition))
    for name,table in tables.items():table.to_csv(OUT/f'{PREFIX}_{name}.csv',index=False)
    render(tables)
    base.storage.atomic_json(OUT/f'{PREFIX}_analysis.json',dict(status='analysis_complete',source_checkpoints=protocol['source_checkpoints'],
        checkpoints={str(s):base.canonical_hash(path(s)) for s in protocol['seeds']},cases=len(cases),
        all_same_speed_converged=all(r['status']=='converged' for r in optimization if r['condition']=='SameSpeed'),scientific_pass=None))
    print(REPORT)


def render(t):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib.rcParams['svg.hashsalt']=PREFIX
    data=t['cases'];primary=data[data.region=='target_unseen_anchor']
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for method in ['LocalOE','NominalPrior','SameSpeed','Complementary']:
        g=primary[primary.method==method]
        for ax,key in zip(axes,['seed','training_speed']):
            grouped=g.groupby(key).yaw_rmse_deg_s
            values=grouped.mean();values[grouped.count()!=grouped.size()]=np.nan
            ax.plot(values.index.astype(str),values,marker='o',label=method)
    axes[0].set(xlabel='Fleet',ylabel='Clean yaw RMSE [deg/s]',title='Unseen anchor speeds: per fleet')
    axes[1].set(xlabel='Target training speed [m/s]',ylabel='Clean yaw RMSE [deg/s]',title='Unseen anchor speeds: per target speed')
    axes[0].legend(fontsize=8)
    for ax in axes:ax.grid(alpha=.2)
    b=io.StringIO();fig.savefig(b,format='svg',metadata={'Date':None});plt.close(fig);svg=b.getvalue();svg=svg[svg.index('<svg'):]
    tab=lambda d:'<div class="scroll">'+d.to_html(index=False,border=0,float_format=lambda x:f'{x:.6g}')+'</div>'
    text=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Complementary-speed versus same-speed peers</title>
<style>body{{font:16px/1.65 system-ui;background:#edf2f5;color:#173348}}main{{max-width:1150px;margin:auto;padding:30px;background:white}}svg{{width:100%;height:auto}}.scroll{{overflow:auto}}table{{border-collapse:collapse;font-size:13px}}th,td{{padding:8px;border-bottom:1px solid #ddd;white-space:nowrap}}</style><main>
<h1>Do complementary peer speeds improve the target model?</h1><p>Same nine physical vehicles in both conditions. For each target, its original local record is fixed. Complementary reuses the converged balanced 13/21/29 m/s fleet. SameSpeed learns from all nine records at the target's speed. The eight peers retain their identities, input, sample count and paired measurement-noise realization; six peers change speed, while two already share the target speed.</p>
<p>Each condition has 108 s of fleet data, including the target's 12 s. Three uniform-speed fits per fleet serve the three target-speed subsets; no target uses its own extra counterfactual records from the other fits. Test conditions and local baselines are unchanged. Peer labels and true parameters are not supplied to identification.</p>
<h2>Convergence and operating checks first</h2>{tab(t['optimization'])}{tab(t['acquisition'])}
<p>Budget exhaustion remains explicit. Failed models are unavailable, with no fallback. Means are withheld when a comparison lacks any expected model. Logical payloads exclude transport overhead and are per fitted condition, not nine independent target deployments. Original baseline fits and complementary optimization are reused.</p>
<h2>Primary paired comparison</h2>{svg}{tab(t['paired'][t['paired'].region=='target_unseen_anchor'])}
<p>Positive reductions and negative mean differences favor Complementary. Unseen anchor speeds are the two other speeds among 13, 21 and 29 m/s. They are observed by complementary peers but absent from the entire SameSpeed condition. These paired cases are not independent fleet replicates.</p>
<h2>All target-speed regions</h2>{tab(t['summary'])}
<p>Target-observed means the target's own speed, on held-out probes. Intermediate reference speeds are 17/25 m/s; outside-reference speeds are 9/33 m/s. These names refer to the fixed original speed grid; they do not imply interpolation within the uniform-speed training condition.</p>
<h2>Interpretation</h2><p>Credit complementary coverage when it improves target-unseen prediction beyond SameSpeed and NominalPrior, consistently across the pilot fleets and without unacceptable target-observed degradation. A gain only over LocalOE is insufficient. If SameSpeed is equally good, extra peer observations or regularization may explain the result without requiring complementary speeds.</p>
<p>Steering amplitude is matched, so yaw response amplitude and sensitivity can differ with speed; these differences are part of the speed intervention. Do not claim a separate geometric-coverage mechanism independent of excitation or signal-to-noise ratio. Common true relaxation lengths favor transfer. Three reused pilot fleets provide development evidence, not independent confirmation. No controller is tested.</p>
<p><a href="../docs/federated_relaxation_peers.html">Frozen protocol and commands</a></p></main></html>'''
    REPORT.parent.mkdir(parents=True,exist_ok=True)
    REPORT.write_text('\n'.join(line.rstrip() for line in text.splitlines())+'\n',encoding='utf-8',newline='\n')


def smoke():
    cfg,protocol=preflight();c=copy.deepcopy(cfg['fleets'][0]['clients'][0]);short=copy.deepcopy(cfg);short['duration_s']=.2
    a,_=base.generate(c,1301,'training',c['training_speed'],short)
    # Pure synthetic paired-record checks, never scientific source data/fits.
    other=next(v for v in protocol['same_speeds'] if v!=c['training_speed'])
    state=base.batch_predict(c['physical_parameters'],[a['steering']],[other],c['geometry_m'],short['sample_time'],short['integration_max_step'])[0,0]
    measured=state[:,1]+a['noise'];np.testing.assert_allclose(measured-state[:,1],a['noise'],atol=1e-16)
    result=continuation.smoke()
    return dict(status='smoke_passed',paired_noise_check=True,optimizer_smoke=result,scientific_fits_run=0)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['preflight','smoke','run','status','analyze','restore'])
    parser.add_argument('--workers',type=int,choices=[1,2,3],default=3)
    parser.add_argument('--seed',type=int,choices=[1301,1302,1303]);args=parser.parse_args()
    cfg,protocol=preflight(execution=args.command in ['run','restore'])
    seeds=[s for s in protocol['seeds'] if args.seed is None or s==args.seed]
    if args.command=='preflight':
        print(json.dumps(dict(status='preflight_passed',fleets=3,targets=27,peers_per_target=8,same_speed_fits=9,
            new_records=54,reused_records=27,new_local_multistart_attempts=162,new_prediction_cases=378,
            total_round_cap=30,new_baseline_fits=0,paired_peer_noise=True)));return
    if args.command=='smoke':
        with ProcessPoolExecutor(max_workers=1,mp_context=multiprocessing.get_context('spawn')) as pool:print(json.dumps(pool.submit(smoke).result()))
        return
    if args.command=='status':
        for seed in seeds:
            if path(seed).exists():
                p=base.storage.read_checkpoint(path(seed));validate(p,cfg,protocol,execution=False)
                print(json.dumps(dict(seed=seed,status=p['status'],records=len(p['records']),local_fits=len(p['local']),
                    conditions={v:dict(status=c['status'],round=c.get('optimization',{}).get('round')) for v,c in p['conditions'].items()},evaluated=len(p['evaluation']))))
            else:print(json.dumps(dict(seed=seed,status='not_started')))
        return
    RAW.mkdir(parents=True,exist_ok=True);lock=RAW/'run.lock'
    try:fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError:raise RuntimeError(f'Lock exists: {lock}; remove only after confirming no peer experiment process is active')
    os.write(fd,str(os.getpid()).encode());os.close(fd)
    try:
        if args.command=='analyze':analyze(cfg,protocol)
        elif args.command=='restore':restore(cfg,protocol,seeds)
        else:
            with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
                for result in as_completed([pool.submit(worker,s) for s in seeds]):print(json.dumps(result.result()),flush=True)
            print('Commit/push seed checkpoints before analyze. See docs/federated_relaxation_peers.html.')
    finally:lock.unlink()


if __name__=='__main__':main()
