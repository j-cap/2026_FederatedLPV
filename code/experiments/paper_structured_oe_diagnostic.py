"""Structured local OE on the frozen fixed-speed calibration pilot."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from multiprocessing import get_context
from pathlib import Path
import hashlib
import html
import json
import sys
import time
import numpy as np
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'code/src'),str(ROOT/'code/experiments')]
import paper_known_steering_diagnostic as known
from federated_lpv.structured_oe import fit_oe,propagate
from federated_lpv.trajectory_calibration import window_indices,array_hash,prediction_diagnostics
parent=known.parent
PREFIX='paper_structured_oe_diagnostic'
CONFIG=ROOT/f'code/config/{PREFIX}.json'
OUT=ROOT/'results/tables'
REPORT=ROOT/'report/structured_oe_diagnostic.html'
METHODS=['KnownOLS','AffineOLS','StructuredOE']


def preflight():
    settings=parent.load(CONFIG); _,cfg,prior=known.preflight()
    for p,sha in settings['source_sha256'].items():
        if parent.file_hash(ROOT/p)!=sha: raise ValueError(f'Frozen source changed: {p}')
    assert settings['seeds']==cfg['pilot_seeds'] and settings['profiles']==list(cfg['profiles'])
    assert settings['estimator']['state_noise_deg']==cfg['state_noise_deg']
    analysis=parent.load(known.OUT/f'{known.PREFIX}_analysis.json')
    assert analysis['provenance']==prior
    for p,sha in analysis['checkpoint_sha256'].items():
        if parent.file_hash(ROOT/p)!=sha: raise ValueError(f'Known-steering checkpoint changed: {p}')
    sources=[CONFIG,Path(__file__),ROOT/'code/src/federated_lpv/structured_oe.py']
    lock=dict(known_provenance=prior,known_checkpoints=analysis['checkpoint_sha256'],
              sources={p.relative_to(ROOT).as_posix():parent.file_hash(p) for p in sources})
    return settings,cfg,lock


def path_for(seed,profile): return OUT/f'{PREFIX}_seed{seed}_{profile}.json'


def read_result(path,lock):
    envelope=parent.load(path);r=envelope['payload']
    assert envelope['payload_sha256']==hashlib.sha256(parent.canonical(r)).hexdigest(),f'Corrupt {path}'
    assert r['provenance']==lock and r['status']=='complete' and len(r['cells'])==83,f'Stale {path}'
    assert path.name==path_for(r['seed'],r['profile']).name
    return r


def affine_prediction(target,offset,trace,cfg):
    # Task-faithful additional metric: training-fitted affine offset, no fitting
    # or re-centering on validation. Exact initial states match historical tests.
    _,ix=window_indices(cfg);scales=np.deg2rad(cfg['regressor_scales_deg'][:2])
    a=np.array(target[:4]).reshape(2,2);b=np.array(target[4:]);offset=np.array(offset)
    steps=round(cfg['prediction_horizon_s']/cfg['sample_time']); errors=[]
    for start in ix[::steps]:
        stop=min(start+steps,ix[-1]+1)
        with np.errstate(over='ignore',invalid='ignore'):
            x=propagate(a,trace['steering'][start:stop,None]*b+offset,trace['state'][start])[1:]
        if not np.isfinite(x).all() or np.max(abs(x/scales))>1e9:
            return None
        errors.append((x-trace['state'][start+1:stop+1])/scales)
    return float(np.sqrt(np.mean(np.concatenate(errors)**2)))


def evaluate_model(model,validation,center,reference,cfg,valid=True):
    if not valid:
        return dict(fit_valid=False,accuracy_pass=False,matrix_error=None,rollout_nrmse=None,
                    affine_rollout_nrmse=None)
    target=np.array(model['target']);delta=target-reference
    metrics=prediction_diagnostics(target,validation,center,cfg)
    err=float(np.linalg.norm(delta)/np.linalg.norm(reference))
    passed=bool(err<=cfg['pilot_sensor_matrix_error'] and metrics['rollout_nrmse'] is not None and
                metrics['rollout_nrmse']<=cfg['pilot_sensor_rollout_nrmse'] and
                metrics['rollout_nrmse']<metrics['persistence_nrmse'])
    return dict(**model,**metrics,fit_valid=True,matrix_error=err,entry_signed_error=delta.tolist(),
        accuracy_pass=passed,affine_rollout_nrmse=affine_prediction(target,model['offset'],validation,cfg))


def fit_cell(training,validation,frozen,reference,cfg,settings):
    _,ix=window_indices(cfg)
    y=training['observed_state'][ix[0]:ix[-1]+2].copy()
    u=training['steering'][ix].copy()
    before=(array_hash(y),array_hash(u))
    started=time.perf_counter()
    # No reference, simulator states or validation data enter the estimator.
    with threadpool_limits(limits=1): result=fit_oe(y,u,settings['estimator'])
    assert before==(array_hash(y),array_hash(u))
    _,center=known.known_observations(validation,cfg)
    np.testing.assert_array_equal(center,frozen['common_validation_center'])
    target=np.array(frozen['KnownSteering']['target']);train_center=np.array(frozen['known_center'])
    a=target[:4].reshape(2,2);b=target[4:]
    baseline=dict(target=target.tolist(),offset=(train_center[:2]-a@train_center[:2]-b*train_center[2]).tolist())
    models=dict(KnownOLS=evaluate_model(baseline,validation,center,reference,cfg),
                AffineOLS=evaluate_model(result['affine_ols'],validation,center,reference,cfg),
                StructuredOE=evaluate_model({k:result[k] for k in ['target','offset','initial_state','spectral_radius']}
                    if result['converged'] else {},validation,center,reference,cfg,result['converged']))
    for key in ['matrix_error','rollout_nrmse','persistence_nrmse']:
        np.testing.assert_allclose(models['KnownOLS'][key],frozen['KnownSteering'][key],atol=1e-10,rtol=1e-9)
    return dict(models=models,optimizer={k:v for k,v in result.items() if k not in ['affine_ols','target','offset','initial_state']},
                training_measurements_sha256=before[0],training_input_sha256=before[1],
                common_validation_center=center.tolist(),elapsed_s=time.perf_counter()-started)


def run_job(seed,profile):
    settings,cfg,lock=preflight();prior=lock['known_provenance']
    source=known.source_result(seed,profile,prior)
    frozen=known.read_result(known.path_for(seed,profile),prior)
    pairs={(v['client'],v['speed']):v for v in frozen['cells']};rows=[]
    for cell in source['cells']:
        training,validation=known.load_traces(source,cell)
        old=pairs[(cell['client'],cell['speed'])]
        result=fit_cell(training,validation,old,np.array(old['reference']),cfg,settings)
        result.update(client=cell['client'],speed=cell['speed'],reference=old['reference'],
            physical_pass=old['physical_pass'],settling_pass=old['settling_pass'],
            clean_matrix_error=cell['cases']['TrajectoryAllClean']['matrix_error'])
        rows.append(result)
    payload=dict(seed=seed,profile=profile,status='complete',provenance=lock,cells=rows)
    parent.save_json(path_for(seed,profile),dict(payload_sha256=hashlib.sha256(parent.canonical(payload)).hexdigest(),payload=payload))
    return dict(seed=seed,profile=profile,cells=len(rows),oe_converged=sum(v['optimizer']['converged'] for v in rows))


def analyze():
    settings,cfg,lock=preflight();rows=[];hashes={}
    for seed in settings['seeds']:
        for profile in settings['profiles']:
            p=path_for(seed,profile)
            if not p.exists() or not parent.tracked_unchanged(p):
                raise ValueError('Complete, commit and push all nine checkpoints before analysis')
            r=read_result(p,lock);hashes[p.relative_to(ROOT).as_posix()]=parent.file_hash(p)
            for cell in r['cells']:
                for method in METHODS:
                    v=cell['models'][method]
                    rows.append(dict(seed=seed,profile=profile,client=cell['client'],speed=cell['speed'],method=method,
                        fit_valid=v['fit_valid'],accuracy_pass=v['accuracy_pass'],matrix_error=v['matrix_error'],
                        rollout_nrmse=v['rollout_nrmse'],affine_rollout_nrmse=v['affine_rollout_nrmse'],
                        physical_pass=cell['physical_pass'],settling_pass=cell['settling_pass'],
                        clean_matrix_error=cell['clean_matrix_error'],
                        spectral_radius=v.get('spectral_radius'),oe_elapsed_s=cell['elapsed_s']))
    # Failed fits remain in denominators; numeric summaries explicitly report n_valid.
    import pandas as pd
    frame=pd.DataFrame(rows)
    def summarize(keys):
        return frame.groupby(keys,sort=False).agg(n=('fit_valid','size'),n_valid=('fit_valid','sum'),
            passing=('accuracy_pass','sum'),median_pct=('matrix_error',lambda x:100*x.median()),
            p95_pct=('matrix_error',lambda x:100*x.quantile(.95)),
            rollout_p95=('rollout_nrmse',lambda x:x.quantile(.95)),
            affine_rollout_p95=('affine_rollout_nrmse',lambda x:x.quantile(.95))).reset_index()
    summary=summarize(['profile','method']);speed=summarize(['profile','method','speed'])
    fleet=summarize(['profile','method','seed'])
    for suffix,data in [('cells',frame),('summary',summary),('by_speed',speed),('by_fleet',fleet)]:
        data.to_csv(OUT/f'{PREFIX}_{suffix}.csv',index=False)
    passing=[]
    for profile in settings['profiles']:
        z=frame[(frame.profile==profile)&(frame.method=='StructuredOE')]
        if z.accuracy_pass.mean()>=cfg['pilot_fraction'] and z.physical_pass.all() and z.settling_pass.all(): passing.append(profile)
    audit=dict(status='analysis_complete',provenance=lock,checkpoint_sha256=hashes,
        profiles_meeting_original_thresholds=passing,confirmation_unlocked=False,
        note='Three reused pilot fleets. No federation, controller or independent confirmation result. Numeric percentiles exclude failed fits; pass rates include them.')
    parent.save_json(OUT/f'{PREFIX}_analysis.json',audit)
    write_report(summary,speed,fleet,passing)
    print(summary.to_string(index=False));print(REPORT)


def write_report(summary,speed,fleet,passing):
    charts=[]
    for profile in ['gentle','standard','strong']:
        elements=[]
        for tick in [.1,1,10,100]:
            y=230-60*(np.log10(tick)+1)
            elements.append(f'<line x1="50" x2="710" y1="{y}" y2="{y}" stroke="#ddd"/><text x="4" y="{y}">{tick}%</text>')
        for method,color in zip(METHODS,['#ba4135','#bf8500','#2463b3']):
            z=speed[(speed.profile==profile)&(speed.method==method)].sort_values('speed')
            pts=' '.join(f'{50+(r.speed-10)*33},{230-60*(np.log10(max(.1,r.median_pct))+1)}' for r in z.itertuples() if np.isfinite(r.median_pct))
            elements.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="3"/>')
        for v in range(10,31,2):elements.append(f'<text x="{50+(v-10)*33}" y="258" text-anchor="middle">{v}</text>')
        charts.append(f'<h3>{profile.title()}</h3><svg viewBox="0 0 740 280">'+''.join(elements)+'</svg>')
    tables=[v.to_html(index=False,float_format=lambda x:f'{x:.4f}',border=0) for v in [summary,speed,fleet]]
    REPORT.write_text('''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Structured OE diagnostic</title><style>body{font:16px/1.55 system-ui;background:#f1f5f8;color:#203345}main{max-width:1150px;margin:auto;padding:30px}section{background:white;padding:24px;margin:18px 0;border-radius:10px}table{border-collapse:collapse;font-size:13px;width:100%}th,td{padding:7px;border-bottom:1px solid #ddd;text-align:right}.scroll{overflow:auto}svg{max-width:740px;width:100%}svg text{font:12px system-ui}</style><main><h1>Local structured output-error estimation</h1><p>Can an estimator designed for noisy states recover the local dynamics from the same fixed-speed driving maneuvers?</p><section><h2>Design</h2><p>KnownOLS is the frozen known-steering fit. AffineOLS adds a free constant offset. StructuredOE estimates the same affine dynamics by simulating the full training segment, with a fitted initial state and C=I. Only measured states and known applied steering enter estimation. Three fixed starts are selected by training loss. The configured measurement variances supply the weights.</p><p>The trajectories, state noise, speed coverage and validation maneuvers are unchanged. This is a local-estimator diagnostic, not a federated-learning result.</p></section><section><h2>Outcome</h2><p>OE profiles meeting the original thresholds: <strong>'''+html.escape(', '.join(passing) or 'none')+'''</strong>. Confirmation remains paused pending review.</p><p>Failed optimizations count as failed cells. Medians and percentiles describe valid fits only; n_valid must be read alongside them.</p><div class="scroll">'''+tables[0]+'''</div></section><section><h2>Speed dependence</h2><p>Median relative matrix error, logarithmic vertical axis; speed [m/s] horizontally. Red: KnownOLS. Gold: AffineOLS. Blue: StructuredOE.</p>'''+''.join(charts)+'''</section><section><h2>How to interpret the metrics</h2><p>matrix_error compares the six physical A/B coefficients with the simulator reference, which is never passed to the estimator. rollout_nrmse uses the historical common validation center and only A/B, allowing comparison with the frozen gates. affine_rollout_nrmse additionally evaluates each complete affine model using its training-derived offset without validation re-centering. Both use exact initial states for each one-second validation window and clean simulator states as evaluation targets; neither is an observer test.</p><p>Compare AffineOLS with KnownOLS to assess offset handling, and StructuredOE with AffineOLS to assess the estimator change. Better prediction alone does not establish matrix recovery. Inspect low speeds, convergence, and all three fleet summaries before proposing confirmation. These are reused development fleets, not independent evidence of generalization.</p></section><section><h2>By speed</h2><div class="scroll">'''+tables[1]+'''</div></section><section><h2>By fleet</h2><div class="scroll">'''+tables[2]+'''</div></section><section><h2>Next decision</h2><p>If OE recovers matrices reliably, freeze the revised local-estimation protocol before independent confirmation and reassess downstream regression assumptions. If it fails, use convergence, affine prediction and speed dependence to distinguish numerical problems from limited information or model mismatch. Do not alter thresholds after seeing these results.</p></section></main></html>''',encoding='utf-8',newline='\n')


def smoke_worker():
    settings,cfg,_=preflight()
    # Independent synthetic affine fixture: never fits a frozen pilot cell.
    rng=np.random.default_rng(813);n=700
    a=np.array([[.94,.018],[-.025,.97]]);b=np.array([.05,.12]);c=np.array([.0001,-.0002])
    u=.01*rng.normal(size=n);x=propagate(a,u[:,None]*b+c,np.array([.002,-.004]))
    y=x+rng.normal(size=x.shape)*np.deg2rad(settings['estimator']['state_noise_deg'])
    start=time.perf_counter()
    with threadpool_limits(limits=1): result=fit_oe(y,u,settings['estimator'])
    assert result['converged'] and result['jacobian_rank']==10
    print(json.dumps(dict(status='smoke_passed',elapsed_s=time.perf_counter()-start,
        attempts=result['attempts'],saved_scientific_results=False)),flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command',choices=['preflight','status','smoke','restore','run','analyze'])
    ap.add_argument('--workers',type=int,default=5);args=ap.parse_args()
    if args.workers<1:ap.error('workers must be positive')
    settings,cfg,lock=preflight();jobs=[(s,p) for s in settings['seeds'] for p in settings['profiles']]
    if args.command in ['preflight','status']:
        complete=present=0
        for s,p in jobs:
            if path_for(s,p).exists():read_result(path_for(s,p),lock);complete+=1
            source=known.source_result(s,p,lock['known_provenance'])
            present+=sum(parent.raw_path('pilot',s,p,c['client_index'],c['speed']).exists() for c in source['cells'])
        print(json.dumps(dict(status='preflight_passed',complete=complete,jobs=len(jobs),raw_present=present,raw_required=747),indent=2))
    elif args.command=='analyze':analyze()
    elif args.command=='smoke':
        with ProcessPoolExecutor(max_workers=1,mp_context=get_context('spawn')) as pool:pool.submit(smoke_worker).result()
    elif args.command=='restore':
        # Delegate to the original hash-exact reconstruction implementation.
        sys.argv=[known.__file__,'restore','--workers',str(args.workers)];known.main()
    else:
        pending=[]
        for s,p in jobs:
            if path_for(s,p).exists():read_result(path_for(s,p),lock);continue
            source=known.source_result(s,p,lock['known_provenance'])
            if any(not parent.raw_path('pilot',s,p,c['client_index'],c['speed']).exists() for c in source['cells']):
                raise FileNotFoundError('Original arrays missing; run restore first. Exact hashes are required.')
            pending.append((s,p))
        with ProcessPoolExecutor(max_workers=args.workers,mp_context=get_context('spawn')) as pool:
            futures=[pool.submit(run_job,*job) for job in pending]
            for f in as_completed(futures):print(json.dumps(f.result()),flush=True)
        print('Execution complete. Commit and push checkpoints before analyze.',flush=True)


if __name__=='__main__':main()
