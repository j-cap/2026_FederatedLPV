r"""Isolate removal of sideslip from physical CT identification.

Windows: python code\experiments\paper_physical_ct_yaw_only.py preflight
No new acquisition or observer design. Existing state-feedback control is an
optional diagnostic, not a sideslip-sensor-free vehicle workflow.
"""
from __future__ import annotations
import os
for name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[name]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import io
import json
import multiprocessing
from pathlib import Path
import subprocess
import sys
import tempfile
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'code/src'),str(ROOT/'code/experiments')]
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import paper_physical_ct_oe as old
from paper_physical_ct_coverage_resume import atomic_json
from federated_lpv.physical_ct_yaw_oe import fit_yaw_oe,validation_prediction
from federated_lpv.physical_ct_oe import discrete_matrices
from federated_lpv.oe_lqi_validation import design_nodes,simulate,interpolate
from federated_lpv.vehicle import VehicleParameters
from federated_lpv.control import augmented_tracking_matrices
from paper_oe_transient_benchmark import profile
from analyze_physical_ct_control_pilot import rk4_origin_matrices
PREFIX='paper_physical_ct_yaw_only'
CONFIG=ROOT/'code/config'/f'{PREFIX}.json'
OUT=ROOT/'results/tables'
REPORT=ROOT/'report/physical_ct_yaw_only.html'
METHODS=['FullStateCT','YawOnlyCT']


def path(seed,control=False):
    return OUT/f'{PREFIX}_{"control_" if control else ""}seed{seed}.json'


def preflight():
    cfg=old.load(CONFIG);prior=old.preflight('pilot')
    for name,sha in cfg['source_sha256'].items():
        if old.file_hash(ROOT/name)!=sha:raise ValueError(f'Frozen input changed: {name}')
    if cfg['seeds']!=prior['cohorts']['pilot']:raise ValueError('Unexpected cohort')
    reference=dict(cfg['estimator']);reference.pop('nuisance_rcond')
    if reference!=prior['estimator']:raise ValueError('Estimator settings changed beyond output selection')
    for seed in cfg['seeds']:
        source=old.sources(seed)[1];full=old.envelope(old.result_path(seed))
        old.validate_result(full,seed,prior,source)
        if full['status']!='execution_complete' or not all(c['fit']['converged'] for c in full['clients']):
            raise ValueError('Full-state comparison incomplete')
    return cfg


def provenance(seed,cfg,control=False):
    files=[CONFIG,Path(__file__),ROOT/'code/src/federated_lpv/physical_ct_yaw_oe.py',
        *old.source_paths(seed),old.result_path(seed)]
    if control:files.extend([path(seed),old.result_path(seed,True)])
    return dict(files={p.relative_to(ROOT).as_posix():old.file_hash(p) for p in files},
        frozen=cfg['source_sha256'],runtime=old.runtime())


def validate(p,seed,cfg,control=False):
    if p['seed']!=seed or p['provenance']!=provenance(seed,cfg,control):raise ValueError('Stale checkpoint; archive before protocol changes')
    if p['status'] not in ['running','execution_complete']:raise ValueError('Invalid status')
    if [c['index'] for c in p['clients']]!=list(range(len(p['clients']))) or len(p['clients'])>30:raise ValueError('Invalid client order')
    if p['status']=='execution_complete' and len(p['clients'])!=30:raise ValueError('Incomplete checkpoint')
    source=old.sources(seed)[1]
    for c in p['clients']:
        src=source['clients'][c['index']]
        if c['client']!=src['client'] or c['observed_speeds']!=src['speeds']:raise ValueError('Client/coverage mismatch')
        if not control:
            if [h['speed'] for h in c['training_hashes']]!=src['speeds']:raise ValueError('Training records mismatch')
            if c['fit']['converged'] and np.asarray(c['grid_targets']).shape!=(11,6):raise ValueError('Invalid model grid')
        else:
            expected={(m,v) for m in METHODS for v in [13,21,29]}
            if {(r['method'],r['speed']) for r in c['rows']}!=expected or len(c['rows'])!=6:raise ValueError('Invalid control cases')


def checkpoint(seed,cfg,control=False):
    p=old.envelope(path(seed,control)) if path(seed,control).exists() else dict(seed=seed,status='running',
        provenance=provenance(seed,cfg,control),clients=[],
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    validate(p,seed,cfg,control);return p


def segment(trace,speed,acfg):
    # Only the yaw channel crosses the estimator boundary. No beta-dependent
    # centering, initialization, regression or model selection is performed.
    _,ix=old.window_indices(acfg)
    return dict(speed=float(speed),measured_yaw=trace['observed_state'][ix[0]:ix[-1]+2,1].copy(),
                applied_input=trace['steering'][ix].copy())


def model_grid(parameters,grid,settings):
    out=[]
    for v in grid:
        a,b=discrete_matrices(parameters,v,settings['geometry_m'],settings['curvature_per_m'],settings['sample_time'])
        out.append(np.r_[a.ravel(),b])
    return np.array(out)


def extra_errors(prediction,reference,cfg):
    error=prediction-reference
    scale=np.deg2rad(cfg['diagnostic_scales_deg'])
    factors=np.r_[(scale[:2][None,:]/scale[:2][:,None]).ravel(),scale[2]/scale[:2]]
    denominator=(reference-np.r_[np.eye(2).ravel(),np.zeros(2)])*factors
    return dict(entry_absolute=np.abs(error).tolist(),
        scaled_increment=(np.linalg.norm(error*factors,axis=1)/np.linalg.norm(denominator,axis=1)).tolist())


def evaluate(seed,index,source,acquired,full,cfg,restore):
    start=time.perf_counter();acfg=old.load(ROOT/'code/config/paper_trajectory_calibration.json')
    train=[];holdout=[];hashes=[]
    for v in source['speeds']:
        cell=next(c for c in acquired['cells'] if c['client_index']==index and c['speed']==v)
        pair,_=old.traces(seed,cell,restore)
        s=segment(pair[0],v,acfg);train.append(s);holdout.append(pair[1])
        hashes.append(dict(speed=v,yaw=old.array_hash(s['measured_yaw']),u=old.array_hash(s['applied_input']),
            original_full_measurements=cell['training_measurements_sha256']))
    fit=fit_yaw_oe(train,cfg['estimator'])
    result=dict(index=index,client=source['client'],observed_speeds=source['speeds'],training_hashes=hashes,
                fit=fit,validation=[],comparisons={})
    grid=np.arange(10,31,2);truth=np.array(source['reference'])
    parameters={'FullStateCT':full['fit']['parameters']}
    if fit['converged']:parameters['YawOnlyCT']=fit['parameters']
    for method,theta in parameters.items():
        prediction=model_grid(theta,grid,cfg['estimator'])
        result['comparisons'][method]=dict(errors=old.dynamic_metrics(prediction,truth,cfg),
            **extra_errors(prediction,truth,cfg))
        if method=='YawOnlyCT':result['grid_targets']=prediction.tolist()
        for v,trace in zip(source['speeds'],holdout):
            s=segment(trace,v,acfg)
            entry=dict(method=method,speed=v,status='complete',
                yaw_sha256=old.array_hash(s['measured_yaw']),input_sha256=old.array_hash(s['applied_input']))
            try:
                with np.errstate(over='raise',invalid='raise'):
                    pred,nuisance=validation_prediction(theta,s,cfg['estimator'],cfg['validation_warmup_transitions'])
                cut=cfg['validation_warmup_transitions']+1
                if not np.isfinite(pred).all():raise ValueError('Nonfinite prediction')
                residual=pred[cut:]-s['measured_yaw'][cut:]
                # True yaw is read only after predictions, for an evaluation-only
                # noise-free score. Never read true beta for initialization.
                _,ix=old.window_indices(acfg)
                exact=trace['state'][ix[0]:ix[-1]+2,1]
                clean=pred[cut:]-exact[cut:]
                entry.update(scored_samples=len(residual),nuisance_prefix_only=nuisance,
                    measured_yaw_rmse_deg_s=float(np.rad2deg(np.sqrt(np.mean(residual**2)))),
                    true_yaw_rmse_deg_s=float(np.rad2deg(np.sqrt(np.mean(clean**2)))))
            except (ValueError,FloatingPointError,np.linalg.LinAlgError) as exc:
                entry.update(status='prediction_failed',reason=str(exc),measured_yaw_rmse_deg_s=None,true_yaw_rmse_deg_s=None)
            result['validation'].append(entry)
    # Reference parameters used only for post-fit evaluation.
    if fit['converged']:
        p=source['parameters'];theta0=[p['front_stiffness']/p['mass'],p['rear_stiffness']/p['mass'],p['mass']/p['yaw_inertia'],.9]
        result['reference_parameters']=theta0
        result['relative_parameter_error']=(np.array(fit['parameters'])/theta0-1).tolist()
    result['elapsed_s']=time.perf_counter()-start
    return result


def run_fleet(seed,restore=False):
    cfg=old.load(CONFIG);p=checkpoint(seed,cfg);acquired,src=old.sources(seed)
    full=old.envelope(old.result_path(seed))
    for i in range(len(p['clients']),30):
        c=evaluate(seed,i,src['clients'][i],acquired,full['clients'][i],cfg,restore)
        p['clients'].append(c);p['status']='execution_complete' if len(p['clients'])==30 else 'running'
        atomic_json(path(seed),p)
        print(json.dumps(dict(seed=seed,saved_clients=len(p['clients']),converged=c['fit']['converged'],elapsed_s=c['elapsed_s'])),flush=True)
    validate(p,seed,cfg)
    return dict(seed=seed,status=p['status'],converged=sum(c['fit']['converged'] for c in p['clients']))


def control_fleet(seed):
    cfg=old.load(CONFIG);fitted=old.envelope(path(seed));validate(fitted,seed,cfg)
    if fitted['status']!='execution_complete':raise ValueError('Finish and commit identification first')
    p=checkpoint(seed,cfg,True);source=old.sources(seed)[1]
    prior=old.envelope(old.result_path(seed,True));cc=old.load(old.storage.CONFIG)
    maneuver=old.load(ROOT/'code/config/paper_oe_transient_benchmark.json')
    for i in range(len(p['clients']),30):
        fit=fitted['clients'][i];original=prior['clients'][i]
        schedule=design_nodes(np.array(fit['grid_targets']),cc) if fit['fit']['converged'] else dict(status='identification_failed',nodes=[])
        rows=[dict(method='FullStateCT',speed=r['speed'],outcome=r['outcome']) for r in original['rows'] if r['method']==old.METHOD]
        for v in maneuver['speeds_mps']:
            if not fit['fit']['converged']:outcome=dict(status='identification_failed',feasible=False,metrics=None,completed_steps=0)
            else:outcome,_=simulate(VehicleParameters(**source['clients'][i]['parameters']),schedule,profile(maneuver,v),cc)
            rows.append(dict(method='YawOnlyCT',speed=v,outcome=outcome))
        p['clients'].append(dict(index=i,client=fit['client'],observed_speeds=fit['observed_speeds'],rows=rows,
            schedules={'FullStateCT':original['schedules'][old.METHOD],'YawOnlyCT':schedule}))
        p['status']='execution_complete' if len(p['clients'])==30 else 'running';atomic_json(path(seed,True),p)
        print(json.dumps(dict(seed=seed,saved_control_clients=len(p['clients']))),flush=True)
    validate(p,seed,cfg,True);return dict(seed=seed,status=p['status'],new_rollouts=90)


def worker(command,seed,restore=False):
    with threadpool_limits(limits=1):
        if command=='restore':return old.restore_fleet(seed)
        if command=='control':return control_fleet(seed)
        return run_fleet(seed,restore)


def aggregate(frame,metrics):
    clients=frame.groupby(['seed','client','method'])[metrics].agg(old.strict_mean).reset_index()
    fleets=clients.groupby(['seed','method'])[metrics].agg(old.strict_mean).reset_index()
    overall=fleets.groupby('method')[metrics].agg(old.strict_mean).reset_index()
    return clients,fleets,overall


def analyze(cfg):
    fits=[];models=[];validation=[];controls=[];hashes={}
    control_present=[path(s,True).exists() for s in cfg['seeds']]
    if any(control_present) and not all(control_present):raise ValueError('Partial controls; finish them before combined analysis')
    for seed in cfg['seeds']:
        p=old.envelope(path(seed));validate(p,seed,cfg)
        if p['status']!='execution_complete':raise ValueError('Finish all pilot fits')
        hashes[path(seed).relative_to(ROOT).as_posix()]=old.file_hash(path(seed))
        for c in p['clients']:
            f=c['fit'];sv=np.array(f.get('information',{}).get('log_parameter_singular_values',[]))
            fits.append(dict(seed=seed,client=c['client'],converged=f['converged'],
                relative_rank=int(sum(sv>sv[0]*1e-8)) if len(sv) else 0,
                near_bound=any(f.get('near_bound',[])),mu=f.get('parameters',[None]*4)[3],
                elapsed_s=c['elapsed_s']))
            for method in METHODS:
                result=c['comparisons'].get(method)
                for j,v in enumerate(range(10,31,2)):
                    row=dict(seed=seed,client=c['client'],method=method,speed=v,seen=v in c['observed_speeds'])
                    for key in ['matrix','increment','a','b','pole_rms_absolute']:
                        row[key]=result['errors'][key][j] if result else np.nan
                    row['scaled_increment']=result['scaled_increment'][j] if result else np.nan
                    for k,name in enumerate(['A11','A12','A21','A22','B1','B2']):row[name+'_absolute']=result['entry_absolute'][j][k] if result else np.nan
                    models.append(row)
                scores=[v for v in c['validation'] if v['method']==method]
                if not scores:scores=[dict(speed=v,status='identification_failed',measured_yaw_rmse_deg_s=None,true_yaw_rmse_deg_s=None) for v in c['observed_speeds']]
                for v in scores:validation.append(dict(seed=seed,client=c['client'],method=method,speed=v['speed'],status=v['status'],
                    measured_yaw_rmse_deg_s=v['measured_yaw_rmse_deg_s'],true_yaw_rmse_deg_s=v['true_yaw_rmse_deg_s']))
        if all(control_present):
            cp=old.envelope(path(seed,True));validate(cp,seed,cfg,True)
            if cp['status']!='execution_complete':raise ValueError('Partial controls; finish them first')
            hashes[path(seed,True).relative_to(ROOT).as_posix()]=old.file_hash(path(seed,True))
            cc=old.load(old.storage.CONFIG);source=old.sources(seed)[1]
            for c in cp['clients']:
                for r in c['rows']:
                    o=r['outcome'];m=o['metrics'] or {};schedule=c['schedules'][r['method']];rho=np.nan
                    if schedule['status']=='designed':
                        gain,_=interpolate(schedule,cc['speed_grid'],[r['speed']])
                        a,b=rk4_origin_matrices(VehicleParameters(**source['clients'][c['index']]['parameters']),r['speed'],cc['sample_time'])
                        aa,bb=augmented_tracking_matrices(a,b,cc['sample_time']);rho=float(max(abs(np.linalg.eigvals(aa-bb@gain))))
                    controls.append(dict(seed=seed,client=c['client'],method=r['method'],speed=r['speed'],status=o['status'],feasible=o['feasible'],
                        tracking_rmse_deg_s=m.get('tracking_rmse_deg_s',np.nan),limiter_fraction=m.get('limiter_fraction',np.nan),
                        steering_rate_rms_deg_s=m.get('steering_rate_rms_deg_s',np.nan),origin_rho=rho))
    fit=pd.DataFrame(fits);model=pd.DataFrame(models);val=pd.DataFrame(validation)
    metrics=['matrix','increment','scaled_increment','a','b','pole_rms_absolute',*[k+'_absolute' for k in ['A11','A12','A21','A22','B1','B2']]]
    clients,fleets,summary=aggregate(model[~model.seen],metrics)
    vc,vf,vs=aggregate(val,['measured_yaw_rmse_deg_s','true_yaw_rmse_deg_s'])
    counts=val.groupby('method').agg(segments=('status','size'),
        complete=('status',lambda x:int((x=='complete').sum()))).reset_index()
    vs=vs.merge(counts,on='method')
    frames=dict(fits=fit,grid_errors=model,clients=clients,fleets=fleets,summary=summary,
        validation=val,validation_clients=vc,validation_fleets=vf,validation_summary=vs)
    cs=None
    if controls:
        cf=pd.DataFrame(controls);cu,fs,cs=aggregate(cf,['tracking_rmse_deg_s','limiter_fraction','steering_rate_rms_deg_s'])
        counts=cf.groupby('method').agg(cases=('status','size'),complete=('status',lambda x:int((x=='complete').sum())),
            feasible=('feasible','sum'),limited=('limiter_fraction',lambda x:int((x>0).sum())),
            origin_unstable=('origin_rho',lambda x:int((x>1+1e-9).sum()))).reset_index()
        cs=cs.merge(counts,on='method');frames.update(control_cases=cf,control_clients=cu,control_fleets=fs,control_summary=cs)
    for suffix,frame in frames.items():frame.to_csv(OUT/f'{PREFIX}_{suffix}.csv',index=False)
    audit=dict(status='analysis_complete',fit_count=len(fit),converged=int(fit.converged.sum()),control_cases=len(controls),
        checkpoint_sha256=hashes,analysis_sha256=old.file_hash(__file__),protocol_sha256=old.file_hash(CONFIG),
        scope=cfg['scope'],aggregation='Strict means: speeds within vehicle, vehicles within fleet, fleets equally. Failures invalidate complete means; no complete-case filtering.')
    old.acquisition.parent.save_json(OUT/f'{PREFIX}_analysis.json',audit)
    report(clients,vc,summary,vs,cs,fit,audit)
    print(summary.to_string(index=False));print(vs.to_string(index=False));print(REPORT)


def report(clients,validation,summary,vs,cs,fits,audit):
    import matplotlib
    matplotlib.use('Agg');matplotlib.rcParams['svg.hashsalt']=PREFIX
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    for ax,frame,metric,label in [(axes[0],clients,'increment','Unseen increment error [%]'),
                                 (axes[1],validation,'true_yaw_rmse_deg_s','Held-out yaw RMSE [deg/s; simulator reference]')]:
        w=frame.pivot(index=['seed','client'],columns='method',values=metric)
        factor=100 if metric=='increment' else 1
        x=w.FullStateCT*factor;y=w.YawOnlyCT*factor;mask=np.isfinite(x)&np.isfinite(y)
        ax.scatter(x[mask],y[mask],s=18,alpha=.6);end=max([*x[mask],*y[mask],1e-6])*1.05
        ax.plot([0,end],[0,end],'--',color='gray');ax.set(xlabel='Full-state-trained CT',ylabel='Yaw-only-trained CT',title=label);ax.grid(alpha=.2)
    buf=io.StringIO();fig.savefig(buf,format='svg',metadata={'Date':None});plt.close(fig)
    svg=buf.getvalue();svg=svg[svg.index('<svg'):]
    def table(df):return '<div class="scroll">'+df.to_html(index=False,border=0,float_format=lambda x:f'{x:.6g}',na_rep='unavailable')+'</div>'
    text=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Yaw-only CT identification diagnostic</title>
<style>body{{font:16px/1.6 system-ui;background:#eef3f6;color:#193142}}main{{max-width:1250px;margin:auto;padding:30px}}section{{background:white;border-radius:10px;padding:24px;margin:20px 0}}.scroll{{overflow:auto}}table{{border-collapse:collapse;font-size:13px;white-space:nowrap}}td,th{{padding:8px;border-bottom:1px solid #ddd}}svg{{width:100%;height:auto}}</style><main><h1>Does local identification depend on observing sideslip?</h1>
<section><h2>Scope and completeness</h2><p>{audit['converged']}/{audit['fit_count']} yaw-only fits converged. Original 90 vehicles and 249 speed segments; one joint model per vehicle. Same physical structure, bounds, four starts, data, coverage and optimizer settings. Existing full-state-trained CT fits are reused. No new acquisition, fleet heterogeneity or sharing method. Acquisition used noisy sideslip feedback; optional control uses true state feedback. This experiment removes sideslip only at the estimator boundary.</p><p>The latent initial sideslip is an affine-coordinate gauge. Free offsets absorb its constant shift. The nuisance projection uses three columns instead of four and checks numerical rank. It does not assume true initial sideslip is zero and does not supply a uniquely identified sideslip offset.</p></section>
<section><h2>Paired comparisons</h2>{svg}<p>Points above the diagonal favor full-state training. Failed/nonfinite values are omitted from plots only. All planned vehicles remain in strict summary means; any missing contributor makes a complete mean unavailable.</p></section>
<section><h2>Model extrapolation at common unseen speeds</h2>{table(summary)}<p>Matrix/increment/A/B/scaled-increment columns are ratios, not percentages. Increment normalization removes the identity contribution. Scaled increment also uses fixed physical scales β=0.25°, r=2°/s, steering=0.3°. These are the pre-existing acquisition regressor scales, not fitted to the outcomes. Entrywise columns are mean absolute errors in the stated physical-state realization. Pole error is absolute. Physical-model identifiability must still be inspected before treating accurate yaw prediction as recovery of all matrix entries.</p></section>
<section><h2>Independent maneuver prediction</h2>{table(vs)}<p>For both methods, dynamics are fixed at their training estimates. Only affine nuisance quantities are estimated from the first 2 s (201 yaw samples) of the separate validation fitting window. The following 18 s (1,800 samples) are predicted without measurement updates and scored. No suffix measurements affect centering or initialization. True yaw is used solely for an additional noise-free evaluation score; measured-yaw RMSE is also reported. No true sideslip enters prediction.</p><p>These validation maneuvers occur at the observed identification speeds. Extrapolation to unobserved speeds is assessed by model diagnostics and optional control, not by claiming an unseen-speed validation trajectory.</p></section>
<section><h2>Optional unchanged state-feedback control</h2>{table(cs) if cs is not None else '<p>Not run. Identification can be reviewed first.</p>'}<p>Common LQI rules, limits and 13/21/29 m/s maneuver. Full-state CT rollouts are reused, yaw-only CT models get 270 new rollouts. True state feedback isolates learned-model quality; no observer is introduced. Feasibility only checks bounds. Limiter activity, steering-rate RMS and the true-plant origin spectral radius are reported separately; local Jacobian stability is not a nonlinear guarantee.</p></section>
<section><h2>Convergence and sensitivity</h2>{table(fits)}<p>Rank uses a relative singular-value threshold of 10⁻⁸. Saved checkpoints include all starts, training losses, bound diagnostics and per-speed sensitivities. Validation and simulator truth never select a start. Inspect conditioning, parameter spread and failures alongside prediction; neither convergence nor full numerical rank alone establishes reliable physical recovery.</p></section>
<p>Reproduce with <code>python code/experiments/paper_physical_ct_yaw_only.py analyze</code>. Protocol and Windows commands: <a href="../docs/physical_ct_yaw_only.html">experiment guide</a>.</p></main></html>'''
    REPORT.write_text('\n'.join(line.rstrip() for line in text.splitlines())+'\n',encoding='utf-8',newline='\n')


def smoke_worker():
    cfg=old.load(CONFIG);rng=np.random.default_rng(63127);theta=np.array([47.,59.,.56,1.02]);segments=[]
    for v in [12.,22.,28.]:
        a,b=discrete_matrices(theta,v,[1.2,1.6],.005,.01)
        u=.012+.008*rng.normal(size=500)
        x=old.propagate(a,u[:,None]*b+[.00013,-.00021],[.002,.09])
        segments.append(dict(speed=v,measured_yaw=x[:,1],applied_input=u))
    with threadpool_limits(limits=1):fit=fit_yaw_oe(segments,cfg['estimator'])
    if not fit['converged'] or fit['weighted_mse']>1e-10:raise AssertionError('Synthetic yaw fit failed')
    np.testing.assert_allclose(fit['parameters'],theta,rtol=2e-4)
    with tempfile.TemporaryDirectory() as directory:
        p=Path(directory)/'checkpoint.json';atomic_json(p,dict(label='β μ',fit=fit))
        if old.envelope(p)['label']!='β μ':raise AssertionError('Checkpoint round-trip failed')
    return dict(status='yaw_only_smoke_passed',spawn=True,synthetic_only=True,new_experimental_results=0)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['preflight','smoke','restore','run','status','analyze','control'])
    parser.add_argument('--workers',type=int,default=3);parser.add_argument('--restore-missing',action='store_true')
    args=parser.parse_args()
    if args.workers<1:parser.error('workers must be positive')
    cfg=preflight()
    if args.command=='preflight':
        missing=sum(not old.original_raw_path(s,c).exists() for s in cfg['seeds'] for c in old.sources(s)[0]['cells'])
        print(json.dumps(dict(status='yaw_only_preflight_passed',fleets=3,new_joint_fits=90,reused_full_state_fits=90,
            original_speed_segments=249,missing_raw_cells=missing,new_acquisition=0,optional_new_control_rollouts=270)))
    elif args.command=='status':
        for seed in cfg['seeds']:
            for control in [False,True]:
                if path(seed,control).exists():
                    p=old.envelope(path(seed,control));validate(p,seed,cfg,control)
                    print(json.dumps(dict(seed=seed,stage='control' if control else 'identification',status=p['status'],clients=len(p['clients']))))
    elif args.command=='analyze':analyze(cfg)
    elif args.command=='smoke':
        with ProcessPoolExecutor(max_workers=1,mp_context=multiprocessing.get_context('spawn')) as pool:
            print(json.dumps(pool.submit(smoke_worker).result()))
    else:
        with old.execution_lock(PREFIX,args.command),old.execution_lock('pilot','identification'),ProcessPoolExecutor(
            max_workers=min(args.workers,3),mp_context=multiprocessing.get_context('spawn')) as pool:
            for f in as_completed([pool.submit(worker,args.command,s,args.restore_missing) for s in cfg['seeds']]):print(json.dumps(f.result()),flush=True)


if __name__=='__main__':main()
