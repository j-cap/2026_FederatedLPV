"""Fixed nominal LTI controller: closed-loop identification and OE-start diagnostic."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from multiprocessing import get_context
from dataclasses import asdict
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
import paper_structured_oe_diagnostic as oe
from federated_lpv.fixed_lti_identification import design_fixed_controller,generate_fixed_lqi,add_nominal_start
from federated_lpv.vehicle import VehicleParameters,family_centers
from federated_lpv.trajectory_calibration import array_hash,window_indices,physical_diagnostics,observations,fit_records
parent=oe.parent
PREFIX='paper_fixed_lti_identification'
CONFIG=ROOT/f'code/config/{PREFIX}.json'
OUT=ROOT/'results/tables'
DATA=ROOT/'results/data'/PREFIX
REPORT=ROOT/'report/fixed_lti_identification.html'
METHODS=['OpenOE3','OpenOE4','ClosedOE3','ClosedOE4','ClosedCleanOLS']


def preflight():
    protocol=parent.load(CONFIG);settings,cfg,prior=oe.preflight()
    assert protocol['seeds']==settings['seeds'] and protocol['profile']=='gentle'
    for path,sha in protocol['source_sha256'].items():
        if parent.file_hash(ROOT/path)!=sha:raise ValueError(f'Frozen source changed: {path}')
    analysis=parent.load(oe.OUT/f'{oe.PREFIX}_analysis.json')
    assert analysis['provenance']==prior
    for path,sha in analysis['checkpoint_sha256'].items():
        if parent.file_hash(ROOT/path)!=sha:raise ValueError(f'OE checkpoint changed: {path}')
    designed=design_fixed_controller(protocol['controller'],cfg['sample_time'])
    controller=protocol['controller_frozen']
    for key in ['a','b','gain','prefilter','design_speed_mps']:
        np.testing.assert_allclose(controller[key],designed[key],atol=1e-10,rtol=1e-9)
    files=[CONFIG,Path(__file__),ROOT/'code/src/federated_lpv/fixed_lti_identification.py']
    lock=dict(oe_provenance=prior,oe_checkpoints=analysis['checkpoint_sha256'],controller=controller,
              sources={p.relative_to(ROOT).as_posix():parent.file_hash(p) for p in files})
    return protocol,settings,cfg,lock


def path_for(seed):return OUT/f'{PREFIX}_seed{seed}.json'
def raw_path(seed,index,speed):return DATA/f'seed{seed}_client{index}_speed{int(speed)}.npz'


def read_result(path,lock):
    envelope=parent.load(path);r=envelope['payload']
    assert envelope['payload_sha256']==hashlib.sha256(parent.canonical(r)).hexdigest(),f'Corrupt {path}'
    assert r['status']=='complete' and r['provenance']==lock and len(r['cells'])==83,f'Stale/incomplete {path}'
    assert path.name==path_for(r['seed']).name
    return r


def inputs(trace,cfg):
    _,ix=window_indices(cfg)
    return trace['observed_state'][ix[0]:ix[-1]+2].copy(),trace['steering'][ix].copy()


def score(fit,validation,reference,cfg):
    _,center=oe.known.known_observations(validation,cfg)
    model={k:fit[k] for k in ['target','offset','initial_state','spectral_radius'] if k in fit}
    return oe.evaluate_model(model,validation,center,reference,cfg,fit['converged'])


def tracking(trace,cfg):
    _,ix=window_indices(cfg)
    return dict(yaw_rmse_deg_s=float(np.rad2deg(np.sqrt(np.mean((trace['state'][ix,1]-trace['yaw_reference'][ix])**2)))),
        yaw_reference_peak_deg_s=float(np.rad2deg(max(abs(trace['yaw_reference'][ix])))),
        limiter_fraction=float(np.mean(trace['limited'][ix])),
        full_run_limiter_fraction=float(np.mean(trace['limited'])))


def preview(trace):
    ix=np.arange(0,len(trace['steering']),10)
    return dict(time=trace['time'][ix].tolist(),beta_deg=np.rad2deg(trace['state'][ix,0]).tolist(),
        yaw_deg_s=np.rad2deg(trace['state'][ix,1]).tolist(),yaw_reference_deg_s=np.rad2deg(trace['yaw_reference'][ix]).tolist(),
        steering_deg=np.rad2deg(trace['steering'][ix]).tolist(),xy=trace['position'][ix].tolist())


def run_job(seed):
    protocol,settings,cfg,lock=preflight();controller=lock['controller']
    source=oe.known.source_result(seed,'gentle',lock['oe_provenance']['known_provenance'])
    previous=oe.read_result(oe.path_for(seed,'gentle'),lock['oe_provenance'])
    old={(v['client'],v['speed']):v for v in previous['cells']}
    rows=[];previews={};start=time.perf_counter()
    with threadpool_limits(limits=1):
        for cell in source['cells']:
            saved=old[(cell['client'],cell['speed'])];reference=np.array(saved['reference'])
            training,validation=oe.known.load_traces(source,cell)
            y,u=inputs(training,cfg)
            assert array_hash(y)==saved['training_measurements_sha256'] and array_hash(u)==saved['training_input_sha256']
            _,open4=add_nominal_start(y,u,settings['estimator'],controller,saved['optimizer'])
            models=dict(OpenOE3=saved['models']['StructuredOE'],OpenOE4=score(open4,validation,reference,cfg))
            parameters=VehicleParameters(**source['clients'][cell['client_index']]['parameters'])
            acqseed=parent.acquisition_seed(seed,cell['client_index'],cell['speed'])
            row=dict(client=cell['client'],client_index=cell['client_index'],speed=cell['speed'],
                acquisition_seed=acqseed,parameters=asdict(parameters),reference=reference.tolist(),
                models=models,open_optimizer=open4,open_physical_pass=saved['physical_pass'],
                open_settling_pass=saved['settling_pass'])
            try:
                ct=generate_fixed_lqi(parameters,cell['speed'],acqseed,cfg,protocol,controller)
                cv=generate_fixed_lqi(parameters,cell['speed'],acqseed,cfg,protocol,controller,True)
            except FloatingPointError as exc:
                row.update(status='simulation_failed',reason=str(exc),physical_pass=False,settling_pass=False)
                for method in ['ClosedOE3','ClosedOE4','ClosedCleanOLS']:
                    models[method]=dict(fit_valid=False,accuracy_pass=False,matrix_error=None,rollout_nrmse=None,affine_rollout_nrmse=None)
                rows.append(row);continue
            arrays=parent.pack_traces(ct,cv)
            row['array_sha256']={k:array_hash(v) for k,v in arrays.items()}
            parent.write_raw(raw_path(seed,cell['client_index'],cell['speed']),arrays)
            cy,cu=inputs(ct,cfg)
            closed3,closed4=add_nominal_start(cy,cu,settings['estimator'],controller)
            models.update(ClosedOE3=score(closed3,cv,reference,cfg),ClosedOE4=score(closed4,cv,reference,cfg))
            records,center=observations(ct,cfg,'clean');target,diag=fit_records(records,center,cfg)
            aa=target[:4].reshape(2,2);bb=target[4:]
            clean=dict(target=target.tolist(),offset=(center[:2]-aa@center[:2]-bb*center[2]).tolist(),converged=diag['fit_valid'])
            models['ClosedCleanOLS']=score(clean,cv,reference,cfg)
            physical=[physical_diagnostics(t,cell['speed'],cfg) for t in [ct,cv]]
            row.update(status='simulated',closed_optimizer=closed4,closed_three_start_optimizer=closed3,
                training=physical[0],validation=physical[1],tracking_train=tracking(ct,cfg),tracking_validation=tracking(cv,cfg),
                physical_pass=all(p['physical_pass'] for p in physical),settling_pass=all(p['settling_pass'] for p in physical),
                training_measurements_sha256=array_hash(cy),training_input_sha256=array_hash(cu))
            if seed==protocol['seeds'][0] and cell['speed'] in [10.,20.,30.] and str(cell['speed']) not in previews:
                previews[str(cell['speed'])]=dict(client=cell['client'],training=preview(ct),validation=preview(cv))
            rows.append(row)
    payload=dict(status='complete',seed=seed,provenance=lock,cells=rows,previews=previews,elapsed_s=time.perf_counter()-start)
    parent.save_json(path_for(seed),dict(payload_sha256=hashlib.sha256(parent.canonical(payload)).hexdigest(),payload=payload))
    return dict(seed=seed,cells=len(rows),physical_pass=sum(c['physical_pass'] for c in rows),
                closed_oe4_pass=sum(c['models']['ClosedOE4']['accuracy_pass'] for c in rows),elapsed_s=payload['elapsed_s'])


def restore_new_job(seed):
    protocol,settings,cfg,lock=preflight();r=read_result(path_for(seed),lock);restored=0
    for cell in r['cells']:
        if cell['status']!='simulated':continue
        path=raw_path(seed,cell['client_index'],cell['speed'])
        if path.exists():
            with np.load(path,allow_pickle=False) as data:arrays={k:data[k] for k in data.files}
        else:
            traces=[generate_fixed_lqi(VehicleParameters(**cell['parameters']),cell['speed'],cell['acquisition_seed'],cfg,protocol,lock['controller'],v) for v in [False,True]]
            arrays=parent.pack_traces(*traces)
        assert set(arrays)==set(cell['array_sha256'])
        assert all(array_hash(arrays[k])==v for k,v in cell['array_sha256'].items()),f'Reconstruction mismatch: {path}'
        if not path.exists():parent.write_raw(path,arrays);restored+=1
    return dict(seed=seed,restored=restored)


def table(headers,rows):
    def display(x):return f'{x:.4f}' if isinstance(x,float) else str(x)
    return '<div class="scroll"><table><thead><tr>'+''.join(f'<th>{html.escape(str(h))}</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join(f'<td>{html.escape(display(x))}</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'


def chart(t,series,ylabel):
    colors=['#1767a8','#bd4c3c'];allv=np.concatenate([np.array(v) for _,v in series]);lo=min(allv);hi=max(allv)
    span=max(hi-lo,1e-5);lo-=.1*span;hi+=.1*span
    xx=lambda v:55+650*v/max(t); yy=lambda v:215-180*(v-lo)/(hi-lo)
    parts=['<svg viewBox="0 0 740 255" role="img" aria-label="'+html.escape(ylabel)+'">']
    for val in np.linspace(lo,hi,4):parts.append(f'<line x1="55" x2="705" y1="{yy(val)}" y2="{yy(val)}" stroke="#ddd"/><text x="2" y="{yy(val)+4}">{val:.2f}</text>')
    for (label,values),color in zip(series,colors):
        pts=' '.join(f'{xx(a):.2f},{yy(b):.2f}' for a,b in zip(t,values))
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2"/>')
    for sec in [0,10,20,24,30,40]:parts.append(f'<text x="{xx(sec)}" y="238" text-anchor="middle">{sec}</text>')
    parts.append('</svg>')
    return '<p>'+html.escape(ylabel)+' · '+html.escape(' / '.join(n for n,_ in series))+'</p>'+''.join(parts)


def analyze():
    protocol,settings,cfg,lock=preflight();results=[];hashes={}
    for seed in protocol['seeds']:
        p=path_for(seed)
        if not p.exists() or not parent.tracked_unchanged(p):raise ValueError('Complete, commit and push all three checkpoints before analysis.')
        results.append(read_result(p,lock));hashes[p.relative_to(ROOT).as_posix()]=parent.file_hash(p)
    cells=[dict(c,seed=r['seed']) for r in results for c in r['cells']]
    rows=[];summaries=[]
    for method in METHODS:
        valid=[c['models'][method] for c in cells if c['models'][method]['fit_valid']]
        errs=[v['matrix_error']*100 for v in valid]
        q=lambda field:float(np.percentile([v[field] for v in valid if v[field] is not None],95)) if any(v[field] is not None for v in valid) else None
        summary=dict(method=method,total=len(cells),valid=len(valid),passing=sum(c['models'][method]['accuracy_pass'] for c in cells),
            median_pct=float(np.median(errs)) if errs else None,p95_pct=float(np.percentile(errs,95)) if errs else None,
            canonical_rollout_p95=q('rollout_nrmse'),affine_rollout_p95=q('affine_rollout_nrmse'))
        summaries.append(summary);rows.append([method,summary['valid'],summary['passing'],summary['median_pct'],summary['p95_pct'],summary['affine_rollout_p95']])
    primary=next(s for s in summaries if s['method']=='ClosedOE4')
    feasible=all(c['physical_pass'] and c['settling_pass'] for c in cells)
    gate=bool(primary['passing']/len(cells)>=protocol['decision']['minimum_accuracy_fraction'] and feasible)
    starts=[c['closed_optimizer']['attempts'][-1] for c in cells if c['status']=='simulated']
    diagnosis=dict(status='analysis_complete',provenance=lock,checkpoint_sha256=hashes,summaries=summaries,
        closed_physical_passing=sum(c['physical_pass'] for c in cells),closed_settling_passing=sum(c['settling_pass'] for c in cells),
        closed_nominal_start_converged=sum(a['converged'] for a in starts),
        closed_nominal_start_selected=sum(c['closed_optimizer'].get('selected_start')=='fixed_nominal_20mps' for c in cells if c['status']=='simulated'),
        identification_gate_pass=gate,confirmation_unlocked=False,lpv_control_benefit_tested=False)
    parent.save_json(OUT/f'{PREFIX}_analysis.json',diagnosis)
    speedrows=[];fleetrows=[]
    for speed in range(10,31,2):
        z=[c for c in cells if c['speed']==speed]
        errors=[c['models']['ClosedOE4']['matrix_error']*100 for c in z if c['models']['ClosedOE4']['fit_valid']]
        tracking_values=[c['tracking_validation']['yaw_rmse_deg_s'] for c in z if c['status']=='simulated']
        limiting=[c['tracking_validation']['limiter_fraction'] for c in z if c['status']=='simulated']
        speedrows.append([speed,len(z),sum(c['models']['ClosedOE4']['accuracy_pass'] for c in z),sum(c['physical_pass'] for c in z),
            float(np.median(errors)) if errors else None,float(np.percentile(errors,95)) if errors else None,
            float(np.median(tracking_values)) if tracking_values else None,
            float(100*np.median(limiting)) if limiting else None])
    for seed in protocol['seeds']:
        z=[c for c in cells if c['seed']==seed]
        fleetrows.append([seed,len(z),sum(c['models']['ClosedOE4']['accuracy_pass'] for c in z),sum(c['physical_pass'] for c in z),sum(c['settling_pass'] for c in z)])
    previews=''
    for r in results:
        for speed,p in r['previews'].items():
            t=p['training'];previews+=f'<h3>{speed} m/s · {html.escape(p["client"])}</h3><p>Training maneuver; time [s]. Identification starts at 24 s. Blue: response; red: yaw reference.</p>'
            previews+=chart(t['time'],[('simulated response',t['yaw_deg_s']),('reference',t['yaw_reference_deg_s'])],'Yaw rate [deg/s]')
            previews+=chart(t['time'],[('physical response',t['beta_deg'])],'Sideslip [deg]')
            previews+=chart(t['time'],[('known applied input',t['steering_deg'])],'Steering [deg]')
    REPORT.write_text('''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Fixed LTI controller identification</title><style>body{font:16px/1.6 system-ui;color:#203345;background:#edf3f7}main{max-width:1120px;margin:auto;padding:30px}section{background:white;padding:24px;border-radius:12px;margin:20px 0}table{width:100%;border-collapse:collapse;font-size:14px}th,td{padding:8px;text-align:right;border-bottom:1px solid #ddd}.scroll{overflow:auto}svg{width:100%;max-width:740px}svg text{font:12px system-ui}pre{white-space:pre-wrap;background:#eef3f7;padding:16px}</style><main><h1>Identification under one fixed nominal LTI controller</h1><p>One 20 m/s nominal model, one fixed LQI gain, one fixed prefilter for every client and speed. Only the prescribed yaw-rate reference depends on speed.</p><section><h2>Goal and scope</h2><p>Test whether controlled, path-inspired maneuvers provide accurate fixed-speed model estimates, and whether the same nominal model is useful as an additional OE start. This is a development diagnostic on three reused fleets. It does not yet compare fixed-LTI and learned-LPV controller performance.</p><p>All runs start from beta = r = 0 at nonzero fixed speed. A four-second curvature ramp establishes cornering; the 20-second fitting segment begins after 20 s settling and 4 s hold. The nominal controller remains active throughout, using noisy state measurements. No independent sideslip reference or spatial-path tracking claim is made.</p></section><section><h2>Identification results</h2><p>Prespecified accuracy plus physical/settling gate: <strong>'''+str(gate)+'''</strong>. Confirmation remains paused.</p>'''+table(['Condition','Valid','Accuracy pass','Median matrix %','p95 matrix %','Affine rollout p95'],rows)+'''<p>OpenOE3 is the frozen gentle open-loop experiment. OpenOE4 adds the fixed nominal start on exactly the same measurements. ClosedOE3/4 use the new fixed-controller maneuvers with three/four starts. ClosedCleanOLS is a diagnostic reference using exact simulated states and known inputs, never used by OE. Numeric percentiles exclude failed fits; failures remain in pass denominators. Physical feasibility includes zero limiter activations during identification; this is stricter than merely staying within applied steering bounds.</p><p>Open versus closed compares acquisition protocols and their resulting data. It does not isolate feedback from reference-shape changes. Three versus four starts isolates initialization on identical data. Model selection uses only training loss.</p></section><section><h2>Low-speed and fleet checks</h2>'''+table(['Speed','Cells','Accuracy pass','Physical pass','Median matrix %','p95 matrix %','Validation yaw RMSE median [deg/s]','Limiter fraction median [%]'],speedrows)+table(['Fleet','Cells','Accuracy pass','Physical pass','Settling pass'],fleetrows)+'''</section><section><h2>Nominal initialization</h2><p>Closed-loop nominal starts converged: '''+str(diagnosis['closed_nominal_start_converged'])+'''. Selected by training loss: '''+str(diagnosis['closed_nominal_start_selected'])+'''. All starts and their physical parameters are saved for review. Selection frequency alone is not evidence of better model accuracy.</p></section><section><h2>Example physical maneuvers</h2>'''+previews+'''</section><section><h2>Interpretation and next step</h2><p>If the gate passes, inspect individual failures, saturation, tracking and start-to-start consistency before freezing this acquisition protocol for independent confirmation. If it fails, retain the successful gentle open-loop OE protocol and identify whether tracking, excitation, local approximation or optimization is responsible. Do not tune the fixed baseline to make LPV appear better.</p><p>The fixed LTI controller is a meaningful baseline for the later LPV controller comparison. That comparison must use identical held-out references, limits, sensing and evaluation criteria. LPV improvement remains an empirical question.</p><p>Canonical rollout uses the common quiet-segment validation center to preserve previous criteria; full-affine rollout uses training-fitted offsets. Both initialize each one-second prediction window at the exact simulator state. Available sideslip, known applied steering and fixed exact speed remain assumptions. Closed-loop noise timing and model mismatch require review.</p></section><section><h2>Frozen controller</h2><pre>'''+html.escape(json.dumps(lock['controller'],indent=2))+'''</pre><p><a href="../docs/fixed_lti_identification.html">Protocol and commands</a></p></section></main></html>''',encoding='utf-8',newline='\n')
    print(json.dumps({k:v for k,v in diagnosis.items() if k not in ['provenance','checkpoint_sha256']},indent=2));print(REPORT)


def smoke_worker():
    protocol,settings,cfg,lock=preflight();controller=lock['controller']
    # Technical fixture on the nominal design vehicle at the two envelope edges.
    # It is not a sampled pilot client and writes no scientific checkpoint.
    for speed in [10.,30.]:
        t=generate_fixed_lqi(family_centers()['nominal'],speed,94601,cfg,protocol,controller)
        y,u=inputs(t,cfg)
        with threadpool_limits(limits=1):base,combined=add_nominal_start(y,u,settings['estimator'],controller)
        assert combined['converged'] and len(combined['attempts'])==4
        assert np.array_equal(t['state'][0],[0.,0.])
        print(json.dumps(dict(smoke_speed=speed,fit_converged=True,physical=physical_diagnostics(t,speed,cfg),
            saved_scientific_results=False)),flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command',choices=['preflight','status','smoke','restore','restore-new','run','analyze'])
    ap.add_argument('--workers',type=int,default=5);args=ap.parse_args()
    if args.workers<1:ap.error('workers must be positive')
    protocol,settings,cfg,lock=preflight()
    if args.command in ['preflight','status']:
        complete=present=0
        for seed in protocol['seeds']:
            if path_for(seed).exists():read_result(path_for(seed),lock);complete+=1
            source=oe.known.source_result(seed,'gentle',lock['oe_provenance']['known_provenance'])
            present+=sum(parent.raw_path('pilot',seed,'gentle',c['client_index'],c['speed']).exists() for c in source['cells'])
        print(json.dumps(dict(status='preflight_passed',jobs=3,complete=complete,original_raw_present=present,
            original_raw_required=249,controller=lock['controller']),indent=2))
    elif args.command=='analyze':analyze()
    elif args.command=='smoke':
        with ProcessPoolExecutor(max_workers=1,mp_context=get_context('spawn')) as pool:pool.submit(smoke_worker).result()
    else:
        pending=[]
        for seed in protocol['seeds']:
            if args.command=='restore-new':
                if path_for(seed).exists():pending.append(seed)
            elif args.command=='restore':
                source=oe.known.source_result(seed,'gentle',lock['oe_provenance']['known_provenance'])
                if any(not parent.raw_path('pilot',seed,'gentle',c['client_index'],c['speed']).exists() for c in source['cells']):
                    pending.append(parent.checkpoint_path('pilot',seed,'gentle'))
            elif path_for(seed).exists():read_result(path_for(seed),lock)
            else:
                source=oe.known.source_result(seed,'gentle',lock['oe_provenance']['known_provenance'])
                if any(not parent.raw_path('pilot',seed,'gentle',c['client_index'],c['speed']).exists() for c in source['cells']):
                    raise FileNotFoundError('Original gentle pilot arrays missing. Run restore first.')
                pending.append(seed)
        fn={'run':run_job,'restore':parent.restore_job,'restore-new':restore_new_job}[args.command]
        with ProcessPoolExecutor(max_workers=min(args.workers,3),mp_context=get_context('spawn')) as pool:
            for future in as_completed([pool.submit(fn,arg) for arg in pending]):print(json.dumps(future.result()),flush=True)
        if args.command=='run':print('Commit and push the three checkpoints, then analyze.',flush=True)


if __name__=='__main__':main()
