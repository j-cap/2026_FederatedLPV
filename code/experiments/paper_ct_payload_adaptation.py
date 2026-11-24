r"""Does centered loading create a useful local-adaptation window?

python code\experiments\paper_ct_payload_adaptation.py preflight
Nine previously identified vehicles, three loading amounts, five causal prefixes.
No fleet prior, new fleet model, covariance tuning or manuscript changes.
"""
from __future__ import annotations
import os
for variable in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[variable] = '1'
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'code/src'),str(ROOT/'code/experiments')]
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import paper_physical_ct_observer as base
from federated_lpv.payload_adaptation import (load_vehicle, loaded_parameters,
    fit_payload, fitting_segment, acquisition_profile)
from federated_lpv.ct_observer_control import (observer_model, kalman_design,
    simulate_observed, trace_hash, origin_closed_loop)
from federated_lpv.physical_ct_oe import discrete_matrices
from federated_lpv.oe_lqi_validation import design_nodes, interpolate
from federated_lpv.vehicle import VehicleParameters, family_centers

PREFIX = 'paper_ct_payload_adaptation'
CONFIG = ROOT/'code/config'/f'{PREFIX}.json'
OUT = ROOT/'results/tables'
RAW = ROOT/'results/data'/PREFIX
REPORT = ROOT/'report/ct_payload_adaptation.html'
METRICS = ['tracking_rmse_deg_s','beta_estimation_rmse_deg','steering_rate_rms_deg_s',
    'tracking_peak_deg_s','beta_peak_deg','steering_rate_peak_deg_s','limiter_fraction']


def key(client, percent):
    return f'{client}_load{percent:02d}'


def path(client, percent):
    return OUT/f'{PREFIX}_{key(client,percent)}.json'


def methods(cfg):
    return ['Previous','KnownChange',*[f'Local{t}' for t in cfg['prefix_seconds']]]


def jobs(cfg):
    return [(c,p) for c in cfg['clients'] for p in cfg['loading_percent']]


def preflight():
    cfg = base.old.load(CONFIG)
    for name, sha in cfg['source_sha256'].items():
        if base.old.file_hash(ROOT/name) != sha:
            raise ValueError(f'Frozen payload input changed: {name}')
    source, _, yaw, controls = base.inputs(cfg['source_seed'])
    if yaw['status'] != 'execution_complete' or controls['status'] != 'execution_complete':
        raise ValueError('Finish pre-change identification/control before this diagnostic')
    selected = [c for c in source['clients'] if c['client'] in cfg['clients']]
    if len(selected) != 9 or len(jobs(cfg)) != 27 or len(set(cfg['clients'])) != 9:
        raise ValueError('Expected nine selected vehicles and 27 loading cases')
    for c in cfg['clients']:
        i = next(i for i,s in enumerate(source['clients']) if s['client'] == c)
        if not yaw['clients'][i]['fit']['converged'] or controls['clients'][i]['schedules']['YawOnlyCT']['status'] != 'designed':
            raise ValueError('Selected pre-change model/controller unavailable')
    if cfg['fit']['sample_time'] != cfg['acquisition']['sample_time'] or cfg['prefix_seconds'][-1] > cfg['acquisition']['duration_s']:
        raise ValueError('Inconsistent prefix/acquisition timing')
    return cfg


def identity(cfg):
    files = [CONFIG,Path(__file__),ROOT/'code/src/federated_lpv/payload_adaptation.py']
    return dict(files={p.relative_to(ROOT).as_posix():base.old.file_hash(p) for p in files},
                frozen=cfg['source_sha256'],runtime=base.old.runtime())


def material(client, percent, cfg):
    source, _, yaw, controls = base.inputs(cfg['source_seed'])
    i = next(i for i,c in enumerate(source['clients']) if c['client'] == client)
    plant = load_vehicle(VehicleParameters(**source['clients'][i]['parameters']),
                         percent/100, cfg['fit']['payload_radius_m'])
    return i, plant, np.array(yaw['clients'][i]['fit']['parameters']), controls['clients'][i]['schedules']['YawOnlyCT']


def sensor_noise(index, percent, stage, speed, samples, cfg):
    rng = np.random.default_rng(np.random.SeedSequence([
        cfg['noise_namespace'],cfg['source_seed'],index,percent,stage,int(1000*speed)]))
    return rng.normal(0.,np.deg2rad(cfg['observer']['yaw_noise_std_deg_s']),samples)


def control_settings(cfg, acquisition=False):
    control = base.old.load(base.old.storage.CONFIG)
    if acquisition:
        control.update(steering_rate_limit_deg_s=cfg['acquisition']['steering_rate_limit_deg_s'],
                       beta_limit_deg=cfg['acquisition']['beta_limit_deg'])
    return control


def acquire(client, percent, cfg):
    i, plant, previous, schedule = material(client,percent,cfg)
    profile = acquisition_profile(cfg['acquisition'])
    a,b = observer_model(previous,cfg['acquisition']['speed_mps'],cfg['observer'])
    observer = kalman_design(a,b,cfg['observer'],len(profile['time']))
    noise = sensor_noise(i,percent,0,cfg['acquisition']['speed_mps'],len(profile['time']),cfg)
    outcome, arrays = simulate_observed(plant,schedule,profile,observer,noise,control_settings(cfg,True))
    valid = bool(outcome['status'] == 'complete' and outcome['feasible'] and
        outcome['metrics']['limiter_fraction'] == 0 and
        outcome['metrics']['lateral_acceleration_peak_mps2'] <= cfg['acquisition']['max_ay_mps2'])
    record = dict(outcome=outcome, operational_checks_passed=valid,
        trace_sha256=trace_hash(arrays),noise_sha256=trace_hash({'noise':noise}),
        observer=observer['diagnostics'],controller='unchanged saved pre-change YawOnlyCT schedule')
    if client == cfg['preview_client'] and percent == cfg['preview_loading_percent']:
        record['preview'] = {k:v[::10].tolist() for k,v in arrays.items()}
    return record, arrays


def raw_path(client, percent, suffix='acquisition'):
    return RAW/f'{key(client,percent)}_{suffix}.npz'


def save_arrays(target, arrays):
    target.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='wb',prefix=target.name+'.',suffix='.tmp',dir=target.parent,delete=False) as stream:
        pending = Path(stream.name);np.savez_compressed(stream,**arrays);stream.flush();os.fsync(stream.fileno())
    for attempt in range(21):
        try:
            os.replace(pending,target);return
        except PermissionError as exc:
            if getattr(exc,'winerror',None) not in [5,32,33] or attempt == 20:
                exc.add_note(f'Original preserved; pending arrays at {pending}');raise
            time.sleep(min(.05*2**min(attempt,4),.5))


def restore_acquisition(client, percent, cfg, saved):
    target = raw_path(client,percent)
    if target.exists():
        with np.load(target,allow_pickle=False) as data:
            arrays = dict(data)
        if trace_hash(arrays) != saved['trace_sha256']:
            raise ValueError('Existing acquisition array hash mismatch; original checkpoint preserved')
    else:
        reconstructed,arrays = acquire(client,percent,cfg)
        if reconstructed['trace_sha256'] != saved['trace_sha256'] or reconstructed['outcome'] != saved['outcome']:
            raise ValueError('Acquisition reconstruction differs from original; checkpoint unchanged. Restore on original numerical platform.')
        save_arrays(target,arrays)
    return arrays


def schedule_for(theta, cfg):
    cc = control_settings(cfg); values = []
    for speed in cc['speed_grid']:
        a,b = discrete_matrices(theta,speed,cfg['fit']['geometry_m'],cfg['controller_curvature_per_m'],cc['sample_time'])
        values.append(np.r_[a.ravel(),b])
    return design_nodes(np.array(values),cc)


def evaluate(client, percent, method, speed, cfg, fitted=None):
    index,plant,previous,old_schedule = material(client,percent,cfg)
    cc = control_settings(cfg)
    maneuver = base.profile(base.old.load(ROOT/'code/config/paper_oe_transient_benchmark.json'),speed)
    noise = sensor_noise(index,percent,1,speed,len(maneuver['time']),cfg)
    record = dict(method=method,speed=speed,noise_sha256=trace_hash({'noise':noise}),
                  trace_sha256=None,true_origin_rho=None,observer=None,schedule=None)
    if method.startswith('Local') and (fitted is None or not fitted['converged']):
        record['outcome'] = dict(status='fit_failed',metrics=None,feasible=False,completed_steps=0)
        return record,None
    theta = (previous if method == 'Previous' else
             loaded_parameters(previous,percent/100,cfg['fit']['payload_radius_m']) if method == 'KnownChange' else
             np.array(fitted['parameters']))
    try:
        schedule = old_schedule if method == 'Previous' else schedule_for(theta,cfg)
        record['schedule'] = schedule
        if schedule['status'] != 'designed':
            raise ValueError('A controller node failed synthesis')
        a,b = observer_model(theta,speed,cfg['observer'])
        observer = kalman_design(a,b,cfg['observer'],len(noise));record['observer'] = observer['diagnostics']
    except (ValueError,np.linalg.LinAlgError,FloatingPointError) as exc:
        record['outcome'] = dict(status='synthesis_failed',reason=str(exc),metrics=None,feasible=False,completed_steps=0)
        return record,None
    outcome,arrays = simulate_observed(plant,schedule,maneuver,observer,noise,cc)
    record.update(outcome=outcome,parameters=theta.tolist(),trace_sha256=trace_hash(arrays))
    ap,bp = base.rk4_origin_matrices(plant,speed,cc['sample_time'])
    gain = interpolate(schedule,cc['speed_grid'],[speed])[0][0]
    record['true_origin_rho'] = float(max(abs(np.linalg.eigvals(origin_closed_loop(
        ap,bp,a,b,gain,observer['diagnostics']['steady_gain'],cc['sample_time'])))))
    if client == cfg['preview_client'] and percent == cfg['preview_loading_percent'] and speed == 21:
        record['preview'] = {k:v[::5].tolist() for k,v in arrays.items()}
    return record,arrays


def validate(payload,client,percent,cfg):
    if payload['client'] != client or payload['loading_percent'] != percent or payload['provenance'] != identity(cfg):
        raise ValueError('Checkpoint identity/protocol/runtime mismatch')
    if payload['status'] not in ['running','execution_complete']:
        raise ValueError('Unknown execution status')
    allowed = {(m,v) for m in methods(cfg) for v in cfg['test_speeds_mps']}
    keys = [(r['method'],r['speed']) for r in payload['rows']]
    if len(set(keys)) != len(keys) or not set(keys) <= allowed:
        raise ValueError('Duplicate/unexpected control case')
    if not set(payload['fits']) <= {str(t) for t in cfg['prefix_seconds']}:
        raise ValueError('Unexpected fitting prefix')
    if payload['status'] == 'execution_complete' and (set(keys) != allowed or len(payload['fits']) != 5 or payload['acquisition'] is None):
        raise ValueError('Incomplete case marked complete')
    if payload['acquisition'] is not None:
        acquisition = payload['acquisition']; outcome = acquisition['outcome']
        if not acquisition['trace_sha256'] or not acquisition['noise_sha256']:
            raise ValueError('Missing acquisition hashes')
        if outcome['status'] == 'complete' and outcome['completed_steps'] != round(cfg['acquisition']['duration_s']/cfg['acquisition']['sample_time']):
            raise ValueError('Acquisition duration mismatch')
    for fit in payload['fits'].values():
        if fit['converged'] and (not cfg['fit']['bounds'][0] <= fit['fraction'] <= cfg['fit']['bounds'][1]
            or np.shape(fit['parameters']) != (4,) or not np.isfinite(fit['parameters']).all()
            or not fit.get('data_sha256')):
            raise ValueError('Invalid saved loading estimate')
    for speed in cfg['test_speeds_mps']:
        hashes = {r['noise_sha256'] for r in payload['rows'] if r['speed'] == speed}
        if len(hashes) > 1:
            raise ValueError('Control methods use different sensor noise')
    cc = control_settings(cfg)
    for row in payload['rows']:
        out = row['outcome']; metrics = out['metrics']
        if out['status'] == 'complete':
            if out['completed_steps'] != 1200 or metrics is None or not np.isfinite(list(metrics.values())).all() or not row['trace_sha256']:
                raise ValueError('Invalid complete metrics')
            feasible = bool(metrics['beta_violation_samples'] == 0 and metrics['steering_peak_deg'] <= cc['steering_limit_deg']+1e-8
                            and metrics['steering_rate_peak_deg_s'] <= cc['steering_rate_limit_deg_s']+1e-8)
            if out['feasible'] != feasible:
                raise ValueError('Invalid feasibility flag')
        elif out['status'] not in ['fit_failed','synthesis_failed','rollout_failed'] or metrics is not None or out['feasible']:
            raise ValueError('Failed trajectories cannot have full-duration metrics')


def run_case(client,percent):
    cfg = base.old.load(CONFIG); target = path(client,percent)
    payload = base.old.envelope(target) if target.exists() else dict(client=client,loading_percent=percent,
        status='running',provenance=identity(cfg),acquisition=None,fits={},rows=[],
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    validate(payload,client,percent,cfg)
    if payload['status'] == 'execution_complete':
        return dict(client=client,loading_percent=percent,status='already_complete')
    with threadpool_limits(limits=1):
        if payload['acquisition'] is None:
            record,arrays = acquire(client,percent,cfg)
            save_arrays(raw_path(client,percent),arrays);payload['acquisition'] = record
            base.atomic_json(target,payload)
        else:
            arrays = restore_acquisition(client,percent,cfg,payload['acquisition'])
        previous = material(client,percent,cfg)[2]
        for seconds in cfg['prefix_seconds']:
            if str(seconds) not in payload['fits']:
                if payload['acquisition']['outcome']['status'] != 'complete':
                    fit = dict(converged=False,reason='Acquisition incomplete; no selective finite-prefix fitting')
                else:
                    segment = fitting_segment(arrays,seconds,cfg['fit']['sample_time'])
                    try:
                        fit = fit_payload(previous,segment,cfg['fit'])
                        fit['data_sha256'] = trace_hash({k:segment[k] for k in ['measured_yaw','applied_input']})
                    except (ValueError,FloatingPointError,np.linalg.LinAlgError) as exc:
                        fit = dict(converged=False,reason=f'{type(exc).__name__}: {exc}')
                payload['fits'][str(seconds)] = fit;base.atomic_json(target,payload)
                print(json.dumps(dict(client=client,loading_percent=percent,fitted_seconds=seconds,converged=fit['converged'])),flush=True)
        done = {(r['method'],r['speed']) for r in payload['rows']}
        for method in methods(cfg):
            fit = payload['fits'].get(method.removeprefix('Local')) if method.startswith('Local') else None
            for speed in cfg['test_speeds_mps']:
                if (method,speed) not in done:
                    row,_ = evaluate(client,percent,method,speed,cfg,fit)
                    payload['rows'].append(row);base.atomic_json(target,payload)
        payload['status'] = 'execution_complete';validate(payload,client,percent,cfg);base.atomic_json(target,payload)
    return dict(client=client,loading_percent=percent,status='execution_complete',control_cases=21)


def tables(frame,cfg):
    frame = frame.copy()
    for metric in METRICS:
        if metric not in frame:
            frame[metric] = np.nan
    units = frame.groupby(['client','loading_percent','method'])[METRICS].agg(base.old.strict_mean).reset_index()
    vehicles = units.groupby(['client','method'])[METRICS].agg(base.old.strict_mean).reset_index()
    summary = vehicles.groupby('method')[METRICS].agg(base.old.strict_mean).reindex(methods(cfg)).reset_index()
    counts = frame.groupby('method').agg(cases=('status','size'),complete=('status',lambda s:int((s=='complete').sum())),
        failures=('status',lambda s:int((s!='complete').sum())),feasible=('feasible','sum'),
        origin_unstable=('true_origin_rho',lambda s:int((s>=1).sum()))).reset_index()
    summary = summary.merge(counts,on='method',validate='one_to_one')
    pairs = []
    for (client,load),group in units.groupby(['client','loading_percent']):
        values = group.set_index('method')
        for seconds in cfg['prefix_seconds']:
            for metric in METRICS:
                old = values.loc['Previous',metric];known = values.loc['KnownChange',metric];local = values.loc[f'Local{seconds}',metric]
                headroom = old-known
                pairs.append(dict(client=client,loading_percent=load,prefix_s=seconds,metric=metric,
                    previous=old,known_change=known,local=local,previous_minus_known=headroom,
                    local_minus_known=local-known,previous_minus_local=old-local,
                    fraction_of_positive_gap_closed=(old-local)/headroom if np.isfinite([old,known,local]).all() and headroom>1e-12 else np.nan))
    sensitivity = []
    for (client,load),group in frame.groupby(['client','loading_percent']):
        known = group[group.method=='KnownChange'].set_index('speed')
        for tolerance in cfg['report_tracking_tolerances_deg_s']:
            good = []
            for seconds in cfg['prefix_seconds']:
                local = group[group.method==f'Local{seconds}'].set_index('speed')
                valid = ((local.status=='complete')&local.feasible&(known.status=='complete')&known.feasible&
                    ((local.tracking_rmse_deg_s-known.tracking_rmse_deg_s)<=tolerance)).all()
                good.append(bool(valid))
            sustained = next((t for j,t in enumerate(cfg['prefix_seconds']) if all(good[j:])),np.nan)
            sensitivity.append(dict(client=client,loading_percent=load,tolerance_deg_s=tolerance,
                first_sustained_prefix_s=sustained,criterion='all three test speeds, both runs complete/feasible; subsequent tested prefixes also satisfy tolerance'))
    return units,vehicles,summary,pd.DataFrame(pairs),pd.DataFrame(sensitivity)


def write_report(summary,pairs,fits,acquisition,sensitivity,previews,cfg):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    def table(data):
        return '<div class="scroll">'+data.to_html(index=False,border=0,float_format=lambda x:f'{x:.6g}',na_rep='unavailable')+'</div>'
    fig,axes = plt.subplots(1,3,figsize=(13,4),layout='constrained')
    values = summary.set_index('method')
    for ax,metric,label in zip(axes,METRICS[:3],['Yaw tracking RMSE [deg/s]','Sideslip estimation RMSE [deg]','Steering-rate RMS [deg/s]']):
        ax.plot(cfg['prefix_seconds'],[values.loc[f'Local{t}',metric] for t in cfg['prefix_seconds']],'-o',label='Local adaptation')
        for method,color in [('Previous','#777777'),('KnownChange','#b65c24')]:
            ax.axhline(values.loc[method,metric],label=method,color=color,linestyle='--')
        ax.set(xlabel='Available post-change data [s]',ylabel=label);ax.grid(alpha=.2)
    axes[0].legend(fontsize=8);chart = base.figure_svg(fig)
    fig,ax = plt.subplots(figsize=(7,4),layout='constrained')
    for load,group in fits.groupby('loading_percent'):
        for client,rows in group.groupby('client'):
            ax.plot(rows.prefix_s,rows.estimated_loading_percent,alpha=.5,marker='.',label=f'{load}%: {client}')
        ax.axhline(load,color='black',linestyle=':',alpha=.5)
    ax.set(xlabel='Available data [s]',ylabel='Estimated added mass [% of pre-change mass]');ax.grid(alpha=.2)
    chart += base.figure_svg(fig)
    for label,arrays in previews:
        p = {k:np.asarray(v) for k,v in arrays.items()}
        fig,axes = plt.subplots(1,3,figsize=(12,3),layout='constrained')
        axes[0].plot(p['time'],np.rad2deg(p['reference']),'k--',label='Reference')
        axes[0].plot(p['time'],np.rad2deg(p['state'][:,1]),label='True yaw')
        axes[1].plot(p['time'],np.rad2deg(p['estimate'][:,0]-p['state'][:,0]))
        axes[2].plot(p['time'][:len(p['steering'])],np.rad2deg(p['steering']))
        for ax,ylabel in zip(axes,['Yaw rate [deg/s]','Sideslip estimation error [deg]','Applied steering [deg]']):
            ax.set(xlabel='Time [s]',ylabel=ylabel);ax.grid(alpha=.2)
        axes[0].legend(fontsize=7);fig.suptitle(label);chart += base.figure_svg(fig)
    text = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Payload adaptation opportunity</title>
<style>body{{font:16px/1.6 system-ui;background:#eef3f6;color:#173044}}main{{max-width:1300px;margin:auto;padding:28px}}section{{background:white;padding:24px;margin:20px 0;border-radius:10px}}.scroll{{overflow:auto}}table{{border-collapse:collapse;white-space:nowrap;font-size:12px}}td,th{{padding:8px;border-bottom:1px solid #ddd}}svg{{width:100%;height:auto}}</style><main>
<h1>Does loading create a meaningful local-adaptation window?</h1>
<section><h2>Scope and comparison</h2><p>Nine reused pilot vehicles, 10/20/30% added mass, centered at the existing CG with known 1 m payload radius of gyration. Geometry, cornering stiffness and friction remain fixed in this controlled simulation. Loading occurs before straight-running initialization, not mid-corner. This is a screen for an adaptation opportunity, not a fleet-learning experiment or a production validation.</p>
<p>Previous uses the same saved pre-change yaw-only CT model. KnownChange applies the true loading fraction to that same identified model; it does not replace the whole model with plant truth and is not a guaranteed performance bound. Local estimates one loading fraction in [0,0.5] using nonlinear yaw-only OE. All use identical controller weights, observer covariances, limits, test references and paired test noise.</p>
<p>One 40-second acquisition trajectory per loading case uses the previous controller/observer at 20 m/s, four smooth left–right curves, nominal lateral demand 1.5 m/s², and no steering dither. Only measured yaw and applied steering enter fitting, at 4/8/16/32/40-second prefixes. Known zero initial state; no affine offsets, latent sideslip pseudo-measurements, future-sample normalization or model updates during acquisition. Noise SD is 0.03°/s. The curves are prescribed and road-inspired, not measured ordinary driving.</p>
<p>135 scalar fits and 567 held-out 12-second control runs at 13/21/29 m/s. Baselines are simulated once per case/speed, then reused in prefix comparisons. Evaluation resets to straight running: these are fixed-model tests after each prefix, not online adaptation rollouts. Pre-change models retain their original full-state-feedback acquisition history.</p></section>
<section><h2>Acquisition operating checks</h2>{table(acquisition)}<p>Checks require full duration, steering/sideslip bounds, no steering limiter activation and actual lateral acceleration ≤6 m/s². Finite violations are disclosed and remain in the analysis; incomplete acquisition causes all local fits to be marked failed. Do not interpret unsafe acquisition as a successful workflow.</p></section>
<section><h2>Control performance versus identification time</h2>{table(summary)}{chart}<p>Means weight speeds, loading levels and vehicles equally. The three loading levels are repeated conditions on each vehicle, not independent vehicles. Any missing full-duration metric invalidates the corresponding full aggregate. Finite constraint violations remain in means. Preview plots are decimated; all metrics use full-rate data.</p></section>
<section><h2>Paired differences and fitting</h2><p>Previous minus KnownChange measures the available benefit of knowing the loading amount. Local minus KnownChange measures the remaining difference after each prefix. Negative values are permitted: a locally fitted loading fraction can partly compensate pre-existing model error. Relative gap closure is unavailable when headroom is nonpositive or numerically zero; assess absolute differences first.</p>{table(pairs[pairs.metric=='tracking_rmse_deg_s'])}{table(fits)}</section>
<section><h2>Threshold sensitivity, not an acceptance standard</h2><p>The reporting tolerances 0.005, 0.01 and 0.02°/s are prespecified descriptive values, not automotive acceptance criteria. The table gives the first tested prefix after which all subsequent tested prefixes remain within the tolerance of KnownChange at all three speeds, with both methods complete and feasible. Also inspect sideslip estimation and steering effort. No automatic practical-success verdict is produced.</p>{table(sensitivity)}</section>
<section><h2>Decision</h2><p>If Previous and KnownChange already perform similarly in absolute terms, loading creates little useful control headroom. If local fitting closes the gap after the first informative curve, the opportunity for fleet priors is narrow. Only a persistent, practically relevant gap motivates testing a donor-derived prior against strong local and fixed-prior baselines. This experiment does not train such a prior or infer its benefit.</p>
<p><a href="../docs/ct_payload_adaptation.html">Protocol, Windows commands and exact trajectory restoration</a></p></section></main></html>'''
    REPORT.parent.mkdir(parents=True,exist_ok=True)
    REPORT.write_text('\n'.join(line.rstrip() for line in text.splitlines())+'\n',encoding='utf-8',newline='\n')


def analyze(cfg):
    records=[];fits=[];acquisition=[];previews=[];hashes={}
    for client,percent in jobs(cfg):
        p = base.old.envelope(path(client,percent));validate(p,client,percent,cfg)
        if p['status'] != 'execution_complete':
            raise ValueError('Finish all 27 loading cases before analysis')
        hashes[path(client,percent).relative_to(ROOT).as_posix()] = base.old.file_hash(path(client,percent))
        acq = p['acquisition'];out = acq['outcome']
        acquisition.append(dict(client=client,loading_percent=percent,status=out['status'],
            operational_checks_passed=acq['operational_checks_passed'],**(out['metrics'] or {})))
        if 'preview' in acq:
            previews.append((f'Acquisition: {client}, {percent}% payload',acq['preview']))
        for seconds in cfg['prefix_seconds']:
            fit = p['fits'][str(seconds)]
            fits.append(dict(client=client,loading_percent=percent,prefix_s=seconds,converged=fit['converged'],
                estimated_loading_percent=100*fit['fraction'] if fit['converged'] else np.nan,
                loading_error_percentage_points=100*fit['fraction']-percent if fit['converged'] else np.nan,
                boundary=fit.get('boundary'),weighted_mse=fit.get('weighted_mse'),
                yaw_sensitivity_norm=fit.get('yaw_sensitivity_norm')))
        for row in p['rows']:
            out = row['outcome']
            records.append(dict(client=client,family=client.rsplit('_',1)[0],loading_percent=percent,
                method=row['method'],speed=row['speed'],status=out['status'],feasible=out['feasible'],
                true_origin_rho=row['true_origin_rho'],**(out['metrics'] or {})))
            if 'preview' in row and row['method'] in ['Previous','KnownChange','Local4','Local40']:
                previews.append((f'{client}, {percent}% payload, {row["method"]}, 21 m/s',row['preview']))
    frame=pd.DataFrame(records);fits=pd.DataFrame(fits);acquisition=pd.DataFrame(acquisition)
    units,vehicles,summary,pairs,sensitivity=tables(frame,cfg)
    for name,table in [('cases',frame),('fits',fits),('acquisition',acquisition),('units',units),
                       ('vehicles',vehicles),('summary',summary),('paired',pairs),('tolerances',sensitivity)]:
        table.to_csv(OUT/f'{PREFIX}_{name}.csv',index=False)
    base.atomic_json(OUT/f'{PREFIX}_analysis.json',dict(status='analysis_complete',cases=len(frame),fits=len(fits),
        loading_cases=len(acquisition),checkpoint_sha256=hashes,provenance=identity(cfg),
        note='Descriptive adaptation screen. No fleet learning, automatic practical threshold or independent confirmation.'))
    write_report(summary,pairs,fits,acquisition,sensitivity,previews,cfg)
    print(summary.to_string(index=False));print(REPORT)


def smoke_worker():
    from federated_lpv.payload_adaptation import yaw_prediction
    cfg=base.old.load(CONFIG);settings=cfg['fit'];theta=np.array([50.,58.,.58,.9])
    dt=settings['sample_time'];u=.008*np.sin(np.arange(400)*dt*2*np.pi*.7)
    expected=.173; y=yaw_prediction(loaded_parameters(theta,expected,settings['payload_radius_m']),u,20.,settings['geometry_m'],dt)[0]
    fit=fit_payload(theta,dict(measured_yaw=y,applied_input=u,speed=20.),settings)
    assert fit['converged'] and abs(fit['fraction']-expected)<1e-5
    plant=load_vehicle(family_centers()['nominal'],.2,1.)
    cc=control_settings(cfg);schedule=schedule_for(theta,cfg)
    a,b=observer_model(theta,21.,cfg['observer']);spec=base.old.load(ROOT/'code/config/paper_oe_transient_benchmark.json')
    profile=base.profile(spec,21.);noise=sensor_noise(0,20,99,21.,len(profile['time']),cfg)
    observer=kalman_design(a,b,cfg['observer'],len(noise))
    outcome,arrays=simulate_observed(plant,schedule,profile,observer,noise,cc)
    assert outcome['status']=='complete'
    with tempfile.TemporaryDirectory() as directory:
        target=Path(directory)/'arrays.npz';save_arrays(target,arrays)
        with np.load(target) as data:
            assert trace_hash(dict(data))==trace_hash(arrays)
        target=Path(directory)/'checkpoint.json';base.atomic_json(target,dict(note='β; °/s',fit=fit))
        assert base.old.envelope(target)['fit']==fit
    return dict(status='payload_smoke_passed',scalar_recovery=fit['fraction'],windows_spawn=True,scientific_results_written=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['preflight','smoke','run','status','analyze','restore','replay'])
    parser.add_argument('--workers',type=int,default=3)
    parser.add_argument('--client',default=None)
    parser.add_argument('--loading-percent',type=int,default=20)
    parser.add_argument('--method',default='Local40')
    parser.add_argument('--speed',type=int,default=21)
    args=parser.parse_args()
    if args.workers<1:parser.error('Positive worker count required')
    cfg=preflight()
    if args.command=='preflight':
        print(json.dumps(dict(status='payload_preflight_passed',vehicles=9,loading_cases=27,
            scalar_fits=135,control_rollouts=567,old_raw_arrays_required=False,fleet_learning=False)))
    elif args.command=='smoke':
        with ProcessPoolExecutor(max_workers=1,mp_context=multiprocessing.get_context('spawn')) as pool:
            print(json.dumps(pool.submit(smoke_worker).result()))
    elif args.command=='run':
        with base.old.execution_lock(PREFIX,'run'),ProcessPoolExecutor(max_workers=min(args.workers,9),mp_context=multiprocessing.get_context('spawn')) as pool:
            for future in as_completed([pool.submit(run_case,c,p) for c,p in jobs(cfg)]):
                print(json.dumps(future.result()),flush=True)
    elif args.command=='status':
        for client,percent in jobs(cfg):
            if path(client,percent).exists():
                p=base.old.envelope(path(client,percent));validate(p,client,percent,cfg)
                print(json.dumps(dict(client=client,loading_percent=percent,status=p['status'],fits=len(p['fits']),control_cases=len(p['rows']))))
            else:print(json.dumps(dict(client=client,loading_percent=percent,status='not_started')))
    elif args.command=='analyze':analyze(cfg)
    else:
        if args.client is not None and args.client not in cfg['clients']:parser.error('Unknown client')
        if args.loading_percent not in cfg['loading_percent']:parser.error('Unknown loading amount')
        selected=jobs(cfg) if args.command=='restore' and args.client is None else [(args.client or cfg['preview_client'],args.loading_percent)]
        with threadpool_limits(limits=1):
            for client,percent in selected:
                p=base.old.envelope(path(client,percent));validate(p,client,percent,cfg)
                if args.command=='restore':
                    if p['acquisition'] is None:raise ValueError('Acquisition not saved yet')
                    arrays=restore_acquisition(client,percent,cfg,p['acquisition'])
                    print(json.dumps(dict(client=client,loading_percent=percent,status='acquisition_hash_verified',path=str(raw_path(client,percent)))))
                else:
                    if args.method not in methods(cfg) or args.speed not in cfg['test_speeds_mps']:parser.error('Unknown method/speed')
                    original=next(r for r in p['rows'] if r['method']==args.method and r['speed']==args.speed)
                    fit=p['fits'].get(args.method.removeprefix('Local'))
                    row,arrays=evaluate(client,percent,args.method,args.speed,cfg,fit)
                    if arrays is None or row['trace_sha256']!=original['trace_sha256'] or row['outcome']!=original['outcome']:
                        raise ValueError('Replay differs from saved original; no checkpoint changed')
                    target=raw_path(client,percent,f'{args.method}_speed{args.speed}');save_arrays(target,arrays)
                    print(json.dumps(dict(status='replay_hash_verified',path=str(target))))


if __name__=='__main__':main()
