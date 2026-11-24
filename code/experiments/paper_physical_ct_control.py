r"""Frozen 90-vehicle closed-loop pilot for the structured CT baseline.

Delegates execution to the already committed control_fleet implementation.
No identification reruns or calibration arrays. Commands: preflight, smoke,
run, status, analyze. Run from Windows CMD with federated_lpv activated.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import io
import json
import multiprocessing
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'code/src'),str(ROOT/'code/experiments')]
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import paper_physical_ct_oe as old
from federated_lpv.oe_lqi_validation import design_nodes, simulate
from federated_lpv.vehicle import family_centers, discrete_bicycle_matrices
from paper_oe_transient_benchmark import profile

CONFIG = ROOT/'code/config/paper_physical_ct_control.json'
OUT = ROOT/'results/tables'
PREFIX = 'paper_physical_ct_control'
REPORT = ROOT/'report/physical_ct_control_pilot.html'


def preflight():
    cfg=old.load(CONFIG)
    for path,sha in cfg['source_sha256'].items():
        if old.file_hash(ROOT/path)!=sha:
            raise ValueError(f'Frozen control-pilot input changed: {path}')
    inherited=old.preflight('pilot')
    if cfg['seeds']!=inherited['cohorts']['pilot'] or cfg['methods']!=['NominalLTI',*inherited['comparators'],old.METHOD]:
        raise ValueError('Unexpected pilot cohort or methods')
    spec=old.load(ROOT/'code/config/paper_oe_transient_benchmark.json')
    control=old.load(old.storage.CONFIG)
    if cfg['speeds_mps']!=spec['speeds_mps'] or control['sample_time']!=.01:
        raise ValueError('Unexpected maneuver or integration settings')
    for seed in cfg['seeds']:
        _,source=old.sources(seed)
        fitted=old.envelope(old.result_path(seed))
        old.validate_result(fitted,seed,inherited,source)
        if fitted['status']!='execution_complete' or not all(c['fit']['converged'] for c in fitted['clients']):
            raise ValueError('Expected all 90 completed, converged pilot identification fits')
        for client in source['clients']:
            if any(v in client['speeds'] for v in cfg['speeds_mps']):
                raise ValueError('Control test speeds must be locally unobserved')
    return cfg


def validate(payload,seed,cfg,source):
    inherited=old.load(old.CONFIG)
    old.validate_result(payload,seed,inherited,source,control=True)
    if payload['identification_sha256']!=old.file_hash(old.result_path(seed)):
        raise ValueError('Control checkpoint no longer matches identification')
    control=old.load(old.storage.CONFIG)
    spec=old.load(ROOT/'code/config/paper_oe_transient_benchmark.json')
    expected={(method,float(speed)) for method in cfg['methods'] for speed in cfg['speeds_mps']}
    for client in payload['clients']:
        keys=[(row['method'],float(row['speed'])) for row in client['rows']]
        if len(keys)!=len(expected) or set(keys)!=expected:
            raise ValueError('Missing, duplicate or unexpected control cases')
        if set(client['schedules'])!=set(cfg['methods']):
            raise ValueError('Missing method schedules')
        for method,schedule in client['schedules'].items():
            if schedule['status'] not in ['designed','synthesis_failed']:
                raise ValueError('Invalid schedule status')
            if len(schedule['nodes'])!=len(control['speed_grid']):
                raise ValueError('Incomplete node synthesis diagnostics')
            if [n['speed'] for n in schedule['nodes']]!=control['speed_grid']:
                raise ValueError('Wrong synthesis grid')
            if schedule['status']=='designed':
                for node in schedule['nodes']:
                    if node['status']!='designed' or len(node['gain'])!=3 or not np.isfinite([*node['gain'],node['prefilter']]).all():
                        raise ValueError('Invalid designed controller')
        for row in client['rows']:
            outcome=row['outcome'];status=outcome['status'];metrics=outcome['metrics']
            if status not in ['complete','synthesis_failed','rollout_failed']:
                raise ValueError('Invalid rollout status')
            if status=='complete':
                if client['schedules'][row['method']]['status']!='designed':
                    raise ValueError('Rollout cannot complete after failed synthesis')
                if outcome['completed_steps']!=round(spec['duration_s']/control['sample_time']):
                    raise ValueError('Incomplete trajectory reported as complete')
                if not metrics or not np.isfinite(list(metrics.values())).all() or min(metrics.values())<0:
                    raise ValueError('Nonfinite/negative complete-run metrics')
                bounded=(metrics['steering_peak_deg']<=control['steering_limit_deg']+1e-8 and
                         metrics['steering_rate_peak_deg_s']<=control['steering_rate_limit_deg_s']+1e-8 and
                         metrics['beta_violation_samples']==0)
                if outcome['feasible']!=bounded:
                    raise ValueError('Constraint status inconsistent with metrics')
            elif metrics is not None or outcome['feasible']:
                raise ValueError('Failed trajectories must not have full-duration metrics or be feasible')


def run_fleet(seed):
    cfg=old.load(CONFIG)
    if old.result_path(seed,True).exists():
        validate(old.envelope(old.result_path(seed,True)),seed,cfg,old.sources(seed)[1])
    outcome=old.control_fleet(seed)
    validate(old.envelope(old.result_path(seed,True)),seed,cfg,old.sources(seed)[1])
    return outcome


def smoke_worker():
    """Three synthetic nominal-vehicle runs, not a selected evaluation recipient."""
    control=old.load(old.storage.CONFIG)
    spec=old.load(ROOT/'code/config/paper_oe_transient_benchmark.json')
    plant=family_centers()['nominal']
    matrices=[]
    for v in control['speed_grid']:
        a,b=discrete_bicycle_matrices(v,plant,control['sample_time'])
        matrices.append(np.r_[a.ravel(),b.ravel()])
    with threadpool_limits(limits=1):
        schedule=design_nodes(matrices,control)
        if schedule['status']!='designed':
            raise AssertionError('Synthetic schedule synthesis failed')
        outcomes=[]
        for v in spec['speeds_mps']:
            outcome,_=simulate(plant,schedule,profile(spec,v),control)
            if outcome['status']!='complete' or outcome['completed_steps']!=1200:
                raise AssertionError('Synthetic control rollout failed')
            outcomes.append(outcome)
    with tempfile.TemporaryDirectory() as directory:
        path=Path(directory)/'smoke.json'
        old.atomic_json(path,dict(outcomes=outcomes,schedule=schedule))
        if old.envelope(path)['outcomes']!=outcomes:
            raise AssertionError('Checkpoint roundtrip failed')
    return dict(status='control_smoke_passed',synthetic_speeds=spec['speeds_mps'],
                windows_spawn=True,scientific_results_written=False)


def tables(frame,cfg):
    """Equal speed/client/fleet weighting. Failed metrics invalidate full means."""
    keys=['tracking_rmse_deg_s','tracking_peak_deg_s','beta_peak_deg','steering_rms_deg',
          'steering_rate_peak_deg_s','lateral_acceleration_peak_mps2','limiter_fraction']
    units=frame.groupby(['seed','method','client'])[keys].agg(old.strict_mean).reset_index()
    fleets=units.groupby(['seed','method'])[keys].agg(old.strict_mean).reset_index()
    summary=fleets.groupby('method')[keys].agg(old.strict_mean).reindex(cfg['methods']).reset_index()
    counts=frame.groupby('method').agg(cases=('status','size'),
        complete=('status',lambda x:int((x=='complete').sum())),
        synthesis_failed=('status',lambda x:int((x=='synthesis_failed').sum())),
        rollout_failed=('status',lambda x:int((x=='rollout_failed').sum())),
        feasible=('feasible','sum')).reindex(cfg['methods']).reset_index()
    summary=counts.merge(summary,on='method',validate='one_to_one')
    speed=frame.groupby(['seed','method','speed'])[keys].agg(old.strict_mean).reset_index()
    paired=[]
    for seed in cfg['seeds']:
        values=fleets[fleets.seed==seed].set_index('method')
        for baseline in cfg['comparators']:
            ct=values.loc[old.METHOD,'tracking_rmse_deg_s'];base=values.loc[baseline,'tracking_rmse_deg_s']
            paired.append(dict(seed=seed,baseline=baseline,baseline_minus_CT_deg_s=base-ct,
                               full_fleet_available=bool(np.isfinite([base,ct]).all())))
    return units,fleets,summary,speed,pd.DataFrame(paired)


def analyze(cfg):
    records=[];hashes={}
    for seed in cfg['seeds']:
        path=old.result_path(seed,True);result=old.envelope(path)
        validate(result,seed,cfg,old.sources(seed)[1])
        if result['status']!='execution_complete':
            raise ValueError(f'Finish all three fleets before analysis: {seed}')
        hashes[path.relative_to(ROOT).as_posix()]=old.file_hash(path)
        for client in result['clients']:
            for row in client['rows']:
                outcome=row['outcome'];metrics=outcome['metrics'] or {}
                records.append(dict(seed=seed,client=client['client'],observed_speeds=str(client['observed_speeds']),
                    method=row['method'],speed=row['speed'],status=outcome['status'],feasible=outcome['feasible'],**metrics))
    frame=pd.DataFrame(records)
    # Define columns even when a whole method or fleet fails before simulation.
    names=['tracking_rmse_deg_s','tracking_peak_deg_s','beta_peak_deg','steering_rms_deg',
           'steering_rate_peak_deg_s','lateral_acceleration_peak_mps2','limiter_fraction']
    for name in names:
        if name not in frame:frame[name]=np.nan
    units,fleets,summary,speed,paired=tables(frame,cfg)
    for suffix,data in [('cases',frame),('clients',units),('fleets',fleets),('summary',summary),('speeds',speed),('paired',paired)]:
        data.to_csv(OUT/f'{PREFIX}_{suffix}.csv',index=False)
    audit=dict(status='analysis_complete',seeds=cfg['seeds'],expected_cases=1620,recorded_cases=len(frame),
        checkpoint_sha256=hashes,protocol_sha256=old.file_hash(CONFIG),analysis_sha256=old.file_hash(__file__),
        full_duration_cases=int((frame.status=='complete').sum()),feasible_cases=int(frame.feasible.sum()),
        note='Reused pilot data; descriptive comparison, no fresh confirmation. Finite infeasible runs remain in means. Any numerical failure makes the full fleet mean unavailable.')
    old.acquisition.parent.save_json(OUT/f'{PREFIX}_analysis.json',audit)
    write_report(summary,fleets,speed,paired,audit)
    print(summary.to_string(index=False));print(REPORT)


def write_report(summary,fleets,speed,paired,audit):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    selected=['NominalLTI','Global','Federated',old.METHOD]
    for method in selected:
        values=fleets[fleets.method==method]
        axes[0].plot(values.seed,values.tracking_rmse_deg_s,marker='o',label=method)
        values=speed[speed.method==method].groupby('speed').tracking_rmse_deg_s.agg(old.strict_mean)
        axes[1].plot(values.index,values.values,marker='o',label=method)
    axes[0].set(xlabel='Pilot fleet',ylabel='Yaw-rate tracking RMSE [deg/s]',xticks=audit['seeds'])
    axes[1].set(xlabel='Test speed [m/s]',ylabel='Yaw-rate tracking RMSE [deg/s]',xticks=[13,21,29])
    axes[0].legend(fontsize=8);axes[0].grid(alpha=.2);axes[1].grid(alpha=.2)
    buf=io.StringIO();fig.savefig(buf,format='svg',bbox_inches='tight',metadata={'Date':None});plt.close(fig)
    chart=buf.getvalue();chart=chart[chart.index('<svg'):]
    def table(df):
        return '<div class="scroll">'+df.to_html(index=False,border=0,float_format=lambda x:f'{x:.6g}',na_rep='unavailable')+'</div>'
    report=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Structured CT closed-loop pilot</title><style>body{{font:16px/1.6 system-ui;background:#eef3f6;color:#173044}}main{{max-width:1250px;margin:auto;padding:30px}}section{{background:white;padding:24px;margin:20px 0;border-radius:10px}}.scroll{{overflow:auto}}table{{border-collapse:collapse;white-space:nowrap;font-size:13px;width:100%}}th,td{{padding:8px;border-bottom:1px solid #dce3e9;text-align:right}}svg{{width:100%;height:auto}}</style><main>
<h1>Does improved local identification benefit closed-loop tracking?</h1><p>90 pilot vehicles × 6 methods × 3 speeds = 1620 cases. No model refitting or controller tuning.</p>
<section><h2>Protocol and interpretation</h2><p>Identical 12-s left-right yaw-rate reference at 13, 21 and 29 m/s, with nominal peak lateral demand 2.5 m/s². Common LQI weights, gain grid, steering bounds (12°, 120°/s), sideslip bound (6°) and zero initial conditions. Noise-free full-state feedback. Structured CT models are fitted around fixed-curvature cornering and are used here without adaptation.</p><p>Check synthesis/numerical failures and feasibility before mean tracking. Limiter activation alone is not infeasibility. Lateral acceleration is reported, but is not an additional feasibility gate in the inherited control benchmark. These are reused pilot vehicles, not independent confirmation.</p></section>
<section><h2>All-method summary</h2>{table(summary)}<p>All methods have 270 planned cases. Complete counts finite full-duration runs, including constraint violations. Feasible requires the inherited steering and sideslip bounds. A missing full-duration error makes the entire method–fleet mean unavailable; no survivor-only means or replacement runs.</p></section>
<section><h2>Tracking comparison</h2>{chart}<p>The plots show Nominal LTI, Global, Federated and PhysicalCTLocal; both local baselines remain in all tables. Gaps denote unavailable full means. Absolute errors and differences matter more than percentage gains when errors are small.</p>{table(fleets)}</section>
<section><h2>Paired fleet differences</h2><p>Positive baseline-minus-CT means lower RMSE for PhysicalCTLocal. No p-values or confirmatory confidence intervals from three reused fleets.</p>{table(paired)}</section>
<section><h2>By fleet and speed</h2>{table(speed)}</section>
<section><h2>Next decision</h2><p>If structured local CT matches or improves Federated tracking without worse feasibility or effort, use it as the practical local baseline in the paper. If identification gains do not translate to tracking, quantify the absolute difference and investigate control sensitivity before changing the experiment. Friction recovery, production sideslip sensing and the need for future federation remain separate questions.</p></section></main></html>'''
    REPORT.write_text('\n'.join(line.rstrip() for line in report.splitlines())+'\n',encoding='utf-8',newline='\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['preflight','smoke','run','status','analyze'])
    parser.add_argument('--workers',type=int,default=3)
    args=parser.parse_args()
    if args.workers<1:parser.error('--workers must be positive')
    cfg=preflight()
    if args.command=='preflight':
        print(json.dumps(dict(status='control_preflight_passed',fleets=3,vehicles=90,methods=cfg['methods'],
                              speeds=cfg['speeds_mps'],planned_cases=1620,refits=0,raw_calibration_arrays_required=False)))
    elif args.command=='smoke':
        with ProcessPoolExecutor(max_workers=1,mp_context=multiprocessing.get_context('spawn')) as pool:
            print(json.dumps(pool.submit(smoke_worker).result()))
    elif args.command=='run':
        with old.execution_lock('pilot','control'),ProcessPoolExecutor(max_workers=min(args.workers,3),mp_context=multiprocessing.get_context('spawn')) as pool:
            for future in as_completed([pool.submit(run_fleet,seed) for seed in cfg['seeds']]):
                print(json.dumps(future.result()),flush=True)
    elif args.command=='status':
        for seed in cfg['seeds']:
            path=old.result_path(seed,True)
            if path.exists():
                payload=old.envelope(path);validate(payload,seed,cfg,old.sources(seed)[1])
                print(json.dumps(dict(seed=seed,status=payload['status'],saved_clients=len(payload['clients']))))
            else:print(json.dumps(dict(seed=seed,status='not_started')))
    else:analyze(cfg)


if __name__=='__main__':main()
