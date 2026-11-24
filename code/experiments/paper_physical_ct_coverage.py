r"""Coverage ablation of local structured CT OE, with unchanged estimator.

Windows: python code\experiments\paper_physical_ct_coverage.py preflight
Commands: preflight, smoke, restore, run, status, control, analyze.
Identification and control are checkpointed separately. See docs/physical_ct_coverage.html.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
import io
import json
import multiprocessing
import os
from pathlib import Path
import sys
import tempfile

# Set before numpy is imported in the parent and Windows spawned workers.
for _name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[_name]='1'
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'code/src'),str(ROOT/'code/experiments')]
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import paper_physical_ct_oe as old
import paper_physical_ct_control as baseline
from federated_lpv.physical_ct_oe import fit_physical_oe,discrete_matrices
from federated_lpv.oe_lqi_validation import design_nodes,simulate
from federated_lpv.vehicle import VehicleParameters
from paper_oe_transient_benchmark import profile

PREFIX='paper_physical_ct_coverage'
CONFIG=ROOT/'code/config'/f'{PREFIX}.json'
OUT=ROOT/'results/tables'
REPORT=ROOT/'report/physical_ct_coverage.html'
KINDS=['full','single','adjacent_pair','multi_20s']


def plan(speeds,index,cfg):
    speeds=sorted(float(v) for v in speeds);n=cfg['transitions_per_original_segment']
    if len(speeds) not in [2,3] or len(set(speeds))!=len(speeds):raise ValueError('Expected two or three original speeds')
    def item(kind,vs,counts=None,reuse=None):
        counts=[n]*len(vs) if counts is None else counts
        windows=[dict(speed=v,start=(n-count)//2,n=count) for v,count in zip(vs,counts)]
        return dict(id=f'client{index:02d}_{kind}_'+('_'.join(f'{v:g}' for v in vs)),
                    index=index,kind=kind,windows=windows,reuse=reuse)
    out=[item('full',speeds,reuse='full')]
    out.extend(item('single',[v]) for v in speeds)
    out.extend(item('adjacent_pair',speeds[k:k+2],reuse='full' if len(speeds)==2 else None) for k in range(len(speeds)-1))
    q,r=divmod(cfg['total_budget_transitions'],len(speeds))
    out.append(item('multi_20s',speeds,[q+(j<r) for j in range(len(speeds))]))
    return out


def fleet_plan(seed,cfg):
    return [s for i,c in enumerate(old.sources(seed)[1]['clients']) for s in plan(c['speeds'],i,cfg)]


def sliced_segments(segments,spec):
    by_speed={float(s['speed']):s for s in segments};out=[]
    for w in spec['windows']:
        s=by_speed[w['speed']];start=w['start'];stop=start+w['n']
        if start<0 or w['n']<10 or stop>len(s['applied_input']):raise ValueError('Invalid fitting window')
        out.append(dict(speed=w['speed'],measured_state=np.asarray(s['measured_state'])[start:stop+1].copy(),
                        applied_input=np.asarray(s['applied_input'])[start:stop].copy()))
    return out


def models(theta,grid,settings):
    values=[]
    for v in grid:
        a,b=discrete_matrices(theta,v,settings['geometry_m'],settings['curvature_per_m'],settings['sample_time'])
        values.append(np.r_[a.ravel(),b])
    return np.asarray(values)


def equivalent_parameters(theta,speed,settings,points):
    """Same fixed-speed A/B for a continuum of friction/stiffness combinations."""
    theta=np.asarray(theta,float);lower,upper=np.asarray(settings['bounds']).T
    def slope(mu):return 1-(settings['curvature_per_m']*speed**2/(mu*9.81))**2
    out=[]
    for mu in np.unique(np.r_[np.linspace(lower[3],upper[3],points),theta[3]]):
        candidate=theta.copy();candidate[:2]*=slope(theta[3])/slope(mu);candidate[3]=mu
        if np.all(candidate>=lower-1e-12) and np.all(candidate<=upper+1e-12):out.append(candidate)
    return np.asarray(out)


def evaluate_fit(fit,spec,source,grid,settings,cfg):
    if not fit['converged']:return dict(grid_targets=None,errors=None,single_speed_equivalence=None)
    target=models(fit['parameters'],grid,settings)
    errors=old.dynamic_metrics(target,np.asarray(source['reference']),{})
    equivalent=None
    if spec['kind']=='single':
        v=spec['windows'][0]['speed'];thetas=equivalent_parameters(fit['parameters'],v,settings,cfg['equivalent_mu_grid_points'])
        ref_observed=models(fit['parameters'],[v],settings)
        rows=[];predictions=[]
        for theta in thetas:
            actual=models(theta,[v],settings)
            np.testing.assert_allclose(actual,ref_observed,rtol=2e-10,atol=2e-12)
            prediction=models(theta,grid,settings);predictions.append(prediction)
            rows.append(dict(parameters=theta.tolist(),max_observed_matrix_difference=float(np.max(abs(actual-ref_observed)))))
        # Variation is measured without choosing a candidate by truth/test error.
        identity=np.r_[np.eye(2).ravel(),np.zeros(2)]
        relative=np.linalg.norm(np.asarray(predictions)-target,axis=2)/np.maximum(np.linalg.norm(target-identity,axis=1),1e-15)
        equivalent=dict(candidates=rows,max_relative_increment_deviation_from_selected_by_speed=relative.max(axis=0).tolist(),
            note='Sampled bounded equivalence family, not confidence intervals or a second fitted estimator. Same dynamics at the single observed speed; no candidate selected by evaluation.')
    return dict(grid_targets=target.tolist(),errors=errors,single_speed_equivalence=equivalent)


def path(seed,control=False):return OUT/f'{PREFIX}_{"control_" if control else ""}seed{seed}.json'


def provenance(seed,cfg,control=False):
    sources={str(CONFIG.relative_to(ROOT)):old.file_hash(CONFIG),str(Path(__file__).relative_to(ROOT)):old.file_hash(__file__)}
    if control:sources[str(path(seed).relative_to(ROOT))]=old.file_hash(path(seed))
    return dict(sources=sources,frozen=cfg['source_sha256'],runtime=old.runtime())


def validate(p,seed,cfg,control=False):
    if p['seed']!=seed or p['provenance']!=provenance(seed,cfg,control):raise ValueError('Stale coverage checkpoint; archive it before changing protocol')
    expected=fleet_plan(seed,cfg);records=p['records'];clients=old.sources(seed)[1]['clients']
    if len(records)>len(expected) or p['status'] not in ['running','execution_complete']:raise ValueError('Invalid coverage checkpoint')
    for r,s in zip(records,expected):
        if r['spec']!=s:raise ValueError('Coverage condition/order mismatch')
        source=clients[s['index']]
        if r['client']!=source['client']:raise ValueError('Coverage client identity mismatch')
        if not control:
            expected_n=sum(w['n'] for w in s['windows'])
            if r['transitions']!=expected_n:raise ValueError('Wrong transition budget')
            if r['fit']['converged'] and (np.asarray(r['grid_targets']).shape!=(11,6) or not np.isfinite(r['grid_targets']).all()):raise ValueError('Invalid fitted grid')
        else:
            if [row['speed'] for row in r['rows']]!=[13,21,29]:raise ValueError('Missing or reordered control cases')
            for row in r['rows']:
                o=row['outcome']
                if o['status']=='complete':
                    m=o['metrics']
                    if o['completed_steps']!=1200 or not np.isfinite(list(m.values())).all():raise ValueError('Invalid complete rollout')
                    bounded=(m['steering_peak_deg']<=12+1e-8 and m['steering_rate_peak_deg_s']<=120+1e-8 and m['beta_violation_samples']==0)
                    if o['feasible']!=bounded:raise ValueError('Inconsistent feasibility')
                elif o['status'] not in ['identification_failed','synthesis_failed','rollout_failed'] or o['metrics'] is not None or o['feasible']:
                    raise ValueError('Invalid failed rollout')
    if p['status']=='execution_complete' and len(records)!=len(expected):raise ValueError('Incomplete checkpoint marked complete')


def preflight():
    cfg=old.load(CONFIG);baseline.preflight();oldcfg=old.load(old.CONFIG)
    for name,sha in cfg['source_sha256'].items():
        if old.file_hash(ROOT/name)!=sha:raise ValueError('Frozen coverage source changed: '+name)
    if cfg['seeds']!=[1301,1302,1303] or cfg['conditions']!=KINDS:raise ValueError('Unexpected cohort or conditions')
    for seed in cfg['seeds']:
        source=old.sources(seed)[1];fitted=old.envelope(old.result_path(seed));old.validate_result(fitted,seed,oldcfg,source)
        control_reference=old.envelope(old.result_path(seed,True))
        baseline.validate(control_reference,seed,old.load(baseline.CONFIG),source)
        if control_reference['status']!='execution_complete':raise ValueError('Incomplete full-coverage control reference')
        if fitted['status']!='execution_complete' or not all(c['fit']['converged'] for c in fitted['clients']):raise ValueError('Incomplete full-coverage reference')
    return cfg


@contextmanager
def lock():
    p=ROOT/'results/data'/PREFIX/'execution.lock';p.parent.mkdir(parents=True,exist_ok=True)
    try:
        with p.open('x',encoding='utf-8') as f:f.write(f'PID {os.getpid()}; remove only after confirming no workers remain.\n')
    except FileExistsError as exc:raise RuntimeError('Coverage run already active; check '+str(p)) from exc
    try:yield
    finally:p.unlink()


def checkpoint(seed,cfg,control=False):
    p=old.envelope(path(seed,control)) if path(seed,control).exists() else dict(seed=seed,status='running',provenance=provenance(seed,cfg,control),records=[])
    validate(p,seed,cfg,control);return p


def run_fleet(seed,restore_missing=False):
    cfg=old.load(CONFIG);settings=old.load(old.CONFIG)['estimator'];acfg=old.load(ROOT/'code/config/paper_trajectory_calibration.json')
    acquired,source=old.sources(seed);reference=old.envelope(old.result_path(seed));todo=fleet_plan(seed,cfg)
    payload=checkpoint(seed,cfg);cached_index=None;segments=None
    for spec in todo[len(payload['records']):]:
        i=spec['index'];c=source['clients'][i];hashes=[]
        if spec['reuse']=='full':
            fit=reference['clients'][i]['fit'];hashes=reference['clients'][i]['training_hashes']
        else:
            if cached_index!=i:
                segments=[]
                for cell in (x for x in acquired['cells'] if x['client_index']==i):
                    pair,_=old.traces(seed,cell,restore=restore_missing)
                    y,u=old.acquisition.inputs(pair[0],acfg)
                    if len(u)!=cfg['transitions_per_original_segment']:raise ValueError('Unexpected original segment length')
                    segments.append(dict(speed=cell['speed'],measured_state=y,applied_input=u))
                cached_index=i
            selected=sliced_segments(segments,spec)
            hashes=[dict(speed=s['speed'],y=old.array_hash(s['measured_state']),u=old.array_hash(s['applied_input'])) for s in selected]
            fit=fit_physical_oe(selected,settings)
        evaluated=evaluate_fit(fit,spec,c,source['grid'],settings,cfg)
        payload['records'].append(dict(spec=spec,client=c['client'],original_speeds=c['speeds'],training_hashes=hashes,
            transitions=sum(w['n'] for w in spec['windows']),fit=fit,**evaluated))
        payload['status']='execution_complete' if len(payload['records'])==len(todo) else 'running'
        old.atomic_json(path(seed),payload)
        print(json.dumps(dict(seed=seed,saved_conditions=len(payload['records']),total=len(todo),converged=fit['converged'])),flush=True)
    validate(payload,seed,cfg)
    return dict(seed=seed,status=payload['status'],conditions=len(todo))


def control_fleet(seed):
    cfg=old.load(CONFIG);fitted=old.envelope(path(seed));validate(fitted,seed,cfg)
    if fitted['status']!='execution_complete':raise ValueError('Finish and commit identification first')
    payload=checkpoint(seed,cfg,True);source=old.sources(seed)[1]
    full=old.envelope(old.result_path(seed,True));control=old.load(old.storage.CONFIG)
    spec_m=old.load(ROOT/'code/config/paper_oe_transient_benchmark.json')
    for r in fitted['records'][len(payload['records']):]:
        i=r['spec']['index']
        if r['spec']['reuse']=='full':
            saved=full['clients'][i];schedule=saved['schedules'][old.METHOD]
            rows=[dict(speed=x['speed'],outcome=x['outcome']) for x in saved['rows'] if x['method']==old.METHOD]
        else:
            schedule=design_nodes(r['grid_targets'],control) if r['fit']['converged'] else dict(status='identification_failed',nodes=[])
            rows=[]
            for v in spec_m['speeds_mps']:
                if not r['fit']['converged']:
                    outcome=dict(status='identification_failed',feasible=False,metrics=None,completed_steps=0)
                else:outcome,_=simulate(VehicleParameters(**source['clients'][i]['parameters']),schedule,profile(spec_m,v),control)
                rows.append(dict(speed=v,outcome=outcome))
        payload['records'].append(dict(spec=r['spec'],client=r['client'],schedule=schedule,rows=rows))
        payload['status']='execution_complete' if len(payload['records'])==len(fitted['records']) else 'running'
        old.atomic_json(path(seed,True),payload)
        print(json.dumps(dict(seed=seed,saved_control_conditions=len(payload['records']))),flush=True)
    validate(payload,seed,cfg,True)
    return dict(seed=seed,status=payload['status'],control_cases=3*len(payload['records']))


def worker(command,seed,restore_missing=False):
    with threadpool_limits(limits=1):
        if command=='restore':return old.restore_fleet(seed)
        if command=='control':return control_fleet(seed)
        return run_fleet(seed,restore_missing)


def smoke_worker():
    cfg=old.load(CONFIG);settings=old.load(old.CONFIG)['estimator'];theta=np.array([47.,59.,.56,1.02]);rng=np.random.default_rng(8271)
    short=dict(cfg,transitions_per_original_segment=600,total_budget_transitions=600)
    segments=[]
    for v in [12.,14.,16.]:
        a,b=discrete_matrices(theta,v,settings['geometry_m'],.005,.01)
        u=.015+.012*rng.normal(size=600)
        y=old.propagate(a,u[:,None]*b+[.00013,-.00021],[.002,.09])
        segments.append(dict(speed=v,measured_state=y,applied_input=u))
    statuses=[]
    with threadpool_limits(limits=1):
        for spec in [plan([12,14,16],0,short)[1],plan([12,14,16],0,short)[4],plan([12,14,16],0,short)[-1]]:
            fit=fit_physical_oe(sliced_segments(segments,spec),settings)
            if not fit['converged'] or fit['weighted_mse']>1e-10:raise AssertionError('Synthetic OE failure')
            if spec['kind']=='single':
                candidates=equivalent_parameters(fit['parameters'],12.,settings,41)
                for t in candidates:np.testing.assert_allclose(models(t,[12],settings),models(theta,[12],settings),atol=1e-8,rtol=1e-7)
            statuses.append(dict(kind=spec['kind'],converged=True))
        schedule=design_nodes(models(theta,np.arange(10,31,2),settings),old.load(old.storage.CONFIG))
        outcome,_=simulate(old.acquisition.family_centers()['nominal'],schedule,
            profile(old.load(ROOT/'code/config/paper_oe_transient_benchmark.json'),21),old.load(old.storage.CONFIG))
        if outcome['status']!='complete':raise AssertionError('Synthetic control failed')
    with tempfile.TemporaryDirectory() as d:
        p=Path(d)/'unicode.json';old.atomic_json(p,dict(label='β μ',outcome=outcome))
        if old.envelope(p)['label']!='β μ':raise AssertionError('UTF-8 checkpoint failure')
    return dict(status='coverage_smoke_passed',windows_spawn=True,synthetic_conditions=statuses,scientific_results_written=False)


def finite_mean(values):
    values=[np.nan if v is None else v for v in values]
    return old.strict_mean(values)


def aggregate(frame,metrics):
    # First average variants within each vehicle; then vehicles; then fleets.
    clients=frame.groupby(['seed','client','kind'])[metrics].agg(finite_mean).reset_index()
    fleets=clients.groupby(['seed','kind'])[metrics].agg(finite_mean).reset_index()
    overall=fleets.groupby('kind')[metrics].agg(finite_mean).reindex(KINDS).reset_index()
    return clients,fleets,overall


def analyze(cfg):
    records=[];controls=[];hashes={};grid=np.arange(10,31,2)
    control_present=[path(s,True).exists() for s in cfg['seeds']]
    if any(control_present) and not all(control_present):raise ValueError('Control is partial; finish all fleets before combined analysis')
    complete_control=all(control_present)
    for seed in cfg['seeds']:
        p=old.envelope(path(seed));validate(p,seed,cfg)
        if p['status']!='execution_complete':raise ValueError('Finish all identification conditions before analysis')
        hashes[str(path(seed).relative_to(ROOT))]=old.file_hash(path(seed))
        for r in p['records']:
            fit=r['fit'];s=r['spec'];seen=[w['speed'] for w in s['windows']]
            common=~np.isin(grid,r['original_speeds']);unseen=~np.isin(grid,seen)
            singular=fit.get('information',{}).get('log_parameter_singular_values',[])
            rank=int(np.sum(np.array(singular)>singular[0]*1e-8)) if singular else 0
            record=dict(seed=seed,client=r['client'],id=s['id'],kind=s['kind'],original_speeds=str(r['original_speeds']),
                selected_speeds=str(seen),speed_span=max(seen)-min(seen),seconds=.01*r['transitions'],
                transitions=r['transitions'],converged=fit['converged'],
                relative_rank=rank,log_parameter_condition=float(singular[0]/singular[-1]) if rank==4 else np.nan,
                mu=fit.get('parameters',[np.nan]*4)[3],weighted_mse=fit.get('weighted_mse',np.nan),
                near_bound=bool(any(fit.get('near_bound',[]))),common_unseen_matrix_pct=np.nan,
                common_unseen_increment_pct=np.nan,all_grid_matrix_pct=np.nan,actual_unseen_matrix_pct=np.nan,
                equivalence_max_increment_deviation_pct=np.nan)
            if fit['converged']:
                e=np.array(r['errors']['matrix']);inc=np.array(r['errors']['increment'])
                record.update(common_unseen_matrix_pct=100*e[common].mean(),common_unseen_increment_pct=100*inc[common].mean(),
                    all_grid_matrix_pct=100*e.mean(),actual_unseen_matrix_pct=100*e[unseen].mean())
                if r['single_speed_equivalence']:
                    record['equivalence_max_increment_deviation_pct']=100*max(r['single_speed_equivalence']['max_relative_increment_deviation_from_selected_by_speed'])
            records.append(record)
        if complete_control:
            c=old.envelope(path(seed,True));validate(c,seed,cfg,True)
            if c['status']!='execution_complete':raise ValueError('Control is partial; finish it before combined analysis')
            hashes[str(path(seed,True).relative_to(ROOT))]=old.file_hash(path(seed,True))
            for r in c['records']:
                for row in r['rows']:
                    o=row['outcome'];m=o['metrics'] or {}
                    controls.append(dict(seed=seed,client=r['client'],id=r['spec']['id'],kind=r['spec']['kind'],speed=row['speed'],
                        status=o['status'],feasible=o['feasible'],tracking_rmse_deg_s=m.get('tracking_rmse_deg_s',np.nan),
                        beta_peak_deg=m.get('beta_peak_deg',np.nan),steering_rms_deg=m.get('steering_rms_deg',np.nan)))
    frame=pd.DataFrame(records)
    metrics=['common_unseen_matrix_pct','common_unseen_increment_pct','all_grid_matrix_pct']
    clients,fleets,summary=aggregate(frame,metrics)
    counts=frame.groupby('kind').agg(conditions=('id','size'),converged=('converged','sum'),rank_below_four=('relative_rank',lambda x:int((x<4).sum())),near_bound=('near_bound','sum')).reset_index()
    summary=summary.merge(counts,on='kind')
    worst=frame.groupby(['seed','client','kind'])[metrics].agg(lambda x:np.nan if not np.isfinite(x).all() else x.max()).reset_index()
    block_clients=frame.groupby(['seed','client','original_speeds','kind'])[metrics].agg(finite_mean).reset_index()
    block_fleets=block_clients.groupby(['seed','original_speeds','kind'])[metrics].agg(finite_mean).reset_index()
    blocks=block_fleets.groupby(['original_speeds','kind'])[metrics].agg(finite_mean).reset_index()
    frames={'fits':frame,'clients':clients,'fleets':fleets,'summary':summary,'worst_variants':worst,'blocks':blocks}
    control_summary=None
    if controls:
        cf=pd.DataFrame(controls);cu,cs,control_summary=aggregate(cf,['tracking_rmse_deg_s','beta_peak_deg','steering_rms_deg'])
        counts=cf.groupby('kind').agg(cases=('status','size'),complete=('status',lambda x:int((x=='complete').sum())),feasible=('feasible','sum')).reset_index()
        control_summary=control_summary.merge(counts,on='kind')
        frames.update(control_cases=cf,control_clients=cu,control_fleets=cs,control_summary=control_summary)
    for suffix,f in frames.items():f.to_csv(OUT/f'{PREFIX}_{suffix}.csv',index=False)
    audit=dict(status='analysis_complete',control_included=bool(controls),condition_records=len(frame),
        control_cases=len(controls),checkpoint_sha256=hashes,script_sha256=old.file_hash(__file__),protocol_sha256=old.file_hash(CONFIG),
        note='All variants preserved. Strict means invalidate a complete mean if any contributing fit/rollout failed. Equal variants within vehicle, vehicles within fleet, fleets overall. No independent confirmation.')
    old.acquisition.parent.save_json(OUT/f'{PREFIX}_analysis.json',audit)
    write_report(frame,clients,summary,control_summary,audit,blocks)
    print(summary.to_string(index=False));print(REPORT)


def write_report(frame,clients,summary,control_summary,audit,blocks=None):
    import matplotlib
    matplotlib.use('Agg');matplotlib.rcParams['svg.hashsalt']=PREFIX
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    labels=['Full','Single','Adjacent pair','Multi-speed 20 s']
    for j,kind in enumerate(KINDS):
        values=clients[clients.kind==kind].common_unseen_increment_pct.to_numpy()
        good=np.isfinite(values)&(values>0)
        axes[0].scatter(j+np.linspace(-.18,.18,len(values))[good],values[good],s=12,alpha=.5)
    axes[0].set(xticks=range(4),xticklabels=labels,yscale='log',ylabel='Common unseen [A−I,B] error [%]',title='Vehicle means; failed values omitted from plot only')
    single=frame[frame.kind=='single']
    axes[1].scatter(single.mu,single.equivalence_max_increment_deviation_pct,s=12,alpha=.5)
    axes[1].set(xlabel='Selected single-speed friction parameter μ',ylabel='Maximum equivalent-model deviation [%]',title='Same observed-speed dynamics, different extrapolation')
    axes[0].tick_params(axis='x',labelrotation=15)
    for ax in axes:ax.grid(alpha=.2)
    buf=io.StringIO();fig.savefig(buf,format='svg',metadata={'Date':None});plt.close(fig)
    chart=buf.getvalue();chart=chart[chart.index('<svg'):]
    def table(f):return '<div class="scroll">'+f.to_html(index=False,border=0,float_format=lambda x:f'{x:.5g}',na_rep='unavailable')+'</div>'
    text=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Local CT coverage diagnostic</title><style>body{{font:16px/1.6 system-ui;background:#eef3f6;color:#193142}}main{{max-width:1200px;margin:auto;padding:30px}}section{{background:white;padding:24px;margin:20px 0;border-radius:10px}}.scroll{{overflow:auto}}table{{border-collapse:collapse;font-size:13px;white-space:nowrap}}td,th{{padding:8px;border-bottom:1px solid #ddd}}svg{{width:100%;height:auto}}</style><main>
<h1>Does limited local coverage impair useful physical models?</h1><p>All 90 existing pilot vehicles. Same tanh plant, structured CT estimator, four starts and bounds. No sharing or new acquisition.</p>
<section><h2>Interpret the contrasts</h2><p>Full: original 2–3 speeds, 40–60 seconds; single: each observed speed, 20 seconds; adjacent pair: each neighboring pair, 40 seconds; multi_20s: all original speeds, 20 seconds total. The two-speed full block is reused for its pair condition. Single variants and pair variants are equally averaged within each vehicle first.</p><p>The multi-speed 20-second condition distributes exactly 2000 transitions across central contiguous windows. It controls recording duration, but changes waveform portions, initial states and nuisance-parameter count. It is a quick existing-data diagnostic, not an identical-excitation or equal-information experiment.</p><p>The primary error uses the same original-unobserved grid speeds for every condition of a vehicle. This prevents the evaluation set from changing when observations are removed. All-grid and condition-specific unseen errors are also saved. Conditions are not selected by their test errors.</p></section>
<section><h2>Identification results</h2>{table(summary)}{chart}<p>Rank uses a relative singular-value tolerance of 10⁻⁸ on log-parameter residual sensitivities. Convergence does not establish identifiability. Single-speed models have an exact stiffness/friction ambiguity even when the optimizer reports success. The equivalence plot samples friction within the original bounds and rescales stiffness to preserve observed-speed dynamics exactly; it is not a confidence interval and is never used to choose the reported fit.</p></section>
<section><h2>Matched control</h2>{table(control_summary) if control_summary is not None else '<p>Not included. Commit identification results before running the separate control stage.</p>'}<p>Same 13/21/29 m/s transient, LQI weights, gain interpolation, steering limits and noise-free state feedback. Full-coverage controls are reused. Every new fitted condition is evaluated at all three speeds, with no repair or fallback. Complete finite constraint violations remain in means; failed fits or trajectories make full means unavailable.</p></section>
<section><h2>Decision</h2><p>Look for increased error at common unseen speeds and meaningful control degradation, together with ambiguity and conditioning. A poor friction estimate alone does not establish a need for federation. If multi-speed 20 s improves over single-speed 20 s, this supports a benefit from operating-condition diversity under this windowing protocol. A subsequent sharing comparison must give all local and shared methods the same restricted client data and the same physical model class.</p><p>At fixed curvature, speed also changes lateral acceleration κv². This study varies both speed and tire utilization. Three reused fleets provide descriptive pilot evidence, not a fresh confirmation or a general stability claim.</p></section>
<section><h2>By original coverage block</h2>{table(blocks) if blocks is not None else '<p>Not supplied.</p>'}<p>Variants are averaged within vehicles, then vehicles within each fleet and block, then fleets equally. Blocks contain different vehicles; cross-block differences are descriptive.</p></section>
<section><h2>Every fitted variant</h2>{table(frame)}</section><p>Analysis provenance: {audit['condition_records']} fit-condition records; {audit['control_cases']} control cases. Source hashes are in the companion analysis JSON.</p></main></html>'''
    REPORT.write_text('\n'.join(line.rstrip() for line in text.splitlines())+'\n',encoding='utf-8',newline='\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['preflight','smoke','restore','run','status','control','analyze'])
    parser.add_argument('--workers',type=int,default=3)
    parser.add_argument('--restore-missing',action='store_true')
    args=parser.parse_args()
    if args.workers<1:parser.error('--workers must be positive')
    cfg=preflight();plans=[x for s in cfg['seeds'] for x in fleet_plan(s,cfg)]
    if args.command=='preflight':
        missing=sum(not old.original_raw_path(s,c).exists() for s in cfg['seeds'] for c in old.sources(s)[0]['cells'])
        print(json.dumps(dict(status='coverage_preflight_passed',fleets=3,vehicles=90,condition_records=len(plans),
            new_fits=sum(x['reuse'] is None for x in plans),reused_conditions=sum(x['reuse'] is not None for x in plans),
            missing_raw_cells=missing,new_control_rollouts=3*sum(x['reuse'] is None for x in plans),new_acquisition=0)))
    elif args.command=='status':
        for seed in cfg['seeds']:
            for control in [False,True]:
                if path(seed,control).exists():
                    p=old.envelope(path(seed,control));validate(p,seed,cfg,control)
                    print(json.dumps(dict(seed=seed,stage='control' if control else 'identification',status=p['status'],saved=len(p['records']))))
    elif args.command=='analyze':analyze(cfg)
    elif args.command=='smoke':
        with ProcessPoolExecutor(max_workers=1,mp_context=multiprocessing.get_context('spawn')) as pool:print(json.dumps(pool.submit(smoke_worker).result()))
    else:
        # Use the legacy identification lock too: restoration writes original raw paths.
        with lock(),old.execution_lock('pilot','identification'),ProcessPoolExecutor(max_workers=min(args.workers,3),mp_context=multiprocessing.get_context('spawn')) as pool:
            for future in as_completed([pool.submit(worker,args.command,s,args.restore_missing) for s in cfg['seeds']]):print(json.dumps(future.result()),flush=True)


if __name__=='__main__':main()
