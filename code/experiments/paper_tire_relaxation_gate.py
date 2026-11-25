r"""Yaw-only observability/distinguishability gate; no control or fleet learning.

Windows: python code\experiments\paper_tire_relaxation_gate.py preflight
         python code\experiments\paper_tire_relaxation_gate.py smoke
         python code\experiments\paper_tire_relaxation_gate.py run --workers 3
         python code\experiments\paper_tire_relaxation_gate.py analyze
"""
from __future__ import annotations
import os
for name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[name]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import importlib.metadata
import io
import json
import multiprocessing
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code/src'))
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from federated_lpv.tire_relaxation import (simulate,trajectory_sensitivities,
    observability,distinguishability,continuous_matrices)

PREFIX='paper_tire_relaxation_gate'
CONFIG=ROOT/'code/config'/f'{PREFIX}.json'
MODULE=ROOT/'code/src/federated_lpv/tire_relaxation.py'
OUT=ROOT/'results/tables'
RAW=ROOT/'results/data'/PREFIX
REPORT=ROOT/'report/tire_relaxation_gate.html'


def digest(payload):
    return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def file_hash(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def trace_hash(arrays):
    h=hashlib.sha256()
    for name,value in sorted(arrays.items()):
        a=np.ascontiguousarray(value,dtype='<f8');h.update(name.encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()


def replace_pending(pending,target):
    for attempt in range(21):
        try:os.replace(pending,target);return
        except PermissionError as exc:
            if getattr(exc,'winerror',None) not in [5,32,33] or attempt==20:
                exc.add_note(f'Original preserved; pending file: {pending}');raise
            time.sleep(min(.05*2**min(attempt,4),.5))


def atomic_json(path,payload):
    path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',newline='\n',prefix=path.name+'.',suffix='.tmp',dir=path.parent,delete=False) as stream:
        pending=Path(stream.name);json.dump(dict(payload=payload,payload_sha256=digest(payload)),stream,indent=2,allow_nan=False)
        stream.write('\n');stream.flush();os.fsync(stream.fileno())
    replace_pending(pending,path)


def read_checkpoint(path):
    envelope=json.loads(path.read_text(encoding='utf-8'));p=envelope['payload']
    if digest(p)!=envelope['payload_sha256']:raise ValueError('Checkpoint digest mismatch')
    return p


def save_arrays(path,arrays):
    path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='wb',prefix=path.name+'.',suffix='.tmp',dir=path.parent,delete=False) as stream:
        pending=Path(stream.name);np.savez_compressed(stream,**arrays);stream.flush();os.fsync(stream.fileno())
    replace_pending(pending,path)


def load_config():return json.loads(CONFIG.read_text(encoding='utf-8'))


def identity(cfg):
    return dict(files={p.relative_to(ROOT).as_posix():file_hash(p) for p in [CONFIG,Path(__file__),MODULE]},
        frozen=cfg['source_sha256'],runtime=dict(python=f'{sys.version_info.major}.{sys.version_info.minor}',
        packages={n:importlib.metadata.version(n) for n in ['numpy','scipy','pandas','threadpoolctl']}))


def jobs(cfg):
    return [(c['client'],v,name) for c in cfg['clients'] for v in cfg['speeds_mps'] for name in cfg['inputs']]


def key(job):return f'{job[0]}_v{job[1]}_{job[2]}'


def path(job):return OUT/f'{PREFIX}_{key(job)}.json'


def client(cfg,name):return next(c for c in cfg['clients'] if c['client']==name)


def preflight():
    cfg=load_config()
    for name,sha in cfg['source_sha256'].items():
        if file_hash(ROOT/name)!=sha:raise ValueError(f'Frozen input changed: {name}')
    if len(jobs(cfg))!=18 or len(set(jobs(cfg)))!=18:raise ValueError('Expected 18 unique probe cases')
    if cfg['integration_max_step']>cfg['grid_bounds_m'][0]/max(cfg['speeds_mps'])/2:
        raise ValueError('Integration step too large for search bounds')
    for c in cfg['clients']:
        for field in ['physical_ratios','previous_estimated_ratios']:
            if np.shape(c[field])!=(4,) or min(c[field])<=0:raise ValueError('Invalid physical-ratio snapshot')
    return cfg


def steering(cfg,name):
    dt=cfg['sample_time'];t=np.arange(round(cfg['duration_s']/dt)+1)*dt;s=cfg['inputs'][name]
    u=sum(w*np.sin(2*np.pi*f*t+p) for f,w,p in zip(s['frequencies_hz'],s['weights'],s['phases_rad']))
    taper=np.sin(np.pi/2*np.minimum(np.minimum(t,cfg['duration_s']-t)/cfg['taper_s'],1))**2
    u*=taper;u*=np.deg2rad(cfg['steering_peak_deg'])/max(abs(u))
    return t,u[:-1]


def generate(job,cfg,max_step=None):
    name,v,probe=job;c=client(cfg,name);t,u=steering(cfg,probe)
    p=np.r_[c['physical_ratios'],cfg['relaxation_lengths_m']]
    x=simulate(p,u,v,c['geometry_m'],cfg['sample_time'],max_step or cfg['integration_max_step'])[0]
    index=jobs(cfg).index(job);noise=np.random.default_rng(np.random.SeedSequence([cfg['noise_seed'],index])).normal(0,np.deg2rad(cfg['yaw_noise_std_deg_s']),len(t))
    arrays=dict(time=t,steering=u,state=x,measured_yaw=x[:,1]+noise,noise=noise)
    metrics=dict(beta_peak_deg=float(np.rad2deg(max(abs(x[:,0])))),ay_peak_mps2=float(max(abs(x[:,2]+x[:,3]))),
        steering_rate_peak_deg_s=float(np.rad2deg(max(abs(np.diff(np.r_[0,u,0]))))/cfg['sample_time']))
    record=dict(trace_sha256=trace_hash(arrays),noise_sha256=trace_hash({'noise':noise}),metrics=metrics,
        operating_checks_passed=all(metrics[k]<=bound for k,bound in cfg['operating_bounds'].items()),
        preview={k:val[::10].tolist() for k,val in arrays.items()})
    return record,arrays


def restore(job,cfg,saved):
    target=RAW/f'{key(job)}.npz'
    if target.exists():
        with np.load(target,allow_pickle=False) as data:arrays=dict(data)
    else:
        _,arrays=generate(job,cfg)
    if trace_hash(arrays)!=saved['trace_sha256']:
        raise ValueError('Exact acquisition hash mismatch. Original preserved; restore on the original numerical platform.')
    if not target.exists():save_arrays(target,arrays)
    return arrays


def negative_controls(cfg):
    p=np.array([50.,37.5,.6,.9,.5,.5]);geom=[1.2,1.6];v=21.
    balanced=observability(p,v,geom,cfg['sample_time'],cfg['observability_horizon_s'],cfg['state_scales'],np.deg2rad(cfg['yaw_noise_std_deg_s']),cfg['svd_rcond'])
    _,jac=trajectory_sensitivities(p,np.zeros(30),v,geom,cfg['sample_time'],cfg['integration_max_step'],cfg['state_scales'],np.deg2rad(cfg['yaw_noise_std_deg_s']))
    zero=distinguishability(jac,cfg['svd_rcond'])
    assert balanced['rank']==2 and zero['conditional']['rank']==0
    sweep=[]
    for epsilon in [0.,1e-4,1e-3,1e-2,.1]:
        trial=p.copy();trial[1]*=1+epsilon
        sweep.append(dict(rear_stiffness_relative_shift=epsilon,**observability(trial,v,geom,cfg['sample_time'],cfg['observability_horizon_s'],cfg['state_scales'],np.deg2rad(cfg['yaw_noise_std_deg_s']),cfg['svd_rcond'])))
    return dict(balanced_known_parameter_state_observability=balanced,zero_steering_parameter_information=zero['conditional'],near_balanced_sweep=sweep,
        note='Linear state observability is distinct from parameter distinguishability; these deliberately degenerate controls are not fleet vehicles.')


def surface(base,arrays,v,geom,cfg):
    axis=np.unique(np.r_[np.geomspace(*cfg['grid_bounds_m'],cfg['grid_points']),cfg['relaxation_lengths_m']])
    front,rear=np.meshgrid(axis,axis,indexing='ij')
    pars=np.column_stack([np.tile(base,(front.size,1)),front.ravel(),rear.ravel()])
    y=simulate(pars,arrays['steering'],v,geom,cfg['sample_time'],cfg['integration_max_step'])[:,:,1]
    scale=np.deg2rad(cfg['yaw_noise_std_deg_s']);surfaces={}
    for target,name in [(arrays['state'][:,1],'clean'),(arrays['measured_yaw'],'noisy')]:
        sse=np.sum(((y-target)/scale)**2,axis=1).reshape(front.shape);i,j=np.unravel_index(np.argmin(sse),sse.shape)
        surfaces[name]=dict(sse=sse.tolist(),minimum_grid_sse=float(sse[i,j]),
            best_grid_lengths_m=[float(axis[i]),float(axis[j])],grid_boundary=bool(i in [0,len(axis)-1] or j in [0,len(axis)-1]),
            front_grid_profile_delta_sse=(sse.min(axis=1)-sse.min()).tolist(),rear_grid_profile_delta_sse=(sse.min(axis=0)-sse.min()).tolist())
    return dict(axis_m=axis.tolist(),objectives=surfaces,
        note='Conditional grid minimum/profiles, not continuous maximum-likelihood fits, confidence intervals or globally profiled nuisance parameters. True lengths included as reference grid coordinates.')


def validate(p,job,cfg):
    if p['job']!=list(job) or p['provenance']!=identity(cfg):raise ValueError('Changed protocol/input/runtime; checkpoint preserved')
    if p['status'] not in ['running','execution_complete']:raise ValueError('Unknown checkpoint status')
    if set(p['surfaces'])-{'true_ct','previous_ct'}:raise ValueError('Unknown conditional surface')
    if p['status']=='execution_complete' and (p['data'] is None or p['sensitivity'] is None or set(p['surfaces'])!={'true_ct','previous_ct'}):raise ValueError('Incomplete checkpoint')


def worker(job):
    cfg=preflight();target=path(job)
    with threadpool_limits(limits=1):
        if target.exists():p=read_checkpoint(target);validate(p,job,cfg)
        else:
            p=dict(job=list(job),status='running',provenance=identity(cfg),data=None,sensitivity=None,surfaces={},
                implementation_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
        if p['status']=='execution_complete':return dict(case=key(job),status='already_complete')
        if p['data'] is None:
            p['data'],arrays=generate(job,cfg);save_arrays(RAW/f'{key(job)}.npz',arrays);atomic_json(target,p)
        else:arrays=restore(job,cfg,p['data'])
        c=client(cfg,job[0]);v=job[1];theta=np.r_[c['physical_ratios'],cfg['relaxation_lengths_m']]
        if p['sensitivity'] is None:
            _,jac=trajectory_sensitivities(theta,arrays['steering'],v,c['geometry_m'],cfg['sample_time'],cfg['integration_max_step'],cfg['state_scales'],np.deg2rad(cfg['yaw_noise_std_deg_s']))
            p['sensitivity']=dict(parameter=distinguishability(jac,cfg['svd_rcond']),
                parameter_rcond_sensitivity={str(t):distinguishability(jac,t) for t in [1e-8,1e-10]},
                state=observability(theta,v,c['geometry_m'],cfg['sample_time'],cfg['observability_horizon_s'],cfg['state_scales'],np.deg2rad(cfg['yaw_noise_std_deg_s']),cfg['svd_rcond']))
            atomic_json(target,p)
        for label,base in [('true_ct',c['physical_ratios']),('previous_ct',c['previous_estimated_ratios'])]:
            if label not in p['surfaces']:
                p['surfaces'][label]=surface(base,arrays,v,c['geometry_m'],cfg);atomic_json(target,p)
        p['status']='execution_complete';atomic_json(target,p)
    return dict(case=key(job),status=p['status'],operating_checks_passed=p['data']['operating_checks_passed'])


def render(records,state_table,parameter_table,cfg,negative):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({'svg.hashsalt':PREFIX,'font.size':10})
    def svg(fig):
        buffer=io.StringIO();fig.savefig(buffer,format='svg',bbox_inches='tight',metadata={'Date':None});plt.close(fig)
        text=buffer.getvalue();return text[text.index('<svg'):]
    def table(frame):return '<div class="scroll">'+frame.to_html(index=False,border=0,float_format=lambda x:f'{x:.5g}',na_rep='rank deficient')+'</div>'
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for ax,probe in zip(axes,cfg['inputs']):
        for nuisance in ['conditional','unknown_ct','unknown_ct_and_initial']:
            rows=parameter_table[(parameter_table.probe==probe)&(parameter_table.assumption==nuisance)]
            for name,group in rows.groupby('client',sort=False):
                ax.semilogy(group.speed,np.maximum(group.smallest_singular,1e-14),'-o',alpha=.7,label=f'{name}: {nuisance}')
        ax.set(title=probe,xlabel='Speed [m/s]',ylabel='Smallest singular value, whitened log-length sensitivity');ax.grid(alpha=.2)
    axes[1].legend(fontsize=6,loc='center left',bbox_to_anchor=(1, .5));charts=svg(fig)
    # Prespecified nominal vehicle at middle speed, both input spectra, all cases stay in tables.
    for p in records:
        if p['job'][0]!=cfg['clients'][0]['client'] or p['job'][1]!=cfg['speeds_mps'][1]:continue
        fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
        for ax,base in zip(axes,['true_ct','previous_ct']):
            s=p['surfaces'][base];axis=np.array(s['axis_m']);z=np.array(s['objectives']['noisy']['sse']);z=np.log10(1+z-z.min())
            mesh=ax.pcolormesh(axis,axis,z.T,shading='nearest',cmap='viridis')
            ax.plot(*cfg['relaxation_lengths_m'],'rx',ms=9,label='Generating lengths')
            ax.plot(*s['objectives']['noisy']['best_grid_lengths_m'],'wo',ms=5,label='Best grid point')
            ax.set(xscale='log',yscale='log',xlabel='Front relaxation length [m]',ylabel='Rear relaxation length [m]',title=base);ax.legend(fontsize=8)
            fig.colorbar(mesh,ax=ax,label='log10(1 + Δ weighted SSE)')
        fig.suptitle(f'{p["job"][0]}, {p["job"][1]} m/s, {p["job"][2]} input');charts+=svg(fig)
        fig,axes=plt.subplots(1,2,figsize=(11,3.6),layout='constrained')
        for ax,which,label in zip(axes,['front','rear'],['Front','Rear']):
            for base,color in [('true_ct','#167c96'),('previous_ct','#bc6b24')]:
                s=p['surfaces'][base]
                for target,style in [('clean','--'),('noisy','-')]:
                    ax.plot(s['axis_m'],s['objectives'][target][f'{which}_grid_profile_delta_sse'],style,color=color,label=f'{base}, {target}')
            ax.set(xscale='log',yscale='symlog',xlabel=f'{label} relaxation length [m]',ylabel='Conditional grid-profile Δ weighted SSE')
            ax.grid(alpha=.2)
        axes[0].legend(fontsize=8);fig.suptitle(f'{p["job"][2]} input — other length minimized on the grid');charts+=svg(fig)
    surface_rows=[]
    for p in records:
        for base,s in p['surfaces'].items():
            for target,val in s['objectives'].items():
                surface_rows.append(dict(client=p['job'][0],speed=p['job'][1],probe=p['job'][2],fixed_ct=base,target=target,
                    front_grid_m=val['best_grid_lengths_m'][0],rear_grid_m=val['best_grid_lengths_m'][1],
                    minimum_grid_sse=val['minimum_grid_sse'],grid_boundary=val['grid_boundary']))
    surfaces=pd.DataFrame(surface_rows)
    text=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Tire relaxation: yaw-only information gate</title>
<style>body{{font:16px/1.6 system-ui;background:#edf2f5;color:#193447}}main{{max-width:1200px;margin:auto;padding:30px}}section{{background:white;padding:24px;border-radius:10px;margin:22px 0}}table{{border-collapse:collapse;font-size:12px;white-space:nowrap}}td,th{{padding:8px;border-bottom:1px solid #ddd}}.scroll{{overflow:auto}}svg{{width:100%;height:auto}}summary{{cursor:pointer}}</style><main><h1>Tire-force relaxation: what can yaw measurements distinguish?</h1>
<section><h2>Scope</h2><p>This is a matched synthetic information diagnostic: three reused pilot vehicles, three fixed speeds, two prescribed open-loop steering probes. State is [β,r,Fyf/m,Fyr/m]. The existing tanh force law is unchanged. True relaxation lengths are 0.4 and 0.7 m; constant axle values are declared simulation assumptions within a published experimental range, not calibrated values for these vehicles. No Pacejka change, controller synthesis, federation, or manuscript update is included.</p><p>Only yaw is used for the information calculations and objective surfaces; applied steering and speed are known. Generated initial states are zero. The target force states and sideslip are not supplied to an estimator. Noise SD is 0.03°/s. Open-loop probes are controlled excitation, not claims about ordinary driving. Operating-bound failures remain visible and do not count as feasible identification procedures.</p></section>
<section><h2>How to read the diagnostics</h2><p><b>State observability:</b> finite-horizon sampled yaw observation matrix at straight running, with known parameters and declared state scales. Full numerical rank is a local linear-model property. It is not parameter identifiability or global nonlinear observability.</p><p><b>Parameter distinguishability:</b> analytic complex-step derivatives of the nonlinear yaw trajectory with respect to log parameters and scaled initial states. Relaxation columns are projected away from nuisance sensitivities. The smallest singular value measures the least visible local log-length direction in noise units. Conditional assumes exact CT ratios and initial states. Unknown CT permits variations in Cf/m, Cr/m, m/Iz and μ. Unknown initial permits all four initial states. No prior regularization is used. Rank and conditioning depend on scaling, horizon and the declared relative SVD threshold; 1e−8 and 1e−10 alternatives are saved too.</p><p><b>Objective surfaces:</b> weighted yaw SSE over a fixed length grid, with clean and noisy targets. True CT fixes other parameters to generating values; Previous CT fixes them to saved pre-change estimates. Comparing them reveals how baseline error can shift the preferred lengths. Marginal grid minima are coarse conditional profiles, not continuously optimized fits, nuisance-profile likelihoods or confidence intervals. The generating lengths are included as reference coordinates.</p><p>These results cannot prove global structural identifiability. Low residuals alone do not prove correct force or sideslip reconstruction.</p></section>
<section><h2>Operating checks and linearized state observability</h2>{table(state_table)}</section>
<section><h2>Information and conditional objective surfaces</h2>{charts}{table(parameter_table)}<details><summary>All conditional grid minima (no independence claim)</summary>{table(surfaces)}</details></section>
<section><h2>Analytical negative checks</h2><p>For equal relaxation times, det(O) for physical-force coordinates is (lf+lr)(lf Cf−lr Cr)²/(Iz³ m τ² v). The normalized-force implementation has the same rank. At lf Cf=lr Cr the sampled four-state observability rank is {negative['balanced_known_parameter_state_observability']['rank']} (expected 2). With zero steering and zero initial state, parameter sensitivity rank is {negative['zero_steering_parameter_information']['rank']} (expected 0). Both assertions must pass.</p>{table(pd.DataFrame(negative['near_balanced_sweep']).drop(columns=['singular_values']))}</section>
<section><h2>Decision before any control comparison</h2><p>Inspect whether the two relaxation directions remain distinguishable after CT and initial-state uncertainty, whether Previous CT moves the best lengths substantially, and whether the required probe remains within the prescribed operating bounds. A rank check alone is not a pass criterion. Broad or boundary-reaching profiles require a narrower parameterization or a separately justified sensor/input change. A useful distinction under ideal assumptions must then survive estimation and independent prediction tests before a control benefit can be claimed.</p><p><a href="../docs/tire_relaxation_gate.html">Equations, assumptions, reproduction and Windows commands</a></p></section></main></html>'''
    REPORT.write_text('\n'.join(line.rstrip() for line in text.splitlines())+'\n',encoding='utf-8',newline='\n')
    return surfaces


def analyze(cfg):
    records=[];state=[];parameter=[]
    for job in jobs(cfg):
        p=read_checkpoint(path(job));validate(p,job,cfg)
        if p['status']!='execution_complete':raise ValueError('Complete all 18 cases first')
        records.append(p)
        state.append(dict(client=job[0],speed=job[1],probe=job[2],operating_checks_passed=p['data']['operating_checks_passed'],
            **p['data']['metrics'],**{k:v for k,v in p['sensitivity']['state'].items() if k!='singular_values'}))
        for name,info in p['sensitivity']['parameter'].items():
            parameter.append(dict(client=job[0],speed=job[1],probe=job[2],assumption=name,**{k:v for k,v in info.items() if k!='singular_values'}))
    neg=negative_controls(cfg);st=pd.DataFrame(state);pt=pd.DataFrame(parameter)
    surfaces=render(records,st,pt,cfg,neg)
    for name,data in [('observability',st),('sensitivity',pt),('grid_minima',surfaces)]:data.to_csv(OUT/f'{PREFIX}_{name}.csv',index=False)
    atomic_json(OUT/f'{PREFIX}_analysis.json',dict(status='analysis_complete',provenance=identity(cfg),negative_controls=neg,
        checkpoint_sha256={path(j).relative_to(ROOT).as_posix():file_hash(path(j)) for j in jobs(cfg)},
        cases=len(records),automatic_scientific_pass=False))
    print(REPORT)


def smoke():
    """Short synthetic checks, no scientific-case checkpoints or result tables."""
    cfg=preflight();neg=negative_controls(cfg);p=np.array([50.,55.,.6,.9,.4,.7]);geom=[1.2,1.6]
    u=.004*np.sin(2*np.pi*.8*np.arange(150)*cfg['sample_time'])
    x,jac=trajectory_sensitivities(p,u,21.,geom,cfg['sample_time'],cfg['integration_max_step'],cfg['state_scales'],np.deg2rad(cfg['yaw_noise_std_deg_s']))
    finer=simulate(p,u,21.,geom,cfg['sample_time'],cfg['integration_max_step']/2)[0]
    assert np.max(abs(x-finer))<1e-7
    h=1e-4;plus=p.copy();minus=p.copy();plus[4]*=np.exp(h);minus[4]*=np.exp(-h)
    y=simulate([plus,minus],u,21.,geom,cfg['sample_time'],cfg['integration_max_step'])[:,:,1]
    numeric=(y[0]-y[1])/(2*h*np.deg2rad(cfg['yaw_noise_std_deg_s']))
    np.testing.assert_allclose(jac[:,4],numeric,atol=1e-7,rtol=1e-5)
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/'utf8.json';atomic_json(path,{'message':'β → μ; Prüfung'});assert read_checkpoint(path)['message']=='β → μ; Prüfung'
    return dict(status='smoke_passed',balanced_rank=neg['balanced_known_parameter_state_observability']['rank'],
        zero_input_parameter_rank=neg['zero_steering_parameter_information']['rank'],
        max_step_halving_state_difference=float(np.max(abs(x-finer))),scientific_cases_run=0)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['preflight','smoke','run','status','analyze','restore'])
    parser.add_argument('--workers',type=int,default=3);parser.add_argument('--case',help='Exact case ID, for restore only')
    args=parser.parse_args();cfg=preflight()
    if args.command=='preflight':print(json.dumps(dict(status='preflight_passed',cases=len(jobs(cfg)),workers=args.workers,control_runs=0,fleet_learning=False)));return
    if args.command=='smoke':
        with ProcessPoolExecutor(max_workers=1,mp_context=multiprocessing.get_context('spawn')) as pool:print(json.dumps(pool.submit(smoke).result()))
        return
    if args.command=='status':
        for job in jobs(cfg):
            if path(job).exists():p=read_checkpoint(path(job));validate(p,job,cfg);status=p['status']
            else:status='missing'
            print(json.dumps(dict(case=key(job),status=status)))
        return
    if args.command=='analyze':analyze(cfg);return
    if args.command=='restore':
        selected=[j for j in jobs(cfg) if args.case is None or key(j)==args.case]
        if not selected:raise ValueError('Unknown case ID')
        for job in selected:
            p=read_checkpoint(path(job));validate(p,job,cfg);restore(job,cfg,p['data']);print(json.dumps(dict(case=key(job),status='exact_arrays_verified')))
        return
    if not 1<=args.workers<=18:raise ValueError('workers must be between 1 and 18')
    RAW.mkdir(parents=True,exist_ok=True);lock=RAW/'run.lock'
    try:fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError:raise RuntimeError(f'Run lock exists: {lock}. Remove only after confirming no other gate process is running.')
    os.write(fd,str(os.getpid()).encode());os.close(fd)
    try:
        with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            for result in as_completed([pool.submit(worker,j) for j in jobs(cfg)]):print(json.dumps(result.result()),flush=True)
    finally:lock.unlink()
    print('Execution complete. Commit/push checkpoints before analyze (see protocol).')


if __name__=='__main__':main()
