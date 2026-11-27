r"""CFL-FF population diagnostic: is a family-specific learned prior useful?

Windows: python code\experiments\paper_cfl_heterogeneity.py preflight
         python code\experiments\paper_cfl_heterogeneity.py smoke
         python code\experiments\paper_cfl_heterogeneity.py run --workers 5
         python code\experiments\paper_cfl_heterogeneity.py analyze
The run command resumes completed starts, clients, rounds and test cases.
"""
from __future__ import annotations
import os
for name in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[name] = '1'
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
import copy
import html
import importlib.metadata
import io
import json
import multiprocessing
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'code/src'), str(ROOT/'code/experiments')]
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import paper_federated_relaxation_coverage as base
from federated_lpv import cfl_population as population
from federated_lpv import federated_relaxation as fl
from federated_lpv.tire_relaxation import continuous_matrices

PREFIX = 'paper_cfl_heterogeneity'
CONFIG = ROOT/'code/config'/f'{PREFIX}.json'
OUT = ROOT/'results/tables'
RAW = ROOT/'results/data'/PREFIX
REPORT = ROOT/'report/cfl_heterogeneity.html'
METHODS = ['LocalOE', 'NominalPrior', 'SharedPrior', 'OracleFamily', 'MixedGroups']
storage = base.storage


def preflight():
    cfg = json.loads(CONFIG.read_text(encoding='utf-8'))
    for name, sha in cfg['source_sha256'].items():
        if base.canonical_hash(ROOT/name) != sha:raise ValueError(f'Source changed: {name}')
    assert cfg['methods'] == METHODS+['Exact']
    assert len(set(cfg['seeds'])) == len(cfg['seeds']) == 3
    assert [f['seed'] for f in cfg['fleets']] == cfg['seeds']
    assert cfg['population']['anchors']['Mini']['source_label'] == 'Compact'
    assert cfg['population']['relaxation_lengths_m'] == [.4, .7]
    np.testing.assert_allclose(cfg['nominal_parameters'], population.nominal_center(cfg['population']), rtol=1e-13)
    bounds = np.array(cfg['fit']['bounds'])
    for fleet in cfg['fleets']:
        expected = population.generate_fleet(fleet['seed'], cfg['population'], cfg['fleet_training_speeds'])
        for saved, generated in zip(fleet['clients'], expected['clients']):
            for name in ['family','training_speed','mixed_group','client']:
                assert saved[name] == generated[name], 'Population assignment changed'
            for name in ['geometry_m','physical_parameters']:
                np.testing.assert_allclose(saved[name],generated[name],rtol=1e-12,atol=1e-14)
            for name in saved['physical_draw']:
                np.testing.assert_allclose(saved['physical_draw'][name],generated['physical_draw'][name],rtol=1e-12,atol=1e-14)
        assert len(fleet['clients']) == 18
        for key, members in blocks(fleet).items():
            assert len(members) == (18 if key == 'SharedPrior_all' else 6)
            assert sorted(c['training_speed'] for c in members) == sorted(cfg['fleet_training_speeds']*(len(members)//3))
            if key.startswith('MixedGroups'):
                assert all(sum(c['family']==f for c in members)==2 for f in cfg['population']['anchors'])
        for c in fleet['clients']:
            theta = np.array(c['physical_parameters'])
            assert np.all((theta > bounds[:, 0]) & (theta < bounds[:, 1]))
    for p in cfg['starts']:
        assert np.all((p > bounds[:, 0]) & (p < bounds[:, 1]))
    assert cfg['integration_max_step'] <= bounds[4:, 0].min()/max(cfg['evaluation_speeds'])/2
    assert cfg['sharing']['weights'] == [10, 10, 10, 0, 4, 4]
    return cfg


def blocks(fleet):
    clients = fleet['clients']
    groups = {'SharedPrior_all': clients}
    groups.update({f'OracleFamily_{f}':[c for c in clients if c['family']==f] for f in ['Mini','Sport','SUV']})
    groups.update({f'MixedGroups_{g}':[c for c in clients if c['mixed_group']==g] for g in range(3)})
    return groups


def identity():
    paths = [CONFIG, Path(__file__), Path(population.__file__), Path(base.__file__),
        Path(fl.__file__), Path(storage.__file__), ROOT/'code/experiments/paper_tire_relaxation_pilot.py',
        *[ROOT/'code/src/federated_lpv'/f'{name}.py' for name in
          ['tire_relaxation', 'tire_relaxation_pilot', 'physical_ct_oe', 'structured_oe']]]
    return dict(files={p.relative_to(ROOT).as_posix():base.canonical_hash(p) for p in paths},
        runtime=dict(python=f'{sys.version_info.major}.{sys.version_info.minor}',
            packages={n:importlib.metadata.version(n) for n in ['numpy','scipy','pandas','scikit-learn','threadpoolctl']}))


def path(seed, stage, key):return OUT/f'{PREFIX}_seed{seed}_{stage}_{key}.json'
def fleet_for(cfg, seed):return next(f for f in cfg['fleets'] if f['seed']==seed)
def client_for(cfg, seed, key):return next(c for c in fleet_for(cfg,seed)['clients'] if c['client']==key)


def load(pth, provenance, strict_runtime=True):
    p = storage.read_checkpoint(pth)
    actual, expected = p['provenance'], provenance
    if actual['files'] != expected['files'] or (strict_runtime and actual['runtime'] != expected['runtime']):
        raise ValueError(f'Code/protocol/runtime mismatch; original preserved: {pth}')
    return p


def start_payload(seed, key, provenance):
    return dict(seed=seed, key=key, status='running', provenance=provenance,
        implementation_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())


@contextmanager
def task_lock(pth):
    """OS releases advisory locks even after a crash; never deletes another writer's lock."""
    pth.parent.mkdir(parents=True, exist_ok=True)
    lock = RAW/'locks'/pth.name
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open('a+b') as stream:
        if stream.tell()==0:stream.write(b'0');stream.flush()
        stream.seek(0)
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:yield
        finally:
            stream.seek(0)
            if os.name=='nt':msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def save(pth, p, stage):
    storage.atomic_json(pth, p)
    print(json.dumps(dict(seed=p['seed'], key=p['key'], saved=stage)), flush=True)


def training(c, seed, cfg, old=None):
    target = RAW/str(seed)/f'{c["client"]}_training.npz'
    if old is not None and target.exists():
        with np.load(target, allow_pickle=False) as saved:arrays = dict(saved)
        record = old
    else:arrays, record = base.generate(c, seed, 'training', c['training_speed'], cfg)
    if old is not None and storage.trace_hash(arrays) != old['trace_sha256']:
        raise ValueError('Training hash mismatch; restore on the original numerical platform')
    if not target.exists():storage.save_arrays(target, arrays)
    observed = dict(speed=c['training_speed'], applied_input=arrays['steering'], measured_yaw=arrays['measured_yaw'])
    return observed, record


def local_task(seed, key, cfg, provenance):
    pth = path(seed, 'local', key);c = client_for(cfg, seed, key)
    with task_lock(pth):
        p = load(pth, provenance) if pth.exists() else dict(start_payload(seed,key,provenance), attempts=[], prior_attempts=[])
        if p['status']=='complete':return dict(seed=seed, key=key, status='already_complete')
        record, data = training(c, seed, cfg, p.get('data'))
        p['data']=data;save(pth,p,'acquired')
        if not data['operating_checks_passed']:
            raise ValueError('Training operating check failed; saved without resampling')
        endpoint = fl.Client([record], c['geometry_m'], base.settings(cfg))
        for j in range(len(p['attempts']), len(cfg['starts'])):
            p['attempts'].append(endpoint.fit(np.log(cfg['starts'][j])))
            save(pth,p,f'local_start{j}')
        p['local']=fl.select_local(p['attempts'])
        if not p['local']['available']:
            save(pth,p,'local_failed');raise ValueError(f'No converged local fit: {seed}/{key}')
        starts = [p['local']['log_parameters'], np.log(cfg['nominal_parameters'])]
        for j in range(len(p['prior_attempts']), len(starts)):
            p['prior_attempts'].append(endpoint.fit(starts[j],np.log(cfg['nominal_parameters']),cfg['sharing']['weights']))
            save(pth,p,f'prior_start{j}')
        p['prior']=fl.select_local(p['prior_attempts'])
        if not p['prior']['available']:
            save(pth,p,'prior_failed');raise ValueError(f'No converged nominal-prior fit: {seed}/{key}')
        p['status']='complete';save(pth,p,'complete')
    return dict(seed=seed,key=key,status='complete')


def group_task(seed, key, cfg, provenance):
    members = blocks(fleet_for(cfg,seed))[key];pth=path(seed,'group',key)
    with task_lock(pth):
        parents = {path(seed,'local',c['client']).name:base.canonical_hash(path(seed,'local',c['client'])) for c in members}
        local = [load(path(seed,'local',c['client']),provenance) for c in members]
        if not all(p['status']=='complete' for p in local):raise ValueError('Local stage incomplete')
        if pth.exists():
            p=load(pth,provenance)
            if p['parents']!=parents:raise ValueError('Local parents changed; group preserved')
        else:
            state=fl.initialize_personal([p['local']['log_parameters'] for p in local],1,0,cfg['sharing']['weights'])
            p=dict(start_payload(seed,key,provenance),parents=parents,client_ids=[c['client'] for c in members],state=state)
        if p['status']=='complete':return dict(seed=seed,key=key,status='already_complete',convergence=p['state']['status'])
        endpoints=[fl.Client([training(c,seed,cfg,lp['data'])[0]],c['geometry_m'],base.settings(cfg)) for c,lp in zip(members,local)]
        sh=cfg['sharing'];state=p['state']
        while state['status']=='running':
            for i,endpoint in enumerate(endpoints):
                if str(i) in state['pending']:continue
                result=endpoint.fit(state['log_parameters'][i],state['centers'][0],sh['weights'],sh['local_round_max_nfev'])
                state['pending'][str(i)]=result;save(pth,p,f'round{state["round"]+1}_client{i}')
            fits=[state['pending'][str(i)] for i in range(len(endpoints))]
            if not all(f['available'] for f in fits):
                save(pth,p,'update_failed');raise ValueError('Unavailable proximal update; no fallback')
            state=fl.finish_personal_round(state,[fl.client_round_message(f) for f in fits],sh['weights'],sh,
                                          all(f['optimizer_converged'] for f in fits))
            state['history'][-1]['local_updates']=[{k:f[k] for k in ['optimizer_converged','nfev','optimality','near_bound']} for f in fits]
            p['state']=state;save(pth,p,f'round{state["round"]}')
        p['status']='complete';save(pth,p,state['status'])
    return dict(seed=seed,key=key,status='complete',convergence=state['status'])


def fitted(seed, c, cfg, provenance, strict_runtime=True):
    local_path=path(seed,'local',c['client']);lp=load(local_path,provenance,strict_runtime)
    if lp['status']!='complete':raise ValueError('Incomplete local fit')
    parameters={'LocalOE':lp['local']['parameters'],'NominalPrior':lp['prior']['parameters']}
    hashes={local_path.name:base.canonical_hash(local_path)};statuses={m:'converged' for m in parameters}
    keys=['SharedPrior_all',f'OracleFamily_{c["family"]}',f'MixedGroups_{c["mixed_group"]}']
    for key in keys:
        pth=path(seed,'group',key);p=load(pth,provenance,strict_runtime)
        if p['status']!='complete':raise ValueError('Incomplete sharing stage')
        for name,sha in p['parents'].items():
            if base.canonical_hash(OUT/name)!=sha:raise ValueError('Group parent changed')
        index=p['client_ids'].index(c['client']);method=key.split('_')[0]
        parameters[method]=np.exp(p['state']['log_parameters'][index]).tolist()
        statuses[method]=p['state']['status'];hashes[pth.name]=base.canonical_hash(pth)
    parameters['Exact']=c['physical_parameters'];statuses['Exact']='reference'
    return parameters,statuses,hashes


def evaluate(c, seed, v, probe, cfg, parameters, statuses):
    arrays,plant=base.generate(c,seed,probe,v,cfg)
    names=list(parameters)
    predicted=base.batch_predict([parameters[m] for m in names],[arrays['steering']],[v],c['geometry_m'],
                                cfg['sample_time'],cfg['integration_max_step'])[:,0]
    rows={}
    for i,name in enumerate(names):
        arrays[f'predicted_{name}']=predicted[i]
        error=predicted[i,:,1]-arrays['state'][:,1]
        rows[name]=dict(yaw_rmse_deg_s=float(np.rad2deg(np.sqrt(np.mean(error**2)))),
            beta_rmse_deg=float(np.rad2deg(np.sqrt(np.mean((predicted[i,:,0]-arrays['state'][:,0])**2)))),
            optimizer_status=statuses[name])
    return dict(speed=v,probe=probe,region=fl.coverage_region(v,c['training_speed'],cfg['fleet_training_speeds']),
        methods=rows,operating_checks_passed=plant['operating_checks_passed'],plant_metrics=plant['metrics'],
        trace_sha256=storage.trace_hash(arrays)),arrays


def eval_task(seed,key,cfg,provenance):
    c=client_for(cfg,seed,key);pth=path(seed,'eval',key)
    with task_lock(pth):
        pars,statuses,parents=fitted(seed,c,cfg,provenance)
        p=load(pth,provenance) if pth.exists() else dict(start_payload(seed,key,provenance),parents=parents,cases={})
        if p['parents']!=parents:raise ValueError('Fit changed after evaluation; results preserved')
        if p['status']=='complete':return dict(seed=seed,key=key,status='already_complete')
        for v in cfg['evaluation_speeds']:
            for probe in ['slow','broadband']:
                case=f'v{v}_{probe}'
                if case in p['cases']:continue
                result,arrays=evaluate(c,seed,v,probe,cfg,pars,statuses)
                storage.save_arrays(RAW/str(seed)/f'{key}_{case}.npz',arrays)
                p['cases'][case]=result;save(pth,p,case)
        p['status']='complete';save(pth,p,'complete')
    return dict(seed=seed,key=key,status='complete')


def execute(job,cfg,provenance):
    stage,seed,key=job
    with threadpool_limits(limits=1):return {'local':local_task,'group':group_task,'eval':eval_task}[stage](seed,key,cfg,provenance)


def run(cfg,workers):
    provenance=identity()
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        for stage in ['local','group','eval']:
            jobs=[(stage,f['seed'],key) for f in cfg['fleets'] for key in
                  (blocks(f) if stage=='group' else [c['client'] for c in f['clients']])]
            futures=[pool.submit(execute,job,cfg,provenance) for job in jobs]
            for future in as_completed(futures):print(json.dumps(future.result()),flush=True)
    analyze(cfg)


def restore(cfg):
    provenance=identity();count=0
    for f in cfg['fleets']:
        seed=f['seed']
        for c in f['clients']:
            key=c['client'];lp=load(path(seed,'local',key),provenance)
            training(c,seed,cfg,lp['data']);count+=1
            ep=path(seed,'eval',key)
            if not ep.exists():continue
            p=load(ep,provenance);pars,statuses,parents=fitted(seed,c,cfg,provenance)
            if p['parents']!=parents:raise ValueError('Frozen parents changed')
            for case,saved in p['cases'].items():
                target=RAW/str(seed)/f'{key}_{case}.npz'
                if target.exists():
                    with np.load(target,allow_pickle=False) as old:arrays=dict(old)
                else:_,arrays=evaluate(c,seed,saved['speed'],saved['probe'],cfg,pars,statuses)
                if storage.trace_hash(arrays)!=saved['trace_sha256']:raise ValueError('Exact restore mismatch; use original runtime')
                if not target.exists():storage.save_arrays(target,arrays)
                count+=1
    print(json.dumps(dict(status='exact_arrays_verified',traces=count)))


def progression(fleet_means,converged,operating_ok,cfg):
    """A descriptive next-step screen, never a test of statistical significance."""
    mean=fleet_means.mean();comparisons={}
    for other in ['SharedPrior','MixedGroups','NominalPrior']:
        comparisons[other]=dict(gain_percent=float(100*(1-mean.OracleFamily/mean[other])),
            absolute_reduction_deg_s=float(mean[other]-mean.OracleFamily),
            fleets_improved=int((fleet_means.OracleFamily<fleet_means[other]).sum()))
    threshold=cfg['screen'];checks=[converged,operating_ok]
    for name in ['SharedPrior','MixedGroups','NominalPrior']:
        checks.extend([comparisons[name]['gain_percent']>=threshold['min_gain_percent'],
                       comparisons[name]['fleets_improved']>=threshold['min_fleets_improved']])
    return dict(status='supports_testing_learned_groups' if all(checks) else 'no_automatic_progression',
        comparisons=comparisons,all_fits_converged=converged,all_operating_checks_passed=operating_ok,
        interpretation='Family labels define one privileged partition, not the optimal partition. A negative result does not rule out other useful groups.')


def analyze(cfg):
    provenance=identity();rows=[];optim=[];sources={};params=[];fleet_description=[]
    for f in cfg['fleets']:
        seed=f['seed']
        for key in blocks(f):
            pth=path(seed,'group',key);p=load(pth,provenance,False)
            if p['status']!='complete':raise ValueError('Complete all fits before analysis')
            s=p['state'];optim.append(dict(seed=seed,group=key,status=s['status'],rounds=s['round'],
                objective=s['history'][-1]['objective'],communication_float64=s['communication_float64']))
            sources[pth.relative_to(ROOT).as_posix()]=base.canonical_hash(pth)
        for c in f['clients']:
            key=c['client'];pars,statuses,parents=fitted(seed,c,cfg,provenance,False)
            pth=path(seed,'eval',key);p=load(pth,provenance,False)
            if p['status']!='complete' or p['parents']!=parents:raise ValueError('Evaluation incomplete or stale')
            expected={f'v{v}_{probe}' for v in cfg['evaluation_speeds'] for probe in ['slow','broadband']}
            if set(p['cases'])!=expected:raise ValueError('Missing evaluation cases')
            sources[pth.relative_to(ROOT).as_posix()]=base.canonical_hash(pth)
            for name,sha in parents.items():sources[(OUT/name).relative_to(ROOT).as_posix()]=sha
            lp=load(path(seed,'local',key),provenance,False)
            for method in ['local','prior']:
                if not lp[method]['available'] or not lp[method]['optimizer_converged']:raise ValueError('Local baseline not converged')
            for case in p['cases'].values():
                if case['methods']['Exact']['yaw_rmse_deg_s']>1e-10:raise ValueError('Exact-model numerical check failed')
                for m,metrics in case['methods'].items():
                    rows.append(dict(seed=seed,client=key,family=c['family'],training_speed=c['training_speed'],
                        speed=case['speed'],probe=case['probe'],region=case['region'],
                        operating_checks_passed=case['operating_checks_passed'],method=m,**metrics))
            for method,theta in pars.items():
                params.append(dict(seed=seed,client=key,family=c['family'],method=method,
                    **{f'{name}_log_error':float(np.log(theta[j]/c['physical_parameters'][j])) for j,name in
                       enumerate(['p1','p2','p3','mu','sigma_f','sigma_r'])}))
            fleet_description.append(dict(seed=seed,client=key,family=c['family'],speed=c['training_speed'],
                lf_m=c['geometry_m'][0],lr_m=c['geometry_m'][1],**c['physical_draw'],
                **{name:value for name,value in zip(['p1','p2','p3','mu','sigma_f','sigma_r'],c['physical_parameters'])}))
    cases=pd.DataFrame(rows)
    primary=cases[(cases.region=='peer_observed') & cases.method.isin(METHODS)]
    vehicle=primary.groupby(['seed','client','family','training_speed','method']).yaw_rmse_deg_s.mean().unstack('method')[METHODS].reset_index()
    fleet=vehicle.groupby('seed')[METHODS].mean()
    by_family=vehicle.groupby(['seed','family'])[METHODS].mean().reset_index()
    by_speed=vehicle.groupby(['seed','training_speed'])[METHODS].mean().reset_index()
    converged=all(o['status']=='converged' for o in optim)
    operating=bool(cases.operating_checks_passed.all())
    decision=progression(fleet,converged,operating,cfg)
    for name,frame in [('cases',cases),('vehicles',vehicle),('fleets',fleet.reset_index()),('families',by_family),
                       ('speeds',by_speed),('optimization',pd.DataFrame(optim)),('parameters',pd.DataFrame(params)),
                       ('population',pd.DataFrame(fleet_description))]:
        frame.to_csv(OUT/f'{PREFIX}_{name}.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    def svg(fig):
        buf=io.StringIO();fig.savefig(buf,format='svg',bbox_inches='tight',metadata={'Date':None});plt.close(fig)
        value=buf.getvalue();return value[value.index('<svg'):]
    fig,ax=plt.subplots(figsize=(8,3.6))
    for seed,row in fleet.iterrows():ax.plot(range(len(METHODS)),row[ METHODS],marker='o',label=f'Fleet {seed}')
    ax.set(xticks=range(len(METHODS)),xticklabels=['Local OE','Nominal prior','Shared prior','Oracle family','Mixed groups'],ylabel='Clean yaw prediction RMSE [deg/s]')
    ax.grid(axis='y',alpha=.25);ax.legend();plot1=svg(fig)
    fig,ax=plt.subplots(figsize=(8,3.6))
    for seed in cfg['seeds']:
        r=by_family[by_family.seed==seed].set_index('family').reindex(['Mini','Sport','SUV'])
        ax.plot(range(3),100*(1-r.OracleFamily/r.SharedPrior),marker='o',label=f'Fleet {seed}')
    ax.axhline(0,color='black',lw=.8);ax.set(xticks=range(3),xticklabels=['Mini','Sport','SUV'],ylabel='Oracle reduction versus shared prior [%]')
    ax.legend();ax.grid(axis='y',alpha=.25);plot2=svg(fig)
    means=fleet.mean();table=pd.DataFrame({'Method':METHODS,'Mean RMSE [deg/s]':means[METHODS].values,
        'Vehicle wins of oracle family':[(vehicle.OracleFamily<vehicle[m]).sum() if m!='OracleFamily' else 0 for m in METHODS]})
    report=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CFL population: is grouping useful?</title><style>body{{font:17px/1.6 system-ui;color:#173341;background:#edf2f5;margin:0}}main{{max-width:1080px;margin:auto;background:white;padding:30px}}svg{{max-width:100%;height:auto}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:9px;border-bottom:1px solid #ddd;text-align:left}}.scroll{{overflow:auto}}.note{{background:#edf6f5;padding:16px}}pre{{white-space:pre-wrap}}</style><main>
<h1>Does the CFL-FF population benefit from family-specific priors?</h1>
<p>Three exploratory fleets; 18 vehicles each; six per family, two at each local speed. Each vehicle supplies one 12 s yaw-only record. No learned clustering and no control experiments.</p>
<p class="note"><b>Progression screen:</b> {html.escape(decision['status'])}. All fits converged: {converged}. All test operating checks passed: {operating}. The screen is a descriptive decision rule, not statistical significance.</p>
<h2>Primary comparison: target-unseen anchor speeds</h2>{plot1}<div class="scroll">{table.to_html(index=False,border=0)}</div>
<p>Each vehicle mean uses four clean-yaw test cases: the other two anchor speeds and two new input probes. Fleet means have equal weights. The 216 tests are not independent fleet replications. All completed cases remain visible; operating failures are not discarded.</p>
<h2>Where does family grouping help?</h2>{plot2}<div class="scroll">{by_family.to_html(index=False,border=0)}</div>
<h2>Decision evidence</h2><pre>{html.escape(json.dumps(decision,indent=2))}</pre>
<p>Progression requires at least 5% pooled reduction and improvement in at least two of three fleets against each of SharedPrior, MixedGroups, and NominalPrior, with converged fits and valid operating conditions. Inspect absolute errors and family/speed variation as well.</p>
<h2>What this experiment isolates</h2><p>SharedPrior learns one regularization center from 18 clients. OracleFamily learns one center per known family from six clients. MixedGroups also uses three six-client centers, but each group has two clients per family and per speed. All methods use identical local records, bounds and penalty weights. Local OE and the population-centered nominal prior establish whether learned sharing adds value.</p>
<p>The oracle is a privileged family partition, not an upper bound over all partitions. Failure of family grouping does not rule out dynamic similarity groups. Success supports investigating learned grouping; it does not establish that groups can be recovered from the short noisy records.</p>
<h2>Population transfer and scope</h2><p>The Mini anchor is the CFL-FF paper's Compact vehicle. Mass, inertia, geometry, payload and tire-slope variability follow its declared distribution, with new draws. Lateral Pacejka zero-slip slopes B C μ Fz are mapped into the current tanh model; the nonlinear Pacejka curve is not imported. E is recorded but does not enter the slope. Plant and estimator remain matched four-state models. Geometry is known per vehicle.</p>
<p>Relaxation lengths remain 0.4/0.7 m for all vehicles; the nominal prior is centered at these values. No family-specific relaxation lengths are invented. Thus this tests CFL-FF-style physical heterogeneity, not heterogeneity of force-relaxation time constants. The engineering prior is built from independent population-design draws, never from the simulated test vehicles or evaluation outcomes.</p>
<h2>Optimization and secondary diagnostics</h2><div class="scroll">{pd.DataFrame(optim).to_html(index=False,border=0)}</div>
<details><summary>Training-speed strata</summary><div class="scroll">{by_speed.to_html(index=False,border=0)}</div></details>
<p>CSV files contain every speed/probe, individual log-parameter errors, bounds diagnostics in local checkpoints, and all optimization histories. Mu is weakly excited and unregularized; prediction, not recovery of all six parameters, is the primary endpoint. No settings were selected using these prediction errors.</p>
</main></html>'''
    REPORT.parent.mkdir(parents=True,exist_ok=True);REPORT.write_text(report,encoding='utf-8',newline='\n')
    outputs=[REPORT,*sorted(OUT.glob(f'{PREFIX}_*.csv'))]
    storage.atomic_json(OUT/f'{PREFIX}_analysis.json',dict(status='analysis_complete',decision=decision,
        source_sha256=sources,protocol_sha256=base.canonical_hash(CONFIG),means_deg_s=means.to_dict(),
        output_sha256={p.relative_to(ROOT).as_posix():base.canonical_hash(p) for p in outputs}))
    print(json.dumps(dict(status='analysis_complete',report=str(REPORT),decision=decision['status'])))


def smoke(cfg):
    """Real numerical checks on shortened records; production outputs untouched."""
    short=copy.deepcopy(cfg);short['duration_s']=.25;short['taper_s']=.05
    c=cfg['fleets'][0]['clients'][0];a,_=base.generate(c,2301,'training',c['training_speed'],short)
    endpoint=fl.Client([dict(speed=c['training_speed'],applied_input=a['steering'],measured_yaw=a['state'][:,1])],
                       c['geometry_m'],base.settings(short))
    q=np.log(c['physical_parameters']);res,jac=endpoint.residual(q,True)
    assert np.max(abs(res))<1e-10
    direction=np.array([.3,-.2,.1,0,.2,-.1]);h=1e-5
    numerical=(endpoint.residual(q+h*direction)-endpoint.residual(q-h*direction))/(2*h)
    np.testing.assert_allclose(numerical,jac@direction,rtol=2e-5,atol=1e-6)
    fit=endpoint.fit(q+1e-3,np.log(cfg['nominal_parameters']),cfg['sharing']['weights'],max_nfev=20)
    assert fit['available'] and fit['objective']<=fit['initial_objective']
    with tempfile.TemporaryDirectory() as directory:
        pth=Path(directory)/'unicode.json';storage.atomic_json(pth,dict(label='Mini / Sport / SUV — yaw β'))
        assert storage.read_checkpoint(pth)['label'].endswith('β')
    print(json.dumps(dict(status='smoke_passed',analytic_jacobian_checked=True,production_outputs_modified=False)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['preflight','smoke','run','analyze','restore'])
    parser.add_argument('--workers',type=int,default=3)
    args=parser.parse_args()
    if args.workers<1:parser.error('--workers must be positive')
    cfg=preflight()
    if args.command=='preflight':
        unstable=[]
        for f in cfg['fleets']:
            for c in f['clients']:
                for v in cfg['evaluation_speeds']:
                    abscissa=max(np.linalg.eigvals(continuous_matrices(c['physical_parameters'],v,c['geometry_m'])[0]).real)
                    if abscissa>=0:unstable.append([f['seed'],c['client'],v,float(abscissa)])
        print(json.dumps(dict(status='preflight_passed',fleets=3,vehicles=54,local_records=54,
            sharing_jobs=21,primary_cases=216,all_test_cases=756,unstable_linearizations=unstable,
            nominal_prior=cfg['nominal_parameters'],new_fits=0)))
    elif args.command=='smoke':
        with ProcessPoolExecutor(max_workers=1,mp_context=multiprocessing.get_context('spawn')) as pool:
            pool.submit(smoke,cfg).result()
    elif args.command=='run':run(cfg,args.workers)
    elif args.command=='analyze':analyze(cfg)
    else:restore(cfg)


if __name__=='__main__':main()
