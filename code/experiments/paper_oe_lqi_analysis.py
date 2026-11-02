"""Audit and describe committed frozen OE-LPV control results; no refitting.

Run: python code/experiments/paper_oe_lqi_analysis.py
The small replay audit reconstructs saved trajectories without changing results.
"""
from __future__ import annotations
from collections import Counter
from io import StringIO
from pathlib import Path
import html
import json
import subprocess
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from threadpoolctl import threadpool_limits

import paper_oe_lqi_validation as run
from federated_lpv.control import augmented_tracking_matrices
from federated_lpv.oe_lqi_validation import interpolate

ROOT=run.ROOT
OUTPUT=ROOT/'results/tables/paper_oe_lqi_validation_analysis.json'
REPORT=ROOT/'report/oe_lqi_validation_analysis.html'
LABELS={'NominalLTI':'Nominal LTI','RestrictedLocal':'Restricted Local','SmoothLocal':'Smooth Local',
        'Global':'Global','Federated':'Federated','ExactLPV':'ExactLPV (reference)'}
COLORS={'NominalLTI':'#657888','RestrictedLocal':'#b86b24','SmoothLocal':'#bd4a59',
        'Global':'#437ab5','Federated':'#00887d','ExactLPV':'#8062a0'}


def require(condition,message):
    if not condition: raise ValueError(message)


def audit_inputs(cfg):
    payloads=[]; rows=[]; schedules=[]; groups=[]
    max_matrix_delta=0.; max_radius_delta=0.
    basis=run.load(ROOT/'code/config/experiment_10a.json')
    profile_set=run.profiles(cfg)
    frozen=run.load(ROOT/'code/config/paper_fixed_lti_identification.json')['controller_frozen']
    locked_files=[run.result_path(s).relative_to(ROOT).as_posix() for s in cfg['seeds']]
    for f in locked_files:
        subprocess.run(['git','ls-files','--error-unmatch',f],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
    subprocess.run(['git','diff','--exit-code','HEAD','--',*locked_files],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
    for seed in cfg['seeds']:
        payload=run.envelope(run.result_path(seed)); run.validate_checkpoint(payload,cfg,seed)
        require(payload['status']=='execution_complete','Incomplete fleet')
        require(payload['protocol']==cfg,'Stored protocol differs from frozen configuration')
        source=run.envelope(run.source_path(seed)); models=source['conditions']['ClosedOE4']['models']
        payloads.append(payload); groups.append(dict(seed=seed,selected_k=source['conditions']['ClosedOE4']['selected_k']))
        for c,original in zip(payload['clients'],source['clients'],strict=True):
            i=c['index']
            for key in ['client','family','parameters']:
                require(c[key]==original[key],f'Client identity/parameters changed: {seed}/{i}/{key}')
            require(c['observed_speeds']==original['speeds'],'Coverage changed')
            require(c['assignment']==models['assignments'][i],'Group assignment changed')
            for method in run.METHODS:
                schedule=c['schedules'][method]
                require([n['speed'] for n in schedule['nodes']]==cfg['speed_grid'],'Schedule grid changed')
                if method=='NominalLTI':
                    expected=np.tile(np.r_[np.asarray(frozen['a']).ravel(),frozen['b']],(11,1))
                else:
                    coefficients=run.coefficient(models,method,i)
                    np.testing.assert_array_equal(c['coefficients'][method],coefficients)
                    expected=run.raw_basis(np.asarray(cfg['speed_grid']),len(coefficients),basis)@coefficients
                actual=np.asarray([n['matrix'] for n in schedule['nodes']])
                max_matrix_delta=max(max_matrix_delta,float(np.max(abs(expected-actual))))
                np.testing.assert_allclose(actual,expected,rtol=1e-10,atol=1e-10)
                for node in schedule['nodes']:
                    if node['status']=='designed':
                        matrix=np.asarray(node['matrix']);gain=np.asarray(node['gain'])
                        if method=='NominalLTI':
                            np.testing.assert_array_equal(gain,frozen['gain'])
                            require(node['prefilter']==frozen['prefilter'],'Nominal prefilter changed')
                        aa,bb=augmented_tracking_matrices(matrix[:4].reshape(2,2),matrix[4:].reshape(2,1),cfg['sample_time'])
                        rho=float(max(abs(np.linalg.eigvals(aa-bb@gain[None,:]))))
                        np.testing.assert_allclose(rho,node['design_rho'],rtol=1e-9,atol=1e-10)
                        require(rho<1+1e-9,'Own-model design check failed')
                diagnostic=run.spectral_diagnostic(schedule,np.asarray(c['coefficients']['ExactLPV']),cfg,basis)
                saved=c['spectral_diagnostics'][method]
                require(diagnostic['status']==saved['status'],'Radius diagnostic status mismatch')
                if saved['status']=='complete':
                    np.testing.assert_allclose(diagnostic['radius'],saved['radius'],rtol=1e-8,atol=1e-9)
                    max_radius_delta=max(max_radius_delta,float(np.max(abs(np.asarray(diagnostic['radius'])-saved['radius']))))
                schedules.append(dict(seed=seed,client=c['client'],family=c['family'],method=method,status=schedule['status'],
                    maximum_radius=saved.get('maximum'),observed_speeds=c['observed_speeds']))
            for row in c['rows']:
                o=row['outcome']; profile=profile_set[row['scenario']]
                if o['status']=='complete':
                    require(o['completed_steps']==1200,'Incomplete trajectory marked complete')
                    require(np.isfinite(list(o['metrics'].values())).all(),'Nonfinite metrics')
                    m=o['metrics']
                    require(all(v>=0 for v in m.values()),'Negative magnitude metric')
                    feasible=(m['beta_violation_samples']==0 and m['steering_peak_deg']<=12+1e-8 and m['steering_rate_peak_deg_s']<=120+1e-8)
                    require(o['feasible']==feasible,'Feasibility mismatch')
                    require((m['beta_peak_deg']>6)==(m['beta_violation_samples']>0),'Sideslip peak/count mismatch')
                    require(all(0<=m[k]<=1 for k in ['limiter_fraction','angle_limiter_fraction','rate_limiter_fraction']),'Limiter fraction invalid')
                    expected_shapes={k:[1201] for k in ['time','speed','reference','integral']}
                    expected_shapes.update(state=[1201,2])
                    expected_shapes.update({k:[1200] for k in ['steering','requested_steering','lateral_acceleration','limited','angle_limited','rate_limited']})
                    require({k:v['shape'] for k,v in row['arrays'].items()}==expected_shapes,'Raw-array shape mismatch')
                    for k in ['time','speed','reference']:
                        require(row['arrays'][k]==run.array_hashes({k:profile[k]})[k],'Profile hash mismatch')
                    require(all(len(v['sha256'])==64 and int(v['sha256'],16)>=0 for v in row['arrays'].values()),'Invalid array hash')
                rows.append(dict(seed=seed,index=i,client=c['client'],family=c['family'],method=row['method'],scenario=row['scenario'],
                    observed_speeds=c['observed_speeds'],**o))
    audit=dict(status='passed',fleets=len(payloads),clients=sum(len(p['clients']) for p in payloads),schedules=len(schedules),cases=len(rows),
        outcomes=dict(Counter(r['status'] for r in rows)),synthesis_status=dict(Counter(s['status'] for s in schedules)),
        result_file_sha256={f:run.file_hash(ROOT/f) for f in locked_files},protocol_sha256=run.file_hash(run.CONFIG),
        execution_commits=sorted({p['git_commit'] for p in payloads}),hosts=[dict(seed=p['seed'],**p['host']) for p in payloads],
        max_saved_model_matrix_abs_difference=max_matrix_delta,max_recomputed_radius_abs_difference=max_radius_delta,
        checks=['Committed unchanged checkpoints','Canonical payload hashes','Frozen source/configuration hashes','All client/method/scenario identities',
                'Source coefficients, assignments, plant parameters and coverage','Saved design matrices and nominal controller identity',
                'Own-model closed-loop radii','Recomputed full-reference frozen-speed radii','Profile array hashes and array dimensions',
                'Metric finiteness, duration and feasibility consistency'])
    return payloads,rows,schedules,groups,audit


def descriptive(values):
    v=np.asarray(values,float)
    if not len(v):return dict(n=0,mean=None,median=None,min=None,max=None)
    return dict(n=len(v),mean=float(v.mean()),median=float(np.median(v)),min=float(v.min()),max=float(v.max()))


def summarize(rows,schedules):
    summary=[]; fleets=[]; pairs=[]; families=[]; coverage=[]
    metric_names=list(next(r['metrics'] for r in rows if r['metrics']))
    seeds=sorted({r['seed'] for r in rows})
    for scenario in run.SCENARIOS:
        for method in run.METHODS:
            rr=[r for r in rows if r['scenario']==scenario and r['method']==method]
            finite=[r for r in rr if r['metrics'] is not None]
            means=[]
            for seed in seeds:
                cells=[r for r in rr if r['seed']==seed]; valid=[r for r in cells if r['metrics'] is not None]
                full=len(valid)==len(cells)==30
                metrics={k:float(np.mean([r['metrics'][k] for r in valid])) if valid else None for k in metric_names}
                fleets.append(dict(seed=seed,scenario=scenario,method=method,cases=len(cells),finite=len(valid),feasible=sum(r['feasible'] for r in cells),
                                   full_fleet=full,metrics=metrics))
                if full:means.append(metrics['tracking_rmse_deg_s'])
            summary.append(dict(scenario=scenario,method=method,cases=len(rr),finite=len(finite),feasible=sum(r['feasible'] for r in rr),
                sideslip_violations=sum(r['metrics']['beta_violation_samples']>0 for r in finite),
                synthesis_failures=sum(r['status']=='synthesis_failed' for r in rr),numerical_failures=sum(r['status']=='rollout_failed' for r in rr),
                full_fleets=len(means),fleet_rmse=descriptive(means),metrics={k:descriptive([r['metrics'][k] for r in finite]) for k in metric_names}))
            for family in sorted({r['family'] for r in rr}):
                cells=[r for r in rr if r['family']==family]
                families.append(dict(scenario=scenario,method=method,family=family,n=len(cells),feasible=sum(r['feasible'] for r in cells),
                    rmse=descriptive([r['metrics']['tracking_rmse_deg_s'] for r in cells if r['metrics']])))
            for block in sorted({tuple(r['observed_speeds']) for r in rr}):
                cells=[r for r in rr if tuple(r['observed_speeds'])==block]
                coverage.append(dict(scenario=scenario,method=method,observed_speeds=list(block),n=len(cells),feasible=sum(r['feasible'] for r in cells),
                    rmse=descriptive([r['metrics']['tracking_rmse_deg_s'] for r in cells if r['metrics']])))
        for baseline in [m for m in run.METHODS if m!='Federated']:
            gains=[]; absolute=[]; detail=[]; wins=ties=client_n=0; feasible_pair_n=0; feasible_pair_gains=[]
            for seed in seeds:
                fed=next(f for f in fleets if (f['seed'],f['scenario'],f['method'])==(seed,scenario,'Federated'))
                base=next(f for f in fleets if (f['seed'],f['scenario'],f['method'])==(seed,scenario,baseline))
                if fed['full_fleet'] and base['full_fleet']:
                    f=fed['metrics']['tracking_rmse_deg_s'];b=base['metrics']['tracking_rmse_deg_s']
                    gain=100*(b-f)/b;delta=b-f;gains.append(gain);absolute.append(delta)
                    detail.append(dict(seed=seed,baseline_mean=b,federated_mean=f,relative_reduction_percent=gain,absolute_reduction_deg_s=delta))
            frows={(r['seed'],r['client']):r for r in rows if r['scenario']==scenario and r['method']=='Federated'}
            for b in [r for r in rows if r['scenario']==scenario and r['method']==baseline]:
                f=frows[(b['seed'],b['client'])]
                if b['metrics'] and f['metrics']:
                    delta=b['metrics']['tracking_rmse_deg_s']-f['metrics']['tracking_rmse_deg_s']
                    wins+=delta>0;ties+=delta==0;client_n+=1
                    if b['feasible'] and f['feasible']:
                        feasible_pair_n+=1;feasible_pair_gains.append(delta)
            pairs.append(dict(scenario=scenario,baseline=baseline,complete_fleet_pairs=len(gains),fleet_wins=sum(g>0 for g in gains),
                paired_fleet_percent=descriptive(gains),paired_fleet_absolute_deg_s=descriptive(absolute),fleet_details=detail,
                finite_client_pairs=client_n,client_wins=wins,client_ties=ties,
                both_feasible_client_pairs=feasible_pair_n,conditional_both_feasible_absolute_deg_s=descriptive(feasible_pair_gains)))
    spectral=[]
    for method in run.METHODS:
        ss=[s for s in schedules if s['method']==method]; values=[s['maximum_radius'] for s in ss if s['maximum_radius'] is not None]
        spectral.append(dict(method=method,n=len(ss),radii=descriptive(values),at_least_one=sum(v>=1 for v in values)))
    return dict(summary=summary,fleets=fleets,pairs=pairs,families=families,coverage=coverage,spectral=spectral)


def replay_audit(payloads,rows,cfg):
    # Fixed first client for every method/scenario, plus worst local RMSE tails.
    selected={(cfg['seeds'][0],0,m,s) for m in run.METHODS for s in run.SCENARIOS}
    for method in ['RestrictedLocal','SmoothLocal']:
        for scenario in run.SCENARIOS:
            worst=max((r for r in rows if r['method']==method and r['scenario']==scenario and r['metrics']),key=lambda r:r['metrics']['tracking_rmse_deg_s'])
            selected.add((worst['seed'],worst['index'],method,scenario))
    lookup={p['seed']:p for p in payloads}; profile_set=run.profiles(cfg); details=[]
    for seed,i,method,scenario in sorted(selected):
        c=lookup[seed]['clients'][i]; row=next(r for r in c['rows'] if (r['method'],r['scenario'])==(method,scenario))
        outcome,arrays=run.simulate(run.VehicleParameters(**c['parameters']),c['schedules'][method],profile_set[scenario],cfg)
        metrics_delta={k:float(outcome['metrics'][k]-v) for k,v in row['outcome']['metrics'].items()} if outcome['metrics'] and row['outcome']['metrics'] else None
        details.append(dict(seed=seed,index=i,client=c['client'],method=method,scenario=scenario,
            arrays_exact=run.array_hashes(arrays)==row['arrays'],outcome_exact=outcome==row['outcome'],
            status_matches=outcome['status']==row['outcome']['status'],feasibility_matches=outcome['feasible']==row['outcome']['feasible'],metric_deltas=metrics_delta))
    return dict(selection='All six methods and two scenarios for seed 1311/client 0, plus maximum-RMSE case for each local baseline/scenario; no outcome replacement.',
        cases=len(details),exact_arrays=sum(d['arrays_exact'] for d in details),exact_outcomes=sum(d['outcome_exact'] for d in details),
        status_and_feasibility_match=all(d['status_matches'] and d['feasibility_matches'] for d in details),details=details,
        scope='Sampled reconstruction only. Original committed metrics remain authoritative; no new scientific fleet and no full 3600-case replay certification.')


def figure_svg(fig,name):
    stream=StringIO(); fig.savefig(stream,format='svg',bbox_inches='tight',metadata={'Date':None})
    temp=Path('/tmp/oe_lqi_analysis_qa');temp.mkdir(exist_ok=True)
    fig.savefig(temp/f'{name}.png',dpi=130,bbox_inches='tight');plt.close(fig)
    value=stream.getvalue();return '\n'.join(line.rstrip() for line in value[value.index('<svg'):].splitlines())


def plots(data,rows):
    plt.rcParams.update({'font.size':10,'font.family':'DejaVu Sans','svg.fonttype':'none','svg.hashsalt':'oe-lqi-analysis',
                         'axes.spines.top':False,'axes.spines.right':False})
    figs={}
    for scenario in run.SCENARIOS:
        fig,ax=plt.subplots(figsize=(9,3.6))
        for i,m in enumerate(run.METHODS):
            rr=[r for r in rows if r['scenario']==scenario and r['method']==m and r['metrics']]
            y=np.array([r['metrics']['tracking_rmse_deg_s'] for r in rr]);x=np.full(len(y),i)
            jitter=np.linspace(-.16,.16,len(y))
            feasible=np.array([r['feasible'] for r in rr])
            ax.scatter((x+jitter)[feasible],y[feasible],s=8,alpha=.28,color=COLORS[m])
            ax.scatter((x+jitter)[~feasible],y[~feasible],s=13,alpha=.6,color='#b22234',marker='x')
            ax.scatter(i,np.mean(y),s=65,color=COLORS[m],edgecolor='black',marker='D',zorder=5)
        ax.set_xticks(range(6),[LABELS[m].replace(' (reference)','\n(reference)').replace(' ','\n',1) for m in run.METHODS])
        ax.set_yscale('log');ax.set_ylabel('Full-maneuver yaw-rate RMSE [deg/s]')
        ax.set_title(f'{scenario.capitalize()}: every client, including infeasible responses',loc='left',weight='bold')
        ax.grid(axis='y',alpha=.2);figs[f'distribution_{scenario}']=figure_svg(fig,f'distribution_{scenario}')
    fig,ax=plt.subplots(figsize=(9,3.3))
    categories=[('NominalLTI','moderate'),('NominalLTI','hard'),('Global','moderate'),('Global','hard')]
    for i,(b,s) in enumerate(categories):
        pair=next(p for p in data['pairs'] if p['baseline']==b and p['scenario']==s)
        y=[v['relative_reduction_percent'] for v in pair['fleet_details']]
        ax.scatter(i+np.linspace(-.13,.13,len(y)),y,color='#00887d' if s=='moderate' else '#437ab5',s=42)
        ax.plot([i-.22,i+.22],[np.mean(y)]*2,color='#172938',lw=2)
    ax.axhline(0,color='#8b2535',lw=1);ax.set_xticks(range(4),[f'{LABELS[b]}\n{s}' for b,s in categories])
    ax.set_ylabel('Federated RMSE reduction [%]');ax.set_title('Paired fleet comparisons: one dot per fleet',loc='left',weight='bold')
    ax.grid(axis='y',alpha=.2);figs['paired']=figure_svg(fig,'paired')
    fig,ax=plt.subplots(figsize=(9,3.2));x=np.arange(6)
    for j,s in enumerate(run.SCENARIOS):
        values=[300-next(r['feasible'] for r in data['summary'] if r['method']==m and r['scenario']==s) for m in run.METHODS]
        bars=ax.bar(x+(j-.5)*.35,values,width=.35,color=['#437ab5','#b86b24'][j],label=s)
        ax.bar_label(bars,padding=3)
    ax.set_xticks(x,[LABELS[m].replace(' (reference)','\n(reference)').replace(' ','\n',1) for m in run.METHODS]);ax.set_ylabel('Infeasible responses / 300')
    ax.set_ylim(0,max(90,ax.get_ylim()[1]*1.1));ax.legend(frameon=False);ax.set_title('Constraint violations are reported alongside tracking error',loc='left',weight='bold')
    figs['feasibility']=figure_svg(fig,'feasibility')
    fig,ax=plt.subplots(figsize=(9,3.2));families=['nominal','heavy','handling'];x=np.arange(3)
    for j,s in enumerate(run.SCENARIOS):
        values=[]
        for family in families:
            f=next(r['rmse']['mean'] for r in data['families'] if (r['scenario'],r['method'],r['family'])==(s,'Federated',family))
            b=next(r['rmse']['mean'] for r in data['families'] if (r['scenario'],r['method'],r['family'])==(s,'Global',family))
            values.append(100*(b-f)/b)
        ax.bar(x+(j-.5)*.35,values,width=.35,color=['#00887d','#437ab5'][j],label=s)
    ax.axhline(0,color='#8b2535',lw=1);ax.set_xticks(x,families);ax.set_ylabel('Reduction vs Global family mean [%]')
    ax.set_title('Aggregate benefit does not imply benefit for every family',loc='left',weight='bold');ax.legend(frameon=False)
    figs['families']=figure_svg(fig,'families')
    return figs


def table(headers,rows):
    return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(str(h))+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table></div>'


def render(data,rows):
    figs=plots(data,rows);summary=data['summary'];audit=data['audit'];replay=data['replay']
    get=lambda m,s:next(r for r in summary if r['method']==m and r['scenario']==s)
    pair=lambda b,s:next(r for r in data['pairs'] if r['baseline']==b and r['scenario']==s)
    metric=lambda m,s,k:get(m,s)['metrics'][k]['mean']
    comparison=[]
    for m in run.METHODS:
        a,b=(get(m,s) for s in run.SCENARIOS)
        comparison.append([LABELS[m],f"{a['fleet_rmse']['mean']:.5f}",f"{b['fleet_rmse']['mean']:.5f}",
                           f"{300-a['feasible']} / {300-b['feasible']}",f"{a['metrics']['tracking_rmse_deg_s']['median']:.5f} / {b['metrics']['tracking_rmse_deg_s']['median']:.5f}"])
    pairrows=[]
    for base in ['NominalLTI','Global']:
        for s in run.SCENARIOS:
            p=pair(base,s);v=p['paired_fleet_percent']
            pairrows.append([LABELS[base],s,f"{v['mean']:.2f}%",f"{v['min']:.2f}% to {v['max']:.2f}%",
                             f"{p['paired_fleet_absolute_deg_s']['mean']:.5f}",f"{p['fleet_wins']}/10",f"{p['client_wins']}/300"])
    effort=[]
    for s in run.SCENARIOS:
        for m in ['NominalLTI','Global','Federated','ExactLPV']:
            effort.append([s,LABELS[m],f"{metric(m,s,'steering_rms_deg'):.4f}",f"{metric(m,s,'steering_rate_rms_deg_s'):.4f}",
                           f"{metric(m,s,'steering_total_variation_deg'):.4f}",f"{100*metric(m,s,'limiter_fraction'):.3f}%"])
    html_text='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>OE–LPV control results: paper readiness</title>
<style>body{margin:0;background:#eef3f6;color:#172938;font:16px/1.62 system-ui,sans-serif}main{max-width:1120px;margin:auto;padding:36px 28px 65px}h1{font-size:2.25rem;line-height:1.2;max-width:24ch}h2{font-size:1.5rem;margin-top:0}h3{font-size:1.1rem}p{max-width:94ch}section{background:white;border:1px solid #d6e1e7;border-radius:10px;padding:25px 28px;margin:24px 0}a{color:#086982}.tag{color:#007c71;text-transform:uppercase;font-size:13px;letter-spacing:.1em}.verdict{border-left:5px solid #00887d;background:#edf9f5;padding:16px 22px}.caution{border-left:4px solid #bf8026;padding:10px 18px;background:#fffaef}nav{display:flex;flex-wrap:wrap;gap:16px;margin:20px 0}table{border-collapse:collapse;font-size:14px;min-width:650px;width:100%}td,th{padding:10px 12px;text-align:right;border-bottom:1px solid #d9e3e9}td:first-child,th:first-child{text-align:left}th{background:#edf4f7}tr:hover{background:#f4f8fa}.scroll{overflow:auto}.figure{margin:22px 0}svg{width:100%;height:auto}.caption,small{color:#506274;font-size:14px}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:14px}.card{border:1px solid #ccdfe5;padding:12px 18px;border-radius:6px}.number{font-size:1.8rem;font-weight:700;display:block;color:#087e73}code,pre{font-family:Consolas,monospace}pre{background:#132e40;color:#f1f7fa;padding:18px;overflow:auto}details{padding:12px 0}summary{cursor:pointer;font-weight:600}li{margin:9px 0}blockquote{border-left:3px solid #00887d;padding:10px 20px;margin-left:0;background:#f2f8f7}@media print{body{background:white}main{padding:0}section{break-inside:avoid}nav{display:none}}</style><main>
<p class="tag">Committed execution results · descriptive control extension</p><h1>OE-derived LPV models are ready for a bounded control claim</h1>
<p class="verdict"><strong>Recommendation: include these results in Section VI.</strong> They complete the link from the frozen OE identification pipeline to nonlinear closed-loop use. The strongest evidence is feasible federated operation and lower fleet-mean tracking error than the original nominal controller. The additional benefit over Global is modest and not uniform across maneuvers, fleets or vehicle families.</p>
<nav><a href="#audit">Integrity</a><a href="#results">Tracking and feasibility</a><a href="#pairs">Paired comparisons</a><a href="#effort">Steering effort</a><a href="#limits">Interpretation</a><a href="#paper">Paper changes</a></nav>'''
    html_text+=f'''<div class="cards"><div class="card"><span class="number">{audit['fleets']}</span>complete saved fleets</div><div class="card"><span class="number">{audit['cases']:,}</span>full finite rollouts</div><div class="card"><span class="number">600 / 600</span>feasible Federated rollouts</div><div class="card"><span class="number">0</span>synthesis / numerical failures</div></div>
<section id="audit"><h2>1. Integrity and protocol audit</h2><p>Source execution commit: <code>{data['execution_results_commit']}</code>. The saved experiment used code commit <code>{audit['execution_commits'][0]}</code>. All ten checkpoint envelopes, frozen source hashes, source coefficients, group assignments, client parameters and profile hashes pass verification. All 1,800 controller schedules contain the intended eleven-speed grid, and all 3,600 rollouts contain 1,200 control steps.</p>
<p>The saved design matrices agree with independently evaluated frozen coefficients to maximum absolute difference {audit['max_saved_model_matrix_abs_difference']:.3g}. Recomputed frozen-speed radii differ by at most {audit['max_recomputed_radius_abs_difference']:.3g}. The original nominal gain and prefilter are unchanged.</p>
<p><strong>Replay sample:</strong> {replay['exact_arrays']}/{replay['cases']} trajectory array sets and {replay['exact_outcomes']}/{replay['cases']} outcome records reproduce exactly on this host. The sample covers all six methods and both maneuvers for the first client, plus each local baseline’s worst-RMSE case in each maneuver. This is a sampled audit, not a claim that all 3,600 trajectories were replayed.</p>
<details><summary>Checks and scope</summary><ul>'''+''.join(f'<li>{html.escape(x)}</li>' for x in audit['checks'])+'''</ul><p>No identification was repeated, no controller was retuned, and no original result was replaced. All ten fleets select three groups. These are the same fleets used in the identification confirmation, now used for a descriptive control extension.</p></details></section>
<section id="results"><h2>2. Tracking and feasibility</h2><p>RMSE is measured in degrees/s. Means average clients within each fleet, then the ten fleet means. Each method has 300 observations per maneuver. Every finite infeasible response remains in the reported mean; all runs are finite in this dataset.</p>'''
    html_text+=table(['Method','Moderate mean','Hard mean','Sideslip violations M / H','Client median M / H'],comparison)
    html_text+='''<p class="caution">The large local-baseline means reflect severe tracking tails, including constraint-violating responses. Passing the actuator and sideslip bounds does not ensure acceptable tracking: some bounded local responses also have large errors. Extreme sideslip is outside credible vehicle operation. Report violations as failed control responses under the tested protocol, not as realistic driving performance or a headline “99% improvement”.</p>'''
    for s in run.SCENARIOS:
        html_text+='<div class="figure">'+figs[f'distribution_{s}']+'<p class="caption">Points: all 300 clients per method. Red crosses: infeasible responses. Diamonds: means. Logarithmic vertical axis exposes both normal errors and severe tails; there is no exclusion of failed responses.</p></div>'
    html_text+='<div class="figure">'+figs['feasibility']+'</div></section><section id="pairs"><h2>3. How much does grouping add?</h2>'
    html_text+=table(['Baseline','Maneuver','Mean fleet reduction','Fleet range','Mean absolute gain [deg/s]','Fleet wins','Client wins'],pairrows)
    html_text+='''<p>Percentages are the mean of ten paired fleet-relative reductions, 100 × (baseline fleet mean − Federated fleet mean) / baseline fleet mean. They are not the percentage difference between pooled means. Absolute gains are baseline minus Federated. Ranges and win counts are descriptive; no confirmatory p-values or confidence intervals are claimed.</p>'''
    html_text+=figs['paired']+'''<p><strong>Interpretation:</strong> the benefit against the fixed nominal controller is consistent at fleet level. Against Global, the moderate maneuver improves in every fleet, but the hard maneuver improves in only eight of ten. The hard-maneuver absolute difference is small. The fact that Federated approaches the ExactLPV reference in aggregate does not establish equivalence or an optimality bound.</p>'''
    html_text+=figs['families']+'''<p class="caution">All 100 handling-family clients have higher Federated RMSE than Global in both maneuvers. All 100 heavy-family clients improve. Nominal-family outcomes improve for 91/100 clients in the moderate maneuver and 52/100 in the hard maneuver. Thus, an aggregate benefit cannot be described as a benefit for every recipient. Family labels are used only for this post-fit diagnostic.</p></section>
<section id="effort"><h2>4. Is lower error bought with much more steering?</h2>'''
    html_text+=table(['Maneuver','Method','RMS steering [deg]','RMS rate [deg/s]','Total variation [deg]','Limited samples'],effort)
    html_text+='''<p>Federated and Global steering effort is nearly identical. In the hard maneuver, Federated uses slightly more steering-rate RMS and slightly more limiting than Global, so a blanket “same or less effort” claim is unwarranted. All applied steering and rate bounds are satisfied. None of NominalLTI, Global, Federated or ExactLPV reaches the 120°/s rate limit; their hard-maneuver limiting is amplitude limiting. These runs do not demonstrate robustness to a tight rate constraint.</p></section>
<section id="limits"><h2>5. What the results establish — and what remains outside scope</h2>
<p>The useful conclusion is that the frozen shared models support feasible scheduled control across varying speed and improve aggregate tracking. Reduced identification error does not guarantee reliable control from a locally extrapolated map. Both local baselines synthesize successfully on their own fitted models, yet some schedules fail when applied to the nonlinear recipient plant.</p>'''
    html_text+=table(['Method','Schedules with reference radius ≥ 1 / 300','Maximum reference radius'],[[LABELS[r['method']],r['at_least_one'],f"{r['radii']['max']:.6f}"] for r in data['spectral']])
    html_text+='''<p>These radii evaluate saved gains against the saved full-reference LPV approximation on 81 frozen speeds. They diagnose mismatch; they do not certify the nonlinear varying-speed system. A radius above one and a sideslip violation are different outcomes and their counts need not match.</p>
<ul><li><strong>Previously analyzed fleets:</strong> describe a supporting control extension, not a new independent control confirmation.</li><li><strong>Available states:</strong> noise-free physical-state feedback isolates model quality. No robustness to measurement noise, state estimation or an observer is established.</li><li><strong>Prescribed speed and yaw reference:</strong> this is nonlinear yaw-rate tracking, not validated full path tracking or longitudinal dynamics.</li><li><strong>Shared controller settings:</strong> Q = diag(25, 50, 1000), R = 0.15 come from the nominal acquisition design. These outcomes concern the common frozen design, not the best individually tuned controller for each model.</li><li><strong>Local extrapolation:</strong> Restricted Local uses its frozen three-function map everywhere. The old nearest-observed-speed fallback is absent. Its failure counts are therefore not directly comparable to the legacy table.</li><li><strong>Actuator protocol:</strong> control uses 12° and 120°/s, whereas acquisition used 30°/s. State this difference. Feasibility additionally requires |β| ≤ 6°.</li><li><strong>Operating-curve mismatch:</strong> the maps were identified around a prescribed cornering condition; the control maneuvers vary the requested turn and apply homogeneous A/B feedback with a steady prefilter. The test is empirical usability beyond that condition.</li></ul></section>
<section id="paper"><h2>6. Recommended manuscript integration</h2>
<p><strong>No additional simulation is required to report this bounded result.</strong> Replace the old Section VI evidence with this experiment. Do not append it to the legacy results or keep their numerical claims.</p>'''
    html_text+=table(['Current material','Required replacement'],[
        ['Seeds 411–420 and fresh map refitting','Reuse frozen ClosedOE4 maps from fleets 1311–1320; no refitting.'],
        ['Controller-selection history and discarded pilot','Explain reuse of the nominal acquisition weights and common settings.'],
        ['Restricted Local nearest-speed fallback','Full-grid evaluation of the saved three-function local map.'],
        ['Legacy RMSE / violation table','Six-method table above, including Nominal LTI and both local failure counts.'],
        ['Legacy gain intervals and “four fleets select one group”','Descriptive paired reductions and fleet wins; all ten saved fits select three groups.'],
        ['Legacy Smooth Local instability / replay paragraph','New local failure counts and radii; sampled replay in reproducibility material only.'],
        ['Abstract and conclusion control statements','Consistent gain over Nominal LTI, modest and nonuniform gain over Global, local extrapolation limitations.']])
    html_text+='''<h3>Compact result paragraph for the next manuscript edit</h3><blockquote>Using the frozen OE-derived maps, all 600 federated rollouts satisfy the tested constraints. Mean paired fleet-relative yaw-rate RMSE reductions against the fixed nominal LTI controller are 24.25% and 25.33% for the moderate and hard maneuvers. The corresponding reductions against Global are 8.36% and 1.29%, with gains in 10/10 and 8/10 fleets. The absolute Global-relative gains are 0.00374 and 0.00203 degrees/s, with similar steering effort. Restricted Local and Smooth Local violate the sideslip bound in 51/77 and 70/62 moderate/hard cases, respectively. These results support the control usability of compatible sharing under the common design, while showing that aggregate improvements do not extend to every recipient.</blockquote>
<p>Use one compact table and, if space permits, the paired fleet plot. The all-client distributions and family breakdown can remain in supporting material. Keep the existing equations only after checking them against the frozen implementation. The abstract, introduction’s control contribution and Section VII should be aligned in the same manuscript pass.</p>
<p>The manuscript has not been modified by this analysis. The results are suitable for inclusion with these qualifications; they do not support a general superiority, real-world deployment or stability-guarantee claim.</p></section>
<section><h2>Reproduce this analysis</h2><pre>python code\\experiments\\paper_oe_lqi_analysis.py</pre>
<p>The script reads committed checkpoints, verifies provenance, produces this self-contained report and writes <code>results/tables/paper_oe_lqi_validation_analysis.json</code>. All input result files remain unchanged. To reconstruct all ignored trajectories separately, use the existing <code>paper_oe_lqi_validation.py restore --workers 5</code> command in the original numeric environment.</p></section></main></html>'''
    REPORT.write_text(html_text,encoding='utf-8',newline='\n')


def main():
    cfg=run.preflight()
    with threadpool_limits(limits=1):
        payloads,rows,schedules,groups,audit=audit_inputs(cfg)
        data=summarize(rows,schedules)
        replay=replay_audit(payloads,rows,cfg)
    require(audit['outcomes']=={'complete':3600},'This report wording requires all 3600 outcomes to be complete')
    require(replay['status_and_feasibility_match'],'Replay outcome discrepancy needs investigation before paper recommendation')
    require(all(g['selected_k']==3 for g in groups),'Update report group-count interpretation')
    data.update(version=1,execution_results_commit=subprocess.check_output(['git','log','-1','--format=%H','--',str(run.result_path(cfg['seeds'][0]).relative_to(ROOT))],cwd=ROOT,text=True).strip(),
        audit=audit,replay=replay,groups=groups,analysis_source_sha256=run.file_hash(Path(__file__)),
        recommendation='Include as descriptive supporting control validation with explicit baseline failures, modest Global-relative gains and recipient tradeoffs. No manuscript edits made.',
        aggregation='Equal clients within fleet, then equal fleets. Percent reductions are means of paired fleet-relative reductions. No inferential claims.')
    OUTPUT.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
    render(data,rows)
    print(json.dumps(dict(status='analysis_complete',audit=audit['status'],cases=audit['cases'],replay_cases=replay['cases'],exact_replays=replay['exact_arrays'],report=str(REPORT))))


if __name__=='__main__':main()
