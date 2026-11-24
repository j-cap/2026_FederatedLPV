"""Identification-only audit while coverage control is incomplete.

No refitting, input reconstruction or nonlinear rollouts. Frozen checkpoints stay
unchanged. Recomputes model grids, errors and sampled single-speed equivalence.
"""
from __future__ import annotations
import io
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import paper_physical_ct_coverage_resume as compat
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

base=compat.base
PREFIX='paper_physical_ct_coverage_review'
REPORT=base.ROOT/'report/physical_ct_coverage_review.html'


def audit():
    compat.install();cfg=base.preflight();settings=base.old.load(base.old.CONFIG)['estimator']
    records=[];hashes={};uncropped_hashes=0;cropped_hashes=0;equivalent_models=0;control=[]
    for seed in cfg['seeds']:
        p=base.old.envelope(base.path(seed));compat.validate(p,seed,cfg)
        if p['status']!='execution_complete':raise ValueError('Identification incomplete')
        for path in [base.path(seed),base.old.result_path(seed),*base.old.source_paths(seed)]:
            hashes[path.relative_to(base.ROOT).as_posix()]=base.old.file_hash(path)
        acquired,source=base.old.sources(seed);cells={(c['client_index'],c['speed']):c for c in acquired['cells']}
        original=base.old.envelope(base.old.result_path(seed));grid=np.array(source['grid'])
        for r in p['records']:
            s=r['spec'];fit=r['fit'];i=s['index'];c=source['clients'][i]
            if s['reuse']=='full' and fit!=original['clients'][i]['fit']:raise ValueError('Reused fit changed')
            if r['original_speeds']!=c['speeds']:raise ValueError('Original coverage changed')
            if [h['speed'] for h in r['training_hashes']]!=[w['speed'] for w in s['windows']]:raise ValueError('Wrong training provenance speeds')
            for h,w in zip(r['training_hashes'],s['windows']):
                if w['n']==2000:
                    cell=cells[i,w['speed']]
                    if h['y']!=cell['training_measurements_sha256'] or h['u']!=cell['training_input_sha256']:raise ValueError('Original training hashes changed')
                    uncropped_hashes+=1
                else:
                    if len(h['y'])!=64 or len(h['u'])!=64:raise ValueError('Invalid cropped-data hash')
                    cropped_hashes+=1
            row=dict(seed=seed,client=c['client'],id=s['id'],kind=s['kind'],block=str(c['speeds']),
                observed=str([w['speed'] for w in s['windows']]),seconds=.01*r['transitions'],converged=fit['converged'],
                rank=0,near_bound=any(fit.get('near_bound',[])),mu=np.nan,matrix_pct=np.nan,increment_pct=np.nan,
                equivalence_max_deviation_pct=np.nan)
            if fit['converged']:
                best=min(a['weighted_mse'] for a in fit['attempts'] if a['converged'])
                np.testing.assert_allclose(best,fit['weighted_mse'],rtol=1e-12,atol=1e-14)
                target=base.models(fit['parameters'],grid,settings)
                np.testing.assert_allclose(target,r['grid_targets'],rtol=2e-10,atol=2e-12)
                errors=base.old.dynamic_metrics(target,np.array(c['reference']),{})
                for key in errors:np.testing.assert_allclose(errors[key],r['errors'][key],rtol=2e-10,atol=2e-12)
                mask=~np.isin(grid,c['speeds']);singular=np.array(fit['information']['log_parameter_singular_values'])
                row.update(rank=int(sum(singular>singular[0]*1e-8)),mu=fit['parameters'][3],
                    matrix_pct=100*np.array(errors['matrix'])[mask].mean(),increment_pct=100*np.array(errors['increment'])[mask].mean())
                if s['kind']=='single':
                    v=s['windows'][0]['speed'];family=base.equivalent_parameters(fit['parameters'],v,settings,cfg['equivalent_mu_grid_points'])
                    saved=r['single_speed_equivalence'];np.testing.assert_allclose(family,[a['parameters'] for a in saved['candidates']],rtol=1e-12,atol=1e-12)
                    deviations=[];same=base.models(fit['parameters'],[v],settings)
                    denominator=np.maximum(np.linalg.norm(target-np.r_[np.eye(2).ravel(),np.zeros(2)],axis=1),1e-15)
                    for theta in family:
                        np.testing.assert_allclose(base.models(theta,[v],settings),same,rtol=2e-10,atol=2e-12)
                        deviations.append(np.linalg.norm(base.models(theta,grid,settings)-target,axis=1)/denominator)
                    maxdev=np.max(deviations,axis=0)
                    np.testing.assert_allclose(maxdev,saved['max_relative_increment_deviation_from_selected_by_speed'],rtol=2e-10,atol=2e-12)
                    row['equivalence_max_deviation_pct']=100*max(maxdev);equivalent_models+=len(family)
            records.append(row)
        if base.path(seed,True).exists():
            path=base.path(seed,True);cp=base.old.envelope(path);compat.validate(cp,seed,cfg,True)
            hashes[path.relative_to(base.ROOT).as_posix()]=base.old.file_hash(path)
            control.append(dict(seed=seed,status=cp['status'],saved_conditions=len(cp['records']),planned_conditions=len(base.fleet_plan(seed,cfg)),new_rollouts_recorded=3*sum(r['spec']['reuse'] is None for r in cp['records'])))
    frame=pd.DataFrame(records);metrics=['matrix_pct','increment_pct']
    clients,fleets,summary=base.aggregate(frame,metrics)
    counts=frame.groupby('kind').agg(conditions=('id','size'),converged=('converged','sum'),rank_three=('rank',lambda x:int((x==3).sum())),near_bound=('near_bound','sum')).reset_index()
    summary=summary.merge(counts,on='kind')
    bc=frame.groupby(['seed','client','block','kind'])[metrics].agg(base.finite_mean).reset_index()
    bf=bc.groupby(['seed','block','kind'])[metrics].agg(base.finite_mean).reset_index()
    blocks=bf.groupby(['block','kind'])[metrics].agg(base.finite_mean).reset_index()
    wide=clients.pivot(index=['seed','client'],columns='kind',values='matrix_pct');paired=[]
    for kind in ['single','adjacent_pair','multi_20s']:
        d=wide[kind]-wide.full
        paired.append(dict(comparison=kind+' minus full',mean_difference_pp=d.mean(),median_difference_pp=d.median(),first_condition_better=int((d<0).sum()),vehicles=len(d)))
    d=wide.single-wide.multi_20s
    paired.append(dict(comparison='single minus multi_20s',mean_difference_pp=d.mean(),median_difference_pp=d.median(),first_condition_better=int((d<0).sum()),vehicles=len(d)))
    paired=pd.DataFrame(paired)
    audit=dict(status='identification_audit_complete',fit_conditions=len(frame),new_fit_conditions=int(sum(frame.kind=='single')+sum(frame.kind=='multi_20s')+138),
        converged_conditions=int(frame.converged.sum()),uncropped_input_hash_pairs_verified=uncropped_hashes,
        cropped_input_hash_pairs_recorded_but_not_recomputed=cropped_hashes,sampled_equivalent_models_checked=equivalent_models,
        control_status=control,control_complete=all(c['status']=='execution_complete' for c in control) and len(control)==3,
        source_sha256=hashes,analysis_sha256=base.old.file_hash(__file__),compatibility_adapter=compat.adapter_identity(),
        note='Identification-only snapshot. No new nonlinear rollouts or identification. Cropped raw-data hashes cannot be recomputed without original arrays. Single-speed models remain structurally non-identifiable. No matched federation comparison.')
    for suffix,df in [('fits',frame),('clients',clients),('fleets',fleets),('summary',summary),('blocks',blocks),('paired',paired)]:df.to_csv(base.OUT/f'{PREFIX}_{suffix}.csv',index=False)
    base.old.acquisition.parent.save_json(base.OUT/f'{PREFIX}.json',audit)
    report(frame,clients,summary,blocks,paired,audit)
    print(summary.to_string(index=False));print(paired.to_string(index=False))
    print('equivalent model deviation median/max',frame.equivalence_max_deviation_pct.median(),frame.equivalence_max_deviation_pct.max());print(REPORT)


def report(frame,clients,summary,blocks,paired,audit):
    import matplotlib
    matplotlib.use('Agg');matplotlib.rcParams['svg.hashsalt']=PREFIX
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    labels=['Full','Single','Adjacent pair','Multi-speed 20 s']
    for j,k in enumerate(base.KINDS):
        v=clients[clients.kind==k].matrix_pct.to_numpy();axes[0].scatter(j+np.linspace(-.16,.16,len(v)),v,s=10,alpha=.5)
    axes[0].set(xticks=range(4),xticklabels=labels,ylabel='Common unseen matrix error [%]',title='Every vehicle; variants averaged first')
    axes[0].tick_params(axis='x',labelrotation=30)
    w=clients.pivot(index=['seed','client'],columns='kind',values='matrix_pct')
    axes[1].scatter(w.single,w.multi_20s,s=15,alpha=.6);maximum=max(w.single.max(),w.multi_20s.max())*1.05
    axes[1].plot([0,maximum],[0,maximum],'--',color='gray')
    axes[1].set(xlabel='Single-speed 20 s error [%]',ylabel='Multi-speed 20 s error [%]',title='Below diagonal: multi-speed improves')
    for k,label in zip(base.KINDS,labels):
        v=blocks[blocks.kind==k].set_index('block');axes[2].plot(range(len(v)),v.matrix_pct,marker='o',label=label)
    axes[2].set(xticks=range(4),xticklabels=['10–14','16–20','22–26','28–30'],xlabel='Original speed block [m/s]',ylabel='Mean unseen matrix error [%]',title='Low-speed result is not monotone')
    axes[2].legend(fontsize=8)
    for ax in axes:ax.grid(alpha=.2)
    buf=io.StringIO();fig.savefig(buf,format='svg',metadata={'Date':None});plt.close(fig);svg=buf.getvalue();svg=svg[svg.index('<svg'):]
    def table(df):return '<div class="scroll">'+df.to_html(index=False,border=0,float_format=lambda x:f'{x:.6g}',na_rep='unavailable')+'</div>'
    saved_cases=3*sum(c['saved_conditions'] for c in audit['control_status'])
    new_cases=sum(c['new_rollouts_recorded'] for c in audit['control_status'])
    control_notice=('Control is complete; consult the separate combined analysis for tracking conclusions.' if audit['control_complete'] else 'Control is incomplete; no restricted-coverage tracking comparison can yet be made.')
    text=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Coverage diagnostic: identification review</title><style>body{{font:16px/1.6 system-ui;background:#edf2f5;color:#193142}}main{{max-width:1350px;margin:auto;padding:30px}}section{{background:white;padding:24px;margin:20px 0;border-radius:10px}}.scroll{{overflow:auto}}table{{border-collapse:collapse;white-space:nowrap;font-size:13px}}td,th{{padding:8px;border-bottom:1px solid #ddd}}svg{{width:100%;height:auto}}.alert{{border-left:5px solid #b47c17}}</style><main>
<h1>Narrower coverage: identification review</h1>
<section class="alert"><h2>Execution status and repair</h2><p>All 588 identification condition records are complete (477 new fits, 111 reused). The checked control checkpoints contain {saved_cases} of 1764 case records, including {new_cases} of 1431 planned new rollouts. {control_notice}</p>{table(pd.DataFrame(audit['control_status']))}<p>The original runner passes saved model lists into a function expecting NumPy arrays. A separate adapter fixes that interface and normalizes Windows/Linux path separators for provenance comparison while preserving every source hash and the frozen runner. Four regression tests pass. Completed identification needs no rerun.</p><p><code>python code/experiments/paper_physical_ct_coverage_resume.py control --workers 3</code></p></section>
<section><h2>Identification result</h2>{table(summary)}<p>Matrix error is relative Frobenius error of [A,B]; increment error uses [A−I,B] in the denominator. All conditions use the same original-unobserved test speeds for each vehicle. First average variants within vehicles, then vehicles within fleets, then fleets equally. Reused conditions are not independent repetitions.</p>{svg}{table(paired)}<p>Full coverage gives 0.386% mean matrix error. Single-speed fitting gives 0.964%; two adjacent speeds give 0.730%. Using the original speeds with only 20 s total gives 0.602%, 37.6% below the single-speed mean and better for 68/90 vehicles. The equal-duration benefit appears in all three fleet means. Cropping changes waveform portions and nuisance-parameter count, so this is evidence within the stated protocol, not a pure information-equivalent experiment.</p></section>
<section><h2>What the rank result means</h2><p>Every one of the 249 single-speed fits converges with relative sensitivity rank 3. All multi-speed conditions have rank 4 at the chosen tolerance. At one speed, only s(v,μ)p₁, s(v,μ)p₂ and p₃ are determined by the local dynamics; different friction/stiffness combinations give exactly the same A/B. A converged optimizer and an interior parameter estimate do not remove that ambiguity.</p><p>The audit recomputes {audit['sampled_equivalent_models_checked']} sampled equivalent models and verifies equality at the observed speed and the saved deviations elsewhere. These alternatives are never selected by test error. Their deviations are model differences, not confidence intervals. Single-speed extrapolation from the selected solution can depend on initialization, bounds and numerical selection along the indistinguishable family.</p></section>
<section><h2>The low-speed block needs care</h2>{table(blocks)}<p>For the 10–14 m/s block, the selected single-speed fits average 0.907% error, while adjacent-pair fits average 1.645% and the short multi-speed fits 1.226%. Thus, adding a second speed is not uniformly better in this finite-data result. The model is weakly sensitive to friction at low speeds; rank 4 alone does not ensure accurate extrapolation. Adjacent pairs have 12 near-bound records, versus one for full coverage and four for multi-speed 20 s. The worst pair has 6.67% matrix error and 24.71% increment error. This supports examining uncertainty and conditioning rather than declaring identifiability solved by two speeds.</p></section>
<section><h2>Implication for the next decision</h2><p>The diagnostic establishes an identification cost of restricted coverage, including a structural single-speed ambiguity. It has not yet established a practically important tracking deficit, or that shared vehicle data resolves it. Even the selected single-speed mean remains below 1% matrix error, so the controller may tolerate much of this difference.</p><p>Resume the unchanged matched control stage before adding Pacejka tires, further restrictions or a sharing method. If control remains strong, parameter ambiguity alone is not enough to motivate the paper. If relevant degradation appears, the next comparison should give local and shared structured estimators the same restricted data. Shared friction or vehicle parameters would require a physical justification; fleet-wide equality is not automatic.</p><p>These are three reused pilot fleets, with full-state measurements and a correct analytical model family. Existing unrestricted Federated results are not a fair comparator for the restricted local fits.</p></section>
<section><h2>Audit scope</h2><p>Validated checkpoint envelopes, portable provenance, condition order and budgets; checked reuse against original fits; reproduced fitted grids and saved error metrics; verified lowest-converged training-loss selection and sampled equivalence. Verified {audit['uncropped_input_hash_pairs_verified']} uncropped measurement/input hash pairs against the original acquisition records. The {audit['cropped_input_hash_pairs_recorded_but_not_recomputed']} cropped pairs have saved hashes, but their raw arrays are unavailable here, so their hashes were not independently recomputed. No new fitting or nonlinear simulation was performed.</p><p>Reproduce: <code>python code/experiments/analyze_physical_ct_coverage_review.py</code>. This is an identification-only review; use the resume adapter's <code>analyze</code> command after the control stage completes.</p></section></main></html>'''
    REPORT.write_text('\n'.join(line.rstrip() for line in text.splitlines())+'\n',encoding='utf-8',newline='\n')


if __name__=='__main__':
    with threadpool_limits(limits=1):audit()
