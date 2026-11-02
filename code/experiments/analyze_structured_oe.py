"""Audit frozen OE checkpoints and render the interpreted HTML report.

No trajectory generation, fitting, or changes to the frozen diagnostic sources.
Run after paper_structured_oe_diagnostic.py analyze; that older command writes
its original generic report, while this command adds the reviewed interpretation.
"""
from __future__ import annotations
import hashlib
import html
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import paper_structured_oe_diagnostic as exp


def percentile(values,q):return float(np.percentile(values,q))


def audit():
    settings,cfg,lock=exp.preflight()
    original=exp.parent.load(exp.OUT/f'{exp.PREFIX}_analysis.json')
    assert original['provenance']==lock
    cells=[];matrix_count=0
    for seed in settings['seeds']:
        for profile in settings['profiles']:
            path=exp.path_for(seed,profile)
            assert exp.parent.file_hash(path)==original['checkpoint_sha256'][path.relative_to(exp.ROOT).as_posix()]
            result=exp.read_result(path,lock)
            known=exp.known.read_result(exp.known.path_for(seed,profile),lock['known_provenance'])
            prior={(v['client'],v['speed']):v for v in known['cells']}
            source=exp.known.source_result(seed,profile,lock['known_provenance'])
            clean={(v['client'],v['speed']):v['cases']['TrajectoryAllClean']['matrix_error'] for v in source['cells']}
            assert len({(v['client'],v['speed']) for v in result['cells']})==83
            for v in result['cells']:
                old=prior[(v['client'],v['speed'])];ref=np.asarray(v['reference'])
                np.testing.assert_array_equal(ref,old['reference'])
                np.testing.assert_array_equal(v['common_validation_center'],old['common_validation_center'])
                assert v['clean_matrix_error']==clean[(v['client'],v['speed'])]
                assert v['physical_pass']==old['physical_pass'] and v['settling_pass']==old['settling_pass']
                for method,m in v['models'].items():
                    if not m['fit_valid']:
                        assert not m['accuracy_pass'];continue
                    target=np.asarray(m['target']);delta=target-ref
                    err=np.linalg.norm(delta)/np.linalg.norm(ref)
                    np.testing.assert_allclose(m['matrix_error'],err,atol=1e-14,rtol=1e-12)
                    np.testing.assert_allclose(m['entry_signed_error'],delta,atol=1e-14,rtol=1e-12)
                    pred=m['rollout_nrmse']
                    passed=bool(err<=cfg['pilot_sensor_matrix_error'] and pred is not None and
                        pred<=cfg['pilot_sensor_rollout_nrmse'] and pred<m['persistence_nrmse'])
                    assert m['accuracy_pass']==passed
                    matrix_count+=1
                base=v['models']['KnownOLS']
                for key in ['target','matrix_error','rollout_nrmse','persistence_nrmse']:
                    np.testing.assert_allclose(base[key],old['KnownSteering'][key],atol=1e-10,rtol=1e-9)
                optimizer=v['optimizer'];converged=[a for a in optimizer['attempts'] if a['converged']]
                assert optimizer['converged']==bool(converged)
                if converged:
                    best=min(converged,key=lambda a:a['weighted_mse'])
                    assert best['start']==optimizer['selected_start'] and best['weighted_mse']==optimizer['weighted_mse']
                    for key in ['target','offset','initial_state','spectral_radius']:
                        np.testing.assert_array_equal(v['models']['StructuredOE'][key],best[key])
                    a=np.array(best['target'][:4]).reshape(2,2)
                    np.testing.assert_allclose(best['spectral_radius'],max(abs(np.linalg.eigvals(a))))
                    near=[a for a in converged if a['weighted_mse']<=best['weighted_mse']*(1+1e-6)]
                    spread=max(np.linalg.norm(np.array(a['target'])-np.array(b['target']))/np.linalg.norm(ref)*100 for a in near for b in near)
                else: near=[];spread=None
                cells.append(dict(**v,seed=seed,profile=profile,near_best_starts=len(near),near_best_spread_pp=spread))
    return settings,cfg,original,cells,matrix_count


def summarize(cells):
    rows=[]
    for profile in ['gentle','standard','strong']:
        z=[c for c in cells if c['profile']==profile]
        for method in exp.METHODS:
            m=[c['models'][method] for c in z];valid=[v for v in m if v['fit_valid']]
            rows.append(dict(profile=profile,method=method,n=len(z),valid=len(valid),passing=sum(v['accuracy_pass'] for v in m),
                median_pct=percentile([100*v['matrix_error'] for v in valid],50),
                p95_pct=percentile([100*v['matrix_error'] for v in valid],95),
                max_pct=max(100*v['matrix_error'] for v in valid),
                canonical_p95=percentile([v['rollout_nrmse'] for v in valid if v['rollout_nrmse'] is not None],95),
                affine_p95=percentile([v['affine_rollout_nrmse'] for v in valid if v['affine_rollout_nrmse'] is not None],95)))
    return rows


def table(headers,rows):
    return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in r)+'</tr>' for r in rows)+'</tbody></table></div>'


def plot(cells,profile):
    # Fixed axes across profiles; exact median values, no fit or smoothing.
    w,h=740,310;left,right,top,bottom=56,715,30,253
    x=lambda v:left+(v-10)/20*(right-left)
    y=lambda e:bottom-(np.log10(max(.01,e))+2)/4*(bottom-top)
    elements=[f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{profile} median matrix error by speed">']
    for tick in [.01,.1,1,10,100]:
        color='#b43a30' if tick==10 else '#dce4eb'
        elements.append(f'<line x1="{left}" x2="{right}" y1="{y(tick)}" y2="{y(tick)}" stroke="{color}" stroke-dasharray="4 4"/><text x="4" y="{y(tick)+4}">{tick:g}%</text>')
    z=[v for v in cells if v['profile']==profile]
    for method,color in [('KnownOLS','#bd4034'),('AffineOLS','#b48211'),('StructuredOE','#1266ae'),('CleanOLS','#84939b')]:
        pts=[]
        for speed in range(10,31,2):
            v=[a for a in z if a['speed']==speed]
            e=percentile([a['clean_matrix_error']*100 if method=='CleanOLS' else a['models'][method]['matrix_error']*100 for a in v],50)
            pts.append(f'{x(speed):.3f},{y(e):.3f}')
        dash=' stroke-dasharray="5 3"' if method=='CleanOLS' else ''
        elements.append(f'<polyline points="{" ".join(pts)}" stroke="{color}" stroke-width="2.5" fill="none"{dash}/>')
    for speed in range(10,31,2):elements.append(f'<text x="{x(speed)}" y="276" text-anchor="middle">{speed}</text>')
    elements.append('<text x="390" y="302" text-anchor="middle">Speed [m/s]</text></svg>')
    return ''.join(elements)


def main():
    settings,cfg,source,cells,count=audit();summary=summarize(cells)
    attempts=[a for c in cells for a in c['optimizer']['attempts']]
    failed=[a for a in attempts if not a['converged']]
    bad=[]
    for c in cells:
        for a in c['optimizer']['attempts']:
            if a['converged']:
                error=100*np.linalg.norm(np.array(a['target'])-c['reference'])/np.linalg.norm(c['reference'])
                if error>10:bad.append((c,a,error))
    selected=[c['models']['StructuredOE'] for c in cells]
    evidence=dict(status='audit_passed',source_analysis_sha256=exp.parent.file_hash(exp.OUT/f'{exp.PREFIX}_analysis.json'),
        checkpoint_sha256=source['checkpoint_sha256'],source_provenance=source['provenance'],
        generator_sha256=exp.parent.file_hash(__file__),matrix_errors_recomputed=count,
        raw_trajectories_replayed=False,baseline_reproduced_from_checkpoints=True,
        selected_oe_pass=sum(m['accuracy_pass'] for m in selected),selected_oe_total=len(cells),
        profiles=summary,attempts=len(attempts),failed_attempts=len(failed),
        converged_attempts_over_10pct_matrix_error=len(bad),
        near_best_objective_relative_tolerance=1e-6,
        min_near_best_starts=min(c['near_best_starts'] for c in cells),
        max_near_best_parameter_spread_pct=max(c['near_best_spread_pp'] for c in cells),
        all_selected_jacobians_rank10=all(c['optimizer']['jacobian_rank']==10 for c in cells),
        selected_radius_min=min(m['spectral_radius'] for m in selected),
        selected_radius_max=max(m['spectral_radius'] for m in selected),
        recommendation='Freeze gentle excitation with structured OE for independent confirmation; do not unlock automatically.',
        historical_selection_changed=False,confirmation_unlocked=False)
    exp.parent.save_json(exp.OUT/f'{exp.PREFIX}_diagnosis.json',evidence)
    # This interpreted report is intentionally specific to the audited outcome.
    assert evidence['selected_oe_pass']==747 and evidence['selected_oe_total']==747
    assert len(attempts)==2241 and len(failed)==42 and len(bad)==5
    assert evidence['min_near_best_starts']>=2 and evidence['all_selected_jacobians_rank10']
    cards=''.join(f'<article><span>{p.title()} excitation</span><strong>{next(r["median_pct"] for r in summary if r["profile"]==p and r["method"]=="StructuredOE"):.2f}%</strong><small>median matrix error · 249/249 pass</small></article>' for p in settings['profiles'])
    overview=table(['Profile','Estimator','Pass / 249','Median error','95th percentile','Worst error'],[[r['profile'],r['method'],r['passing'],f'{r["median_pct"]:.3f}%',f'{r["p95_pct"]:.3f}%',f'{r["max_pct"]:.3f}%'] for r in summary])
    low=[];fleet=[];prediction=[]
    for profile in settings['profiles']:
        z=[c for c in cells if c['profile']==profile]
        for speed in [10,12,14,16]:
            v=[c for c in z if c['speed']==speed]
            low.append([profile,speed,f'{percentile([c["models"]["KnownOLS"]["matrix_error"]*100 for c in v],50):.2f}%',f'{percentile([c["models"]["StructuredOE"]["matrix_error"]*100 for c in v],50):.2f}%',f'{max(c["models"]["StructuredOE"]["matrix_error"]*100 for c in v):.2f}%',f'{sum(c["models"]["StructuredOE"]["accuracy_pass"] for c in v)}/{len(v)}'])
        for seed in settings['seeds']:
            v=[c for c in z if c['seed']==seed];errs=[c['models']['StructuredOE']['matrix_error']*100 for c in v]
            fleet.append([profile,seed,f'{percentile(errs,50):.3f}%',f'{percentile(errs,95):.3f}%',f'{max(errs):.3f}%','83/83'])
        for r in [a for a in summary if a['profile']==profile]:prediction.append([profile,r['method'],f'{r["canonical_p95"]:.5f}',f'{r["affine_p95"]:.5f}'])
    worst=sorted(cells,key=lambda c:c['models']['StructuredOE']['matrix_error'],reverse=True)[:12]
    worsttable=table(['Profile','Fleet','Vehicle','Speed','OE error','Known OLS error'],[[c['profile'],c['seed'],c['client'],c['speed'],f'{100*c["models"]["StructuredOE"]["matrix_error"]:.3f}%',f'{100*c["models"]["KnownOLS"]["matrix_error"]:.3f}%'] for c in worst])
    badtable=table(['Profile','Fleet','Vehicle','Speed','Start','Matrix error','Loss / selected loss'],[[c['profile'],c['seed'],c['client'],c['speed'],a['start'],f'{error:.1f}%',f'{a["weighted_mse"]/c["optimizer"]["weighted_mse"]:.2f}'] for c,a,error in bad])
    charts=''.join(f'<h3>{p.title()} excitation</h3>'+plot(cells,p) for p in settings['profiles'])
    content='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Structured OE — results and decision</title><style>
:root{color-scheme:light}body{margin:0;background:#eef3f7;color:#203245;font:16px/1.65 system-ui,sans-serif}main{max-width:1120px;margin:auto;padding:36px 24px}h1{font-size:36px;line-height:1.2;letter-spacing:-1px}h2{font-size:23px;line-height:1.3}h3{font-size:18px}p{max-width:95ch}.eyebrow{font-size:12px;font-weight:700;letter-spacing:1.5px;color:#406377}section{background:white;padding:26px 30px;border:1px solid #dce5ec;border-radius:12px;margin:22px 0}.lead{font-size:20px}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}.cards article{background:#e7f4ed;border:1px solid #c2e0cd;border-radius:10px;padding:20px}.cards span,.cards small{display:block}.cards strong{font-size:36px;display:block;color:#126441}.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:10px 9px;text-align:right;border-bottom:1px solid #e3e9ed;white-space:nowrap}th:first-child,td:first-child{text-align:left}th{color:#476174;font-size:12px}tr:nth-child(even){background:#f7f9fb}.note{background:#f0f5fa;padding:16px 20px;border-left:4px solid #427b9d}.caution{background:#fff7e6;border-left-color:#b58014}svg{width:100%;max-width:850px}svg text{font:12px system-ui;fill:#405365}a{color:#1266ae}summary{cursor:pointer;font-weight:650}pre{white-space:pre-wrap;background:#edf3f7;padding:18px;font-size:13px}.legend span{margin-right:20px}footer{font-size:13px;color:#536c7c}@media(max-width:650px){.cards{grid-template-columns:1fr}section{padding:20px}h1{font-size:29px}}
</style><main><p class="eyebrow">FEDERATED LPV · LOCAL CALIBRATION DIAGNOSTIC</p><h1>Output-error fitting resolves the observed pilot calibration failure</h1><p class="lead">All 747 selected OE fits pass the original accuracy criteria. The improvement also holds for the weakest excitation and every low-speed case.</p>'''+cards+'''
<section><h2>What this experiment answers</h2><p>The previous pilot produced accurate local matrices with exact simulated states, but large errors with noisy measured states—even when steering was known. This diagnostic keeps the physical trajectories, noisy states, speed coverage and applied steering fixed and changes how the local dynamics are fitted.</p><p><strong>KnownOLS</strong> is the frozen one-step fit with quiet-segment centering. <strong>AffineOLS</strong> adds a fitted constant term. <strong>StructuredOE</strong> simulates the full training segment and fits the same affine A/B dynamics together with an initial state. Physical coordinates are fixed by C = I. The clean OLS reference is for diagnosis only.</p><div class="note">The question is whether these driving maneuvers support local identification with measurement noise. This experiment does not yet test federated sharing, full-envelope interpolation, changing-speed behavior, or controller performance.</div></section>
<section><h2>The main result</h2>'''+overview+'''<p>OE improves matrix error in every paired cell against both OLS variants. Affine OLS has the same passing counts as the original known-input fit and nearly identical errors. Allowing a constant offset alone therefore does not explain the recovery; fitting the simulated trajectory with an estimated initial state is the effective change in this diagnostic.</p><p>A passing cell must have matrix error ≤10%, one-second normalized rollout error ≤0.5, and lower rollout error than persistence. The profile threshold is ≥95% passing cells, with all physical and settling checks passing. All three profiles now pass descriptively, with no threshold changes.</p></section>
<section><h2>The low-speed failure is removed in this pilot</h2><p>At 10 m/s the gentle profile changes from 57.30% median matrix error under known-input OLS to 1.00% under OE. Its worst cell is 6.06%, below the unchanged 10% threshold. The original low-speed tail is therefore not an unavoidable consequence of these trajectories.</p><p class="legend"><span style="color:#bd4034">● Known OLS</span><span style="color:#b48211">● Affine OLS</span><span style="color:#1266ae">● OE</span><span style="color:#84939b">– – Clean OLS</span></p><p>All plots use the same logarithmic error axis. Curves show medians; the dashed red line marks the per-cell matrix threshold. Overlapping OLS curves indicate very similar errors.</p>'''+charts+table(['Profile','Speed [m/s]','OLS median','OE median','OE worst','OE pass'],low)+'''<p>Standard and strong profiles share capped excitation at 10 and 12 m/s, so their results there are identical. These repeated cells are not additional independent evidence.</p></section>
<section><h2>Prediction improves too, but keep the two metrics separate</h2>'''+table(['Profile','Estimator','Canonical rollout p95','Full-affine rollout p95'],prediction)+'''<p>The canonical metric evaluates A/B using the common validation center, preserving the original gates. The full-affine metric uses each model’s training-derived offset, without fitting a new validation offset. Both evaluate independent chirp maneuvers and restart one-second windows at the exact simulator state. Values are dimensionless normalized RMSE, not percentages or an observer performance measure.</p><p>The stronger excitation gives slightly better matrix-error tails but higher OE validation prediction errors. This pattern is consistent with increased local-linear approximation error at larger amplitudes; this comparison does not isolate that mechanism. Gentle already meets the required matrix accuracy and has the lowest OE prediction-error tail.</p></section>
<section><h2>Numerical audit: keep the multiple starts</h2><p>Every selected fit converged, has a full-rank 10-parameter Jacobian, and has spectral radius between '''+f'{evidence["selected_radius_min"]:.4f} and {evidence["selected_radius_max"]:.4f}'+'''. No stability constraint was imposed. These checks support numerical usability but do not prove global optimality or general identifiability.</p><p>Of 2241 starts, 42 reached the evaluation limit. Five additional starts converged to poor solutions with large matrix errors. All five had higher training loss and were rejected by the predeclared selection rule. Every case has at least two starts within 10⁻⁶ relative training objective of the best solution; their largest pairwise matrix difference is only '''+f'{evidence["max_near_best_parameter_spread_pct"]:.6f}%'+''' of the reference matrix norm.</p><div class="note caution">Do not replace the three-start procedure with one arbitrary initialization. “Solver converged” alone was insufficient for several starts. The saved lowest-training-loss selection is part of the estimator protocol.</div><details><summary>Inspect the five poor converged starts (none selected)</summary>'''+badtable+'''</details></section>
<section><h2>Consistency across fleets</h2>'''+table(['Profile','Fleet seed','OE median','OE p95','OE worst','OE pass'],fleet)+'''<p>All nine fleet/profile runs pass 83/83 cells. These are three reused development fleets, not nine independent fleets or 747 independent fleet replications. The results establish feasibility within the simulated pilot, not out-of-sample confirmation.</p><details><summary>Inspect the twelve largest selected OE errors</summary>'''+worsttable+'''</details></section>
<section><h2>Recommended next step</h2><p><strong>Freeze gentle excitation with structured OE as the candidate for independent confirmation.</strong> It is the smallest tested excitation and passes all original criteria. Keep the existing state-noise levels, known applied steering, physical coordinates, three starts, offset/initial-state handling and acceptance thresholds fixed.</p><ol><li>Define the revised confirmation protocol using the untouched confirmation seeds. Preserve the original rejected OLS selection as historical evidence.</li><li>Confirm local matrix recovery first. Then pass the resulting local A/B estimates into the federated speed-basis regression and rerun the relevant comparisons with the same estimator for every method.</li><li>Check how the changed local estimator affects regression weights, local-prior equivalence statements and operating-point handling before manuscript integration.</li></ol><p>The present analysis does not unlock the old confirmation runner or change the historical pilot selection. It does not justify a claim about ordinary driving, production sensing, noisy speed measurements or real-vehicle validation. Both sideslip and yaw rate are measured here, and the estimator uses the declared simulation measurement variances.</p></section>
<section><h2>Provenance and reproducibility</h2><p>Nine checkpoint hashes and nested source locks verified. All '''+str(count)+''' saved matrix errors were independently recomputed from A/B and the frozen references. The known-steering baseline, common validation centers, pass flags and training-loss selection were checked. The original summary CSVs, analysis JSON and generic report were regenerated without differences before this interpreted report was created.</p><p>Raw trajectories are ignored by git and were unavailable in this workspace. This audit did not rerun fits or validation rollouts; it verifies saved numerical outputs and their provenance. No new scientific simulation was run.</p><pre>python code/experiments/analyze_structured_oe.py</pre><p>This command rebuilds the interpreted report and the diagnosis JSON. The older <code>paper_structured_oe_diagnostic.py analyze</code> command recreates the original generic report; run this audit command afterward to restore the interpretation.</p><p><a href="../docs/structured_oe_diagnostic.html">Frozen diagnostic protocol</a> · <a href="../results/tables/paper_structured_oe_diagnostic_diagnosis.json">Audit JSON</a> · <a href="../results/tables/paper_structured_oe_diagnostic_summary.csv">Original summary CSV</a></p></section><footer>Results checkpoint: 699369c · Original analysis: e1344ec · Scientific scope: fixed-speed local calibration, three reused pilot fleets.</footer></main></html>'''
    exp.REPORT.write_text(content,encoding='utf-8',newline='\n')
    print(json.dumps({k:evidence[k] for k in ['status','matrix_errors_recomputed','selected_oe_pass','failed_attempts','converged_attempts_over_10pct_matrix_error','min_near_best_starts','max_near_best_parameter_spread_pct']},indent=2))
    print(exp.REPORT)


if __name__=='__main__':main()
