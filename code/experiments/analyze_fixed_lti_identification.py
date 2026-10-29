"""Audit saved fixed-LTI pilot results and add an interpreted HTML report.

No new simulation or model fitting; frozen experimental sources remain unchanged.
"""
from pathlib import Path
import json
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import paper_fixed_lti_identification as run


def percentiles(values):
    return dict(zip(['min','median','p95','max'],map(float,np.percentile(values,[0,50,95,100]))))


def main():
    protocol,settings,cfg,lock=run.preflight()
    source=run.parent.load(run.OUT/f'{run.PREFIX}_analysis.json')
    assert source['provenance']==lock
    cells=[];count=0;bounds_ok=0;settled=0
    for seed in protocol['seeds']:
        path=run.path_for(seed)
        assert run.parent.file_hash(path)==source['checkpoint_sha256'][path.relative_to(run.ROOT).as_posix()]
        result=run.read_result(path,lock)
        old=run.oe.read_result(run.oe.path_for(seed,'gentle'),lock['oe_provenance'])
        oldmap={(c['client'],c['speed']):c for c in old['cells']}
        assert len({(c['client'],c['speed']) for c in result['cells']})==83
        for c in result['cells']:
            assert c['status']=='simulated'
            original=oldmap[(c['client'],c['speed'])]
            np.testing.assert_array_equal(c['reference'],original['reference'])
            ref=np.array(c['reference'])
            for key in ['target','matrix_error','rollout_nrmse','affine_rollout_nrmse']:
                np.testing.assert_allclose(c['models']['OpenOE3'][key],original['models']['StructuredOE'][key],atol=1e-12)
            for name,m in c['models'].items():
                assert m['fit_valid']
                delta=np.asarray(m['target'])-ref;err=np.linalg.norm(delta)/np.linalg.norm(ref)
                np.testing.assert_allclose(m['matrix_error'],err,atol=1e-14,rtol=1e-12)
                np.testing.assert_allclose(m['entry_signed_error'],delta,atol=1e-14)
                pred=m['rollout_nrmse']
                passed=bool(err<=cfg['pilot_sensor_matrix_error'] and pred is not None and
                    pred<=cfg['pilot_sensor_rollout_nrmse'] and pred<m['persistence_nrmse'])
                assert passed==m['accuracy_pass'];count+=1
            for optname,modelname in [('open_optimizer','OpenOE4'),('closed_optimizer','ClosedOE4'),('closed_three_start_optimizer','ClosedOE3')]:
                opt=c[optname];best=min([a for a in opt['attempts'] if a['converged']],key=lambda a:a['weighted_mse'])
                assert best['start']==opt['selected_start'] and best['weighted_mse']==opt['weighted_mse']
                for key in ['target','offset','initial_state','spectral_radius']:
                    np.testing.assert_array_equal(best[key],c['models'][modelname][key])
                target=c['models'][modelname]['target']
                np.testing.assert_allclose(max(abs(np.linalg.eigvals(np.array(target[:4]).reshape(2,2)))),best['spectral_radius'])
            nominal=np.r_[np.array(lock['controller']['a']).ravel(),lock['controller']['b']]
            for key in ['open_optimizer','closed_optimizer']:
                np.testing.assert_array_equal(c[key]['attempts'][-1]['initial_target'],nominal)
            state_bounds=[];state_settling=[];physical=[]
            for phase in ['training','validation']:
                p=c[phase]
                bounded=bool(p['max_beta_deg']<=cfg['max_beta_deg'] and p['max_ay']<=cfg['max_ay'] and
                    p['max_steering_deg']<=cfg['steering_limit_deg']+1e-8 and p['max_rate_deg_s']<=cfg['steering_rate_deg_s']+1e-8)
                settle=bool(p['quiet_yaw_error']<=max(cfg['settling_abs_yaw'],cfg['settling_relative_yaw']*c['speed']*cfg['curvature_per_m']) and p['quiet_drift']<=cfg['max_quiet_drift'])
                assert p['physical_pass']==(bounded and p['excitation_limited_steps']==0)
                assert p['settling_pass']==settle
                tr=c['tracking_train' if phase=='training' else 'tracking_validation']
                np.testing.assert_allclose(tr['limiter_fraction'],p['excitation_limited_steps']/round(cfg['excitation_s']/cfg['sample_time']))
                state_bounds.append(bounded);state_settling.append(settle);physical.append(p['physical_pass'])
            assert c['physical_pass']==all(physical) and c['settling_pass']==all(state_settling)
            bounds_ok+=all(state_bounds);settled+=all(state_settling)
            cells.append(dict(c,seed=seed))
    summary=[]
    for method in run.METHODS:
        m=[c['models'][method] for c in cells]
        row=dict(method=method,passing=sum(v['accuracy_pass'] for v in m),
            matrix_error_pct=percentiles([100*v['matrix_error'] for v in m]),
            canonical_rollout=percentiles([v['rollout_nrmse'] for v in m]),
            affine_rollout=percentiles([v['affine_rollout_nrmse'] for v in m]))
        saved=next(s for s in source['summaries'] if s['method']==method)
        np.testing.assert_allclose([saved['median_pct'],saved['p95_pct']],[row['matrix_error_pct']['median'],row['matrix_error_pct']['p95']])
        assert saved['passing']==row['passing'];summary.append(row)
    nominal_stats={}
    for optname in ['open_optimizer','closed_optimizer']:
        extra=[c[optname]['attempts'][-1] for c in cells]
        assert all(a['converged'] for a in extra)
        nominal_stats[optname]=dict(converged=len(extra),selected=sum(c[optname]['selected_start']=='fixed_nominal_20mps' for c in cells),
            nfev=percentiles([a['nfev'] for a in extra]),
            relative_objective_gap=percentiles([(a['weighted_mse']-c[optname]['weighted_mse'])/c[optname]['weighted_mse'] for c,a in zip(cells,extra)]),
            parameter_difference_pct=percentiles([100*np.linalg.norm(np.array(a['target'])-c[optname]['target'])/np.linalg.norm(c['reference']) for c,a in zip(cells,extra)]),
            matrix_error_pct=percentiles([100*np.linalg.norm(np.array(a['target'])-c['reference'])/np.linalg.norm(c['reference']) for c,a in zip(cells,extra)]))
    physical={k:percentiles([c[p][k] for c in cells for p in ['training','validation']]) for k in ['max_beta_deg','max_ay','max_steering_deg','max_rate_deg_s','max_jerk','excitation_limited_steps']}
    limiter={p:percentiles([100*c[k]['limiter_fraction'] for c in cells]) for p,k in [('training','tracking_train'),('validation','tracking_validation')]}
    speed=[]
    for v in range(10,31,2):
        z=[c for c in cells if c['speed']==v]
        speed.append(dict(speed=v,n=len(z),matrix_error_pct=percentiles([100*c['models']['ClosedOE4']['matrix_error'] for c in z]),
            yaw_rmse_deg_s=percentiles([c['tracking_validation']['yaw_rmse_deg_s'] for c in z]),
            limiter_pct=percentiles([100*c['tracking_validation']['limiter_fraction'] for c in z])))
    audit=dict(status='audit_passed',source_provenance=lock,checkpoint_sha256=source['checkpoint_sha256'],
        source_analysis_sha256=run.parent.file_hash(run.OUT/f'{run.PREFIX}_analysis.json'),generator_sha256=run.parent.file_hash(__file__),
        matrix_errors_recomputed=count,summary=summary,nominal_initialization=nominal_stats,by_speed=speed,
        bounded_cells=bounds_ok,settled_cells=settled,physical=physical,limiter_pct=limiter,
        closed_matrix_improves_over_open=sum(c['models']['ClosedOE4']['matrix_error']<c['models']['OpenOE3']['matrix_error'] for c in cells),
        all_selected_closed_jacobians_rank10=all(c['closed_optimizer']['jacobian_rank']==10 for c in cells),
        max_selected_closed_spectral_radius=max(c['closed_optimizer']['spectral_radius'] for c in cells),
        gate_pass=False,confirmation_unlocked=False,lpv_control_benefit_tested=False,
        raw_trajectories_replayed=False,
        interpretation='Local identification succeeds; the inherited zero-clipping requirement fails. Applied bounds and settling pass. Nominal initialization matches the best solution to numerical precision.',
        recommendation='Retain one fixed nominal controller; develop common feedback-noise handling before freezing acquisition. Preserve raw measurements for OE and rerun a paired diagnostic before confirmation.')
    assert count==1245 and bounds_ok==249 and settled==249
    assert all(c['models']['ClosedOE4']['accuracy_pass'] for c in cells)
    assert all(c[p]['excitation_limited_steps']>0 for c in cells for p in ['training','validation'])
    assert source['identification_gate_pass'] is False
    run.parent.save_json(run.OUT/f'{run.PREFIX}_diagnosis.json',audit)
    # Rebuild the original figures/details without editing their frozen generator.
    analysis_before=(run.OUT/f'{run.PREFIX}_analysis.json').read_bytes()
    run.analyze()
    assert (run.OUT/f'{run.PREFIX}_analysis.json').read_bytes()==analysis_before
    base=run.REPORT.read_text()
    methodtable=run.table(['Acquisition / starts','Pass','Median matrix error [%]','p95 [%]','Worst [%]'],[
        [r['method'],f'{r["passing"]}/249',r['matrix_error_pct']['median'],r['matrix_error_pct']['p95'],r['matrix_error_pct']['max']] for r in summary])
    boundtable=run.table(['Check','Observed worst','Threshold / rule','Outcome'],[
        ['Sideslip magnitude',physical['max_beta_deg']['max'],'5 deg','All pass'],
        ['Lateral acceleration magnitude',physical['max_ay']['max'],'6 m/s²','All pass'],
        ['Applied steering magnitude',physical['max_steering_deg']['max'],'12 deg','All pass'],
        ['Applied steering rate magnitude',physical['max_rate_deg_s']['max'],'30 deg/s (+ numerical tolerance)','All pass'],
        ['Settling','249/249','Original yaw/drift criteria','All pass'],
        ['Limiter activations / 2000 fitting steps',physical['excitation_limited_steps']['max'],'Zero','All 498 runs fail']])
    nomtable=run.table(['Data','Nominal converged','Nominal selected','Median evaluations','Max loss gap relative to best','Max matrix difference [% of reference norm]'],[
        [name.replace('_optimizer',''),d['converged'],d['selected'],d['nfev']['median'],f'{d["relative_objective_gap"]["max"]:.2e}',f'{d["parameter_difference_pct"]["max"]:.6f}'] for name,d in nominal_stats.items()])
    outcomes='''<section style="border-left:5px solid #25724d"><h2>Reviewed outcome</h2><p><strong>The fixed-controller maneuvers support accurate identification: 249/249 selected OE fits pass.</strong> Their median matrix error is 0.232%, versus 0.315% in the gentle open-loop reference. The overall protocol gate remains failed because every training and validation maneuver activates the steering-rate limiter.</p><p>Three separate conclusions follow: the local estimator works on these closed-loop data; the fixed nominal initialization is effective; and the feedback implementation still needs refinement if we require unclipped, smooth acquisition.</p>'''+methodtable+'''<p>Closed-loop matrix error improves in 166/249 paired cells, but the acquisition comparison changes both feedback and excitation construction. It cannot attribute the difference to feedback alone. The nominal-start differences are numerical only. All comparisons reuse three development fleets.</p></section>
<section><h2>What “physical failure” means here</h2>'''+boundtable+'''<p>The failed flag does <strong>not</strong> mean that applied steering or vehicle-response bounds were exceeded. The historical acquisition check combines those bounds with a stricter requirement: no steering-limit activation during the identification segment. All cells fail that last requirement. The limiter enforces the applied-input bounds successfully.</p><p>The median limiter duty is 6.55% in both training and validation. Across all 498 maneuvers, 58–195 of the 2000 fitting steps are limited (2.90–9.75%). The maximum reported lateral jerk is 32.76 m/s³; jerk had no acceptance threshold, so it is a diagnostic, not an additional gate failure.</p><p>The accurate fitted models remain useful evidence: the estimator uses the actual limited steering, so limiter activity does not create an unrecorded input mismatch. It is not yet evidence for the intended smooth acquisition protocol. Keep the original failed gate in the record; changing an acceptance rule would require a new explicitly defined protocol.</p></section>
<section><h2>The nominal model is a useful initialization</h2>'''+nomtable+'''<p>All 498 added nominal starts converge. Their largest relative training-loss gap to the selected best fit is below 6×10⁻¹², and their largest matrix difference is below 0.00015% of the reference norm. Selection frequencies of 63/249 and 64/249 reflect tiny numerical differences, not meaningful identification improvements.</p><p>Use the fixed nominal model as the primary start in the eventual implementation. Keep the existing fallbacks through independent confirmation; this reused pilot is not sufficient to guarantee one-start convergence on unseen vehicles.</p></section>
<section><h2>What the fixed controller already achieves</h2><p>The same model, gain and prefilter are used everywhere. Median validation yaw-rate RMSE decreases from 0.0715 deg/s at 10 m/s to 0.0305 deg/s at 30 m/s. All 249 cases settle successfully, and the worst selected matrix error is 4.683%, below the unchanged 10% criterion.</p><p>This fixed LTI baseline tracks the tested mild references well across the full speed range. These results do not show that LPV control is necessary or will substantially outperform it. The paper's main question remains whether cooperative identification improves model accuracy across locally unobserved speeds. Control-performance benefits from LPV models require a separate matched test.</p></section>
<section><h2>Likely source of limiter activity, and the clean next step</h2><p>The controller feeds noisy beta and yaw measurements directly into a fixed gain at 100 Hz. With its frozen gains, the instantaneous measurement-noise contribution to the requested steering has standard deviation about 0.0773 deg, before accounting for the resulting feedback dynamics. This provides a plausible explanation for repeated rate-limit activation, but the saved statistics alone do not isolate causality.</p><p><strong>Keep the single nominal LTI controller and test common feedback-noise handling.</strong> A fixed measurement filter for the feedback path is a focused next candidate. Keep the raw measured states for OE fitting and record the actual applied steering, so filtering the control signal does not silently change the estimator's measurement-noise model. Select any filter settings only on development data and apply the same sensing pipeline to later fixed-LTI and LPV controllers.</p><p>Compare raw and filtered feedback on the same maneuver, noise realization, physical limits and estimator settings. Assess tracking, limiter duty, jerk and matrix recovery together. Do not tune the nominal baseline to enlarge the eventual LPV performance gap. Freeze acquisition only after this practical issue is resolved or a different prospective limit-duty requirement is justified.</p></section>
<section><h2>Audit and reproducibility</h2><p>All three checkpoint hashes and nested source locks were verified. All 1245 matrix errors were recomputed from the saved physical matrices and frozen references. The original open-loop baseline, training-loss selection, nominal initial matrices, spectral radii and physical/settling flags were checked. All selected closed-loop Jacobians are reported rank 10; the largest selected spectral radius is 0.97595.</p><p>No new scientific simulation or fit was run. Raw NPZ trajectories are ignored by git and unavailable here, so prediction/tracking metrics and physical extrema were checked for internal consistency rather than independently replayed. The existing report previews are decimated to 10 Hz from the 100 Hz simulation and cannot establish high-frequency smoothness; limit-duty and jerk summaries use the original full-rate computations.</p><p>Rebuild this reviewed report with <code>python code/experiments/analyze_fixed_lti_identification.py</code>. The older experiment <code>analyze</code> command recreates the generic report; run this audit command afterward to restore the interpretation.</p><p><a href="../results/tables/paper_fixed_lti_identification_diagnosis.json">Audited numerical summary</a> · <a href="../docs/fixed_lti_identification.html">Frozen experiment protocol</a></p></section>'''
    marker='<section><h2>Goal and scope</h2>'
    assert base.count(marker)==1
    base=base.replace(marker,outcomes+marker,1)
    base=base.replace('<title>Fixed LTI controller identification</title>','<title>Fixed LTI identification — reviewed results</title>')
    base=base.replace('<h1>Identification under one fixed nominal LTI controller</h1>','<h1>Accurate identification under a fixed controller, with a remaining input-smoothness issue</h1>')
    run.REPORT.write_text(base,encoding='utf-8',newline='\n')
    print(json.dumps(dict(audit='passed',matrix_errors=count,bounded_cells=bounds_ok,settled_cells=settled,gate_pass=False),indent=2))
    print(run.REPORT)


if __name__=='__main__':main()
