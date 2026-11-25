"""Read-only numerical audit and interpretation of the frozen relaxation gate.

Replays existing cases only. Original source hashes, checkpoints and surfaces
are preserved. Exact LF/CRLF source variants are verified explicitly.
"""
from __future__ import annotations
import os
for name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[name]='1'
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'code/src'),str(ROOT/'code/experiments')]
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import paper_tire_relaxation_gate as run
from federated_lpv.tire_relaxation import trajectory_sensitivities,observability,distinguishability,project_nuisance

PREFIX='paper_tire_relaxation_gate_review'
REPORT=ROOT/'report/tire_relaxation_gate_review.html'


def portable_validate(p,job,cfg):
    expected=run.identity(cfg);original=p['provenance'];forms={}
    if original['frozen']!=expected['frozen'] or original['runtime']!=expected['runtime']:
        raise ValueError('Frozen input/runtime mismatch')
    if set(original['files'])!=set(expected['files']):raise ValueError('Different source list')
    for name,sha in original['files'].items():
        raw=(ROOT/name).read_bytes();lf=raw.replace(b'\r\n',b'\n')
        candidates={'workspace_bytes':raw,'LF':lf,'CRLF':lf.replace(b'\n',b'\r\n')}
        matched=[key for key,value in candidates.items() if hashlib.sha256(value).hexdigest()==sha]
        if not matched:raise ValueError(f'Source content differs beyond line endings: {name}')
        forms[name]=matched[0]
    comparable=copy.deepcopy(p);comparable['provenance']['files']=expected['files']
    run.validate(comparable,job,cfg)
    return forms


def compare_nested(actual,expected):
    if isinstance(expected,dict):
        assert set(actual)==set(expected)
        for k in expected:compare_nested(actual[k],expected[k])
    elif isinstance(expected,list) and expected and isinstance(expected[0],dict):
        assert len(actual)==len(expected)
        for a,e in zip(actual,expected):compare_nested(a,e)
    elif isinstance(expected,(list,float,int)) and not isinstance(expected,bool):
        np.testing.assert_allclose(actual,expected,rtol=1e-8,atol=1e-9)
    else:assert actual==expected


def audit(cfg):
    replays=[];state=[];parameter=[];minima=[];extra=[];hashes={};forms={}
    max_surface_error=0.;max_sensitivity_error=0.;saved=[]
    for job in run.jobs(cfg):
        p=run.read_checkpoint(run.path(job));forms[run.key(job)]=portable_validate(p,job,cfg)
        assert p['status']=='execution_complete';saved.append(p)
        hashes[run.path(job).relative_to(ROOT).as_posix()]=run.file_hash(run.path(job))
        record,arrays=run.generate(job,cfg);old=p['data']
        assert record['operating_checks_passed']==old['operating_checks_passed']
        compare_nested(record['metrics'],old['metrics'])
        preview_diff=0.
        for name,value in old['preview'].items():
            other=arrays[name][::10];np.testing.assert_allclose(other,value,rtol=1e-10,atol=1e-10)
            preview_diff=max(preview_diff,float(np.max(abs(other-value))))
        replays.append(dict(client=job[0],speed=job[1],probe=job[2],exact_trace_hash_match=record['trace_sha256']==old['trace_sha256'],
            exact_noise_hash_match=record['noise_sha256']==old['noise_sha256'],original_trace_sha256=old['trace_sha256'],
            reconstructed_trace_sha256=record['trace_sha256'],max_preview_abs_difference=preview_diff,
            max_metric_abs_difference=max(abs(record['metrics'][k]-old['metrics'][k]) for k in old['metrics'])))
        c=run.client(cfg,job[0]);theta=np.r_[c['physical_ratios'],cfg['relaxation_lengths_m']]
        _,jac=trajectory_sensitivities(theta,arrays['steering'],job[1],c['geometry_m'],cfg['sample_time'],cfg['integration_max_step'],cfg['state_scales'],np.deg2rad(cfg['yaw_noise_std_deg_s']))
        info=distinguishability(jac,cfg['svd_rcond']);compare_nested(info,p['sensitivity']['parameter'])
        for rcond in [1e-8,1e-10]:compare_nested(distinguishability(jac,rcond),p['sensitivity']['parameter_rcond_sensitivity'][str(rcond)])
        obs=observability(theta,job[1],c['geometry_m'],cfg['sample_time'],cfg['observability_horizon_s'],cfg['state_scales'],np.deg2rad(cfg['yaw_noise_std_deg_s']),cfg['svd_rcond'])
        compare_nested(obs,p['sensitivity']['state'])
        state.append(dict(client=job[0],speed=job[1],probe=job[2],operating_checks_passed=old['operating_checks_passed'],**old['metrics'],
            **{k:v for k,v in p['sensitivity']['state'].items() if k!='singular_values'}))
        for assumption,val in p['sensitivity']['parameter'].items():
            parameter.append(dict(client=job[0],speed=job[1],probe=job[2],assumption=assumption,**{k:v for k,v in val.items() if k!='singular_values'}))
            max_sensitivity_error=max(max_sensitivity_error,abs(val['smallest_singular']-info[assumption]['smallest_singular']))
        # Additional audit decomposition: determine whether friction alone drives
        # the loss of information. Uses the same trajectory derivatives, no refit.
        target=jac[:,4:6];nuisance=jac[:,[0,1,2,3,6,7,8,9]]
        effective,_=project_nuisance(target,nuisance,cfg['svd_rcond'])
        _,singular,vh=np.linalg.svd(effective,full_matrices=False)
        log_step=.1*vh[-1];comp=-np.linalg.pinv(nuisance,rcond=cfg['svd_rcond'])@(target@log_step)
        mu_fixed,_=project_nuisance(target,jac[:,[0,1,2,6,7,8,9]],cfg['svd_rcond'])
        extra.append(dict(client=job[0],speed=job[1],probe=job[2],mu_fixed_smallest_singular=float(np.linalg.svd(mu_fixed,compute_uv=False)[-1]),
            all_ct_free_smallest_singular=float(singular[-1]),sigma_front_log_step=float(log_step[0]),sigma_rear_log_step=float(log_step[1]),
            **{f'{name}_compensating_log_step':float(comp[i]) for i,name in enumerate(['p1','p2','p3','mu'])},
            note='Infinitesimal linear compensation scaled to norm 0.1 for illustration; not a verified finite nonlinear ambiguity.'))
        for base,ratios in [('true_ct',c['physical_ratios']),('previous_ct',c['previous_estimated_ratios'])]:
            reproduced=run.surface(ratios,arrays,job[1],c['geometry_m'],cfg);original=p['surfaces'][base]
            np.testing.assert_allclose(reproduced['axis_m'],original['axis_m'],rtol=1e-14,atol=1e-15)
            for target_name,values in original['objectives'].items():
                compare_nested(reproduced['objectives'][target_name],values)
                max_surface_error=max(max_surface_error,float(np.max(abs(np.array(reproduced['objectives'][target_name]['sse'])-values['sse']))))
                minima.append(dict(client=job[0],speed=job[1],probe=job[2],fixed_ct=base,target=target_name,
                    front_grid_m=values['best_grid_lengths_m'][0],rear_grid_m=values['best_grid_lengths_m'][1],
                    minimum_grid_sse=values['minimum_grid_sse'],grid_boundary=values['grid_boundary']))
        print(json.dumps(dict(audited_case=run.key(job),cases=len(saved))),flush=True)
    st,pt,gt=pd.DataFrame(state),pd.DataFrame(parameter),pd.DataFrame(minima)
    for name,data in [('observability',st),('sensitivity',pt),('grid_minima',gt)]:
        pd.testing.assert_frame_equal(pd.read_csv(run.OUT/f'{run.PREFIX}_{name}.csv'),data,check_dtype=False,rtol=1e-10,atol=1e-10)
    analysis=run.read_checkpoint(run.OUT/f'{run.PREFIX}_analysis.json')
    assert analysis['checkpoint_sha256']==hashes
    compare_nested(run.negative_controls(cfg),analysis['negative_controls'])
    for name,sha in hashes.items():assert run.file_hash(ROOT/name)==sha
    rp=pd.DataFrame(replays)
    metadata=dict(status='audit_complete',cases=18,unique_state_observability_designs=9,checked_conditional_surfaces=36,
        checked_clean_and_noisy_objective_arrays=72,checked_parameter_assumptions=72,
        max_objective_sse_abs_difference=max_surface_error,max_smallest_singular_abs_difference=max_sensitivity_error,
        max_replay_metric_abs_difference=float(rp.max_metric_abs_difference.max()),max_replay_preview_abs_difference=float(rp.max_preview_abs_difference.max()),
        exact_reconstructed_trace_hash_matches=int(rp.exact_trace_hash_match.sum()),exact_reconstructed_noise_hash_matches=int(rp.exact_noise_hash_match.sum()),
        source_line_ending_matches=forms,checkpoint_sha256=hashes,review_script_sha256=run.file_hash(__file__),
        numerical_comparison_tolerance=dict(rtol=1e-8,atol=1e-9),grid_coordinate_tolerance=dict(rtol=1e-14,atol=1e-15),
        new_scientific_cases=0,refits=0,original_files_modified=False,
        note='Source bytes match saved hashes after verified LF-to-CRLF translation. Replay agreement is numerical, not asserted byte-identical recovery. Nuisance decomposition is an additional local audit, not a finite nonlinear or globally profiled fit.')
    return st,pt,gt,pd.DataFrame(extra),rp,metadata


def summarize(pt,gt,cfg):
    grouped=pt.groupby(['probe','assumption']).agg(min_smin=('smallest_singular','min'),median_smin=('smallest_singular','median'),max_smin=('smallest_singular','max'),
        min_local_delta_sse=('weak_direction_delta_sse_for_log_step_0p1','min'),max_local_delta_sse=('weak_direction_delta_sse_for_log_step_0p1','max')).reset_index()
    keys=['client','speed','probe'];s=pt.pivot(index=keys,columns='assumption',values='smallest_singular')
    s['remaining_weak_information_ratio']=(s.unknown_ct_and_initial/s.conditional)**2
    s['initial_only_smin_ratio']=s.unknown_initial/s.conditional
    shifts=gt.copy();shifts['front_error_pct']=100*(shifts.front_grid_m/cfg['relaxation_lengths_m'][0]-1)
    shifts['rear_error_pct']=100*(shifts.rear_grid_m/cfg['relaxation_lengths_m'][1]-1)
    prior=pd.DataFrame([dict(client=c['client'],**{name:float(x) for name,x in zip(['p1_error_pct','p2_error_pct','p3_error_pct','mu_error_pct'],100*(np.array(c['previous_estimated_ratios'])/c['physical_ratios']-1))}) for c in cfg['clients']])
    return grouped,s.reset_index(),shifts,prior


def render(st,pt,gt,extra,meta,grouped,ratios,shifts,prior,cfg):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({'svg.hashsalt':PREFIX,'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    def svg(fig,name):
        fig.savefig(Path(tempfile.gettempdir())/f'{name}.png',dpi=140)
        stream=io.StringIO();fig.savefig(stream,format='svg',metadata={'Date':None});plt.close(fig);s=stream.getvalue();return s[s.index('<svg'):]
    def table(d):return '<div class="scroll">'+d.to_html(index=False,border=0,float_format=lambda x:f'{x:.5g}')+'</div>'
    colors={'conditional':'#147d91','unknown_ct':'#bc6c25','unknown_ct_and_initial':'#753c8a'}
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for ax,probe in zip(axes,['slow','broadband']):
        for assumption,color in colors.items():
            for k,(name,rows) in enumerate(pt[(pt.probe==probe)&(pt.assumption==assumption)].groupby('client')):
                ax.plot(rows.speed,rows.smallest_singular,color=color,marker=['o','s','^'][k],alpha=.8,
                    label={'conditional':'Exact CT and initial state','unknown_ct':'CT parameters free','unknown_ct_and_initial':'CT and initial state free'}[assumption] if k==0 else None)
        ax.set(yscale='log',xlabel='Speed [m/s]',ylabel='Smallest whitened sensitivity singular value',title=probe.capitalize()+' steering probe');ax.grid(alpha=.2)
    axes[0].legend(fontsize=8);information=svg(fig,'tire_gate_review_information')
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for ax,probe in zip(axes,['slow','broadband']):
        data=shifts[(shifts.probe==probe)&(shifts.fixed_ct=='previous_ct')]
        ax.axvline(.4,color='#777',ls=':');ax.axhline(.7,color='#777',ls=':');ax.plot(.4,.7,'k*',ms=13,label='Generating lengths')
        for target,marker,color in [('clean','o','#147d91'),('noisy','x','#bc6c25')]:
            rows=data[data.target==target];ax.scatter(rows.front_grid_m,rows.rear_grid_m,marker=marker,color=color,s=55,label=target.capitalize()+' yaw target')
        ax.set(xlabel='Best front grid value [m]',ylabel='Best rear grid value [m]',title=probe.capitalize()+' — previous CT fixed',xlim=(.32,.58),ylim=(.5,.9));ax.grid(alpha=.2)
    axes[0].legend(fontsize=8);gridplot=svg(fig,'tire_gate_review_grid')
    nominal=pt[(pt.client=='nominal_00')&(pt.speed==21)&(pt.probe=='broadband')][['assumption','smallest_singular','weak_direction_delta_sse_for_log_step_0p1']]
    b=grouped[grouped.assumption.isin(['conditional','unknown_ct_and_initial'])][['probe','assumption','min_smin','max_smin']]
    examples=shifts[(shifts.client=='nominal_00')&(shifts.speed==29)&(shifts.target=='clean')][['probe','fixed_ct','front_grid_m','rear_grid_m','minimum_grid_sse']]
    frac=ratios.groupby('probe').remaining_weak_information_ratio.median()
    text=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Tire relaxation — scientific assessment</title>
<style>body{{margin:0;background:#edf2f5;color:#193447;font:16px/1.65 system-ui}}main{{max-width:1100px;margin:auto;padding:30px}}section{{background:white;padding:25px;border-radius:12px;margin:22px 0}}h1{{line-height:1.2;font-size:34px}}h2{{line-height:1.3}}.lead{{font-size:20px}}.decision{{border-left:6px solid #bc6c25}}.scroll{{overflow:auto}}table{{border-collapse:collapse;font-size:13px;white-space:nowrap}}td,th{{padding:9px;border-bottom:1px solid #ddd;text-align:right}}td:first-child,th:first-child{{text-align:left}}svg{{width:100%;height:auto}}a{{color:#147d91}}.note{{color:#526875;font-size:14px}}</style><main>
<h1>Tire-force states are observable;<br>joint parameter recovery remains conditional</h1>
<p class="lead">The extension is viable as a model. The main uncertainty is separating relaxation from errors in the existing CT parameters, rather than reconstructing four states from one output.</p>
<section class="decision"><h2>Decision</h2><p><b>Proceed with the augmented model, but do not claim reliable joint identification from yaw alone yet.</b> All nine distinct vehicle–speed linearizations have full state-observability rank. Broadband steering distinguishes the relaxation lengths well when CT parameters are exact. Allowing those parameters to vary substantially reduces the remaining information, even though numerical rank stays two.</p><p>The next small comparison should include a strong local augmented-model fit and independent prediction checks. A known-parameter augmented reference can establish whether the extra physics matters for control; it cannot validate the local identification method. Fleet learning is not justified by this diagnostic alone.</p></section>
<section><h2>1. State observability and operating conditions</h2><p>All 18 probe records pass their operating checks. The maximum sideslip is {st.beta_peak_deg.max():.3f}°, lateral acceleration {st.ay_peak_mps2.max():.3f} m/s², and steering rate {st.steering_rate_peak_deg_s.max():.3f}°/s. These are gentle, prescribed open-loop probes. They do not demonstrate robust control or normal-driving identification.</p><p>The nine distinct known-parameter linearizations have rank 4, with scaled finite-horizon condition numbers {st.condition.min():.1f}–{st.condition.max():.1f}. The two probes repeat each same state-observability calculation. The analytical balanced negative case has rank 2, and zero steering has zero relaxation-parameter information. Thus a single yaw output is not by itself an obstacle to a four-state model.</p><p class="note">This is local linearized observability at straight running, with known parameters and the documented state scales. It does not establish global nonlinear observability, observer accuracy with uncertain parameters, or structural parameter identifiability.</p></section>
<section><h2>2. Parameter information depends strongly on the CT assumptions</h2>{table(b)}{information}<p>Each line is one vehicle: circles indicate handling_00, squares heavy_00, and triangles nominal_00. Colors identify the parameter assumptions.</p><p>The smallest singular value measures the least visible local combination of log front/rear relaxation lengths, after whitening by yaw noise. Larger values mean a larger change in measured yaw for the same local parameter perturbation. Full rank alone says nothing about adequate precision.</p><p>After allowing CT and initial-state variations, the median ratio of smallest information eigenvalues to the exact-CT result is {100*frac['slow']:.1f}% for slow steering and {100*frac['broadband']:.1f}% for broadband steering. The weakest direction can rotate, so this ratio compares minima rather than a fixed physical direction. Initial-state uncertainty alone changes the broadband smallest singular values by less than 0.2%; uncertainty in the CT dynamics dominates.</p><p>For nominal_00 at 21 m/s with broadband steering:</p>{table(nominal)}<p>A log-length perturbation of Euclidean norm 0.1 produces a minimum locally predicted weighted squared output change of 11.25 with exact CT parameters, but only 0.53 after CT/initial-state compensation. This is an infinitesimal sensitivity calculation extrapolated to a small illustrative step, not an observed finite-parameter experiment, confidence interval, or statistical acceptance rule.</p><p>All relaxation sensitivity matrices have rank 2 at all three saved relative SVD cutoffs. The reduction in information is therefore not a numerical rank-threshold artifact. Holding μ fixed in an additional decomposition of the same derivatives does not remove the loss: for nominal_00 at 13 m/s broadband, the smallest singular value is about 2.72 with μ fixed, versus 2.72 with μ free, compared with 35.69 under exact CT parameters.</p></section>
<section><h2>3. Existing model errors can shift the preferred relaxation lengths</h2><p>The generating lengths are 0.4 m front and 0.7 m rear. With exact CT parameters and noise-free data, all 18 grid surfaces have their minimum at these values. Because the generating values were explicitly included in the grid, this is a consistency check, not proof that an optimizer recovers unknown parameters or that the continuous optimum is globally unique.</p>{gridplot}<p>For nominal_00 at 29 m/s, even clean yaw data produce the following conditional grid minima:</p>{table(examples)}<p>With the previous CT estimates fixed, the slow-probe rear length shifts to 0.548 m (−21.7%); for broadband it shifts to 0.633 m (−9.6%). This shows compensation for baseline-model error. These are coarse grid minima, so the percentages are not continuous-estimator bias estimates. All grid minima are interior, which does not exclude a broad valley.</p><p>Noise adds further variability. Across slow probes with previous CT fixed, the largest front-length grid deviation is +36.9% and the largest positive rear deviation is +20.5%. The broadband results are much closer to the generating values, but each record uses only one noise realization; no estimator variance claim follows.</p><details><summary>Existing CT parameter errors relative to the generating ratios</summary>{table(prior)}</details></section>
<section><h2>4. Implications for the next controlled comparison</h2><p><b>Keep the force-state extension.</b> The state-observability concern is not a blocker for these vehicles and speeds. <b>Use the broadband records for identification development.</b> The slow records are a useful weak-information diagnostic, not a strong basis for learning two relaxation lengths.</p><p><b>Avoid presenting a two-length fit with the old CT coefficients frozen as physical parameter recovery.</b> The saved CT errors are large enough to move the preferred lengths. The next pilot should compare this conditional fit with a joint structured local fit of the relevant CT ratios and relaxation lengths, using the available multiple-speed records and reporting the treatment of μ and initial states explicitly. Nonlinear objective profiles and held-out trajectory predictions must check whether the local sensitivity warning persists at finite parameter changes.</p><p>That identification stage can be part of the small controlled model/control comparison already planned. Keep a known-parameter augmented reference to separate the value of representing the physics from the quality of estimation. Do not add Pacejka coefficients, fleet priors, or more heterogeneity simultaneously. If the joint local model remains weakly distinguished, discuss a justified reduced parameterization or an acceleration measurement before expanding the study.</p><p class="note">No joint fit, additional-speed pooling, new trajectory, control comparison, or federation experiment was performed in this review. The proposed next stage has not been validated by these results.</p></section>
<section><h2>Audit and reproducibility</h2><p>All 18 original checkpoints, three CSV tables, negative controls, 72 parameter-assumption records, all saved SVD-cutoff variants, and 72 clean/noisy objective arrays were checked. Replayed maximum objective-SSE difference: {meta['max_objective_sse_abs_difference']:.3g}; maximum smallest-singular-value difference: {meta['max_smallest_singular_abs_difference']:.3g}; maximum operating-metric difference: {meta['max_replay_metric_abs_difference']:.3g}.</p><p>The three implementation/configuration files match their original Windows hashes exactly after LF-to-CRLF translation. No source or result hash was replaced. Full reconstructed trace hashes match {meta['exact_reconstructed_trace_hash_matches']}/18; noise hashes match {meta['exact_reconstructed_noise_hash_matches']}/18. Numerical replays are distinguished from byte-identical recovery. Original results, protocol, and strict restore command are unchanged.</p><p><a href="tire_relaxation_gate.html">Original complete diagnostic</a> · <a href="../docs/tire_relaxation_gate.html">Frozen protocol</a></p></section></main></html>'''
    REPORT.write_text('\n'.join(line.rstrip() for line in text.splitlines())+'\n',encoding='utf-8',newline='\n')


def main():
    cfg=run.preflight()
    with threadpool_limits(limits=1):st,pt,gt,extra,replays,metadata=audit(cfg)
    grouped,ratios,shifts,prior=summarize(pt,gt,cfg)
    for name,table in [('summary',grouped),('information_ratios',ratios),('grid_errors',shifts),('prior_errors',prior),('nuisance_decomposition',extra),('replays',replays)]:
        table.to_csv(run.OUT/f'{PREFIX}_{name}.csv',index=False)
    run.atomic_json(run.OUT/f'{PREFIX}_audit.json',metadata)
    render(st,pt,gt,extra,metadata,grouped,ratios,shifts,prior,cfg)
    print(json.dumps({k:v for k,v in metadata.items() if k not in ['source_line_ending_matches','checkpoint_sha256']},indent=2));print(REPORT)


if __name__=='__main__':main()
