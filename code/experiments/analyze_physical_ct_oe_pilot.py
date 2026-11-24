"""Audit and interpretation of the committed four-parameter OE pilot.

No fitting, raw-data reconstruction, controller synthesis, or new simulations.
Recompute every grid prediction/metric from saved models and check summaries.
Writes an additional self-contained HTML review; leaves frozen runner/results intact.
"""
from __future__ import annotations
import hashlib
import html
import io
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'code/src'), str(ROOT/'code/experiments')]
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import paper_physical_ct_oe as run
from federated_lpv.physical_ct_oe import discrete_matrices

PREFIX = 'paper_physical_ct_oe_pilot_review'
OUT = ROOT/'results/tables'
REPORT = ROOT/'report/physical_ct_oe_pilot_review.html'
METRICS = ['matrix','a','b','increment','pole_rms_absolute','pulse_beta','pulse_yaw']
LABELS = {'PhysicalCTLocal':'Structured local CT', 'Federated':'Federated',
          'Global':'Global', 'SmoothLocal':'Smooth Local', 'RestrictedLocal':'Restricted Local'}
BLOCKS = ['[10.0, 12.0, 14.0]', '[16.0, 18.0, 20.0]', '[22.0, 24.0, 26.0]', '[28.0, 30.0]']
COLORS = ['#246995','#d38820','#43845a','#8b5c91']


def close(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=2e-10, atol=2e-12)


def audit():
    cfg = run.preflight('pilot')
    seeds = cfg['cohorts']['pilot']
    archived = run.load(OUT/'paper_physical_ct_oe_pilot_analysis.json')
    saved_grid = pd.read_csv(OUT/'paper_physical_ct_oe_pilot_grid_errors.csv')
    rows, fit_rows, attempts, validation = [], [], [], []
    hashes = {}
    training_pairs = 0
    for seed in seeds:
        acquired, integrated = run.sources(seed)
        result = run.envelope(run.result_path(seed))
        run.validate_result(result, seed, cfg, integrated)
        if result['status'] != 'execution_complete':
            raise ValueError('Incomplete pilot')
        sha = run.file_hash(run.result_path(seed))
        if sha != archived['checkpoints'][run.result_path(seed).name]:
            raise ValueError('Analysis is stale relative to a result checkpoint')
        hashes[run.result_path(seed).relative_to(ROOT).as_posix()] = sha
        for path in run.source_paths(seed):
            hashes[path.relative_to(ROOT).as_posix()] = run.file_hash(path)
        cells = {(c['client_index'], float(c['speed'])):c for c in acquired['cells']}
        models = integrated['conditions']['ClosedOE4']['models']
        grid = np.array(integrated['grid'])
        basis = run.load(ROOT/'code/config/experiment_10a.json')
        for c in result['clients']:
            i = c['index']; fit = c['fit']; source = integrated['clients'][i]
            if not fit['converged']:
                raise ValueError('This interpretation expects the fully converged pilot; revise it if fits fail')
            if len(c['training_hashes']) != len(source['speeds']):
                raise ValueError('Missing training provenance')
            if {h['speed'] for h in c['training_hashes']} != set(source['speeds']):
                raise ValueError('Unexpected training speeds')
            for record in c['training_hashes']:
                cell = cells[i, float(record['speed'])]
                if record['y'] != cell['training_measurements_sha256'] or record['u'] != cell['training_input_sha256']:
                    raise ValueError('Different measurements or applied steering entered the fit')
                training_pairs += 1
            candidates = [a for a in fit['attempts'] if a['converged']]
            selected = next(a for a in candidates if a['start']==fit['selected_start'])
            close(fit['weighted_mse'], min(a['weighted_mse'] for a in candidates))
            close(selected['parameters'], fit['parameters'])
            for a in fit['attempts']:
                attempts.append(dict(seed=seed,client=c['client'],**a))
            theta = np.array(fit['parameters'])
            targets = []
            for v in grid:
                a,b = discrete_matrices(theta,v,cfg['estimator']['geometry_m'],
                    cfg['estimator']['curvature_per_m'],cfg['estimator']['sample_time'])
                targets.append(np.r_[a.ravel(),b])
            close(targets,c['grid_targets'])
            truth_parameters = source['parameters']
            physical = np.array([truth_parameters['front_stiffness']/truth_parameters['mass'],
                truth_parameters['rear_stiffness']/truth_parameters['mass'],
                truth_parameters['mass']/truth_parameters['yaw_inertia'], .9])
            close(c['reference_parameters'],physical)
            close(c['relative_parameter_error'],theta/physical-1)
            a0 = fit['attempts'][0]['parameters']
            spread = max(np.max(np.abs(np.array(a['parameters'])-a0)/theta) for a in candidates)
            fit_rows.append(dict(seed=seed,client=c['client'],family=source['family'],block=str(source['speeds']),
                mu=theta[3],condition=fit['information']['condition'],rank=fit['information']['rank'],
                near_bound=any(fit['near_bound']), weakest_parameter=int(np.argmax(np.abs(fit['information']['weakest_log_parameter_direction']))),
                max_start_parameter_relative_spread=spread,
                max_start_loss_spread=max(a['weighted_mse'] for a in candidates)-min(a['weighted_mse'] for a in candidates),
                elapsed_s=c['elapsed_s'],single_speed_ranks=','.join(str(z['relative_rank']) for z in fit['single_speed_information']),
                **{name+'_abs_relative_pct':100*abs(theta[j]/physical[j]-1) for j,name in enumerate(['p1','p2','p3','mu'])}))
            for v in c['validation']:
                old = cells[i,float(v['speed'])]['models']['ClosedOE4']['affine_rollout_nrmse']
                close(old,v['original_oe_affine_nrmse'])
                validation.append(dict(seed=seed,client=c['client'],**v))
            for method in cfg['comparators']+[run.METHOD]:
                if method == run.METHOD:
                    prediction = np.array(targets)
                else:
                    coeff = run.lpv.model_for(models,method,i)
                    prediction = run.raw_basis(grid,len(coeff),basis)@coeff
                metrics = run.dynamic_metrics(prediction,np.array(source['reference']),cfg)
                if method == run.METHOD:
                    for key in metrics:
                        close(metrics[key],c['errors'][key])
                else:
                    # Historical combined A/B errors are independently recomputed.
                    close(metrics['matrix'],integrated['conditions']['ClosedOE4']['errors'][method][i]['matrix'])
                for j,v in enumerate(grid):
                    rows.append(dict(seed=seed,client=c['client'],block=str(source['speeds']),method=method,
                        speed=float(v),seen=bool(v in source['speeds']),
                        **{k:metrics[k][j] for k in ['matrix','a','b','increment','pole_rms_absolute','spectral_radius']},
                        pulse_beta=metrics['pulse_relative_by_state'][j][0],pulse_yaw=metrics['pulse_relative_by_state'][j][1]))
    frame, fits, validation = pd.DataFrame(rows),pd.DataFrame(fit_rows),pd.DataFrame(validation)
    index = ['seed','client','method','speed']
    generated = frame.set_index(index).sort_index()
    archived_grid = saved_grid.set_index(index).sort_index()
    if not generated.index.equals(archived_grid.index):
        raise ValueError('Archived grid has missing or extra rows')
    close(generated[METRICS+['spectral_radius']],archived_grid[METRICS+['spectral_radius']])
    units = frame.loc[~frame.seen].groupby(['seed','client','method'])[METRICS].agg(run.strict_mean).reset_index()
    fleets = units.groupby(['seed','method'])[METRICS].agg(run.strict_mean).reset_index()
    summary = fleets.groupby('method')[METRICS].agg(run.strict_mean)
    original = pd.read_csv(OUT/'paper_physical_ct_oe_pilot_summary.csv').set_index('method')
    close(summary.sort_index(),original[METRICS].sort_index())
    matrix = units.pivot(index=['seed','client'],columns='method',values='matrix')
    paired = matrix.reset_index().merge(fits,on=['seed','client'],validate='one_to_one')
    paired['CT_minus_Federated_pp'] = 100*(paired[run.METHOD]-paired.Federated)
    exceptions = paired.loc[paired.CT_minus_Federated_pp >= 0]
    block = paired.groupby('block')[cfg['comparators']+[run.METHOD]].mean().reindex(BLOCKS)*100
    block.insert(0,'vehicles',paired.groupby('block').size())
    stability = frame.assign(unstable=frame.spectral_radius>=1).groupby('method').agg(
        unstable_grid_pairs=('unstable','sum'),evaluated_grid_pairs=('unstable','size'),maximum_radius=('spectral_radius','max'))
    parameters = fits.groupby('block').agg(vehicles=('client','size'),mu_min=('mu','min'),mu_median=('mu','median'),
        mu_max=('mu','max'),condition_median=('condition','median'),condition_max=('condition','max'),bound_hits=('near_bound','sum')).reindex(BLOCKS)
    raw_validation_mean = validation[['physical_ct_affine_nrmse','original_oe_affine_nrmse']].mean()
    equal_client_validation = validation.groupby(['seed','client'])[['physical_ct_affine_nrmse','original_oe_affine_nrmse']].mean().groupby('seed').mean().mean()
    wins = {}
    for method in cfg['comparators']:
        wins[method] = int((matrix[run.METHOD]<matrix[method]).sum())
    summary_values = summary.loc[run.METHOD]
    audit = dict(status='audited',seeds=seeds,clients=len(fits),training_pairs_verified=training_pairs,
        prediction_grid_pairs_verified=len(frame),attempts=len(attempts),converged_attempts=sum(a['converged'] for a in attempts),
        checkpoint_sha256=hashes,analysis_script_sha256=run.file_hash(__file__),
        all_joint_ranks_four=bool((fits['rank']==4).all()),all_single_speed_ranks_three=bool(all(set(s.split(','))=={'3'} for s in fits.single_speed_ranks)),
        weakest_direction_mu_clients=int((fits.weakest_parameter==3).sum()),bound_hits=int(fits.near_bound.sum()),
        maximum_start_loss_spread=float(fits.max_start_loss_spread.max()),
        maximum_start_parameter_relative_spread=float(fits.max_start_parameter_relative_spread.max()),
        mean_unseen_matrix_pct={m:float(100*summary.loc[m,'matrix']) for m in summary.index},
        relative_reduction_vs_federated=float(1-summary_values['matrix']/summary.loc['Federated','matrix']),
        client_wins=wins,ct_client_error_quantiles_pct={str(q):float(v) for q,v in (100*matrix[run.METHOD]).quantile([.5,.9,.95,1.]).items()},
        validation_cell_wins=int((validation.physical_ct_affine_nrmse<validation.original_oe_affine_nrmse).sum()),
        validation_cells=len(validation),validation_cell_means=raw_validation_mean.to_dict(),
        validation_equal_client_fleet_means=equal_client_validation.to_dict(),
        control_results_present=any(run.result_path(s,True).exists() for s in seeds),
        limitations=['Three previously inspected pilot fleets; no fresh independent confirmation.',
            'Known correct tanh-tire structure, common known geometry, prescribed curvature and noisy full-state measurements.',
            'Saved validation scores checked for provenance and baseline consistency; raw validation trajectories unavailable here, so not replayed.',
            'Sensitivity is local and descriptive; full numerical rank does not imply precise friction recovery.',
            'No new closed-loop evaluation or yaw-only identification in this audit.'])
    for path in [OUT/'paper_physical_ct_oe_pilot_summary.csv',OUT/'paper_physical_ct_oe_pilot_grid_errors.csv']:
        audit['checkpoint_sha256'][path.relative_to(ROOT).as_posix()] = run.file_hash(path)
    run.acquisition.parent.save_json(OUT/f'{PREFIX}.json',audit)
    paired.to_csv(OUT/f'{PREFIX}_clients.csv',index=False)
    block.to_csv(OUT/f'{PREFIX}_coverage.csv')
    stability.to_csv(OUT/f'{PREFIX}_stability.csv')
    write_report(audit,summary,fleets,paired,exceptions,block,parameters,stability,fits,validation)
    print(json.dumps(audit,indent=2))
    print(REPORT)


def write_report(audit,summary,fleets,paired,exceptions,block,parameters,stability,fits,validation):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.hashsalt':'physical-ct-pilot-review-v1'})
    def svg(fig):
        stream=io.StringIO();fig.savefig(stream,format='svg',bbox_inches='tight',metadata={'Date':None});plt.close(fig)
        raw=stream.getvalue();return raw[raw.index('<svg'):]
    fig,axes=plt.subplots(1,2,figsize=(11.4,4.0),layout='constrained')
    display=['SmoothLocal','Global','Federated',run.METHOD]
    values=100*summary.loc[display,'matrix']
    axes[0].bar(range(4),values,color=['#9aa6af','#cf8e33','#506aa1','#287f68'],width=.65)
    for j,method in enumerate(display):
        fleet=100*fleets.loc[fleets.method==method,'matrix'].to_numpy()
        axes[0].plot(j+np.linspace(-.12,.12,3),fleet,'o',color='#182b38',ms=4)
        axes[0].text(j,max(values.iloc[j],fleet.max())+.12,f'{values.iloc[j]:.3f}%',ha='center',fontsize=9)
    axes[0].set(xticks=range(4),xticklabels=['Smooth\nLocal','Global','Federated','Structured\nlocal CT'],ylabel='Mean unobserved-speed matrix error [%]',ylim=(0,5.5))
    axes[0].set_title('Fleet means: every pilot improves')
    for name,color in zip(BLOCKS,COLORS):
        z=paired[paired.block==name]
        axes[1].scatter(100*z.Federated,100*z[run.METHOD],c=color,label=name,alpha=.8,s=24)
    axes[1].plot([.02,10],[.02,10],color='#536271',ls='--',lw=1)
    axes[1].set(xscale='log',yscale='log',xlim=(.15,10),ylim=(.025,10),xlabel='Federated error [%]',ylabel='Structured local CT error [%]')
    axes[1].set_title('87 of 90 vehicles below equal-error line')
    axes[1].legend(title='Observed speeds [m/s]',fontsize=7,title_fontsize=8,loc='upper left')
    figure1=svg(fig)
    fig,axes=plt.subplots(1,2,figsize=(11.4,4.0),layout='constrained')
    for j,(name,color) in enumerate(zip(BLOCKS,COLORS)):
        z=fits[fits.block==name].sort_values(['seed','client'])
        offset=np.linspace(-.22,.22,len(z))
        axes[0].scatter(j+offset,z.mu,c=color,s=24,alpha=.8)
    axes[0].axhline(.9,color='#1c3441',ls='--',lw=1,label='Simulator truth: 0.9')
    axes[0].axhline(1.5,color='#a14536',ls=':',lw=1,label='Upper bound: 1.5')
    axes[0].set(xticks=range(4),xticklabels=['10–14','16–20','22–26','28–30'],xlabel='Observed speed block [m/s]',ylabel='Estimated friction μ',ylim=(.6,1.58))
    axes[0].set_title('Friction is least informative at low speeds');axes[0].legend(fontsize=8)
    # Match points explicitly by vehicle, not incidental DataFrame row ordering.
    for name,color in zip(BLOCKS,COLORS):
        z=paired[paired.block==name]
        axes[1].scatter(z.condition,100*z[run.METHOD],c=color,s=24,alpha=.8)
    axes[1].set(xscale='log',yscale='log',xlabel='Projected log-parameter sensitivity condition',ylabel='Structured local CT error [%]')
    axes[1].set_title('Full numerical rank still permits weak recovery')
    figure2=svg(fig)
    def table(frame,index=False):
        return '<div class="scroll">'+frame.to_html(index=index,border=0,float_format=lambda x:f'{x:.4g}',na_rep='unavailable')+'</div>'
    comparison=summary.loc[['RestrictedLocal','SmoothLocal','Global','Federated',run.METHOD],['matrix','a','b','increment']]*100
    comparison=comparison.rename(index=LABELS,columns={'matrix':'Combined [%]','a':'A [%]','b':'B [%]','increment':'[A−I,B] [%]'})
    fleet_table=fleets.pivot(index='seed',columns='method',values='matrix')[['Global','Federated',run.METHOD]]*100
    exceptions=exceptions[['seed','client','block','mu','condition','near_bound','Federated',run.METHOD,'CT_minus_Federated_pp']].copy()
    exceptions[['Federated',run.METHOD]]*=100
    exceptions=exceptions.rename(columns={'Federated':'Federated [%]',run.METHOD:'Structured local CT [%]'})
    response=summary.loc[['Global','Federated',run.METHOD],['pulse_beta','pulse_yaw','pole_rms_absolute']].copy()
    response[['pulse_beta','pulse_yaw']]*=100
    response=response.rename(index=LABELS,columns={'pulse_beta':'Sideslip pulse error [%]','pulse_yaw':'Yaw pulse error [%]','pole_rms_absolute':'Discrete pole RMS'})
    parameter_errors=fits[[x+'_abs_relative_pct' for x in ['p1','p2','p3','mu']]].quantile([.5,.95,1.])
    parameter_errors.index=['Median','95th percentile','Maximum']
    parameter_errors.columns=['Cf/m error [%]','Cr/m error [%]','m/Iz error [%]','μ error [%]']
    reduction=100*audit['relative_reduction_vs_federated']
    body=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Physical CT OE pilot — scientific assessment</title><style>body{{font:16px/1.6 system-ui;background:#eef3f7;color:#173044;margin:0}}main{{max-width:1200px;margin:auto;padding:32px}}h1,h2{{line-height:1.2}}section{{background:white;padding:25px;border-radius:10px;margin:20px 0}}.lead{{border-left:5px solid #287f68}}.note{{border-left:5px solid #d38820}}.scroll{{overflow:auto}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{text-align:right;padding:9px;border-bottom:1px solid #dae2e7}}svg{{width:100%;height:auto}}.cards{{display:flex;flex-wrap:wrap;gap:15px}}.card{{flex:1;min-width:170px;padding:18px;background:#eaf3f1;border-radius:8px}}strong.big{{font-size:28px;display:block}}a{{color:#15678f}}</style><main>
<h1>Analytical structure changes the conclusion</h1><p>Audit of pilot fleets 1301–1303 • 90 vehicles • 249 original vehicle–speed records • no new fitting or control runs</p>
<section class="lead"><h2>Main finding</h2><p>The four-parameter local model reduces mean unobserved-speed matrix error to <strong>0.386%</strong>, versus <strong>2.266% for Federated</strong> and <strong>4.500% for Global</strong>. This supports Markus’s concern that much of the apparent need for sharing in this vehicle benchmark comes from the flexible scheduling representation. It does not establish that federation is generally unnecessary.</p>
<div class="cards"><div class="card"><strong class="big">{reduction:.1f}%</strong>lower mean matrix error than Federated</div><div class="card"><strong class="big">87 / 90</strong>vehicles improved over Federated</div><div class="card"><strong class="big">90 / 90</strong>joint fits converged</div><div class="card"><strong class="big">1 / 90</strong>fit at a parameter bound</div></div></section>
<section><h2>1. Evidence and comparison</h2>{table(comparison,True)}{figure1}{table(fleet_table,True)}
<p>Metrics average unobserved speeds within each vehicle, vehicles within a fleet, then the three fleet means. Dots in the left plot are individual fleet means, not confidence intervals. Restricted Local has 20.87% combined error and is included in the table; it is omitted from the bar plot to keep the four smaller values readable.</p>
<p>Structured local CT also beats Smooth Local and Global on all 90 vehicles. Its median vehicle error is 0.246%, 95th percentile 1.228%, and maximum 2.103%. These are descriptive results on previously inspected pilot data; no new independent confirmation or population-level significance is claimed.</p></section>
<section><h2>2. Why this matters for the paper</h2><p>The earlier rank-deficiency statement is valid for the seven-function, 42-coefficient representation. It does not show that a vehicle lacks enough information to identify an appropriately structured model. The current fit links all matrix entries and speeds through Cf/m, Cr/m, m/Iz and μ, exploiting known geometry and the correct tire-law structure.</p>
<p>The comparison changes both parameterization and joint across-speed estimation. It is not a clean attribution to fewer parameters alone. Nonetheless, it is the relevant practical baseline: the vehicle-focused paper must explain why such available structure would be insufficient before presenting federation as necessary for speed coverage.</p>
<p>The identification solver receives only noisy state measurements, known applied steering, observed speeds and fixed model settings. It uses neither recipient truth, family labels, validation trajectories nor the old fitted matrix targets. True parameters are used afterward to assess parameter recovery.</p></section>
<section class="note"><h2>3. The remaining limitation is friction information at low speeds</h2>{table(block,True)}{figure2}{table(parameters,True)}
<p>All 249 single-speed projected sensitivity matrices have numerical rank three under the configured relative threshold; all 90 joint matrices have rank four. In all 90 joint fits the largest component of the weakest log-parameter direction is μ. Condition numbers range from {fits.condition.min():.1f} to {fits.condition.max():.1f}; these are local sensitivity diagnostics, not confidence intervals.</p>
<p>The three vehicles where Federated performs better all observe only 10, 12 and 14 m/s:</p>{table(exceptions)}
<p>The strongest exception, seed 1302 / handling_04, estimates μ=1.5 at the upper bound although the simulator uses μ=0.9. Its other parameter errors remain below 0.5%. Do not call this reliable physical friction recovery simply because the optimizer converged.</p>
<p>Analytically, s(v)=1−κ²v⁴/(μ²g²), so ∂s/∂log μ=2κ²v⁴/(μ²g²). At fixed curvature this part of friction sensitivity grows as v⁴. This explains why low-speed observations can be structurally sufficient across speeds yet weakly informative about friction in practice. It is a possible motivation for future sharing, not evidence that sharing is already needed on average here.</p>{table(parameter_errors,True)}</section>
<section><h2>4. The gain survives more informative diagnostics</h2>{table(response,True)}
<p>The identity-free metric normalizes by [A−I,B]; its mean is 1.555% for structured local CT versus 8.399% for Federated. Thus the finding is not confined to the historical denominator containing the identity. This metric remains coordinate-dependent.</p>
<p>The pulse test compares local linear model responses to 0.3° steering applied for 0.2 s over a 2.01-s horizon. It evaluates beta and yaw rate separately. It is not a new nonlinear maneuver. Very large pulse errors for Restricted Local and Smooth Local are driven by unstable model predictions and should not be described as realistic vehicle responses.</p>{table(stability,True)}
<p>Federated’s seven unstable grid pairs occur at 30 m/s in one learned group in pilot fleet 1302. They help explain its large mean sideslip pulse error. This is a pilot finding; it does not contradict the manuscript’s separate ten-fleet evaluation result. Frozen-speed eigenvalues below one also do not prove scheduled closed-loop stability.</p></section>
<section><h2>5. Validation and optimization checks</h2><p>Saved independent chirp-validation scores favor structured local CT in <strong>{audit['validation_cell_wins']} / {audit['validation_cells']}</strong> observed-speed cases. With equal-vehicle/equal-fleet averaging, normalized affine rollout error is <strong>{audit['validation_equal_client_fleet_means']['physical_ct_affine_nrmse']:.6f}</strong> versus <strong>{audit['validation_equal_client_fleet_means']['original_oe_affine_nrmse']:.6f}</strong> for the original local affine OE fits. This is an observed-speed comparison, not evidence about unseen nonlinear trajectories.</p>
<p>All {audit['converged_attempts']} of {audit['attempts']} starts converged. The maximum within-vehicle spread of training losses is {audit['maximum_start_loss_spread']:.3g}; the maximum relative parameter spread is {audit['maximum_start_parameter_relative_spread']:.3g}. The result is not explained by one unusually favorable start.</p>
<p>Audit checks: original source and checkpoint hashes; all {audit['training_pairs_verified']} training measurement/input hash pairs; training-loss selection; all saved structured grid matrices; all {audit['prediction_grid_pairs_verified']} method–vehicle–speed metric rows; historical baseline errors; aggregate CSV reproduction. Validation scores were checked against saved records and original baseline values, but could not be replayed here because the original byte-identical raw trajectories are not available.</p></section>
<section><h2>6. Recommended decision before further manuscript work</h2><ol><li>Use the structured model as the new local baseline. Pause claims that incomplete local speed coverage alone establishes a need for federation in this plant.</li><li>Run the already prepared pilot closed-loop comparison next. Keep weights, maneuvers and all bounds unchanged; better identification does not automatically imply better tracking.</li><li>If the control result is consistent, apply the unchanged diagnostic to the ten existing evaluation fleets. These data have already been inspected and should be described as an extension, not fresh confirmation.</li><li>Discuss the paper’s scope with Markus. A future federated structured model would need a demonstrated practical local limitation, such as weak excitation or unavailable sideslip. Neither changing heterogeneity to favor grouping nor widening the scope without new evidence would address the current criticism.</li></ol>
<p>The report does not authorize or execute these next experiments. The manuscript is unchanged. No yaw-only identification or new state observer has been tested.</p></section>
<p><a href="physical_ct_oe_pilot.html">Open the original detailed pilot report</a> · <a href="../docs/physical_ct_oe.html">Experiment protocol and commands</a></p></main></html>'''
    REPORT.write_text('\n'.join(line.rstrip() for line in body.splitlines())+'\n',encoding='utf-8',newline='\n')


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        audit()
