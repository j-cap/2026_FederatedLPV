"""Audit frozen OE-to-LPV results and add reviewed interpretation to the HTML.

No acquisition, local OE optimization, clustering rerun, or parameter selection.
"""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_oe_lpv_integration as run


def main():
    cfg, basis, mixture, acq, lock = run.preflight()
    analysis_path = run.OUT / f'{run.PREFIX}_analysis.json'
    source = run.parent.load(analysis_path)
    if source['provenance'] != lock:
        raise ValueError('Analysis provenance mismatch')
    results, families, unstable, groups = [], [], [], []
    errors_checked = sensitivity_checked = 0
    for seed in cfg['seeds']:
        path = run.path_for(seed)
        assert run.parent.tracked_unchanged(path)
        assert run.parent.file_hash(path) == source['checkpoint_sha256'][path.relative_to(run.ROOT).as_posix()]
        r = run.read_result(path, seed, lock)
        original = run.acquisition.read_result(run.acquisition.path_for(seed), lock['acquisition_provenance'])
        pilot = run.acquisition.oe.known.source_result(seed, 'gentle', lock['acquisition_provenance']['oe_provenance']['known_provenance'])
        assert r['clients'] == pilot['clients'] and r['grid'] == basis['speed_grid']
        for condition, f in r['conditions'].items():
            speeds, targets, diagnostic = run.assemble_inputs(original, pilot, condition, acq)
            assert diagnostic == f['local_identification']
            for actual, frozen in zip(targets, r['inputs'][condition]):
                np.testing.assert_array_equal(actual, frozen)
            recomputed, sensitivity = run.evaluate(f, r['clients'], speeds, basis)
            for method in run.METHODS:
                for i, (actual, stored) in enumerate(zip(recomputed[method], f['errors'][method])):
                    for metric in actual:
                        np.testing.assert_allclose(actual[metric], stored[metric], atol=1e-11, rtol=1e-8)
                    errors_checked += len(r['grid'])
                    for j, rho in enumerate(actual['spectral_radius']):
                        if rho >= 1:
                            truth = np.asarray(r['clients'][i]['reference'][j])
                            unstable.append(dict(seed=seed, condition=condition, method=method,
                                client=r['clients'][i]['client'], speed=r['grid'][j], rho=rho,
                                true_rho=float(max(abs(np.linalg.eigvals(truth[:4].reshape(2, 2))))),
                                group=f['models']['assignments'][i] if method == 'Federated' else None))
            for strength, actual in sensitivity.items():
                np.testing.assert_allclose(actual, f['sensitivity_errors'][strength], atol=1e-11, rtol=1e-8)
                sensitivity_checked += np.asarray(actual).size
            # An independent pooled solve verifies saved models without rerunning assignments.
            pooled = run.central_ridge(np.concatenate(speeds), np.vstack(targets), basis, mixture['ridge_strength'])
            np.testing.assert_allclose(pooled, f['models']['Global'], atol=1e-8, rtol=0)
            selected = min(f['candidates'], key=lambda row: row['bic'])
            assert selected['k'] == f['selected_k']
            assert sorted(i for g in f['groups'] for i in g['members']) == list(range(len(r['clients'])))
            for g in f['groups']:
                members = np.flatnonzero(np.asarray(f['models']['assignments']) == g['group'])
                assert members.tolist() == g['members']
                observed = np.concatenate([speeds[i] for i in members])
                info = run.information(observed, basis)
                for key in ['rank', 'full_rank', 'observations', 'unique_speeds']:
                    assert info[key] == g[key]
                if info['condition'] is not None:
                    np.testing.assert_allclose(info['condition'], g['condition'], rtol=1e-8)
                central = run.central_ridge(observed, np.vstack([targets[i] for i in members]), basis, mixture['ridge_strength'])
                for name in ['Federated', 'CentralSameGroups']:
                    np.testing.assert_allclose(central, f['models'][name][g['group']], atol=1e-8, rtol=0)
                groups.append(dict(seed=seed, target_condition=condition, **g))
            for family in sorted({c['family'] for c in r['clients']}):
                members = [i for i, c in enumerate(r['clients']) if c['family'] == family]
                info = run.information(np.concatenate([speeds[i] for i in members]), basis)
                families.append(dict(seed=seed, target_condition=condition, family=family, **info))
        results.append(r)
    units, sensitivity, speed_summary, comparisons = run.summarize(results, cfg)
    for name, frame in [('fleet_summary', units), ('smooth_sensitivity', sensitivity), ('speed_summary', speed_summary), ('comparisons', pd.DataFrame(comparisons))]:
        saved = pd.read_csv(run.OUT / f'{run.PREFIX}_{name}.csv')
        pd.testing.assert_frame_equal(saved, frame, check_exact=False, atol=1e-12, rtol=1e-10)
    assert len(comparisons) == len(source['comparisons'])
    for actual, stored in zip(comparisons, source['comparisons']):
        for key in actual:
            if isinstance(actual[key], float):
                np.testing.assert_allclose(actual[key], stored[key], atol=1e-10)
            else:
                assert actual[key] == stored[key]
    primary = units[units.condition == 'ClosedOE4']
    primary_unseen = primary[primary.region == 'unseen']
    candidates = sensitivity[(sensitivity.condition == 'ClosedOE4') & (sensitivity.region == 'unseen')]
    candidate_means = candidates.groupby('strength').matrix.mean()
    best = float(candidate_means.idxmin())
    baseline = candidates[candidates.strength == best].set_index('seed').matrix
    fed = primary_unseen[primary_unseen.method == 'Federated'].set_index('seed').matrix
    best_gains = 100*(1-fed/baseline)
    for row in candidates.itertuples():
        assert fed.loc[row.seed] < row.matrix
    client_wins = {b: 0 for b in ['RestrictedLocal', 'SmoothLocal', 'Global']}
    for r in results:
        errors = r['conditions']['ClosedOE4']['errors']
        averages = {m: np.asarray([np.mean(np.asarray(e['matrix'])[run.region_mask(r['grid'], c['speeds'], 'unseen')])
                      for e, c in zip(errors[m], r['clients'])]) for m in [*client_wins, 'Federated']}
        for baseline in client_wins:
            client_wins[baseline] += int(np.sum(averages['Federated'] < averages[baseline]))
    primary_groups = [g for g in groups if g['target_condition'] == 'ClosedOE4']
    affected = [v for v in unstable if v['condition'] == 'ClosedOE4' and v['method'] == 'Federated']
    counts = {m: sum(v['condition'] == 'ClosedOE4' and v['method'] == m for v in unstable) for m in run.PRIMARY}
    summary = {m: {region: float(100*primary[(primary.method == m) & (primary.region == region)].matrix.mean())
                  for region in run.REGIONS} for m in run.PRIMARY}
    audit = dict(status='audited_development_result', provenance=lock,
        analysis_sha256=run.parent.file_hash(analysis_path), checkpoint_sha256=source['checkpoint_sha256'],
        generator_sha256=run.parent.file_hash(__file__), matrix_errors_recomputed=errors_checked,
        sensitivity_matrix_errors_recomputed=sensitivity_checked, primary_matrix_error_pct=summary,
        primary_unseen_client_wins=client_wins, clients=90, grid_evaluations=990,
        best_smooth_strength_diagnostic=best, best_smooth_unseen_error_pct=100*float(candidate_means.loc[best]),
        gain_over_best_smooth_pct=float(best_gains.mean()), gain_over_best_smooth_fleet_wins=int((best_gains > 0).sum()),
        primary_unstable_client_speed_counts=counts, unstable_cases=unstable, family_coverage=families,
        primary_groups_full_rank=sum(g['full_rank'] for g in primary_groups), primary_groups=len(primary_groups),
        primary_groups_cover_all_grid_speeds=sum(set(g['unique_speeds']) == set(basis['speed_grid']) for g in primary_groups),
        maximum_saved_equivalence_coefficient_error=max(f['equivalence']['coefficient_max_abs'] for r in results for f in r['conditions'].values()),
        original_acquisition_gate_pass=False, confirmation_unlocked=False, control_benefit_tested=False,
        conclusion='Positive development evidence for complementary sharing under OE. Gains persist against the full prespecified smoothness sweep. One shared high-speed group matrix is unstable; full rank alone does not guarantee physically consistent extrapolation.',
        next_step='Freeze this identification pipeline for independent confirmation, reporting the boundary-model diagnostic prospectively. Do not retune acquisition, smoothness, groups or stability from confirmation outcomes. Controller evaluation remains separate.')
    assert errors_checked == 13860 and sensitivity_checked == 13860
    assert all(f['equivalence']['passed'] for r in results for f in r['conditions'].values())
    run.parent.save_json(run.OUT / f'{run.PREFIX}_diagnosis.json', audit)
    # Recreate the generic report using saved results, never modifying frozen sources/tables.
    run.write_report(results, cfg, units, sensitivity, speed_summary, comparisons, source, run.REPORT)
    table = run.acquisition.table
    rows = [[m, summary[m]['seen'], summary[m]['unseen'], summary[m]['full']] for m in run.PRIMARY]
    def paired_plot(ax):
        view = primary_unseen.pivot(index='seed', columns='method', values='matrix')
        for seed, values in view.iterrows():
            ax.plot(range(4), [100*values[m] for m in run.PRIMARY], marker='o', label=f'Fleet {seed}')
        ax.set(xticks=range(4), xticklabels=run.PRIMARY, ylabel='Unobserved-speed matrix error [%]')
        ax.legend(); ax.grid(alpha=.2)
    def coverage_plot(ax):
        coverage = np.array([[v in g['unique_speeds'] for v in basis['speed_grid']] for g in primary_groups])
        from matplotlib.colors import ListedColormap
        ax.imshow(coverage, cmap=ListedColormap(['#f4d2aa', '#277e9c']), aspect='auto', vmin=0, vmax=1)
        ax.set(xticks=range(11), xticklabels=basis['speed_grid'], xlabel='Speed [m/s]',
               yticks=range(9), yticklabels=[f"{g['seed']} / group {g['group']}" for g in primary_groups])
        ax.set_title('Blue: observed somewhere in group; orange: absent from group')
    def radius_plot(ax):
        r = next(r for r in results if r['seed'] == 1302)
        f = r['conditions']['ClosedOE4']
        group = next(g for g in f['groups'] if g['group'] == affected[0]['group'])
        matrix = run.predict(np.asarray(f['models']['Federated'][group['group']]), r['grid'], basis)
        rho = np.max(abs(np.linalg.eigvals(matrix[:, :4].reshape(-1, 2, 2))), axis=1)
        truth = np.array([np.max(abs(np.linalg.eigvals(np.asarray(r['clients'][i]['reference'])[:, :4].reshape(-1, 2, 2))), axis=1) for i in group['members']])
        ax.fill_between(r['grid'], truth.min(axis=0), truth.max(axis=0), alpha=.3, color='#277e9c', label='Reference range: seven clients')
        ax.plot(r['grid'], truth.mean(axis=0), color='#277e9c', linewidth=2, label='Reference mean')
        ax.plot(r['grid'], rho, 'o-', color='#ad3d36', label='Shared learned group model')
        ax.axhline(1, color='black', linestyle='--', label='Discrete-time stability boundary')
        ax.axvspan(26, 30, color='#f4d2aa', alpha=.25, label='Beyond group observations')
        ax.set(xlabel='Speed [m/s]', ylabel='Spectral radius of A')
        ax.legend(loc='upper left', fontsize=8); ax.grid(alpha=.2)
    reviewed = '''<section style="border-left:5px solid #25724d"><h2>Reviewed result: the OE-based method shows a sharing benefit</h2><p><strong>Federated groups reduce unobserved-speed matrix error in all three development fleets against both local baselines and global pooling.</strong> Mean errors are 2.27% for Federated, 4.17% for SmoothLocal, 4.50% for Global and 20.87% for RestrictedLocal. This is the intended incomplete-coverage effect under the new acquisition and local estimator.</p>'''
    reviewed += table(['Method', 'Seen [%]', 'Unseen [%]', 'Full envelope [%]'], rows) + run.svg_figure(paired_plot)
    reviewed += '''<p>The mean of paired fleet gains is 46.0% against SmoothLocal and 49.8% against Global. These are means of three relative improvements, not ratios of the aggregate errors. At the client level, Federated wins on unseen-speed average error for 69/90 clients against SmoothLocal, 72/90 against Global and 87/90 against RestrictedLocal. Benefits are not universal.</p><p>Sharing trades local accuracy for wider coverage: observed-speed error increases from 0.36% (SmoothLocal) to 1.07% (Federated). The scientific claim should be better full-envelope identification with incomplete local coverage, not uniform superiority at every speed.</p></section>
<section><h2>The result is not explained by a weak smoothness setting</h2><p>Federated beats every one of the seven prespecified SmoothLocal strengths on unseen-speed mean error in each fleet. The best development setting, 0.01, gives 4.07%, versus 4.17% for the inherited 10⁶ setting. Against that best setting, the mean paired federated gain remains ''' + f"{best_gains.mean():.1f}%" + '''. This is a post-result sensitivity diagnosis, not a newly selected primary baseline.</p><p>The exact-state diagnostic also shows a benefit: unseen error is 2.11% for Federated, 3.53% for inherited SmoothLocal and 4.49% for Global. Therefore the sharing result does not depend entirely on noisy local estimation. Clean-state and OE grouping assignments can differ; their difference is not a noise-only causal estimate.</p></section>
<section><h2>Group coverage and the boundary-model limitation</h2><p>Each fleet selects three learned groups. All nine group designs have rank seven. Eight groups cover every grid speed; the remaining group in fleet 1302 covers 10–26 m/s and extrapolates to 28 and 30 m/s. Family ARI is 1.000, 0.731 and 0.731: the groups are related to the planted families but do not simply reproduce them in every fleet.</p>'''
    reviewed += run.svg_figure(coverage_plot) + run.svg_figure(radius_plot)
    reviewed += '''<p><strong>One shared group model has spectral radius 1.0163 at 30 m/s.</strong> It is used by seven clients, giving seven unstable client–speed predictions out of 990 grid evaluations (all seven are locally unobserved). Their reference models are stable, with spectral radii about 0.971. This is an identified-model extrapolation limitation; no closed-loop instability has been tested. SmoothLocal has 83 unstable client–speed predictions, RestrictedLocal 49 and Global zero. A smaller count than a baseline does not establish stability guarantees.</p><p>Full design rank does not guarantee reliable extrapolation. The family-label diagnostic makes this particularly clear: fleet 1303's nominal family has no data at 10–14 m/s, and its rank-seven design has condition number about 3399. Its unseen error is 21.81% with OE and 31.67% with clean-state targets. Known family labels alone do not ensure sufficient useful coverage. The learned groups in that fleet include all grid speeds and avoid this large failure, though they mix some family members.</p></section>
<section><h2>Decision and next step</h2><p><strong>Proceed to independent confirmation of the complete identification pipeline.</strong> Keep the fixed nominal acquisition, four-start OE, scheduling basis, mixture settings and inherited smooth-local baseline frozen. Filtering and further local-estimator tuning are not prerequisites supported by this result. Record unseen-speed accuracy, per-client outcomes, group coverage and boundary spectral radii prospectively on the untouched fleets 1311–1320.</p><p>Do not alter the learned matrices or hide the seven unstable predictions in this pilot. A controller comparison requires its own matched design and evaluation; low matrix error alone does not establish controller feasibility or performance. The three reused development fleets support this next step, not a final confirmatory paper claim.</p></section>
<section><h2>Additional audit</h2><p>Checkpoint hashes and provenance were verified. All 13,860 saved method/client/speed matrix errors, their A/B and entrywise diagnostics and spectral radii, plus 13,860 smoothness-sensitivity matrix errors, were recomputed from saved coefficients and references. Every scheduling input matches its frozen acquisition target. Fleet summaries and paired gains reproduce the committed CSVs. Independent pooled solves confirm the saved fixed-partition group models; the largest saved coefficient discrepancy is 2.27×10⁻¹³.</p><p>No acquisition, OE fitting or clustering search was rerun. Affine offsets and initial states remain local OE quantities. The original acquisition gate failure and the prospective permission for recorded actuator limiting remain visible.</p><p>Rebuild this reviewed report with <code>python code/experiments/analyze_oe_lpv_integration.py</code>. The original runner's <code>analyze</code> command rebuilds the generic report. <a href="../results/tables/paper_oe_lpv_integration_diagnosis.json">Detailed numerical audit</a>.</p></section>'''
    marker='<section><h2>Question and scope</h2>'
    text = run.REPORT.read_text(encoding='utf-8')
    assert text.count(marker) == 1
    text = text.replace(marker, reviewed+marker)
    text = '\n'.join(line.rstrip() for line in text.splitlines())
    run.REPORT.write_text(text, encoding='utf-8', newline='\n')
    print(json.dumps(dict(status='audit_passed', matrix_errors=errors_checked,
        sensitivity_errors=sensitivity_checked, unseen_error_pct=summary['Federated']['unseen'],
        gain_over_best_smooth_pct=float(best_gains.mean()), unstable_federated_cases=len(affected)), indent=2))
    print(run.REPORT)


if __name__ == '__main__':
    main()
