"""Audit frozen CFL-family results and write an interpretation without fitting.

From the repository root:
    python code/experiments/analyze_cfl_heterogeneity_review.py
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_cfl_heterogeneity as experiment
import numpy as np
import pandas as pd


def main():
    e = experiment
    cfg = e.preflight()
    provenance = e.identity()
    manifest_path = e.OUT / f'{e.PREFIX}_analysis.json'
    manifest = e.storage.read_checkpoint(manifest_path)
    locked = {**manifest['source_sha256'], **manifest['output_sha256'],
              e.CONFIG.relative_to(e.ROOT).as_posix(): manifest['protocol_sha256']}
    for name, sha in locked.items():
        assert e.base.canonical_hash(e.ROOT / name) == sha, f'Frozen file changed: {name}'
    original_hashes = dict(locked)
    rows, parameter_rows, optimization = [], [], []
    bound_counts = {m: np.zeros(6, dtype=int) for m in e.METHODS}
    expected_paths = set()
    implementation_commits, numerical_runtimes = set(), set()
    training_checks = 0
    bounds = np.log(np.array(cfg['fit']['bounds']))

    def checkpoint(path):
        expected_paths.add(path)
        p = e.load(path, provenance, strict_runtime=False)
        assert p['status'] == 'complete', f'Incomplete checkpoint: {path}'
        implementation_commits.add(p['implementation_commit'])
        numerical_runtimes.add(json.dumps(p['provenance']['runtime'], sort_keys=True))
        for name, sha in p.get('parents', {}).items():
            assert e.base.canonical_hash(e.OUT / name) == sha, f'Stale parent: {name}'
        return p

    for fleet in cfg['fleets']:
        seed = fleet['seed']
        for key, members in e.blocks(fleet).items():
            p = checkpoint(e.path(seed, 'group', key))
            state = p['state']
            assert p['client_ids'] == [c['client'] for c in members]
            assert set(p['parents']) == {e.path(seed, 'local', c['client']).name for c in members}
            assert state['status'] == 'converged' and not state['pending']
            assert state['assignments'] == [0] * len(members)
            assert len(state['history']) == state['round']
            assert state['history'][-1]['all_local_optimizers_converged']
            objectives = np.array([h['objective'] for h in state['history']])
            assert np.all(np.diff(objectives) <= 1e-6 * np.maximum(1, abs(objectives[:-1])))
            assert state['communication_float64'] == len(members) * (6 + 13 * state['round'])
            optimization.append(dict(seed=seed, group=key, rounds=state['round'], status=state['status']))

        for c in fleet['clients']:
            key = c['client']
            local = checkpoint(e.path(seed, 'local', key))
            assert local['data']['operating_checks_passed']
            training_checks += 1
            for name in ['local', 'prior']:
                fit = local[name]
                assert fit['available'] and fit['optimizer_converged']
                assert np.isfinite(fit['objective'])
                assert fit['selected_start'] < len(local['attempts' if name == 'local' else 'prior_attempts'])
            parameters, statuses, parents = e.fitted(seed, c, cfg, provenance, strict_runtime=False)
            for method, theta in parameters.items():
                q = np.log(theta)
                assert np.all(np.isfinite(q))
                assert np.all(q >= bounds[:, 0] - 1e-10) and np.all(q <= bounds[:, 1] + 1e-10)
                if method in bound_counts:
                    bound_counts[method] += (np.minimum(q - bounds[:, 0], bounds[:, 1] - q) < 1e-3)
                parameter_rows.append(dict(seed=seed, client=key, family=c['family'], method=method,
                    **{f'{name}_log_error': float(np.log(theta[j] / c['physical_parameters'][j]))
                       for j, name in enumerate(['p1', 'p2', 'p3', 'mu', 'sigma_f', 'sigma_r'])}))
            evaluation = checkpoint(e.path(seed, 'eval', key))
            assert evaluation['parents'] == parents
            expected_cases = {f'v{v}_{probe}' for v in cfg['evaluation_speeds'] for probe in ['slow', 'broadband']}
            assert set(evaluation['cases']) == expected_cases
            for case_key, case in evaluation['cases'].items():
                assert case_key == f'v{case["speed"]}_{case["probe"]}'
                assert set(case['methods']) == set(cfg['methods'])
                assert case['operating_checks_passed']
                region = e.fl.coverage_region(case['speed'], c['training_speed'], cfg['fleet_training_speeds'])
                assert case['region'] == region
                for method, metrics in case['methods'].items():
                    assert metrics['optimizer_status'] == statuses[method]
                    assert np.isfinite(metrics['yaw_rmse_deg_s']) and metrics['yaw_rmse_deg_s'] >= 0
                    rows.append(dict(seed=seed, client=key, family=c['family'], training_speed=c['training_speed'],
                        speed=case['speed'], probe=case['probe'], region=region,
                        operating_checks_passed=case['operating_checks_passed'], method=method, **metrics))

    assert len(expected_paths) == 129
    assert expected_paths == set(e.OUT.glob(f'{e.PREFIX}_seed*.json'))
    cases = pd.DataFrame(rows)
    assert len(cases) == 4536
    assert len(cases.drop_duplicates(['seed', 'client', 'speed', 'probe'])) == 756
    assert cases[cases.method == 'Exact'].yaw_rmse_deg_s.max() <= 1e-10
    pd.testing.assert_frame_equal(cases, pd.read_csv(e.OUT / f'{e.PREFIX}_cases.csv'), check_exact=False, rtol=1e-12, atol=1e-14)
    pd.testing.assert_frame_equal(pd.DataFrame(parameter_rows), pd.read_csv(e.OUT / f'{e.PREFIX}_parameters.csv'),
                                  check_exact=False, rtol=1e-12, atol=1e-14)
    primary = cases[(cases.region == 'peer_observed') & cases.method.isin(e.METHODS)]
    assert len(primary) == 1080
    assert primary.groupby(['seed', 'client', 'method']).size().eq(4).all()
    vehicles = primary.groupby(['seed', 'client', 'family', 'training_speed', 'method']).yaw_rmse_deg_s.mean().unstack('method')[e.METHODS].reset_index().rename_axis(columns=None)
    pd.testing.assert_frame_equal(vehicles, pd.read_csv(e.OUT / f'{e.PREFIX}_vehicles.csv'),
                                  check_exact=False, rtol=1e-12, atol=1e-14)
    fleets = vehicles.groupby('seed')[e.METHODS].mean()
    pd.testing.assert_frame_equal(fleets.reset_index(), pd.read_csv(e.OUT / f'{e.PREFIX}_fleets.csv'),
                                  check_exact=False, rtol=1e-12, atol=1e-14)
    means = fleets.mean()
    for method in e.METHODS:
        np.testing.assert_allclose(means[method], manifest['means_deg_s'][method], rtol=1e-12, atol=1e-14)
    decision = e.progression(fleets, True, True, cfg)
    assert decision == manifest['decision']
    family = vehicles.groupby('family')[e.METHODS].mean()
    speed = vehicles.groupby('training_speed')[e.METHODS].mean()
    strata = vehicles.groupby(['seed', 'family'])[e.METHODS].mean()
    region = cases[cases.method.isin(e.METHODS)].groupby(['region', 'method']).yaw_rmse_deg_s.mean().unstack()[e.METHODS]
    wins = {m: int((vehicles.OracleFamily < vehicles[m]).sum()) for m in e.METHODS if m != 'OracleFamily'}
    family_gain = 100 * (1 - family.OracleFamily / family.SharedPrior)
    speed_gain = 100 * (1 - speed.OracleFamily / speed.SharedPrior)
    gain_by_fleet = 100 * (1 - fleets.OracleFamily / fleets.SharedPrior)
    family_contribution = 100 * (family.SharedPrior - family.OracleFamily) / (family.SharedPrior - family.OracleFamily).sum()
    near_bound = {m: counts.tolist() for m, counts in bound_counts.items()}
    result = dict(status='frozen_results_audit_passed', decision=decision,
        checkpoint_count=129, selected_local_and_nominal_fits=108, learned_group_jobs=21,
        training_operating_checks_passed=training_checks, test_records=756, primary_test_records=216,
        methods=e.METHODS, means_deg_s=means.to_dict(), vehicle_wins_of_oracle= wins,
        family_means_deg_s=family.to_dict(orient='index'), family_gain_vs_shared_percent=family_gain.to_dict(),
        family_contribution_to_absolute_gain_percent=family_contribution.to_dict(),
        training_speed_gain_vs_shared_percent={str(k):v for k,v in speed_gain.items()},
        fleet_gain_vs_shared_percent={str(k):v for k,v in gain_by_fleet.items()},
        improved_family_fleet_strata=int((strata.OracleFamily < strata.SharedPrior).sum()),
        family_fleet_strata=9, region_means_deg_s=region.to_dict(orient='index'),
        vehicle_error_quantiles_deg_s={str(k):v for k,v in vehicles[e.METHODS].quantile([.5,.9,.95,1]).to_dict(orient='index').items()},
        parameter_order=['p1', 'p2', 'p3', 'mu', 'sigma_f', 'sigma_r'], near_bound_counts=near_bound,
        implementation_commits=sorted(implementation_commits),
        original_numerical_runtimes=[json.loads(v) for v in sorted(numerical_runtimes)],
        source_sha256={manifest_path.relative_to(e.ROOT).as_posix():e.base.canonical_hash(manifest_path), **original_hashes},
        new_optimizer_calls=0, new_simulations=0,
        scope='Three exploratory fleets, oracle fixed family labels, matched tanh/relaxation plant, known geometry, yaw prediction. Learned grouping and control benefits remain untested.')
    for name, sha in original_hashes.items():
        assert e.base.canonical_hash(e.ROOT / name) == sha, f'Review modified frozen evidence: {name}'

    rows_md = '\n'.join(f'| {m} | {means[m]:.6f} | {wins.get(m, "—")} |' for m in e.METHODS)
    family_md = '\n'.join(f'| {f} | {family.loc[f,"SharedPrior"]:.6f} | {family.loc[f,"OracleFamily"]:.6f} | {family_gain[f]:.1f}% |'
                          for f in ['Compact', 'Sport', 'SUV'])
    fleet_md = '\n'.join(f'| {s} | {fleets.loc[s,"SharedPrior"]:.6f} | {fleets.loc[s,"OracleFamily"]:.6f} | {gain_by_fleet[s]:.1f}% |'
                         for s in cfg['seeds'])
    review = f'''# CFL-FF heterogeneity: result review

The frozen oracle-family diagnostic passes its predeclared progression screen.
Family-compatible learned priors improve clean yaw prediction at a vehicle's
locally unobserved anchor speeds. This supports testing learned grouping.

## Evidence integrity

All 129 result checkpoints pass their payload checksums, source/protocol hashes,
parent hashes and completion checks. All 108 selected local/nominal-prior fits
and 21 learned-prior jobs converge. All 54 training and 756 test records pass
their operating checks. Exact-model test prediction error is zero.
Checkpoint metrics reproduce the saved case, parameter, vehicle and
fleet tables and the original progression decision. Original result files are
unchanged. This review performs no optimization or simulation.

## Primary result

Each of 54 vehicles contributes four clean-yaw tests: two new steering probes
at each of its two locally unobserved anchor speeds. The 216 test records are
repeated evaluations within three fleets, not 216 independent replications.
Errors below are mean vehicle RMSE, equally averaged across fleets, in deg/s.

| Method | Mean yaw RMSE [deg/s] | Vehicles improved by oracle / 54 |
|---|---:|---:|
{rows_md}

OracleFamily reduces mean error by **37.6% versus SharedPrior**, **37.4% versus
NominalPrior**, **39.3% versus MixedGroups**, and **45.7% versus LocalOE**.
The absolute reduction versus SharedPrior is **0.002928 deg/s**.

| Fleet | Shared prior [deg/s] | Oracle family [deg/s] | Reduction |
|---|---:|---:|---:|
{fleet_md}

The oracle beats each of the shared, nominal and mixed-group controls in all
three fleets. The screen required at least 5% pooled reduction and improvement
in at least two fleets for each comparison, with convergence and valid
operating conditions. All requirements pass.

## Mechanism and limits of the comparison

SharedPrior and NominalPrior are almost equal on average: 0.007777 versus
0.007742 deg/s. A single learned center therefore adds no average prediction
advantage over the strong fixed population prior in this diagnostic. The
shared center is 0.45% worse on this pooled mean. MixedGroups is 2.71% worse
than SharedPrior despite having the same number and size of groups as
OracleFamily. These controls support attributing the oracle gain to the
chosen family-compatible partition, beyond generic partitioning or prior use.
Only one predeclared mixed partition is tested per fleet.

| Family | Shared prior [deg/s] | Oracle family [deg/s] | Reduction |
|---|---:|---:|---:|
{family_md}

Sport contributes 76.3% of the pooled absolute reduction versus SharedPrior,
but Compact and SUV also improve when pooled across fleets. Eight of nine
family/fleet strata improve; Compact in fleet 2303 becomes 2.95% worse.
OracleFamily beats SharedPrior for 42/54 vehicles, so it is not an individual
guarantee. The 90th percentile of vehicle error falls from 0.018468 to 0.009861
deg/s, supporting a reduction beyond only the mean.

All three local training-speed strata improve versus SharedPrior: 40.3% at
13 m/s, 27.0% at 21 m/s and 39.1% at 29 m/s. Most absolute improvement comes
from low-speed-trained vehicles. Secondary error reductions are 9.7% at the
locally observed speed, 35.4% for fleet interpolation and 30.9% for fleet
extrapolation. These secondary endpoints do not replace the frozen primary
endpoint or select a setting.

## What can be claimed

In this controlled heterogeneous population, one fleet-wide regularization
center is inferior to the three chosen family-specific centers for unseen-speed
yaw prediction. Keeping individual vehicle parameters while sharing compatible
priors provides a useful way to combine operating-envelope completion with
persistent vehicle differences.

The experiment does not establish that three groups are necessary or optimal,
that family labels can be recovered from short local records, or that clustering
improves closed-loop control. Labels are privileged information. The three
fleets are exploratory. The physical CFL-FF anchors and variations are mapped
to the existing matched tanh/relaxation plant; full nonlinear Pacejka mismatch
is not tested. Geometry is known and relaxation lengths are common to all
vehicles. Absolute yaw errors are small, so control relevance remains open.

Friction is weakly excited and unregularized; it reaches a bound in
{near_bound['OracleFamily'][3]}/54 oracle fits. Successful yaw prediction therefore
does not establish reliable identification of every physical parameter.

## Recommended next experiment

Replace oracle labels with groups learned from local training information,
keeping these fleets, records, bounds, penalty weights and test endpoints
fixed. Keep LocalOE, NominalPrior, SharedPrior and OracleFamily as references.
Report performance and per-vehicle losses alongside retrospective family
agreement, rather than selecting grouping settings using the outer test.
First test whether a practical learned partition retains the oracle advantage;
unknown group count and control validation can follow if it does.

No new experiment is started by this review.

## Reproduce this review

```bat
python code\\experiments\\analyze_cfl_heterogeneity_review.py
```

After regenerating review outputs locally, commit any resulting changes:

```bat
git add docs/cfl_heterogeneity_result_review.md
git add results/tables/paper_cfl_heterogeneity_review.json
git diff --cached --stat
git commit -m "Audit CFL family-prior heterogeneity results"
git push origin main
```
'''
    target = e.ROOT / 'docs/cfl_heterogeneity_result_review.md'
    target.write_text(review, encoding='utf-8', newline='\n')
    result['review_sha256'] = e.base.canonical_hash(target)
    result['review_code_sha256'] = e.base.canonical_hash(Path(__file__))
    e.storage.atomic_json(e.OUT / f'{e.PREFIX}_review.json', result)
    print(json.dumps({k:result[k] for k in ['status', 'checkpoint_count', 'means_deg_s', 'vehicle_wins_of_oracle', 'family_gain_vs_shared_percent']}, indent=2))


if __name__ == '__main__':
    main()
