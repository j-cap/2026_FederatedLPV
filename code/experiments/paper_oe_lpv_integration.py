"""Resumable LPV comparison using the frozen fixed-LTI identification pilot.

Commands: preflight, smoke, run, status, analyze. No trajectory generation or OE
refitting. See docs/oe_lpv_integration.html for the prospective protocol amendment.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import html
import io
import json
from multiprocessing import get_context
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'code/src'), str(ROOT / 'code/experiments')]
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from sklearn.metrics import adjusted_rand_score
import paper_fixed_lti_identification as acquisition
from experiment_10a_basis_coverage_audit import raw_basis
from experiment_10b_oracle_complementary_gate import ridge_fit, predict
from experiment_10e_federated_latent_groups import fit_federated_mixture, client_statistics, aggregate_scale
from federated_lpv.local_scheduling import curvature_factor, fit_smooth_local

parent = acquisition.parent
PREFIX = 'paper_oe_lpv_integration'
CONFIG = ROOT / f'code/config/{PREFIX}.json'
OUT = ROOT / 'results/tables'
REPORT = ROOT / 'report/oe_lpv_integration.html'
PRIMARY = ['RestrictedLocal', 'SmoothLocal', 'Global', 'Federated']
METHODS = PRIMARY + ['CentralSameGroups', 'FamilyLabels', 'ExactLPV']
REGIONS = ['full', 'seen', 'unseen']


def preflight():
    cfg = parent.load(CONFIG)
    _, _, acquisition_cfg, old_lock = acquisition.preflight()
    for path, expected in cfg['source_sha256'].items():
        if parent.file_hash(ROOT / path) != expected:
            raise ValueError(f'Frozen integration input changed: {path}')
    basis = parent.load(ROOT / 'code/config/experiment_10a.json')
    mixture = parent.load(ROOT / 'code/config/experiment_10e.json')
    if cfg['seeds'] != [1301, 1302, 1303] or cfg['conditions'] != ['ClosedOE4', 'ClosedCleanOLS']:
        raise ValueError('This integration pilot must reuse the three development fleets')
    if set(cfg['seeds']) & set(acquisition_cfg['confirmation_seeds']):
        raise ValueError('Development and confirmation must be disjoint')
    if cfg['smooth_strength'] != parent.load(OUT / 'paper_local_smoothness_selection.json')['selected_strength']:
        raise ValueError('Primary smoothness strength must remain the inherited selection')
    sources = dict(cfg['source_sha256'])
    for path in [CONFIG, Path(__file__)]:
        sources[path.relative_to(ROOT).as_posix()] = parent.file_hash(path)
    lock = dict(sources=sources, acquisition_provenance=old_lock,
                protocol_amendment=cfg['protocol_amendment'], confirmation_unlocked=False)
    return cfg, basis, mixture, acquisition_cfg, lock


def bounded(p, cfg):
    return bool(p['max_beta_deg'] <= cfg['max_beta_deg'] and p['max_ay'] <= cfg['max_ay']
                and p['max_steering_deg'] <= cfg['steering_limit_deg'] + 1e-8
                and p['max_rate_deg_s'] <= cfg['steering_rate_deg_s'] + 1e-8)


def assemble_inputs(result, source, condition, cfg):
    """Match all saved cells exactly; no filtering by reference accuracy or labels."""
    clients = source['clients']
    index = {(c['client'], float(c['speed'])): c for c in result['cells']}
    expected = {(c['client'], float(v)) for c in clients for v in c['speeds']}
    if len(index) != len(result['cells']) or set(index) != expected:
        raise ValueError('Missing, duplicate or extra OE cells')
    speeds, targets, diagnostics = [], [], []
    for i, client in enumerate(clients):
        observed = np.asarray(client['speeds'], dtype=float)
        values = []
        for speed in observed:
            c = index[(client['client'], float(speed))]
            if c['client_index'] != i or c['parameters'] != client['parameters']:
                raise ValueError('Client identity/parameters changed')
            model = c['models'][condition]
            if c['status'] != 'simulated' or not model['fit_valid']:
                raise ValueError('Invalid local fit: abort the fleet; do not drop the cell')
            if not all(bounded(c[p], cfg) and c[p]['settling_pass'] for p in ['training', 'validation']):
                raise ValueError('Applied bounds or settling failed: abort the fleet')
            value = np.asarray(model['target'], dtype=float)
            if value.shape != (6,) or not np.isfinite(value).all():
                raise ValueError('Expected six finite physical A/B entries')
            values.append(value)
            diagnostics.append(dict(client=client['client'], speed=float(speed),
                original_physical_pass=c['physical_pass'], accuracy_pass=model['accuracy_pass'],
                matrix_error=model['matrix_error'],
                limiter_fraction_train=c['tracking_train']['limiter_fraction'],
                limiter_fraction_validation=c['tracking_validation']['limiter_fraction']))
        speeds.append(observed)
        targets.append(np.asarray(values))
    return speeds, targets, diagnostics


def information(speeds, basis):
    design = raw_basis(np.asarray(speeds), 7, basis)
    singular = np.linalg.svd(design, compute_uv=False)
    rank = int(np.linalg.matrix_rank(design))
    return dict(rank=rank, full_rank=rank == 7, observations=len(speeds),
                unique_speeds=sorted(set(map(float, speeds))),
                condition=float(singular[0] / singular[-1]) if rank == 7 else None)


def central_ridge(speeds, targets, basis, strength):
    """Independent pooled augmented-LS solution of the same ridge objective."""
    design = raw_basis(np.asarray(speeds), 7, basis)
    augmented = np.vstack([design, np.sqrt(strength + 1e-12) * np.eye(7)])
    right = np.vstack([targets, np.zeros((7, 6))])
    return np.linalg.lstsq(augmented, right, rcond=None)[0]


def fit_models(speeds, targets, seed, cfg, basis, mixture):
    """Label-free estimator interface: only local speeds and fitted A/B targets."""
    scale = aggregate_scale([client_statistics(v, y, basis) for v, y in zip(speeds, targets)])
    if not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError('Degenerate target scaling: the original group criterion is undefined')
    selected, candidates, global_model, communication = fit_federated_mixture(
        speeds, targets, mixture['candidate_clusters'], mixture['mixture_restarts'],
        mixture['mixture_iterations'], mixture['minimum_cluster_size'], seed, basis,
        strength=mixture['ridge_strength'])
    factor = curvature_factor(cfg['quadrature_points'])
    local, smooth, sensitivity = [], [], {str(s): [] for s in cfg['smooth_strength_candidates']}
    for observed, y in zip(speeds, targets):
        local.append(ridge_fit(observed, y, 3, basis, strength=1e-8).tolist())
        for strength in cfg['smooth_strength_candidates']:
            sensitivity[str(strength)].append(fit_smooth_local(
                raw_basis(observed, 7, basis), y, factor, strength,
                epsilon=cfg['smooth_numerical_ridge']).tolist())
        smooth.append(sensitivity[str(cfg['smooth_strength'])][-1])
    groups, central = [], []
    grid = np.asarray(basis['speed_grid'])
    global_central = central_ridge(np.concatenate(speeds), np.vstack(targets), basis, mixture['ridge_strength'])
    coefficient_delta = float(np.max(abs(global_model - global_central)))
    prediction_delta = float(np.max(abs(predict(global_model, grid, basis) - predict(global_central, grid, basis))))
    for g in range(selected['k']):
        members = np.flatnonzero(selected['assignments'] == g)
        observed = np.concatenate([speeds[i] for i in members])
        pooled = central_ridge(observed, np.vstack([targets[i] for i in members]), basis, mixture['ridge_strength'])
        central.append(pooled.tolist())
        coefficient_delta = max(coefficient_delta, float(np.max(abs(pooled - selected['models'][g]))))
        prediction_delta = max(prediction_delta, float(np.max(abs(predict(pooled, grid, basis) - predict(selected['models'][g], grid, basis)))))
        groups.append(dict(group=g, members=members.tolist(), clients=len(members), **information(observed, basis)))
    return dict(models=dict(RestrictedLocal=local, SmoothLocal=smooth, Global=global_model.tolist(),
                Federated=[m.tolist() for m in selected['models']], CentralSameGroups=central,
                assignments=selected['assignments'].tolist()),
        smooth_sensitivity=sensitivity, selected_k=int(selected['k']),
        candidates=candidates.to_dict('records'), groups=groups,
        global_information=information(np.concatenate(speeds), basis), communication=communication,
        equivalence=dict(coefficient_max_abs=coefficient_delta, prediction_max_abs=prediction_delta,
            passed=bool(coefficient_delta <= cfg['equivalence_tolerance'] and prediction_delta <= cfg['equivalence_tolerance']),
            scope='Pooled augmented least squares versus summed statistics for the SAME selected partition; no claim of independent clustering agreement.'))


def add_diagnostic_models(fitted, clients, speeds, targets, basis, mixture):
    """Labels and full-envelope references enter only after primary fits are fixed."""
    families = sorted({c['family'] for c in clients})
    family_models = []
    for family in families:
        indices = [i for i, c in enumerate(clients) if c['family'] == family]
        family_models.append(central_ridge(np.concatenate([speeds[i] for i in indices]),
            np.vstack([targets[i] for i in indices]), basis, mixture['ridge_strength']).tolist())
    fitted['models']['FamilyLabels'] = family_models
    fitted['models']['family_assignments'] = [families.index(c['family']) for c in clients]
    fitted['models']['ExactLPV'] = [ridge_fit(basis['speed_grid'], np.asarray(c['reference']), 7,
        basis, strength=1e-10).tolist() for c in clients]
    fitted['family_ari'] = float(adjusted_rand_score(fitted['models']['family_assignments'], fitted['models']['assignments']))
    blocks = [str(list(map(float, v))) for v in speeds]
    fitted['coverage_block_ari'] = float(adjusted_rand_score(blocks, fitted['models']['assignments']))


def model_for(models, method, i):
    if method in ['RestrictedLocal', 'SmoothLocal', 'ExactLPV']:
        return np.asarray(models[method][i])
    if method == 'Global':
        return np.asarray(models[method])
    assignment = models['family_assignments' if method == 'FamilyLabels' else 'assignments'][i]
    return np.asarray(models[method][assignment])


def error_values(prediction, truth):
    delta = prediction - truth
    return dict(matrix=np.linalg.norm(delta, axis=1) / np.linalg.norm(truth, axis=1),
                a=np.linalg.norm(delta[:, :4], axis=1) / np.linalg.norm(truth[:, :4], axis=1),
                b=np.linalg.norm(delta[:, 4:], axis=1) / np.linalg.norm(truth[:, 4:], axis=1),
                spectral_radius=np.max(abs(np.linalg.eigvals(prediction[:, :4].reshape(-1, 2, 2))), axis=1),
                entry_absolute=np.abs(delta))


def evaluate(fitted, clients, speeds, basis):
    grid = np.asarray(basis['speed_grid'])
    errors = {method: [] for method in METHODS}
    sensitivity = {key: [] for key in fitted['smooth_sensitivity']}
    for i, client in enumerate(clients):
        truth = np.asarray(client['reference'])
        if truth.shape != (len(grid), 6) or not np.isfinite(truth).all():
            raise ValueError('Invalid evaluation reference')
        for method in METHODS:
            estimate = predict(model_for(fitted['models'], method, i), grid, basis)
            errors[method].append({k: v.tolist() for k, v in error_values(estimate, truth).items()})
        for strength, models in fitted['smooth_sensitivity'].items():
            estimate = predict(np.asarray(models[i]), grid, basis)
            sensitivity[strength].append(error_values(estimate, truth)['matrix'].tolist())
    return errors, sensitivity


def run_payload(seed, cfg, basis, mixture, acquisition_cfg, lock):
    start = time.perf_counter()
    result = acquisition.read_result(acquisition.path_for(seed), lock['acquisition_provenance'])
    source = acquisition.oe.known.source_result(seed, 'gentle', lock['acquisition_provenance']['oe_provenance']['known_provenance'])
    grid = list(basis['speed_grid'])
    # Check the references used in the local diagnostic against the stored full grid.
    for cell in result['cells']:
        np.testing.assert_array_equal(cell['reference'], source['clients'][cell['client_index']]['reference'][grid.index(cell['speed'])])
    conditions, inputs = {}, {}
    with threadpool_limits(limits=1):
        for condition in cfg['conditions']:
            speeds, targets, diagnostics = assemble_inputs(result, source, condition, acquisition_cfg)
            fitted = fit_models(speeds, targets, seed, cfg, basis, mixture)
            add_diagnostic_models(fitted, source['clients'], speeds, targets, basis, mixture)
            errors, sensitivity = evaluate(fitted, source['clients'], speeds, basis)
            conditions[condition] = dict(**fitted, errors=errors, sensitivity_errors=sensitivity,
                                         local_identification=diagnostics)
            inputs[condition] = [t.tolist() for t in targets]
    return dict(status='execution_complete', seed=seed, provenance=lock,
                clients=source['clients'], grid=grid, inputs=inputs, conditions=conditions,
                elapsed_s=time.perf_counter() - start, confirmation_unlocked=False,
                control_benefit_tested=False, original_acquisition_gate_pass=False)


def path_for(seed, directory=None):
    return (OUT if directory is None else directory) / f'{PREFIX}_seed{seed}.json'


def save_result(payload, directory=None):
    parent.save_json(path_for(payload['seed'], directory), dict(payload=payload,
        payload_sha256=hashlib.sha256(parent.canonical(payload)).hexdigest()))


def read_result(path, seed, lock):
    saved = parent.load(path)
    payload = saved['payload']
    if saved['payload_sha256'] != hashlib.sha256(parent.canonical(payload)).hexdigest():
        raise ValueError(f'Corrupt checkpoint: {path}')
    if payload['provenance'] != lock or payload['seed'] != seed or payload['status'] != 'execution_complete':
        raise ValueError(f'Stale checkpoint: {path}; move aside explicitly, never silently overwrite')
    return payload


def worker(seed):
    cfg, basis, mixture, acquisition_cfg, lock = preflight()
    payload = run_payload(seed, cfg, basis, mixture, acquisition_cfg, lock)
    save_result(payload)
    return dict(seed=seed, status='checkpoint_saved', elapsed_s=round(payload['elapsed_s'], 2))


def region_mask(grid, observed, region):
    seen = np.isin(grid, observed)
    return np.ones(len(grid), dtype=bool) if region == 'full' else (seen if region == 'seen' else ~seen)


def fleet_mean(values, clients, grid, region):
    """Equal client weights within each fleet, then equal fleet weights in analysis."""
    return float(np.mean([np.mean(np.asarray(row)[region_mask(grid, c['speeds'], region)])
                          for c, row in zip(clients, values)]))


def summarize(results, cfg):
    units, sensitivity, per_speed = [], [], []
    for result in results:
        for condition, fitted in result['conditions'].items():
            for method, values in fitted['errors'].items():
                for region in REGIONS:
                    units.append(dict(seed=result['seed'], condition=condition, method=method, region=region,
                        **{metric: fleet_mean([v[metric] for v in values], result['clients'], result['grid'], region)
                           for metric in ['matrix', 'a', 'b']},
                        unstable_fraction=fleet_mean([np.asarray(v['spectral_radius']) >= 1 for v in values], result['clients'], result['grid'], region)))
                for j, speed in enumerate(result['grid']):
                    for region in ['seen', 'unseen']:
                        selected = [v['matrix'][j] for c, v in zip(result['clients'], values)
                                    if bool(speed in c['speeds']) == (region == 'seen')]
                        if selected:
                            per_speed.append(dict(seed=result['seed'], condition=condition, method=method,
                                                  region=region, speed=speed, matrix=float(np.mean(selected))))
            for strength, values in fitted['sensitivity_errors'].items():
                for region in REGIONS:
                    sensitivity.append(dict(seed=result['seed'], condition=condition, strength=float(strength), region=region,
                        matrix=fleet_mean(values, result['clients'], result['grid'], region)))
    units, sensitivity, per_speed = map(pd.DataFrame, [units, sensitivity, per_speed])
    comparisons = []
    for condition in cfg['conditions']:
        for region in REGIONS:
            view = units[(units.condition == condition) & (units.region == region)].pivot(index='seed', columns='method', values='matrix')
            for baseline in ['RestrictedLocal', 'SmoothLocal', 'Global']:
                gains = 100 * (1 - view.Federated / view[baseline])
                comparisons.append(dict(condition=condition, region=region, baseline=baseline,
                    improvement_pct=float(gains.mean()), minimum_pct=float(gains.min()), maximum_pct=float(gains.max()),
                    wins=int((gains > 0).sum()), fleets=len(gains)))
    return units, sensitivity, per_speed, comparisons


def svg_figure(draw):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with plt.rc_context({'svg.hashsalt': PREFIX, 'font.size': 10}):
        fig, ax = plt.subplots(figsize=(9, 4), layout='constrained')
        draw(ax)
        out = io.StringIO()
        fig.savefig(out, format='svg', metadata={'Date': None})
        plt.close(fig)
    return out.getvalue()[out.getvalue().index('<svg'):]


def write_report(results, cfg, units, sensitivity, per_speed, comparisons, manifest, destination):
    table = acquisition.table
    primary = units[units.condition == 'ClosedOE4']
    rows = []
    for method in METHODS:
        view = primary[primary.method == method]
        rows.append([method, *[100 * view[view.region == r].matrix.mean() for r in REGIONS],
                     100 * view[view.region == 'unseen'].b.mean()])
    def bars(ax):
        width = .25
        for i, region in enumerate(REGIONS):
            view = primary[primary.region == region].groupby('method').matrix.mean()
            ax.bar(np.arange(4) + (i-1)*width, [100*view[m] for m in PRIMARY], width, label=region)
        ax.set(xticks=np.arange(4), xticklabels=PRIMARY, ylabel='Mean relative matrix error [%]')
        ax.legend(); ax.grid(axis='y', alpha=.2)
    def curves(ax):
        view = per_speed[(per_speed.condition == 'ClosedOE4') & (per_speed.region == 'unseen')]
        for method in PRIMARY:
            values = view[view.method == method].groupby('speed').matrix.mean()
            ax.plot(values.index, 100*values, marker='o', label=method)
        ax.set(xlabel='Speed [m/s]', ylabel='Unobserved-speed matrix error [%]')
        ax.legend(); ax.grid(alpha=.2)
    def smooth(ax):
        view = sensitivity[(sensitivity.condition == 'ClosedOE4') & (sensitivity.region == 'unseen')]
        for seed, group in view.groupby('seed'):
            group = group.sort_values('strength')
            ax.semilogx(group.strength, 100*group.matrix, marker='o', label=f'Fleet {seed}')
        ax.axvline(cfg['smooth_strength'], color='black', linestyle='--', label='Inherited primary strength')
        ax.set(xlabel='Smooth-local curvature penalty', ylabel='Unobserved-speed matrix error [%]')
        ax.legend(); ax.grid(alpha=.2)
    groups, checks = [], []
    for r in results:
        for condition, f in r['conditions'].items():
            checks.append([r['seed'], condition, f['selected_k'], f['equivalence']['passed'],
                           f"{f['equivalence']['coefficient_max_abs']:.2e}", f['family_ari'], f['coverage_block_ari']])
            for g in f['groups']:
                groups.append([r['seed'], condition, g['group'], g['clients'], g['rank'],
                               ', '.join(f'{v:g}' for v in g['unique_speeds']), g['condition']])
    paired = [[c['region'], c['baseline'], c['improvement_pct'], c['minimum_pct'], c['maximum_pct'],
               f"{c['wins']}/{c['fleets']}"] for c in comparisons if c['condition'] == 'ClosedOE4']
    clean_rows = []
    for condition in cfg['conditions']:
        for method in PRIMARY:
            v = units[(units.condition == condition) & (units.region == 'unseen') & (units.method == method)]
            clean_rows.append([condition, method, 100*v.matrix.mean(), 100*v.a.mean(), 100*v.b.mean(), 100*v.unstable_fraction.mean()])
    text = '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>OE to federated LPV: development integration</title><style>body{font:16px/1.6 system-ui;color:#203345;background:#edf3f7;margin:0}main{max-width:1120px;margin:auto;padding:28px}section{background:white;padding:24px;border-radius:12px;margin:20px 0}table{width:100%;border-collapse:collapse;font-size:14px}th,td{padding:8px;text-align:right;border-bottom:1px solid #ddd}.scroll{overflow:auto}svg{width:100%;height:auto}pre{white-space:pre-wrap;overflow-wrap:anywhere}code{background:#edf3f7}h1{line-height:1.2}</style></head><body><main><h1>From local OE fits to full-envelope LPV models</h1><p>Three reused development fleets · fixed nominal LTI acquisition · no new trajectories or OE fits</p><section><h2>Question and scope</h2><p>Does learned grouping combine accurate but locally incomplete speed coverage into better full-envelope models? The primary comparison uses the same 249 saved noisy-state, known-applied-steering OE targets for every deployable method. References and family labels do not enter their fitting or group selection.</p><p>This is a development integration result, not independent confirmation. The original zero-limiter-activation gate remains failed. The prospective amendment permits limiter activity for this comparison because actual applied steering is recorded, applied bounds and settling pass, and no cells are excluded by reference accuracy. It makes no smooth-driving or LPV-controller-performance claim.</p></section><section><h2>Primary matrix comparison</h2><p>Equal client weights within each fleet, then equal weights across the three fleets. Seen/unseen refers to each recipient's own identification speeds. Unseen speeds include both within-block interpolation gaps (if any) and extrapolation outside the local block.</p>'''
    text += table(['Method', 'Full [%]', 'Seen [%]', 'Unseen [%]', 'Unseen B error [%]'], rows) + svg_figure(bars) + svg_figure(curves)
    text += '''<p>RestrictedLocal uses the original three-function physical span. SmoothLocal uses the seven-function span and the inherited curvature penalty 10⁶. Global and Federated use the same seven-function span and the original mixture settings. CentralSameGroups is an equivalence check with the federated assignments fixed. FamilyLabels uses privileged family labels; ExactLPV fits the full reference grid. Both are diagnostic references, not deployable competitors or guaranteed lower bounds.</p></section><section><h2>Paired fleet comparisons</h2><p>Positive values mean lower matrix error for Federated. With only three reused fleets, the range and win count are descriptive; no confirmatory confidence interval is claimed.</p>'''
    text += table(['Region', 'Baseline', 'Mean gain [%]', 'Minimum [%]', 'Maximum [%]', 'Fleet wins'], paired)
    text += '<h3>Individual fleet means</h3>' + table(['Fleet', 'Method', 'Unseen [%]'], [[int(r.seed), r.method, 100*r.matrix] for r in primary[primary.region == 'unseen'].itertuples()])
    text += '</section><section><h2>Smooth-local sensitivity</h2><p>The inherited primary strength is unchanged. All seven prespecified strengths are shown to expose dependence on regularization. These development reference scores do not automatically select a new primary model or unlock confirmation.</p>' + svg_figure(smooth)
    text += '</section><section><h2>Noise and matrix diagnostics</h2><p>ClosedCleanOLS repeats the scheduling comparison using exact simulated states from the same closed-loop maneuvers. It diagnoses local measurement-noise effects; finite-amplitude approximation remains. A and B errors are reported separately so diagonal A entries cannot conceal B errors. Spectral radius ≥1 is reported without clipping or removing models; it does not establish closed-loop stability.</p>'
    text += table(['Targets', 'Method', 'Unseen matrix [%]', 'A [%]', 'B [%]', 'Unstable fitted A [%]'], clean_rows)
    text += '</section><section><h2>Grouping and centralized equivalence</h2><p>The independent centralized check solves pooled augmented least squares for exactly the same partition and ridge objective. It does not rerun clustering. Any failed equivalence check requires investigation of numerical conditioning and solver agreement before interpreting sharing benefits. Rank-deficient groups remain visible and receive no silent fallback. Family and coverage-block ARI are evaluation-only diagnostics.</p>'
    text += table(['Fleet', 'Targets', 'K', 'Equivalent', 'Max coefficient difference', 'Family ARI', 'Block ARI'], checks)
    text += table(['Fleet', 'Targets', 'Group', 'Clients', 'Rank / 7', 'Covered speeds [m/s]', 'Condition'], groups)
    text += '</section><section><h2>What to decide next</h2><p>Inspect paired unseen-speed errors against both local baselines and Global, smoothness sensitivity, group coverage, and the clean-state diagnostic together. A positive development result supports freezing the complete pipeline for fresh fleets; a weak or negative result should be reported and traced to local-estimation noise, grouping, or coverage. No scientific success threshold or confirmation unlock is inferred from a favorable single aggregate.</p></section><section><h2>Reproducibility</h2><p>All local targets, full-grid evaluation references, learned coefficients, assignments, candidate scores, sufficient-statistic communication counts and per-cell errors are saved in the three execution checkpoints. Raw trajectories are unnecessary for this stage. No family labels, unseen references, validation tracking errors or local accuracy flags select learned groups. OE affine offsets and initial states remain local nuisance parameters; only the six A/B entries enter scheduling regression.</p><p><a href="../docs/oe_lpv_integration.html">Protocol and Windows commands</a> · <a href="../results/tables/paper_oe_lpv_integration_analysis.json">Analysis manifest</a></p><pre>'
    text += html.escape(json.dumps({k: v for k, v in manifest.items() if k not in ['provenance', 'checkpoint_sha256', 'comparisons']}, indent=2)) + '</pre></section></main></body></html>'
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text, encoding='utf-8', newline='\n')


def analyze():
    cfg, _, _, _, lock = preflight()
    results, hashes = [], {}
    for seed in cfg['seeds']:
        path = path_for(seed)
        if not path.exists() or not parent.tracked_unchanged(path):
            raise ValueError('Complete, commit and push all execution checkpoints before analyze.')
        results.append(read_result(path, seed, lock))
        hashes[path.relative_to(ROOT).as_posix()] = parent.file_hash(path)
    units, sensitivity, per_speed, comparisons = summarize(results, cfg)
    manifest = dict(status='development_analysis_complete', provenance=lock, checkpoint_sha256=hashes,
        fleets=len(results), conditions=cfg['conditions'],
        local_cells_per_condition=sum(len(c['speeds']) for r in results for c in r['clients']),
        all_fixed_partition_equivalence_pass=all(f['equivalence']['passed'] for r in results for f in r['conditions'].values()),
        original_acquisition_gate_pass=False, protocol_amendment=cfg['protocol_amendment'],
        confirmation_unlocked=False, control_benefit_tested=False, comparisons=comparisons)
    parent.save_json(OUT / f'{PREFIX}_analysis.json', manifest)
    for suffix, frame in [('fleet_summary', units), ('smooth_sensitivity', sensitivity), ('speed_summary', per_speed), ('comparisons', pd.DataFrame(comparisons))]:
        frame.to_csv(OUT / f'{PREFIX}_{suffix}.csv', index=False, lineterminator='\n')
    write_report(results, cfg, units, sensitivity, per_speed, comparisons, manifest, REPORT)
    print(REPORT)


def smoke_worker():
    """Synthetic integration fixture; never fit or inspect the scientific fleets."""
    cfg, basis, mixture, _, _ = preflight()
    mixture = {**mixture, 'candidate_clusters': [1, 2], 'minimum_cluster_size': 2,
               'mixture_restarts': 2, 'mixture_iterations': 5}
    speeds = [np.asarray(v, dtype=float) for v in [[10, 12, 14], [16, 18, 20], [22, 24, 26], [28, 30]] * 2]
    grid = np.asarray(basis['speed_grid'])
    clients, targets = [], []
    for i, observed in enumerate(speeds):
        def values(v):
            z = (np.asarray(v) - 20) / 10
            return np.column_stack([.94+.01*z, -.01+.002*z, .10-.01*z,
                                    .93+.01*z, .02+.002*z, .30+.03*z]) * (1+.015*(i//4))
        targets.append(values(observed))
        clients.append(dict(client=f'fixture{i}', family=f'family{i//4}', speeds=observed.tolist(), reference=values(grid).tolist()))
    with threadpool_limits(limits=1):
        fitted = fit_models(speeds, targets, 1300, cfg, basis, mixture)
        add_diagnostic_models(fitted, clients, speeds, targets, basis, mixture)
        errors, sensitivity = evaluate(fitted, clients, speeds, basis)
    if not fitted['equivalence']['passed']:
        raise AssertionError('Synthetic centralized equivalence failed')
    result = dict(seed=1300, clients=clients, grid=grid.tolist(), conditions={c: dict(**fitted, errors=errors, sensitivity_errors=sensitivity) for c in cfg['conditions']})
    units, sensitivity, per_speed, comparisons = summarize([result], cfg)
    with tempfile.TemporaryDirectory(prefix='oe_lpv_smoke_') as tmp:
        path = Path(tmp) / 'report.html'
        write_report([result], cfg, units, sensitivity, per_speed, comparisons, dict(status='synthetic_smoke'), path)
        if path.read_text(encoding='utf-8').count('<svg') != 3:
            raise AssertionError('Missing report plots')
    return dict(status='smoke_passed', scientific_fleets_run=False, report_plots=3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['preflight', 'smoke', 'run', 'status', 'analyze'])
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    cfg, _, _, acquisition_cfg, lock = preflight()
    if args.command == 'preflight':
        for seed in cfg['seeds']:
            r = acquisition.read_result(acquisition.path_for(seed), lock['acquisition_provenance'])
            s = acquisition.oe.known.source_result(seed, 'gentle', lock['acquisition_provenance']['oe_provenance']['known_provenance'])
            for condition in cfg['conditions']:
                assemble_inputs(r, s, condition, acquisition_cfg)
        print(json.dumps(dict(status='preflight_passed', fleets=3, reused_oe_fits=249,
            conditions=cfg['conditions'], raw_arrays_required=False, new_simulations=0,
            primary_smooth_strength=cfg['smooth_strength'], confirmation_unlocked=False)))
    elif args.command == 'smoke':
        with ProcessPoolExecutor(max_workers=1, mp_context=get_context('spawn')) as pool:
            print(json.dumps(pool.submit(smoke_worker).result()))
    elif args.command == 'analyze':
        analyze()
    else:
        pending = []
        for seed in cfg['seeds']:
            path = path_for(seed)
            if path.exists():
                read_result(path, seed, lock)
                print(f'{seed}: complete, hashes verified', flush=True)
            else:
                pending.append(seed)
                print(f'{seed}: pending', flush=True)
        if args.command == 'run' and pending:
            with ProcessPoolExecutor(max_workers=min(args.workers, len(pending)), mp_context=get_context('spawn')) as pool:
                for future in as_completed([pool.submit(worker, seed) for seed in pending]):
                    print(json.dumps(future.result()), flush=True)
        if args.command == 'run':
            print('Execution complete. Commit and push the three checkpoints, then run analyze.', flush=True)


if __name__ == '__main__':
    main()
