"""10H-H: repair and validate the K=1 measured-output LPV estimator.

Runs opened development fleets only. No confirmation execution option exists.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from federated_lpv.innovation_likelihood import (
    InnovationLikelihood, MeasuredDataset, PARAMETER_NAMES,
    information_scale, projected_gradient, physical_coupling_diagnostic,
)
from experiment_10h_higher_order_lpv_gate import sample_fleet
from experiment_10hf_output_identifiability import (
    measurement_covariance, nominal_effective_parameters, simulate_measurements,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / 'code/config/experiment_10hh.json'
OUT = ROOT / 'results/tables'
FIGURE = ROOT / 'results/figures/experiment_10hh_single_model_repair.pdf'
PREFIX = 'experiment_10hh'


def load_configuration():
    cfg = json.loads(CONFIG.read_text())
    if set(cfg['development_seeds']) & set(cfg['reserved_confirmation_seeds']):
        raise ValueError('Development and confirmation seeds overlap')
    inherited = {
        name: json.loads((ROOT/f'code/config/experiment_{name}.json').read_text())
        for name in ['10h', '10ha', '10hf']
    }
    return cfg, inherited


def measured_datasets(seed, cfg, inherited):
    """Simulator boundary: only (speed, command, measured outputs) leave it."""
    h, ha, hf = (inherited[name] for name in ['10h', '10ha', '10hf'])
    rng = np.random.default_rng(seed + 1_700_000)
    records = {'train': [], 'heldout': []}
    split_rows = []
    for index, client in enumerate(sample_fleet(seed, h)):
        block = index % len(h['coverage_blocks'])
        band = (index // len(h['coverage_blocks'])) % len(h['frequency_bands_hz'])
        split = ('heldout' if index % cfg['heldout_client_modulus'] ==
                 cfg['heldout_client_remainder'] else 'train')
        for speed in h['coverage_blocks'][block]:
            u, y = simulate_measurements(
                client, speed, h['frequency_bands_hz'][band], hf, h, ha, rng
            )
            count = cfg['samples_per_segment']
            records[split].append((float(speed), u[:count], y[:count]))
            split_rows.append(dict(seed=seed, client_position=index, split=split,
                                   speed=speed, samples=count))
    datasets = {
        name: MeasuredDataset(np.array([r[0] for r in rows]),
                              np.array([r[1] for r in rows]),
                              np.array([r[2] for r in rows]))
        for name, rows in records.items()
    }
    return datasets, pd.DataFrame(split_rows)


def dataset_hash(data):
    digest = hashlib.sha256()
    for array in [data.speeds, data.commands, data.measurements]:
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def optimize(initial, nominal, scale, lower, upper, evaluator, cfg, method):
    z = (initial - nominal) / scale
    phase_rows = []
    evaluations = 0
    invalid = 0
    stages = cfg['stages'] if method == 'staged' else []
    for phase, active in [
        *[(f'stage_{i+1}', np.array(indices)) for i, indices in enumerate(stages)],
        ('joint_polish', np.arange(9)),
    ]:
        base = z.copy()

        def objective(values):
            nonlocal evaluations, invalid
            evaluations += 1
            local = base.copy()
            local[active] = values
            try:
                value, gradient = evaluator.value_gradient(nominal + scale*local)
                if not np.isfinite(value) or not np.isfinite(gradient).all():
                    raise FloatingPointError('nonfinite likelihood')
                return value, gradient[active] * scale[active]
            except (np.linalg.LinAlgError, FloatingPointError):
                invalid += 1
                return 1e12, np.zeros(len(active))

        budget = (cfg['optimizer_max_iterations'] if phase == 'joint_polish'
                  else cfg['stage_max_iterations'])
        result = minimize(
            objective, base[active], jac=True, method='L-BFGS-B',
            bounds=list(zip(((lower-nominal)/scale)[active],
                            ((upper-nominal)/scale)[active])),
            options=dict(maxiter=budget, ftol=cfg['optimizer_ftol'],
                         gtol=cfg['optimizer_gtol'], maxls=40, maxcor=20),
        )
        z[active] = result.x
        phase_rows.append(dict(phase=phase, success=bool(result.success),
                               iterations=int(result.nit), objective=float(result.fun),
                               message=str(result.message)))
    fitted = nominal + scale*z
    return fitted, result, phase_rows, evaluations, invalid


def gradient_audit(evaluator, nominal, cfg):
    rows = []
    step = cfg['gradient_check_log_step']
    for point, coordinates in [('nominal', nominal),
                                ('perturbed', nominal + np.linspace(-.08, .10, 9))]:
        analytic = evaluator.value_gradient(coordinates)[1]
        numerical = np.array([
            (evaluator.value(coordinates + np.eye(9)[j]*step)
             - evaluator.value(coordinates - np.eye(9)[j]*step))/(2*step)
            for j in range(9)
        ])
        error = np.linalg.norm(analytic-numerical) / max(np.linalg.norm(numerical), 1e-8)
        for j, name in enumerate(PARAMETER_NAMES):
            rows.append(dict(point=point, parameter=name, analytic=analytic[j],
                             central_difference=numerical[j], norm_relative_error=error))
    return pd.DataFrame(rows)


def curvature_audit(evaluator, fitted, lower, upper, cfg):
    step = cfg['hessian_log_step']
    hessian = np.column_stack([
        (evaluator.value_gradient(fitted + np.eye(9)[j]*step)[1]
         - evaluator.value_gradient(fitted - np.eye(9)[j]*step)[1])/(2*step)
        for j in range(9)
    ])
    asymmetry = np.linalg.norm(hessian-hessian.T)/max(np.linalg.norm(hessian), 1e-12)
    eigenvalues, eigenvectors = np.linalg.eigh((hessian+hessian.T)/2)
    directions = [(name, np.eye(9)[j]) for j, name in enumerate(PARAMETER_NAMES)]
    directions.append(('weakest_hessian_direction', eigenvectors[:, 0]))
    baseline = evaluator.value(fitted)
    profile_rows, edge_increases = [], []
    for name, direction in directions:
        values = []
        for offset in cfg['profile_log_offsets']:
            proposed = fitted + offset*direction
            candidate = np.clip(proposed, lower, upper)
            objective = evaluator.value(candidate)
            values.append(objective)
            profile_rows.append(dict(direction=name, log_offset=offset,
                                     actual_log_distance=float(np.linalg.norm(candidate-fitted)),
                                     clipped=not np.allclose(candidate, proposed, atol=1e-12, rtol=0),
                                     objective=objective,
                                     increase_pct=100*(objective-baseline)/baseline))
        if name != 'weakest_hessian_direction':
            edge_increases.append(100*(min(values[0], values[-1])-baseline)/baseline)
    return (eigenvalues, asymmetry, min(edge_increases), pd.DataFrame(profile_rows),
            pd.DataFrame({'eigenvalue_index':np.arange(9), 'eigenvalue':eigenvalues}))


def improvement(reference, value):
    return 100*(reference-value)/reference


def evaluate_seed(seed):
    cfg, inherited = load_configuration()
    if seed not in cfg['development_seeds']:
        raise ValueError('Only predeclared opened development seeds may be run')
    datasets, split = measured_datasets(seed, cfg, inherited)
    h, ha, hf = (inherited[name] for name in ['10h', '10ha', '10hf'])
    noise_scale = json.loads((OUT/'experiment_10ha_frozen_selection.json').read_text())['process_noise_scale']
    q = np.diag(np.asarray(ha['base_process_noise_diagonal'])*noise_scale)
    r = measurement_covariance(ha)
    train = InnovationLikelihood(datasets['train'], h['sample_time'], q, r)
    heldout = InnovationLikelihood(datasets['heldout'], h['sample_time'], q, r)
    nominal = np.log(nominal_effective_parameters(hf))
    lower, upper = (nominal+np.log(cfg[name]) for name in
                    ['parameter_lower_multiplier', 'parameter_upper_multiplier'])
    nominal_result = train.evaluate(nominal, gradient=True, information=True, diagnostics=True)
    nominal_test = heldout.evaluate(nominal, diagnostics=True)
    scale = information_scale(nominal_result['information'],
                              cfg['information_scale_minimum'], cfg['information_scale_maximum'])
    gradient_checks = gradient_audit(train, nominal, cfg).assign(seed=seed)
    rng = np.random.default_rng(seed+1_900_000)
    initials = [nominal]
    initials.extend(np.clip(nominal+rng.normal(0, cfg['restart_log_standard_deviation'], 9),
                            lower, upper) for _ in range(cfg['restart_count']-1))
    runs, restarts, parameters, profiles, eigenvalues, phases = [], [], [], [], [], []
    for method in cfg['methods']:
        fitted_restarts, objective_values, restart_summaries = [], [], []
        started = time.perf_counter()
        for restart, initial in enumerate(initials):
            fitted, result, phase_rows, calls, invalid = optimize(
                initial, nominal, scale, lower, upper, train, cfg, method)
            final = train.evaluate(fitted, gradient=True, diagnostics=True)
            fitted_restarts.append(fitted)
            objective_values.append(final['objective'])
            gradient_norm = np.max(np.abs(projected_gradient(
                fitted, final['gradient'], lower, upper)))
            row = dict(seed=seed, method=method, restart=restart,
                       success=bool(result.success), objective=final['objective'],
                       joint_iterations=int(result.nit), objective_gradient_evaluations=calls,
                       invalid_evaluations=invalid, projected_gradient_max=float(gradient_norm),
                       message=str(result.message))
            restart_summaries.append(row)
            restarts.append(row)
            phases.extend(dict(seed=seed, method=method, restart=restart, **p) for p in phase_rows)
            print(f'10H-H seed={seed} method={method} restart={restart} '
                  f'objective={final["objective"]:.6f} success={result.success} '
                  f'iterations={result.nit} projected_gradient={gradient_norm:.2g}', flush=True)
        best_index = int(np.argmin(objective_values))
        fitted = fitted_restarts[best_index]
        final = train.evaluate(fitted, gradient=True, diagnostics=True)
        test = heldout.evaluate(fitted, diagnostics=True)
        ev, asymmetry, profile_edge, profile, eigen = curvature_audit(
            train, fitted, lower, upper, cfg)
        exponentials = np.exp(fitted_restarts)
        cv = np.std(exponentials, axis=0)/np.mean(exponentials, axis=0)
        boundary = np.isclose(fitted, lower, atol=1e-5, rtol=0) | np.isclose(fitted, upper, atol=1e-5, rtol=0)
        best_row = restart_summaries[best_index]
        runs.append(dict(
            seed=seed, method=method, best_restart=best_index,
            all_restart_success=all(x['success'] for x in restart_summaries),
            optimizer_success=best_row['success'],
            train_objective_nominal=nominal_result['objective'], train_objective_fitted=final['objective'],
            heldout_objective_nominal=nominal_test['objective'], heldout_objective_fitted=test['objective'],
            heldout_objective_improvement_pct=improvement(nominal_test['objective'],test['objective']),
            heldout_sensor_mse_nominal=nominal_test['sensor_normalized_mse'],
            heldout_sensor_mse_fitted=test['sensor_normalized_mse'],
            heldout_sensor_mse_improvement_pct=improvement(nominal_test['sensor_normalized_mse'],test['sensor_normalized_mse']),
            heldout_yaw_rmse_deg_s=test['yaw_rmse_deg_s'],
            heldout_acceleration_rmse_mps2=test['acceleration_rmse_mps2'],
            heldout_steering_rmse_deg=test['steering_rmse_deg'],
            restart_objective_spread_pct=100*(max(objective_values)-min(objective_values))/min(objective_values),
            maximum_restart_parameter_cv=float(cv.max()),
            maximum_restart_projected_gradient=max(x['projected_gradient_max'] for x in restart_summaries),
            minimum_hessian_eigenvalue=float(ev[0]), maximum_hessian_eigenvalue=float(ev[-1]),
            hessian_asymmetry=float(asymmetry), minimum_profile_edge_increase_pct=profile_edge,
            boundary_fraction=float(boundary.mean()), joint_iterations=best_row['joint_iterations'],
            invalid_evaluations=sum(x['invalid_evaluations'] for x in restart_summaries),
            elapsed_seconds=time.perf_counter()-started,
            gradient_check_relative_error=float(gradient_checks.norm_relative_error.max()),
            train_data_sha256=dataset_hash(datasets['train']), heldout_data_sha256=dataset_hash(datasets['heldout']),
        ))
        parameters.extend(dict(seed=seed, method=method, parameter=name,
                               nominal=float(np.exp(nominal[j])), fitted=float(np.exp(fitted[j])),
                               information_scale=scale[j], restart_cv=cv[j], at_boundary=bool(boundary[j]))
                          for j,name in enumerate(PARAMETER_NAMES))
        profiles.append(profile.assign(seed=seed, method=method))
        eigenvalues.append(eigen.assign(seed=seed, method=method))
    # Re-score archived 10H-G parameters under the repaired objective, without
    # using its retrospective truth column or selecting any model on test data.
    archived = pd.read_csv(OUT/'experiment_10hg_parameters.csv', usecols=['seed','parameter','fitted'])
    old = archived[archived.seed == seed].set_index('parameter').loc[list(PARAMETER_NAMES),'fitted'].to_numpy()
    legacy_test = heldout.evaluate(np.log(old), diagnostics=True)
    ablation = pd.DataFrame([dict(seed=seed, method='archived_10HG_rescored',
                                heldout_objective_fitted=legacy_test['objective'],
                                heldout_objective_improvement_pct=improvement(nominal_test['objective'],legacy_test['objective']),
                                heldout_sensor_mse_improvement_pct=improvement(nominal_test['sensor_normalized_mse'],legacy_test['sensor_normalized_mse']))])
    frames = dict(runs=pd.DataFrame(runs), restarts=pd.DataFrame(restarts),
                  parameters=pd.DataFrame(parameters), profiles=pd.concat(profiles),
                  hessian_eigenvalues=pd.concat(eigenvalues), phases=pd.DataFrame(phases),
                  gradient_checks=gradient_checks, splits=split, archived_ablation=ablation)
    for name, frame in frames.items():
        frame.to_csv(OUT/f'{PREFIX}_seed{seed}_{name}.csv', index=False)
    return seed


def summarize():
    cfg, inherited = load_configuration()
    names = ['runs','restarts','parameters','profiles','hessian_eigenvalues','phases',
             'gradient_checks','splits','archived_ablation']
    frames = {}
    for name in names:
        paths = [OUT/f'{PREFIX}_seed{seed}_{name}.csv' for seed in cfg['development_seeds']]
        if not all(p.exists() for p in paths):
            raise RuntimeError('All predeclared development fleets must finish before summarizing')
        frames[name] = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
        frames[name].to_csv(OUT/f'{PREFIX}_{name}.csv', index=False)
    runs = frames['runs']
    conclusions, summary_rows = {}, []
    for method in cfg['methods']:
        local = runs[runs.method == method]
        aggregate = dict(
            mean_heldout_objective_improvement_pct=float(local.heldout_objective_improvement_pct.mean()),
            minimum_heldout_objective_improvement_pct=float(local.heldout_objective_improvement_pct.min()),
            mean_heldout_sensor_mse_improvement_pct=float(local.heldout_sensor_mse_improvement_pct.mean()),
            minimum_heldout_sensor_mse_improvement_pct=float(local.heldout_sensor_mse_improvement_pct.min()),
            all_restart_success_rate=float(local.all_restart_success.mean()),
            maximum_projected_gradient=float(local.maximum_restart_projected_gradient.max()),
            maximum_restart_objective_spread_pct=float(local.restart_objective_spread_pct.max()),
            maximum_restart_parameter_cv=float(local.maximum_restart_parameter_cv.max()),
            minimum_hessian_eigenvalue=float(local.minimum_hessian_eigenvalue.min()),
            minimum_profile_edge_increase_pct=float(local.minimum_profile_edge_increase_pct.min()),
            maximum_boundary_fraction=float(local.boundary_fraction.max()),
            maximum_gradient_check_relative_error=float(local.gradient_check_relative_error.max()),
            maximum_hessian_asymmetry=float(local.hessian_asymmetry.max()),
            invalid_evaluations=int(local.invalid_evaluations.sum()),
        )
        gates = dict(
            analytic_gradient_gate=aggregate['maximum_gradient_check_relative_error'] <= cfg['maximum_gradient_relative_error'],
            optimizer_success_gate=bool(local.all_restart_success.all()),
            stationarity_gate=aggregate['maximum_projected_gradient'] <= cfg['maximum_projected_gradient'],
            heldout_likelihood_gate=aggregate['minimum_heldout_objective_improvement_pct'] >= cfg['minimum_heldout_objective_improvement_pct'],
            heldout_fixed_weight_prediction_gate=aggregate['minimum_heldout_sensor_mse_improvement_pct'] > cfg['minimum_heldout_sensor_mse_improvement_pct'],
            restart_objective_gate=aggregate['maximum_restart_objective_spread_pct'] <= cfg['maximum_restart_objective_spread_pct'],
            restart_parameter_gate=aggregate['maximum_restart_parameter_cv'] <= cfg['maximum_restart_parameter_cv'],
            hessian_gate=aggregate['minimum_hessian_eigenvalue'] >= cfg['minimum_hessian_eigenvalue'],
            conditional_profile_gate=aggregate['minimum_profile_edge_increase_pct'] >= cfg['minimum_profile_edge_increase_pct'],
            boundary_gate=aggregate['maximum_boundary_fraction'] <= cfg['maximum_boundary_fraction'],
        )
        gates['numerical_repair_gate'] = all(gates[name] for name in [
            'analytic_gradient_gate', 'optimizer_success_gate', 'stationarity_gate',
            'heldout_likelihood_gate', 'heldout_fixed_weight_prediction_gate',
            'restart_objective_gate', 'restart_parameter_gate',
        ])
        gates['development_gate_pass'] = all(gates.values())
        conclusions[method] = {'aggregate':aggregate, 'gates':gates}
        summary_rows.append(dict(method=method, **aggregate,
                                 numerical_repair_gate=gates['numerical_repair_gate'],
                                 development_gate_pass=gates['development_gate_pass']))
    pd.DataFrame(summary_rows).to_csv(OUT/f'{PREFIX}_summary.csv', index=False)
    hashes = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
              for p in [CONFIG, Path(__file__), ROOT/'code/src/federated_lpv/innovation_likelihood.py',
                        ROOT/'code/config/experiment_10h.json', ROOT/'code/config/experiment_10ha.json',
                        ROOT/'code/config/experiment_10hf.json', OUT/'experiment_10ha_frozen_selection.json']}
    output = dict(methods=conclusions, confirmation_run=False,
                  reserved_confirmation_seeds=cfg['reserved_confirmation_seeds'],
                  leakage_statement='Only measured arrays and public design covariances, scheduling, nominal coordinates, and bounds enter the learner. Hidden fleet quantities and labels remain behind the simulator boundary.',
                  interpretation='Paired development repair, not blind confirmation. Steady-state Gaussian quasi-likelihood; fixed Q and unmodeled quiet-bias estimation uncertainty limit stochastic interpretation. Conditional profiles do not establish global uniqueness.',
                  inherited_covariance_tuning='Process-noise scale 0.01 was originally selected in 10H-A using latent-state estimation errors. No current-fleet truths are used to retune it. A fully measured-output-only covariance-selection pipeline remains to be validated.',
                  provenance_sha256=hashes)
    (OUT/f'{PREFIX}_conclusions.json').write_text(json.dumps(output, indent=2)+'\n')
    plot(frames)
    print(json.dumps(output, indent=2), flush=True)


def plot(frames):
    runs = frames['runs']
    fig, axes = plt.subplots(1,3,figsize=(13,3.8))
    colors = dict(joint='#0072B2', staged='#D55E00')
    for method in ['joint','staged']:
        rows = runs[runs.method == method]
        axes[0].plot(rows.seed.astype(str),rows.heldout_objective_improvement_pct,'o-',color=colors[method],label=method)
        axes[1].plot(rows.seed.astype(str),rows.heldout_sensor_mse_improvement_pct,'o-',color=colors[method],label=method)
        axes[2].plot(rows.seed.astype(str),rows.maximum_restart_parameter_cv,'o-',color=colors[method],label=method)
    old = frames['archived_ablation']
    axes[0].plot(old.seed.astype(str),old.heldout_objective_improvement_pct,'x--',color='#777777',label='10H-G rescored')
    axes[1].plot(old.seed.astype(str),old.heldout_sensor_mse_improvement_pct,'x--',color='#777777',label='10H-G rescored')
    axes[0].axhline(2,color='black',linestyle=':')
    axes[1].axhline(0,color='black',linestyle=':')
    axes[2].axhline(.08,color='black',linestyle=':')
    for ax,title,ylabel in zip(axes,['Corrected held-out likelihood','Fixed sensor-weight prediction','Restart coordinate dispersion'],['Improvement (%)','Improvement (%)','Maximum coefficient CV']):
        ax.set(title=title,xlabel='Opened development fleet',ylabel=ylabel)
        ax.legend(fontsize=8)
    fig.tight_layout()
    FIGURE.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(FIGURE)
    plt.close(fig)


def diagnose_bounds():
    """Post-hoc interpretation only: never changes fits or predeclared gates."""
    cfg, inherited = load_configuration()
    fitted_table = pd.read_csv(OUT/f'{PREFIX}_parameters.csv')
    rows, coordinate_rows = [], []
    for seed in cfg['development_seeds']:
        datasets, _ = measured_datasets(seed, cfg, inherited)
        h, ha, hf = (inherited[name] for name in ['10h','10ha','10hf'])
        noise_scale = json.loads((OUT/'experiment_10ha_frozen_selection.json').read_text())['process_noise_scale']
        q = np.diag(np.asarray(ha['base_process_noise_diagonal'])*noise_scale)
        evaluator = InnovationLikelihood(datasets['train'], h['sample_time'], q, measurement_covariance(ha))
        nominal = np.log(nominal_effective_parameters(hf))
        lower = nominal+np.log(cfg['parameter_lower_multiplier'])
        upper = nominal+np.log(cfg['parameter_upper_multiplier'])
        for method in cfg['methods']:
            local = fitted_table[(fitted_table.seed == seed)&(fitted_table.method == method)]
            fitted = np.log(local.set_index('parameter').loc[list(PARAMETER_NAMES),'fitted'].to_numpy())
            gradient = evaluator.value_gradient(fitted)[1]
            step = cfg['hessian_log_step']
            hessian = np.column_stack([
                (evaluator.value_gradient(fitted+np.eye(9)[j]*step)[1]
                 - evaluator.value_gradient(fitted-np.eye(9)[j]*step)[1])/(2*step)
                for j in range(9)
            ])
            hessian = (hessian+hessian.T)/2
            at_lower = np.isclose(fitted,lower,atol=1e-5,rtol=0)
            at_upper = np.isclose(fitted,upper,atol=1e-5,rtol=0)
            free = ~(at_lower|at_upper)
            free_eigen = np.linalg.eigvalsh(hessian[np.ix_(free,free)])
            boundary_names = []
            for j,name in enumerate(PARAMETER_NAMES):
                if not free[j]:
                    side = 'lower' if at_lower[j] else 'upper'
                    boundary_names.append(name+':'+side)
                    # Positive gradient at a lower bound means moving inward
                    # raises loss; a zero clipped outward profile is not flatness.
                    coordinate_rows.append(dict(seed=seed,method=method,parameter=name,
                                                bound=side,gradient=gradient[j],
                                                kkt_sign_satisfied=bool(gradient[j]>=0 if at_lower[j] else gradient[j]<=0)))
            rows.append(dict(seed=seed,method=method,active_coordinates=';'.join(boundary_names),
                             free_dimension=int(free.sum()),
                             minimum_free_hessian_eigenvalue=float(free_eigen[0]),
                             free_hessian_condition=float(free_eigen[-1]/free_eigen[0]),
                             maximum_free_gradient=float(np.max(np.abs(gradient[free]))),
                             **physical_coupling_diagnostic(fitted)))
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT/f'{PREFIX}_bound_diagnostics.csv',index=False)
    pd.DataFrame(coordinate_rows).to_csv(OUT/f'{PREFIX}_bound_coordinate_gradients.csv',index=False)
    output = dict(posthoc_diagnostic=True,predeclared_gates_unchanged=True,
                  minimum_free_hessian_eigenvalue=float(frame.minimum_free_hessian_eigenvalue.min()),
                  minimum_physical_coupling_discrepancy_pct=float(100*frame.relative_coupling_discrepancy.min()),
                  maximum_physical_coupling_discrepancy_pct=float(100*frame.relative_coupling_discrepancy.max()),
                  interpretation='Boundary KKT and free-face curvature distinguish a constrained minimum from an unconverged fit. Independent nine-coordinate fits need not obey the exact common-inertial-ratio bicycle identity; this diagnostic uses only fitted coefficients, not simulator truth.',
                  estimator_sha256=hashlib.sha256((ROOT/'code/src/federated_lpv/innovation_likelihood.py').read_bytes()).hexdigest())
    (OUT/f'{PREFIX}_bound_diagnostic_conclusions.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output,indent=2),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=5)
    parser.add_argument('--seed',type=int,help='One predeclared development seed only')
    parser.add_argument('--summarize-only',action='store_true')
    parser.add_argument('--diagnose-bounds',action='store_true',help='Post-hoc bound and physical-coupling interpretation')
    parser.add_argument('--resume',action='store_true')
    args = parser.parse_args()
    cfg,_ = load_configuration()
    OUT.mkdir(parents=True,exist_ok=True)
    if args.diagnose_bounds:
        diagnose_bounds()
        return
    if args.summarize_only:
        summarize()
        return
    seeds = [args.seed] if args.seed is not None else cfg['development_seeds']
    if any(seed not in cfg['development_seeds'] for seed in seeds):
        parser.error('Only opened development seeds are allowed; confirmation is sealed')
    if args.resume:
        required = ['runs','restarts','parameters','profiles','hessian_eigenvalues',
                    'phases','gradient_checks','splits','archived_ablation']
        seeds = [seed for seed in seeds if not all(
            (OUT/f'{PREFIX}_seed{seed}_{name}.csv').exists() for name in required
        )]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(evaluate_seed,seed):seed for seed in seeds}
        for future in as_completed(futures):
            print(f'Completed 10H-H development fleet {future.result()}',flush=True)
    if args.seed is None:
        summarize()


if __name__ == '__main__':
    main()
