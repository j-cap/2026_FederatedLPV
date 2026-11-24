"""Numerical correctness and scientific invariants of the reduced OE baseline."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.linalg import expm
from scipy.optimize._numdiff import approx_derivative

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'code/src'), str(ROOT/'code/experiments')]
from federated_lpv.physical_ct_oe import (
    continuous_matrices, discrete_matrices, projected_response, fit_physical_oe,
)
from federated_lpv.structured_oe import propagate
from federated_lpv.vehicle import family_centers, nonlinear_bicycle_rhs
from experiment_10a_basis_coverage_audit import equilibrium


def settings():
    return json.loads((ROOT/'code/config/paper_physical_ct_oe.json').read_text(encoding='utf-8'))['estimator']


@pytest.mark.parametrize('family', list(family_centers()))
@pytest.mark.parametrize('speed', [10., 20., 30.])
def test_analytical_model_matches_nonlinear_jacobian(family, speed):
    p = family_centers()[family]
    physical = [p.front_stiffness/p.mass, p.rear_stiffness/p.mass, p.mass/p.yaw_inertia, .9]
    state = equilibrium(p, speed, .005)
    numerical = approx_derivative(lambda z: nonlinear_bicycle_rhs(z[:2], z[2], speed, p),
                                  state, method='3-point', abs_step=1e-6)
    a, b = continuous_matrices(physical, speed, [p.front_length, p.rear_length], .005)
    np.testing.assert_allclose(np.column_stack([a, b]), numerical, rtol=2e-8, atol=1e-8)
    augmented = np.zeros((3, 3)); augmented[:2] = numerical
    ad, bd = discrete_matrices(physical, speed, [1.2, 1.6], .005, .01)
    np.testing.assert_allclose(np.column_stack([ad, bd]), expm(.01*augmented)[:2], atol=2e-9)


def fixture(noise=0., n=500, speeds=(12., 28.)):
    rng = np.random.default_rng(8271)
    truth = np.array([47., 59., .56, 1.02])
    cfg = settings(); segments = []
    for v in speeds:
        ad, bd = discrete_matrices(truth, v, cfg['geometry_m'], .005, .01)
        u = .015+.012*rng.normal(size=n)
        x = propagate(ad, u[:, None]*bd + [0.00013, -0.00021], [.002, .09])
        y = x + noise*rng.normal(size=x.shape)*np.deg2rad(cfg['state_noise_deg'])
        segments.append(dict(speed=v, measured_state=y, applied_input=u))
    return truth, cfg, segments


def test_exact_variable_projection_jacobian_with_nonzero_residual():
    _, cfg, segments = fixture(noise=1.)
    q = np.log([55., 48., .65, .85])
    residual, jac, _ = projected_response(q, segments[0], cfg)
    numerical = approx_derivative(lambda z: projected_response(z, segments[0], cfg, False)[0], q,
                                  method='3-point', rel_step=1e-5)
    assert np.linalg.norm(residual) > 1
    np.testing.assert_allclose(jac, numerical, atol=3e-5, rtol=2e-5)


def test_noiseless_joint_recovery_and_single_speed_ambiguity():
    truth, cfg, segments = fixture()
    result = fit_physical_oe(segments, cfg)
    assert result['converged']
    np.testing.assert_allclose(result['parameters'], truth, rtol=2e-5)
    assert result['information']['rank'] == 4
    assert all(v['relative_rank'] == 3 for v in result['single_speed_information'])
    for fitted in result['trajectories']:
        np.testing.assert_allclose(fitted['offset'], [0.00013, -0.00021], atol=1e-8)
        np.testing.assert_allclose(fitted['initial_state'], [.002, .09], atol=1e-7)


def test_inputs_unchanged_and_noisy_smoke():
    _, cfg, segments = fixture(noise=1., n=700)
    before = [s['measured_state'].copy() for s in segments]
    result = fit_physical_oe(segments, cfg)
    assert result['converged'] and result['information']['rank'] == 4
    assert result['weighted_mse'] < 1.3
    for s, original in zip(segments, before):
        np.testing.assert_array_equal(s['measured_state'], original)


def test_invalid_curvature_and_nonfinite_data_rejected():
    with pytest.raises(ValueError, match='equilibrium'):
        continuous_matrices([50., 50., .6, .4], 30., [1.2, 1.6], .005)
    _, cfg, segments = fixture()
    segments[0]['measured_state'][0, 0] = np.nan
    with pytest.raises(ValueError, match='Nonfinite'):
        fit_physical_oe(segments, cfg)


def test_identity_free_metric_and_zero_response_errors():
    import paper_physical_ct_oe as runner
    truth, cfg, _ = fixture()
    targets = []
    for speed in [12., 28.]:
        a, b = discrete_matrices(truth, speed, cfg['geometry_m'], .005, .01)
        targets.append(np.r_[a.ravel(), b])
    targets = np.array(targets)
    metrics = runner.dynamic_metrics(targets, targets, {})
    for key in ['matrix', 'a', 'b', 'increment', 'pole_rms_absolute', 'pulse_relative_by_state']:
        np.testing.assert_array_equal(metrics[key], np.zeros_like(metrics[key]))
    perturbed = targets.copy(); perturbed[:, 0] += .001
    metrics = runner.dynamic_metrics(perturbed, targets, {})
    expected = np.linalg.norm(perturbed-targets, axis=1)/np.linalg.norm(
        targets-np.r_[np.eye(2).ravel(), np.zeros(2)], axis=1)
    np.testing.assert_allclose(metrics['increment'], expected)


def test_missing_results_cannot_be_averaged_away():
    import paper_physical_ct_oe as runner
    import pandas as pd
    assert np.isnan(runner.strict_mean([1., np.nan, 2.]))
    frame = pd.DataFrame({'fleet':[1,1,2,2], 'error':[1.,np.nan,2.,4.]})
    result = frame.groupby('fleet').error.agg(runner.strict_mean)
    assert np.isnan(result[1]) and result[2] == 3.


def test_original_source_and_training_inputs_are_required(monkeypatch, tmp_path):
    import paper_physical_ct_oe as runner
    acquired, _ = runner.sources(1301)
    cell = acquired['cells'][0]
    raw = tmp_path/'fake.npz'
    monkeypatch.setattr(runner, 'original_raw_path', lambda seed, cell: raw)
    with pytest.raises(FileNotFoundError, match='restore'):
        runner.traces(1301, cell)
    np.savez(raw, train_steering=np.ones(10))
    with pytest.raises(ValueError, match='hash mismatch'):
        runner.traces(1301, cell)


def test_stale_checkpoint_rejected():
    import paper_physical_ct_oe as runner
    with pytest.raises(ValueError, match='Stale'):
        runner.validate_result({'seed':1301, 'provenance':{}}, 1301,
                               runner.load(runner.CONFIG), {})


def test_html_report_utf8_and_figures(tmp_path):
    import paper_physical_ct_oe as runner
    import pandas as pd
    methods = ['Global', 'PhysicalCTLocal']; rows = []
    keys = ['matrix','a','b','increment','pole_rms_absolute','pulse_beta','pulse_yaw']
    for method in methods:
        for speed in [12,28]:
            rows.append(dict(seed=1301, method=method, speed=speed, seen=False,
                             **{key:.01 for key in keys}))
    errors = pd.DataFrame(rows)
    summary = errors.groupby('method')[keys].mean().reset_index()
    fleets = summary.copy(); fleets['seed'] = 1301
    parameters = pd.DataFrame([dict(mu=.9, condition=10.)])
    counts = pd.DataFrame([dict(seed=1301, expected=1, converged=1, near_bound=0)])
    audit = dict(cohort='synthetic smoke', converged_clients=1, expected_clients=1)
    path = tmp_path/'report.html'
    runner.write_report(path, summary, fleets, counts, parameters, errors,
                        pd.DataFrame(), pd.DataFrame(), audit)
    text = path.read_text(encoding='utf-8')
    assert text.count('<svg') == 2 and 'A−I' in text
    assert 'Not run yet' in text


def test_matched_transient_control_synthesis():
    import paper_physical_ct_oe as runner
    from federated_lpv.oe_lqi_validation import design_nodes, simulate
    from paper_oe_transient_benchmark import profile
    from federated_lpv.vehicle import VehicleParameters
    cfg = runner.load(runner.storage.CONFIG)
    spec = runner.load(ROOT/'code/config/paper_oe_transient_benchmark.json')
    p = family_centers()['nominal']
    physical = [p.front_stiffness/p.mass, p.rear_stiffness/p.mass, p.mass/p.yaw_inertia,.9]
    targets = []
    for speed in cfg['speed_grid']:
        a,b = discrete_matrices(physical,speed,[1.2,1.6],.005,.01)
        targets.append(np.r_[a.ravel(),b])
    schedule = design_nodes(targets,cfg)
    assert schedule['status'] == 'designed'
    outcome, arrays = simulate(p,schedule,profile(spec,21.),cfg)
    assert outcome['status'] == 'complete' and outcome['completed_steps'] == 1200
    assert len(arrays['state']) == 1201
