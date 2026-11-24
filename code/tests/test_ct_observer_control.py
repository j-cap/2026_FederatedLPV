"""Observer timing, physical coordinates, closed-loop Jacobian and reporting."""
import copy
import inspect
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'code/src'), str(ROOT/'code/experiments')]
import paper_physical_ct_observer as run
import federated_lpv.ct_observer_control as model
from federated_lpv.vehicle import family_centers, discrete_bicycle_matrices
from federated_lpv.oe_lqi_validation import rk4, interpolate


def settings():
    return run.old.load(run.CONFIG)['observer']


def test_physical_coordinate_reconstruction_and_friction_independence():
    p = family_centers()['nominal']
    theta = [p.front_stiffness/p.mass, p.rear_stiffness/p.mass, p.mass/p.yaw_inertia, .9]
    for speed in [10, 21, 30]:
        a, b = model.observer_model(theta, speed, settings())
        expected_a, expected_b = discrete_bicycle_matrices(speed, p, .01)
        np.testing.assert_allclose(a, expected_a, atol=1e-14)
        np.testing.assert_allclose(b, expected_b.ravel(), atol=1e-14)
        theta[-1] = .55
        other_a, other_b = model.observer_model(theta, speed, settings())
        np.testing.assert_array_equal(a, other_a)
        np.testing.assert_array_equal(b, other_b)


def test_covariance_timing_and_exact_zero_initial_state():
    a, b = run.nominal_matrices()
    obs = model.kalman_design(a, b, settings(), 1201)
    q = np.array(obs['diagnostics']['q']); r = obs['diagnostics']['r']
    np.testing.assert_array_equal(obs['gains'][0], np.zeros(2))
    np.testing.assert_allclose(obs['gains'][1], q[:, 1]/(q[1, 1]+r))
    np.testing.assert_allclose(obs['gains'][-1], obs['diagnostics']['steady_gain'], atol=1e-12)
    assert obs['diagnostics']['error_rho'] < 1
    assert obs['diagnostics']['observability_rank'] == 2


def test_detectable_unobservable_hidden_state_is_not_rejected():
    obs = model.kalman_design(np.diag([.9, .8]), np.ones(2), settings(), 10)
    assert obs['diagnostics']['observability_rank'] == 1
    assert obs['diagnostics']['observability_condition'] is None
    with pytest.raises((np.linalg.LinAlgError, ValueError)):
        model.kalman_design(np.diag([1.1, .8]), np.ones(2), settings(), 10)


def test_update_control_prediction_and_saturation_use_only_available_signals():
    cfg = run.old.load(run.old.storage.CONFIG)
    a, b = run.nominal_matrices()
    prior = np.array([.02, .03]); gain = np.array([1., 2., 3.]); l = np.array([.1, .3])
    result = model.feedback_step(prior, .1, 0., .04, 10., gain, 2., l, a, b, cfg)
    posterior = prior+l*(.04-prior[1])
    np.testing.assert_allclose(result['posterior'], posterior)
    assert result['limited'] and result['rate_limited']
    np.testing.assert_allclose(result['applied'], np.deg2rad(120)*.01)
    np.testing.assert_allclose(result['next_prior'], a@posterior+b*result['applied'])
    assert not np.allclose(result['next_prior'], a@posterior+b*result['requested'])
    assert result['integral'] == .1  # outward integral increment is blocked
    assert set(inspect.signature(model.feedback_step).parameters) == {
        'prior','integral','previous_input','measured_yaw','reference',
        'gain','prefilter','observer_gain','a','b','cfg'}


def test_combined_jacobian_matches_nonlinear_feedback_finite_difference():
    cfg = run.old.load(run.old.storage.CONFIG)
    plant = family_centers()['heavy']; speed = 21.
    ap, bp = run.rk4_origin_matrices(plant, speed, .01)
    a, b = run.nominal_matrices()
    obs = model.kalman_design(a, b, settings(), 1)
    l = np.array(obs['diagnostics']['steady_gain'])
    schedule = run.old.storage.nominal_schedule(cfg)
    gain, prefilter = interpolate(schedule, cfg['speed_grid'], [speed])
    gain = gain[0]; prefilter = prefilter[0]
    def transition(z):
        fb = model.feedback_step(z[2:4], z[4], 0., z[1], 0., gain, prefilter, l, a, b, cfg)
        assert not fb['limited']
        return np.r_[rk4(z[:2], fb['applied'], speed, plant, .01), fb['next_prior'], fb['integral']]
    h = 1e-7
    numerical = np.column_stack([(transition(h*e)-transition(-h*e))/(2*h) for e in np.eye(5)])
    expected = model.origin_closed_loop(ap, bp, a, b, gain, l, .01)
    np.testing.assert_allclose(numerical, expected, rtol=1e-8, atol=1e-10)


def test_common_noise_and_reproducible_full_trace(tmp_path):
    cfg = run.old.load(run.CONFIG); cc = run.old.load(run.old.storage.CONFIG)
    spec = run.old.load(run.ROOT/'code/config/paper_oe_transient_benchmark.json')
    spec['duration_s'] = .1
    profile = run.profile(spec, 21.)
    a, b = run.nominal_matrices()
    obs = model.kalman_design(a, b, settings(), len(profile['time']))
    noise = model.yaw_noise(1301, 0, 21, len(profile['time']), settings())
    np.testing.assert_array_equal(noise, model.yaw_noise(1301, 0, 21, len(noise), settings()))
    assert not np.array_equal(noise, model.yaw_noise(1301, 1, 21, len(noise), settings()))
    schedule = run.old.storage.nominal_schedule(cc)
    args = (family_centers()['nominal'], schedule, profile, obs, noise, cc)
    outcome, arrays = model.simulate_observed(*args)
    repeated, again = model.simulate_observed(*args)
    assert outcome == repeated and model.trace_hash(arrays) == model.trace_hash(again)
    np.testing.assert_allclose(arrays['measured_yaw'], arrays['state'][:, 1]+noise)
    np.testing.assert_allclose(arrays['integral'][1], cc['sample_time']*noise[0])
    target = tmp_path/'trace.npz'; np.savez_compressed(target, **arrays)
    with np.load(target) as saved:
        assert model.trace_hash(dict(saved)) == model.trace_hash(arrays)
    broken = dict(cc, numerical_abort_abs_state=1e-20)
    failure, _ = model.simulate_observed(*args[:-1], broken)
    assert failure['status'] == 'rollout_failed' and failure['metrics'] is None


def fake_records(cfg):
    return [dict(seed=seed, client='synthetic', method=method, speed=speed,
        status='complete', feasible=True, true_origin_rho=.9, **{k: .1 for k in run.METRICS})
        for seed in cfg['seeds'] for method in cfg['methods'] for speed in cfg['speeds_mps']]


def test_failure_invalidates_full_means_and_remains_in_counts(tmp_path, monkeypatch):
    cfg = run.old.load(run.CONFIG); frame = pd.DataFrame(fake_records(cfg))
    frame.loc[0, ['status','feasible']] = ['rollout_failed', False]
    frame.loc[0, run.METRICS] = np.nan
    _, fleets, summary, speeds, paired = run.aggregate(frame, cfg)
    nominal = summary.set_index('method').loc['Nominal/Nominal']
    assert nominal['cases'] == 9 and nominal['complete'] == 8 and nominal['rollout_failed'] == 1
    assert np.isnan(nominal['tracking_rmse_deg_s'])
    target = tmp_path/'report.html'; monkeypatch.setattr(run, 'REPORT', target)
    run.write_report(summary, fleets, speeds, paired, {}, cfg)
    text = target.read_text(encoding='utf-8')
    assert 'unavailable' in text and text.count('<svg') == 4 and '0.03°/s' in text


def test_checkpoint_rejects_duplicate_cases_false_feasibility_and_noise_mismatch(monkeypatch):
    cfg = run.old.load(run.CONFIG)
    metrics = {k: .1 for k in run.METRICS}
    metrics.update(beta_violation_samples=0)
    rows = [dict(method=m, speed=v, noise_sha256=f'noise{v}', trace_sha256='trace',
        outcome=dict(status='complete', completed_steps=1200, feasible=True, metrics=copy.deepcopy(metrics)))
        for m in cfg['methods'] for v in cfg['speeds_mps']]
    payload = dict(seed=1301, provenance={'fixture': True}, status='running', clients=[
        dict(index=0, client='nominal_00', rows=rows)])
    run.validate(payload, 1301, cfg, {'fixture': True})
    broken = copy.deepcopy(payload); broken['clients'][0]['rows'][-1] = rows[0]
    with pytest.raises(ValueError, match='Missing/duplicate'):
        run.validate(broken, 1301, cfg, {'fixture': True})
    broken = copy.deepcopy(payload); broken['clients'][0]['rows'][0]['noise_sha256'] = 'different'
    with pytest.raises(ValueError, match='Unmatched sensor'):
        run.validate(broken, 1301, cfg, {'fixture': True})
    broken = copy.deepcopy(payload); broken['clients'][0]['rows'][0]['outcome']['metrics']['beta_violation_samples'] = 1
    with pytest.raises(ValueError, match='feasibility'):
        run.validate(broken, 1301, cfg, {'fixture': True})


def test_saved_controller_reuse_and_no_nuisance_consumption(monkeypatch):
    cfg = run.old.load(run.CONFIG); data = run.inputs(1301)
    def fake_simulate(plant, schedule, profile, observer, noise, control):
        assert schedule == data[3]['clients'][0]['schedules']['YawOnlyCT']
        expected = model.observer_model(data[2]['clients'][0]['fit']['parameters'], 21., settings())
        np.testing.assert_array_equal(observer['a'], expected[0])
        return dict(status='complete', completed_steps=1200, feasible=True, metrics={}), {'noise': noise}
    monkeypatch.setattr(run, 'simulate_observed', fake_simulate)
    row, _ = run.case(1301, 0, 'YawOnlyCT/Matched', 21., cfg, data)
    changed = copy.deepcopy(data)
    changed[2]['clients'][0]['fit']['offset'] = [1e10, -1e10]
    changed[2]['clients'][0]['fit']['initial_state'] = [1e10, -1e10]
    other, _ = run.case(1301, 0, 'YawOnlyCT/Matched', 21., cfg, changed)
    assert row == other


def test_resume_reuses_completed_vehicle_after_interrupted_checkpoint(tmp_path, monkeypatch):
    cfg = run.old.load(run.CONFIG); data = run.inputs(1301)
    monkeypatch.setattr(run, 'OUT', tmp_path)
    monkeypatch.setattr(run, 'inputs', lambda seed: data)
    calls = []
    def fake_case(seed, index, method, speed, cfg, data):
        calls.append(index)
        metrics = {k: .1 for k in run.METRICS}; metrics['beta_violation_samples'] = 0
        return dict(method=method, speed=speed, noise_sha256=f'noise{speed}',
            trace_sha256='synthetic', outcome=dict(status='complete', completed_steps=1200,
            feasible=True, metrics=metrics)), None
    monkeypatch.setattr(run, 'case', fake_case)
    writer = run.atomic_json
    def interrupt_after_first(path, payload):
        if len(payload['clients']) == 2:
            raise PermissionError('Synthetic checkpoint interruption')
        writer(path, payload)
    monkeypatch.setattr(run, 'atomic_json', interrupt_after_first)
    with pytest.raises(PermissionError, match='Synthetic'):
        run.run_fleet(1301)
    assert len(run.old.envelope(run.path(1301))['clients']) == 1
    calls.clear(); monkeypatch.setattr(run, 'atomic_json', writer)
    result = run.run_fleet(1301)
    assert result['status'] == 'execution_complete'
    assert 0 not in calls and len(calls) == 29*15
