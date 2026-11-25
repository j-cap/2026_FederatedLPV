"""Scientific invariants and restart/report behavior of the small pilot."""
import copy
from pathlib import Path
import sys

import numpy as np
import pytest

from federated_lpv import tire_relaxation_pilot as model
from federated_lpv.tire_relaxation import simulate

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'code/experiments'))
import paper_tire_relaxation_pilot as runner


@pytest.fixture
def cfg():
    return runner.preflight()


def test_multispeed_candidates_and_jacobian(cfg):
    pars = np.array([[50., 55., .6, .9, .4, .7], [53., 48., .58, 1., .8, .3]])
    speeds = [13., 21., 29.]; t = np.arange(40)*.01
    inputs = np.array([.004*np.sin(2*np.pi*f*t) for f in [.5, .8, 1.1]])
    out = model.batch_predict(pars, inputs, speeds, [1.2, 1.6], .01, .001)
    for i, p in enumerate(pars):
        for j, v in enumerate(speeds):
            reference = simulate(p, inputs[j], v, [1.2, 1.6], .01, .001)[0]
            np.testing.assert_allclose(out[i, j], reference, atol=1e-13, rtol=1e-13)
    for n in [4, 6]:
        p = pars[0, :n]; s = runner.settings(cfg)
        _, jac = model.response_jacobian(np.log(p), p, list(range(n)), inputs, speeds, [1.2, 1.6], s)
        for k in range(n):
            trials = np.tile(p, (2, 1)); trials[:, k] *= np.exp([1e-4, -1e-4])
            x = model.batch_predict(trials, inputs, speeds, [1.2, 1.6], .01, .001)
            numeric = ((x[0, :, :, 1]-x[1, :, :, 1])/2e-4).ravel()
            np.testing.assert_allclose(jac[:, k], numeric, atol=1e-8, rtol=2e-5)


def test_estimator_rejects_hidden_measurements():
    record = dict(speed=21., applied_input=np.zeros(20), measured_yaw=np.zeros(21), beta=np.zeros(21))
    with pytest.raises(ValueError, match='accepts yaw'):
        model.observed_records([record])


def test_initial_state_and_integration_refinement():
    p = [50., 55., .6, .9, .4, .7]; u = .005*np.sin(np.arange(80)*.09)
    for n in [4, 6]:
        a = model.batch_predict(p[:n], [u], [29.], [1.2, 1.6], .01, .001)
        b = model.batch_predict(p[:n], [u], [29.], [1.2, 1.6], .01, .0005)
        np.testing.assert_array_equal(a[:, :, 0], 0.)
        np.testing.assert_allclose(a, b, atol=1e-7, rtol=1e-5)


def test_lqi_equilibrium_and_measured_feedback(cfg):
    ccfg = runner.control_settings(cfg)
    for p in [[50., 55., .6, .9], [50., 55., .6, .9, .4, .7]]:
        d = model.design(p, 21., [1.2, 1.6], ccfg, 101)
        n = len(d['b']); a, b = d['a'], d['b']; c = np.zeros(n); c[1] = 1.
        eq = np.linalg.solve(np.block([[np.eye(n)-a, -b[:, None]], [c[None, :], np.zeros((1, 1))]]), np.r_[np.zeros(n), 1.])
        ref = .01; x = eq[:-1]*ref; expected_u = eq[-1]*ref
        posterior, prior, integ, command, _ = model.feedback(x, 0., expected_u, ref, ref, d, 50, ccfg)
        np.testing.assert_allclose(prior, x, atol=1e-12)
        np.testing.assert_allclose(command, expected_u, atol=1e-12)
        assert integ == 0.
        # Changing measured yaw affects output even though no latent state argument exists.
        changed = model.feedback(x, 0., expected_u, ref+.001, ref, d, 50, ccfg)
        assert changed[3] != command
        assert max(d['diagnostics']['control_rho'], d['diagnostics']['observer_rho']) < 1


def test_selection_does_not_promote_unconverged_fit():
    attempts = [dict(converged=False, weighted_sse=0.), dict(converged=True, weighted_sse=10.),
                dict(converged=True, weighted_sse=9.)]
    assert model.select_attempt(attempts)['selected_start'] == 2
    assert not model.select_attempt(attempts[:1])['converged']


def test_noise_roles_and_validation_are_disjoint(cfg):
    gcfg = runner.gate.load_config()
    _, train = runner.gate.steering(gcfg, 'broadband')
    _, validation = runner.signal(cfg, 'broadband')
    assert not np.array_equal(train, validation)
    a = runner.noise(cfg, 0, 21, 0, 1, 100)
    np.testing.assert_array_equal(a, runner.noise(cfg, 0, 21, 0, 1, 100))
    assert not np.array_equal(a, runner.noise(cfg, 0, 21, 0, 2, 100))


def test_run_resume_restore_and_report(tmp_path, monkeypatch, cfg):
    """Tiny synthetic orchestration test; real simulation/control, stub optimizer.

    This exercises every method/status/report path without running scientific fits.
    Checkpointed starts and conditions must never be repeated during resume.
    """
    cfg = copy.deepcopy(cfg); cfg['duration_s'] = .2
    cfg['training_speeds_mps'] = [21]; cfg['evaluation_speeds_mps'] = [21]
    cfg['fit']['length_starts_m'] = [[.3, .9]]; cfg['fit']['profile_factors'] = [1.]
    monkeypatch.setattr(runner, 'OUT', tmp_path/'tables'); monkeypatch.setattr(runner, 'RAW', tmp_path/'raw')
    monkeypatch.setattr(runner, 'REPORT', tmp_path/'report.html'); monkeypatch.setattr(runner, 'preflight', lambda: cfg)
    calls = []
    def fit(records, geometry, settings, start, free):
        calls.append(1)
        return dict(converged=True, parameters=list(start), weighted_sse=1., weighted_mse=1., nfev=1)
    monkeypatch.setattr(model, 'fit_attempt', fit)
    def generate(job, gate_cfg):
        c = next(c for c in cfg['clients'] if c['client'] == job[0])
        t, u = runner.signal(cfg, 'broadband')
        x = model.batch_predict(c['physical_ratios']+cfg['relaxation_lengths_m'], [u], [21.], c['geometry_m'], .01, .001)[0, 0]
        return {}, dict(time=t, steering=u, state=x, measured_yaw=x[:, 1], noise=np.zeros(len(t)))
    monkeypatch.setattr(runner.gate, 'generate', generate)
    def training(c, cfg):
        _, arrays = generate((c['client'], 21, 'broadband'), cfg)
        return [dict(speed=21., applied_input=arrays['steering'], measured_yaw=arrays['measured_yaw'])], {'21': runner.store_trace(c['client'], 'training_v21', arrays)}
    monkeypatch.setattr(runner, 'training', training)
    for c in cfg['clients']:
        runner.worker(c['client'])
    assert len(calls) == 9
    nominal_path = runner.path('nominal_00'); p = runner.read_checkpoint(nominal_path)
    p['status'] = 'running'; removed = next(iter(p['control'])); del p['control'][removed]
    runner.atomic_json(nominal_path, p)
    runner.worker('nominal_00'); assert len(calls) == 9
    runner.worker('nominal_00'); assert len(calls) == 9
    dest = runner.RAW/'nominal_00'/'training_v21.npz'; dest.unlink()
    runner.restore(cfg, ['nominal_00']); assert dest.exists()
    runner.analyze(cfg)
    html = runner.REPORT.read_text(encoding='utf-8')
    assert html.count('<svg') == 2 and '\ufffd' not in html
    # Analysis does not mutate the frozen execution checkpoint.
    assert runner.read_checkpoint(nominal_path)['status'] == 'execution_complete'
    incomplete = runner.read_checkpoint(nominal_path)
    incomplete['control'].pop(next(iter(incomplete['control'])))
    with pytest.raises(ValueError, match='missing/extra'):
        runner.validate(incomplete, cfg, 'nominal_00')
    # Reports must remain available even when every controller fails synthesis.
    for c in cfg['clients']:
        saved = runner.read_checkpoint(runner.path(c['client']))
        for record in saved['control'].values():
            record.update(status='synthesis_failed', metrics=None, feasible=False)
            record.pop('preview', None)
        runner.atomic_json(runner.path(c['client']), saved)
    runner.analyze(cfg)
    summary = runner.pd.read_csv(runner.OUT/f'{runner.PREFIX}_summary.csv')
    control = summary[summary.stage == 'control']
    assert (control.complete == 0).all() and control.mean_rmse_deg_s.isna().all()


@pytest.mark.parametrize('n', [4, 6])
def test_joint_clean_synthetic_fit(cfg, n):
    p = np.array([50., 55., .6, .9, .4, .7])[:n]
    speeds = [13., 21., 29.]; t = np.arange(120)*.01
    u = np.array([.006*np.sin(2*np.pi*.8*t)]*3)
    x = model.batch_predict(p, u, speeds, [1.2, 1.6], .01, .001)
    records = [dict(speed=v, applied_input=u[k], measured_yaw=x[0, k, :, 1]) for k, v in enumerate(speeds)]
    fit = model.fit_attempt(records, [1.2, 1.6], dict(runner.settings(cfg), max_nfev=35), p*1.01, list(range(n)))
    assert fit['converged'] and fit['weighted_mse'] < 1e-10
    np.testing.assert_allclose(fit['parameters'], p, rtol=1e-3)


def test_actual_plant_origin_jacobian(cfg):
    p = np.array([50., 55., .6, .9, .4, .7]); v = 21.; geom = [1.2, 1.6]
    ccfg = runner.control_settings(cfg)
    from federated_lpv.tire_relaxation import sampled_matrices
    ap, bp = sampled_matrices(p, v, geom, .01)
    for pars in [p[:4], p]:
        d = model.design(pars, v, geom, ccfg, 1); n = len(d['b'])
        steady = np.array(d['diagnostics']['steady_observer_gain'])
        d['observer_gains'][0] = steady
        def step(z):
            x, prior, integral = z[:4], z[4:4+n], z[-1]
            _, next_prior, next_integral, command, _ = model.feedback(prior, integral, 0., x[1], 0., d, 0, ccfg)
            return np.r_[ap@x+bp*command, next_prior, next_integral]
        eps = 1e-7; matrix = np.column_stack([(step(eps*e)-step(-eps*e))/(2*eps) for e in np.eye(5+n)])
        np.testing.assert_allclose(model.closed_loop_matrix(p, v, geom, d, ccfg), matrix, rtol=1e-9, atol=1e-10)
