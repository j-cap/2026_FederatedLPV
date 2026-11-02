"""OE-to-LPV boundaries: immutable inputs, no accuracy filtering, matched pooling."""
import copy
import json
from pathlib import Path
import sys

import numpy as np
import unittest
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'code/src'), str(ROOT / 'code/experiments')]
import paper_oe_lpv_integration as integration
from experiment_10e_federated_latent_groups import client_statistics, solve_statistics


def setup():
    return integration.preflight()


def saved_inputs(setup):
    cfg, basis, mixture, acquisition_cfg, lock = setup
    acquisition = integration.acquisition
    result = acquisition.read_result(acquisition.path_for(1301), lock['acquisition_provenance'])
    source = acquisition.oe.known.source_result(1301, 'gentle', lock['acquisition_provenance']['oe_provenance']['known_provenance'])
    return result, source, acquisition_cfg


def check_frozen_scope_reuses_estimates_and_reserves_confirmation(setup):
    cfg, basis, mixture, acquisition_cfg, lock = setup
    assert cfg['seeds'] == [1301, 1302, 1303]
    assert not set(cfg['seeds']) & set(acquisition_cfg['confirmation_seeds'])
    assert cfg['smooth_strength'] == 1e6
    assert mixture['candidate_clusters'] == [1, 2, 3, 4, 5]
    assert lock['confirmation_unlocked'] is False


def check_local_targets_exactly_reused_without_accuracy_selection(saved_inputs):
    result, source, cfg = saved_inputs
    before = integration.parent.canonical(result)
    speeds, targets, diagnostics = integration.assemble_inputs(result, source, 'ClosedOE4', cfg)
    assert sum(map(len, speeds)) == 83 and len(targets) == 30
    # A failed reference-accuracy flag must not remove a usable local estimate.
    changed = copy.deepcopy(result)
    changed['cells'][0]['models']['ClosedOE4']['accuracy_pass'] = False
    changed['cells'][0]['models']['ClosedOE4']['matrix_error'] = 123.
    other_speeds, other_targets, other_diag = integration.assemble_inputs(changed, source, 'ClosedOE4', cfg)
    for a, b in zip(targets, other_targets):
        np.testing.assert_array_equal(a, b)
    assert len(other_diag) == len(diagnostics)
    assert before == integration.parent.canonical(result)
    assert not all(d['original_physical_pass'] for d in diagnostics)


def check_bad_inputs_stop_whole_fleet(saved_inputs, mutation):
    result, source, cfg = saved_inputs
    result = copy.deepcopy(result)
    c = result['cells'][0]
    if mutation == 'duplicate':
        result['cells'].append(copy.deepcopy(c))
    elif mutation == 'missing':
        result['cells'].pop()
    elif mutation == 'invalid':
        c['models']['ClosedOE4']['fit_valid'] = False
    elif mutation == 'bounds':
        c['training']['max_ay'] = 100.
    elif mutation == 'nan':
        c['models']['ClosedOE4']['target'][0] = float('nan')
    else:
        c['client_index'] = 99
    with unittest.TestCase().assertRaises(ValueError):
        integration.assemble_inputs(result, source, 'ClosedOE4', cfg)


def check_pooled_augmented_solve_matches_client_statistics_for_fixed_partition(setup):
    _, basis, _, _, _ = setup
    rng = np.random.default_rng(23)
    for observed in [[10, 12, 14], list(range(10, 31, 2))]:
        speeds = [np.asarray(observed, dtype=float)] * 4
        targets = [rng.normal(size=(len(observed), 6)) for _ in speeds]
        stats = [client_statistics(s, y, basis) for s, y in zip(speeds, targets)]
        fed = solve_statistics(stats, np.arange(4), 1e-8)
        central = integration.central_ridge(np.concatenate(speeds), np.vstack(targets), basis, 1e-8)
        np.testing.assert_allclose(integration.predict(fed, basis['speed_grid'], basis),
                                   integration.predict(central, basis['speed_grid'], basis), atol=1e-6, rtol=1e-7)
        assert integration.information(np.concatenate(speeds), basis)['rank'] == min(len(observed), 7)


def check_region_average_weights_clients_equally():
    grid = [10, 12, 14, 16]
    clients = [dict(speeds=[10, 12]), dict(speeds=[10, 12, 14])]
    values = [[100, 100, 1, 1], [100, 100, 100, 9]]
    assert integration.fleet_mean(values, clients, grid, 'unseen') == 5.
    assert integration.fleet_mean(values, clients, grid, 'seen') == 100.
    np.testing.assert_array_equal(integration.region_mask(grid, [10, 12], 'unseen'), [False, False, True, True])


def check_checkpoint_corruption_and_stale_provenance_rejected(tmp_path):
    lock = {'version': 1}
    payload = dict(seed=1301, status='execution_complete', provenance=lock)
    integration.save_result(payload, tmp_path)
    path = integration.path_for(1301, tmp_path)
    assert integration.read_result(path, 1301, lock) == payload
    with unittest.TestCase().assertRaisesRegex(ValueError, 'Stale'):
        integration.read_result(path, 1301, {'version': 2})
    saved = json.loads(path.read_text(encoding='utf-8'))
    saved['payload']['seed'] = 1302
    path.write_text(json.dumps(saved), encoding='utf-8')
    with unittest.TestCase().assertRaisesRegex(ValueError, 'Corrupt'):
        integration.read_result(path, 1301, lock)


def check_diagnostic_labels_and_references_do_not_mutate_primary_fits(setup):
    cfg, basis, mixture, _, _ = setup
    speeds = [np.array([10., 20., 30.])] * 4
    targets = [np.tile(np.array([.9, .01, .1, .95, .02, .3]), (3, 1)) * (1+.01*i) for i in range(4)]
    mixture = {**mixture, 'candidate_clusters': [1], 'minimum_cluster_size': 2}
    with integration.threadpool_limits(limits=1):
        fitted = integration.fit_models(speeds, targets, 1300, cfg, basis, mixture)
    before = integration.parent.canonical(fitted)
    clients = [dict(family=f'arbitrary{i%2}', reference=np.tile(targets[i][0]*3, (11, 1)).tolist()) for i in range(4)]
    integration.add_diagnostic_models(fitted, clients, speeds, targets, basis, mixture)
    original = json.loads(before)
    for key in original['models']:
        assert fitted['models'][key] == original['models'][key]
    for key in ['selected_k', 'candidates', 'equivalence', 'smooth_sensitivity']:
        assert fitted[key] == original[key]


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.setup = setup()
        cls.inputs = saved_inputs(cls.setup)

    def test_frozen_scope(self):
        check_frozen_scope_reuses_estimates_and_reserves_confirmation(self.setup)

    def test_exact_reuse_without_accuracy_selection(self):
        check_local_targets_exactly_reused_without_accuracy_selection(self.inputs)

    def test_bad_inputs(self):
        for mutation in ['duplicate', 'missing', 'invalid', 'bounds', 'nan', 'identity']:
            with self.subTest(mutation=mutation):
                check_bad_inputs_stop_whole_fleet(self.inputs, mutation)

    def test_pooled_equivalence(self):
        check_pooled_augmented_solve_matches_client_statistics_for_fixed_partition(self.setup)

    def test_equal_client_weights(self):
        check_region_average_weights_clients_equally()

    def test_checkpoint_integrity(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            check_checkpoint_corruption_and_stale_provenance_rejected(Path(folder))

    def test_diagnostic_isolation(self):
        check_diagnostic_labels_and_references_do_not_mutate_primary_fits(self.setup)

    def test_synthetic_execution_checkpoint_and_report(self):
        cfg, basis, mixture, acquisition_cfg, lock = self.setup
        cfg = {**cfg, 'seeds': [1300]}
        mixture = {**mixture, 'candidate_clusters': [1], 'minimum_cluster_size': 2}
        grid = basis['speed_grid']
        clients, cells = [], []
        for i in range(4):
            observed = [[10., 12., 14.], [16., 18., 20.], [22., 24., 26.], [28., 30.]][i]
            truth = [[v*(1+.01*i) for v in [.94, -.01, .1, .93, .02, .3]] for _ in grid]
            clients.append(dict(client=f'fixture{i}', family=f'family{i%2}', parameters={},
                                speeds=observed, reference=truth))
            for speed in observed:
                target = truth[grid.index(speed)]
                cell = dict(client=f'fixture{i}', client_index=i, speed=speed, parameters={},
                            reference=target, status='simulated', physical_pass=False,
                            training=dict(max_beta_deg=1., max_ay=1., max_steering_deg=1., max_rate_deg_s=30., settling_pass=True),
                            validation=dict(max_beta_deg=1., max_ay=1., max_steering_deg=1., max_rate_deg_s=30., settling_pass=True),
                            tracking_train=dict(limiter_fraction=.05), tracking_validation=dict(limiter_fraction=.05),
                            models={c: dict(target=target, fit_valid=True, accuracy_pass=True, matrix_error=0.) for c in cfg['conditions']})
                cells.append(cell)
        with patch.object(integration.acquisition, 'read_result', return_value=dict(cells=cells)), patch.object(integration.acquisition.oe.known, 'source_result', return_value=dict(clients=clients)):
            payload = integration.run_payload(1300, cfg, basis, mixture, acquisition_cfg, lock)
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            directory = Path(folder)
            integration.save_result(payload, directory)
            saved = integration.read_result(integration.path_for(1300, directory), 1300, lock)
            self.assertEqual(saved, payload)
            with patch.object(integration, 'OUT', directory), patch.object(integration, 'REPORT', directory/'report.html'), patch.object(integration, 'preflight', return_value=(cfg, basis, mixture, acquisition_cfg, lock)), patch.object(integration.parent, 'tracked_unchanged', return_value=True):
                integration.analyze()
            report = (directory/'report.html').read_text(encoding='utf-8')
            self.assertEqual(report.count('<svg'), 3)
            self.assertTrue((directory/'paper_oe_lpv_integration_fleet_summary.csv').exists())
            self.assertFalse(saved['confirmation_unlocked'])
            self.assertFalse(saved['control_benefit_tested'])
            self.assertTrue(all(f['equivalence']['passed'] for f in saved['conditions'].values()))

    def test_degenerate_group_scaling_is_explicit(self):
        cfg, basis, mixture, _, _ = self.setup
        speeds = [np.array([10., 20., 30.])] * 4
        targets = [np.zeros((3, 6)) for _ in speeds]
        with self.assertRaisesRegex(ValueError, 'Degenerate target scaling'):
            integration.fit_models(speeds, targets, 1300, cfg, basis, mixture)

    def test_smoke_report_with_windows_default_encoding(self):
        original_open = Path.open

        def windows_open(path, mode='r', buffering=-1, encoding=None, errors=None, newline=None):
            if 'b' not in mode and encoding in (None, 'locale'):
                encoding = 'cp1252'
            return original_open(path, mode, buffering, encoding, errors, newline)

        # Reproduce Windows text decoding while preserving explicit UTF-8 calls.
        with patch.object(Path, 'open', new=windows_open):
            result = integration.smoke_worker()
        self.assertEqual(result['status'], 'smoke_passed')

    def test_checkpoint_before_analysis(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(integration, 'OUT', Path(folder)), patch.object(integration, 'preflight', return_value=self.setup):
            with self.assertRaisesRegex(ValueError, 'commit and push'):
                integration.analyze()


if __name__ == '__main__':
    unittest.main(verbosity=2)
