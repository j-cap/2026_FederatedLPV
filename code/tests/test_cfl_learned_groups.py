"""Matched peer records, training-only grouping, provenance and crash recovery."""
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'code/experiments'), str(ROOT/'code/src')]
import paper_cfl_learned_groups as run


@pytest.fixture(scope='module')
def cfg():
    return run.preflight()


def test_same_speed_control_preserves_vehicles_targets_and_budget(cfg):
    seed = cfg['simulation']['seeds'][0]
    original_clients = run.clients_for(cfg, 'development', seed)
    with run.scenario(cfg, 'development', 'same13') as flat:
        same = run.original.fleet_for(flat, seed)['clients']
        assert len(same) == len(original_clients) == 18
        assert flat['fleet_training_speeds'] == [13]
        for old, new in zip(original_clients, same):
            assert new['training_speed'] == 13
            for key in ['client','family','geometry_m','physical_parameters','physical_draw','mixed_group']:
                assert old[key] == new[key]
        assert len(run.target_ids(cfg, 'development', seed, 'same13')) == 6
        target = next(c for c in original_clients if c['training_speed'] == 13)
        transformed = next(c for c in same if c['client'] == target['client'])
        short = dict(flat, duration_s=.12, taper_s=.03)
        a, _ = run.original.base.generate(target, seed, 'training', 13, short)
        b, _ = run.original.base.generate(transformed, seed, 'training', 13, short)
        assert run.storage.trace_hash(a) == run.storage.trace_hash(b)
    assert run.original.PREFIX == run.SOURCE_PREFIX


def test_group_initialization_ignores_unregularized_friction(cfg):
    q = np.repeat(np.log(cfg['simulation']['starts']), 3, axis=0)
    modified = q.copy()
    modified[:,3] += [3., -2., 6., -4., 5., 0., 7., -1., 2.]
    weights = cfg['simulation']['sharing']['weights']
    a = run.fl.initialize_personal(q, 3, 1701, weights)
    b = run.fl.initialize_personal(modified, 3, 1701, weights)
    assert a['assignments'] == b['assignments']


def test_restart_selection_uses_converged_training_objective_only():
    def candidate(objective, status='converged', test_error=None):
        return dict(status='complete', test_error=test_error,
                    state=dict(status=status, round=3, history=[dict(objective=objective)]))
    candidates = {'r0':candidate(3., test_error=0.), 'r1':candidate(2., test_error=999.),
                  'r2':candidate(1., status='budget_exhausted')}
    assert run.choose_restart(candidates)['selected'] == 'r1'
    with pytest.raises(ValueError, match='No converged'):
        run.choose_restart({'r2':candidates['r2']})


def test_compatibility_and_coverage_are_separate_decisions(cfg):
    fleet = pd.DataFrame(dict(LocalOE=[1.1]*3, NominalPrior=[1.]*3, SharedPrior=[1.]*3,
                              OracleFamily=[.7]*3, MixedGroups=[1.05]*3, LearnedGroups=[.8]*3))
    paired = pd.DataFrame(dict(complement=[.8]*3, same13=[1.]*3))
    verdict = run.decision(fleet, paired, True, cfg, 'development')
    assert verdict['compatible_sharing_passed'] and verdict['complementary_coverage_passed']
    paired.complement = 1.1
    verdict = run.decision(fleet, paired, True, cfg, 'development')
    assert verdict['status'] == 'compatible_sharing_only'
    assert not run.decision(fleet, paired, False, cfg, 'development')['compatible_sharing_passed']


@pytest.fixture
def tiny(tmp_path, monkeypatch, cfg):
    config = copy.deepcopy(cfg)
    config['confirmation_fleets'] = config['confirmation_fleets'][:1]
    config['confirmation_seeds'] = config['confirmation_seeds'][:1]
    config['simulation'].update(duration_s=.12, taper_s=.03, evaluation_speeds=[13,21,29])
    config['simulation']['sharing']['personal_max_rounds'] = 3
    provenance = run.identity()
    monkeypatch.setattr(run, 'OUT', tmp_path/'tables')
    monkeypatch.setattr(run, 'RAW', tmp_path/'raw')
    monkeypatch.setattr(run, 'REPORT', tmp_path/'report')
    monkeypatch.setattr(run, 'ROOT', tmp_path)
    monkeypatch.setattr(run, 'identity', lambda:provenance)
    calls = []
    def fit(self, start, center=None, weights=None, max_nfev=None):
        calls.append(1)
        q = np.log(config['simulation']['nominal_parameters']) + self.geometry[0]*np.array([.002,-.001,.001,0,0,0])
        eta = q if center is None else np.asarray(center)
        w = np.zeros(6) if weights is None else np.asarray(weights)
        penalty = float(np.sum((w*(q-eta))**2))
        return dict(available=True, optimizer_converged=True, log_parameters=q.tolist(), parameters=np.exp(q).tolist(),
            objective=1.+penalty, data_sse=1., prior_penalty=penalty, initial_objective=1.+penalty,
            nfev=1, optimality=0., near_bound=[False]*6)
    monkeypatch.setattr(run.fl.Client, 'fit', fit)
    return config, provenance, calls


def acquire(tiny, condition='complement'):
    cfg, provenance, calls = tiny
    seed = cfg['confirmation_seeds'][0]
    for c in run.clients_for(cfg, 'confirmation', seed):
        run.execute(('local','confirmation',condition,seed,c['client']), cfg, provenance)
    return seed


def test_mid_round_resume_uses_saved_updates(tiny, monkeypatch):
    cfg, provenance, calls = tiny
    seed = acquire(tiny)
    key = 'LearnedGroups_r1701'
    saved = run.original.save
    def interrupt(path, payload, stage):
        saved(path, payload, stage)
        if stage == 'round1_client0':
            raise RuntimeError('interrupted')
    monkeypatch.setattr(run.original, 'save', interrupt)
    with pytest.raises(RuntimeError, match='interrupted'):
        run.execute(('group','confirmation','complement',seed,key),cfg,provenance)
    monkeypatch.setattr(run.original, 'save', saved)
    before = len(calls)
    run.execute(('group','confirmation','complement',seed,key),cfg,provenance)
    assert len(calls)-before == 35  # Remaining 17 updates, then one full 18-client round.
    p = run.load(run.checkpoint_path('confirmation','complement',seed,'group',key),provenance)
    assert p['state']['status'] == 'converged'


def test_family_labels_do_not_change_learned_updates(tiny):
    cfg, provenance, calls = tiny
    seed = acquire(tiny)
    run.execute(('group','confirmation','complement',seed,'LearnedGroups_r1701'),cfg,provenance)
    altered = copy.deepcopy(cfg)
    for c in altered['confirmation_fleets'][0]['clients']:
        c['family'] = 'hidden'
    # Distinct restart key, same numerical initialization seed for the comparison.
    with run.scenario(altered,'confirmation','complement') as flat:
        original_state = run.load(run.original.path(seed,'group','LearnedGroups_r1701'),provenance)['state']
        path = run.original.path(seed,'group','LearnedGroups_r1701')
        original_bytes = path.read_bytes()
        path.unlink()
        run.learned_job(seed,'LearnedGroups_r1701',dict(altered,flat=flat),provenance)
        hidden_state = run.load(path,provenance)['state']
        assert original_state == hidden_state
        path.write_bytes(original_bytes)


def test_unchanged_target_cache_moves_without_simulation(tiny, monkeypatch):
    cfg, provenance, calls = tiny
    seed = acquire(tiny)
    key = run.target_ids(cfg,'confirmation',seed,'same13')[0]
    before = len(calls)
    def forbidden(*args, **kwargs):
        raise AssertionError('An unchanged target must use its existing record')
    monkeypatch.setattr(run.original.base,'generate',forbidden)
    run.execute(('local','confirmation','same13',seed,key),cfg,provenance)
    with run.scenario(cfg,'confirmation','same13') as flat:
        lp = run.load(run.original.path(seed,'local',key),provenance)
        c = run.original.client_for(flat,seed,key)
        _, record = run.original.training(c,seed,flat,lp['data'])
        assert record['trace_sha256'] == lp['data']['trace_sha256']
    assert len(calls) == before


def test_cached_original_evaluation_is_reused_exactly(tiny, monkeypatch):
    cfg, provenance, calls = tiny
    flat = cfg['simulation']
    c = flat['fleets'][0]['clients'][0]
    seed = flat['seeds'][0]
    pars = {m:flat['nominal_parameters'] for m in run.BASE_METHODS}
    pars['Exact'] = c['physical_parameters']
    statuses = {m:'converged' for m in pars}
    statuses['Exact'] = 'reference'
    old, arrays = run.original.evaluate(c,seed,29,'slow',flat,pars,statuses)
    cache = run.ROOT/'results/data'/run.SOURCE_PREFIX/str(seed)/f'{c["client"]}_v29_slow.npz'
    run.storage.save_arrays(cache,arrays)
    source = run.OUT/f'{run.SOURCE_PREFIX}_seed{seed}_eval_{c["client"]}.json'
    run.storage.atomic_json(source,dict(cases={'v29_slow':old}))
    original_hash = run.file_hash(source)
    def forbidden(*args, **kwargs):
        raise AssertionError('Original arrays must not be regenerated when cached')
    monkeypatch.setattr(run.original.base,'generate',forbidden)
    pars['LearnedGroups'] = flat['nominal_parameters']
    statuses['LearnedGroups'] = 'converged'
    new, reconstructed = run.evaluate(seed,c,29,'slow',dict(cfg,flat=flat),pars,statuses,'development','complement')
    assert new['methods']['NominalPrior'] == old['methods']['NominalPrior']
    assert run.storage.trace_hash({k:v for k,v in reconstructed.items() if k!='predicted_LearnedGroups'}) == old['trace_sha256']
    assert run.file_hash(source) == original_hash


def test_confirmation_requires_freeze_and_rejects_changed_evidence(tiny):
    cfg, provenance, _ = tiny
    with pytest.raises(ValueError, match='gated'):
        run.require_confirmation_lock(cfg, provenance)
    run.OUT.mkdir(parents=True)
    execution = run.summary_path('development','execution')
    run.storage.atomic_json(execution, dict(status='execution_complete',source_sha256={}))
    analysis = run.summary_path('development','analysis')
    run.storage.atomic_json(analysis, dict(provenance=provenance, decision=dict(compatible_sharing_passed=True),
        output_sha256={}, execution_sha256=run.file_hash(execution)))
    run.freeze(cfg)
    run.require_confirmation_lock(cfg,provenance)
    run.storage.atomic_json(execution,dict(changed=True))
    with pytest.raises(ValueError,match='evidence changed'):
        run.require_confirmation_lock(cfg,provenance)


def test_runtime_mismatch_blocks_resume_but_allows_scalar_analysis(tiny):
    _, provenance, _ = tiny
    path = run.OUT/'runtime.json'
    run.storage.atomic_json(path,dict(status='complete',provenance=provenance))
    changed = copy.deepcopy(provenance)
    changed['runtime']['system'] = 'different_system'
    with pytest.raises(ValueError,match='runtime mismatch'):
        run.load(path,changed)
    assert run.load(path,changed,strict_runtime=False)['status'] == 'complete'


def test_workflow_reuse_restore_analysis_and_stale_parent(tiny, monkeypatch):
    cfg, provenance, calls = tiny
    seed = acquire(tiny)
    before = len(calls)
    acquire(tiny,'same13')
    assert len(calls)-before == 12*5, 'Unchanged target fits must be reused'
    for condition in cfg['conditions']:
        with run.scenario(cfg,'confirmation',condition) as flat:
            keys = run.group_keys(flat,seed,cfg)
        for key in keys:
            run.execute(('group','confirmation',condition,seed,key),cfg,provenance)
        with run.scenario(cfg,'confirmation',condition) as flat:
            run.selection_job(seed,dict(cfg,flat=flat),provenance)
    for condition in cfg['conditions']:
        for key in run.target_ids(cfg,'confirmation',seed,condition):
            run.execute(('eval','confirmation',condition,seed,key),cfg,provenance)
    before = len(calls)
    acquire(tiny)
    acquire(tiny,'same13')
    assert len(calls) == before
    source_paths = [p for condition in cfg['conditions'] for p in run.OUT.glob(f'{run.prefix("confirmation",condition)}_seed*.json')]
    run.storage.atomic_json(run.summary_path('confirmation','execution'),dict(status='execution_complete',
        provenance=provenance,source_sha256={p.relative_to(run.ROOT).as_posix():run.file_hash(p) for p in source_paths}))
    scalar_runtime = copy.deepcopy(provenance)
    scalar_runtime['runtime']['system'] = 'scalar_analysis_system'
    with monkeypatch.context() as context:
        context.setattr(run, 'identity', lambda:scalar_runtime)
        run.analyze(cfg,'confirmation')
    assert (run.REPORT/'cfl_learned_groups_confirmation.html').exists()
    analysis = run.storage.read_checkpoint(run.summary_path('confirmation','analysis'))
    assert analysis['provenance'] == provenance
    assert analysis['analysis_runtime'] == scalar_runtime['runtime']
    hashes = {p:run.file_hash(p) for p in run.OUT.glob('*.json')}
    c = cfg['confirmation_fleets'][0]['clients'][0]
    raw = run.RAW/'confirmation'/'complement'/str(seed)
    (raw/f'{c["client"]}_training.npz').unlink()
    (raw/f'{c["client"]}_v21_slow.npz').unlink()
    run.restore(cfg,'confirmation')
    assert hashes == {p:run.file_hash(p) for p in hashes}
    assert len(calls) == before
    parent = run.checkpoint_path('confirmation','complement',seed,'local',c['client'])
    p = run.storage.read_checkpoint(parent); p['tampered'] = True
    run.storage.atomic_json(parent,p)
    with pytest.raises(ValueError,match='parent'):
        run.execute(('eval','confirmation','complement',seed,c['client']),cfg,provenance)
