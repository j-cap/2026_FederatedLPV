r"""Learned compatible priors and matched same-speed peers.

Windows, from the repository root:
    python code\experiments\paper_cfl_learned_groups.py preflight
    python code\experiments\paper_cfl_learned_groups.py smoke
    python code\experiments\paper_cfl_learned_groups.py run --workers 5
    python code\experiments\paper_cfl_learned_groups.py analyze

Run saves execution checkpoints first. Analyze is a separate, read-only fit audit
and report step. Repeating run resumes without rerunning completed work.
"""
from __future__ import annotations

import os
for variable in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[variable] = '1'
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
import copy
import html
import json
import multiprocessing
from pathlib import Path
import platform
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'code/src'), str(ROOT / 'code/experiments')]
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits
import paper_cfl_heterogeneity as original

PREFIX = 'paper_cfl_learned_groups'
CONFIG = ROOT / 'code/config' / f'{PREFIX}.json'
OUT = ROOT / 'results/tables'
RAW = ROOT / 'results/data' / PREFIX
REPORT = ROOT / 'report'
SOURCE_PREFIX = original.PREFIX
BASE_METHODS = original.METHODS
METHODS = BASE_METHODS + ['LearnedGroups']
storage = original.storage
fl = original.fl
file_hash = original.base.canonical_hash


def validate_fleet(saved, regenerated):
    """Keep frozen draws; tolerate only roundoff when checking another platform."""
    assert saved.keys() == regenerated.keys(), 'Fleet fields changed'
    assert saved['seed'] == regenerated['seed'], 'Fleet seed changed'
    assert len(saved['clients']) == len(regenerated['clients']), 'Fleet size changed'
    for actual, expected in zip(saved['clients'], regenerated['clients']):
        assert actual.keys() == expected.keys(), 'Client fields changed'
        for name in ['client', 'family', 'training_speed', 'mixed_group']:
            assert actual[name] == expected[name], f'Population assignment changed: {name}'
        for name in ['geometry_m', 'physical_parameters']:
            np.testing.assert_allclose(actual[name], expected[name], rtol=1e-12, atol=1e-14,
                                       equal_nan=False, err_msg=f'Frozen vehicle values changed: {name}')
        assert actual['physical_draw'].keys() == expected['physical_draw'].keys(), 'Physical draw fields changed'
        for name in actual['physical_draw']:
            np.testing.assert_allclose(actual['physical_draw'][name], expected['physical_draw'][name],
                rtol=1e-12, atol=1e-14, equal_nan=False, err_msg=f'Frozen physical draw changed: {name}')


def preflight():
    cfg = json.loads(CONFIG.read_text(encoding='utf-8'))
    for name, sha in cfg['source_sha256'].items():
        if file_hash(ROOT / name) != sha:
            raise ValueError(f'Frozen source changed: {name}')
    cfg['simulation'] = original.preflight()
    assert cfg['learned']['groups'] == 3
    assert len(set(cfg['learned']['restart_seeds'])) == 3
    assert cfg['simulation']['sharing']['weights'][3] == 0
    assert cfg['conditions'] == ['complement', 'same13']
    assert [f['seed'] for f in cfg['confirmation_fleets']] == cfg['confirmation_seeds']
    assert len(set(cfg['confirmation_seeds'])) == 10
    assert set(cfg['confirmation_seeds']).isdisjoint(cfg['simulation']['seeds'])
    bounds = np.array(cfg['simulation']['fit']['bounds'])
    for fleet in cfg['confirmation_fleets']:
        expected = original.population.generate_fleet(fleet['seed'], cfg['simulation']['population'],
                                                       cfg['simulation']['fleet_training_speeds'])
        validate_fleet(fleet, expected)
        for c in fleet['clients']:
            assert np.all(np.array(c['physical_parameters']) > bounds[:, 0])
            assert np.all(np.array(c['physical_parameters']) < bounds[:, 1])
    return cfg


def identity():
    previous = original.identity()
    return dict(files={**previous['files'], CONFIG.relative_to(ROOT).as_posix(): file_hash(CONFIG),
                       Path(__file__).relative_to(ROOT).as_posix(): file_hash(Path(__file__))},
                runtime=dict(previous['runtime'], python_full=platform.python_version(),
                             system=platform.system(), machine=platform.machine()))


def fleets(cfg, phase):
    return cfg['simulation']['fleets'] if phase == 'development' else cfg['confirmation_fleets']


def prefix(phase, condition):
    return f'{PREFIX}_{phase}_{condition}'


def checkpoint_path(phase, condition, seed, stage, key):
    return OUT / f'{prefix(phase, condition)}_seed{seed}_{stage}_{key}.json'


def summary_path(phase, kind):
    return OUT / f'{PREFIX}_{phase}_{kind}.json'


def clients_for(cfg, phase, seed):
    return next(f['clients'] for f in fleets(cfg, phase) if f['seed'] == seed)


def target_ids(cfg, phase, seed, condition):
    return [c['client'] for c in clients_for(cfg, phase, seed)
            if condition == 'complement' or c['training_speed'] == 13]


@contextmanager
def scenario(cfg, phase, condition):
    """One process handles one job at a time; original evidence stays untouched."""
    saved = {k: getattr(original, k) for k in ['PREFIX', 'OUT', 'RAW']}
    original.PREFIX = prefix(phase, condition)
    original.OUT = OUT
    original.RAW = RAW / phase / condition
    flat = copy.deepcopy(cfg['simulation'])
    flat['fleets'] = copy.deepcopy(fleets(cfg, phase))
    flat['seeds'] = [f['seed'] for f in flat['fleets']]
    if condition == 'same13':
        flat['fleet_training_speeds'] = [13]
        for fleet in flat['fleets']:
            for client in fleet['clients']:
                client['training_speed'] = 13
    try:
        yield flat
    finally:
        for key, value in saved.items():
            setattr(original, key, value)


def load(path, provenance, strict_runtime=True, visited=None):
    """Check payload, fitting code, runtime and the full checkpoint parent chain."""
    p = original.load(path, provenance, strict_runtime)
    seen = set() if visited is None else visited
    if path in seen:
        return p
    seen.add(path)
    for name, sha in p.get('parents', {}).items():
        parent = OUT / name
        if file_hash(parent) != sha:
            raise ValueError(f'Checkpoint parent changed: {name}')
        if name.startswith(PREFIX + '_'):
            load(parent, provenance, strict_runtime, seen)
        else:
            storage.read_checkpoint(parent)
    return p


def inherit(path, source, provenance, parents=None):
    """Save an explicitly attributed copy, with no optimizer call."""
    if path.exists():
        p = load(path, provenance)
        if p.get('reused_from') != dict(name=source.name, sha256=file_hash(source)):
            raise ValueError('Reused source changed; checkpoint preserved')
        return dict(key=path.name, status='already_complete')
    prior = storage.read_checkpoint(source)
    if prior['status'] != 'complete':
        raise ValueError('Cannot reuse an unfinished source')
    p = copy.deepcopy(prior)
    p['provenance'] = provenance
    p['source_implementation_commit'] = prior['implementation_commit']
    p['implementation_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=original.ROOT, text=True).strip()
    p['reused_from'] = dict(name=source.name, sha256=file_hash(source))
    p['parents'] = {source.name: file_hash(source), **(parents or {})}
    storage.atomic_json(path, p)
    return dict(key=path.name, status='reused', source=source.name)


def reuse_training_cache(path, source_cache):
    """Use the original recorded arrays when available, including across OSes."""
    p = storage.read_checkpoint(path)
    target = original.RAW / str(p['seed']) / f'{p["key"]}_training.npz'
    cached = target if target.exists() else source_cache
    if not cached.exists():
        return  # The normal training loader regenerates and checks the saved hash.
    with np.load(cached, allow_pickle=False) as saved:
        arrays = dict(saved)
    if storage.trace_hash(arrays) != p['data']['trace_sha256']:
        raise ValueError('Cached training data differ from the frozen record')
    if not target.exists():
        storage.save_arrays(target, arrays)


def local_job(seed, key, cfg, provenance, phase, condition):
    path = original.path(seed, 'local', key)
    if phase == 'development' and condition == 'complement':
        with original.task_lock(path):
            result = inherit(path, OUT / f'{SOURCE_PREFIX}_seed{seed}_local_{key}.json', provenance)
            reuse_training_cache(path, ROOT / 'results/data' / SOURCE_PREFIX / str(seed) / f'{key}_training.npz')
            return result
    source_client = next(c for c in clients_for(cfg, phase, seed) if c['client'] == key)
    if condition == 'same13' and source_client['training_speed'] == 13:
        source = checkpoint_path(phase, 'complement', seed, 'local', key)
        with original.task_lock(path):
            load(source, provenance)
            result = inherit(path, source, provenance)
            reuse_training_cache(path, RAW / phase / 'complement' / str(seed) / f'{key}_training.npz')
            return result
    if path.exists():
        load(path, provenance)
    return original.local_task(seed, key, cfg['flat'], provenance)


def group_keys(flat, seed, cfg):
    fleet = original.fleet_for(flat, seed)
    return list(original.blocks(fleet)) + [f'LearnedGroups_r{r}' for r in cfg['learned']['restart_seeds']]


def learned_job(seed, key, cfg, provenance):
    flat = cfg['flat']
    members = original.fleet_for(flat, seed)['clients']
    path = original.path(seed, 'group', key)
    sh = flat['sharing']
    with original.task_lock(path):
        local_paths = [original.path(seed, 'local', c['client']) for c in members]
        local = [load(p, provenance) for p in local_paths]
        if not all(p['status'] == 'complete' for p in local):
            raise ValueError('Local fitting is incomplete')
        parents = {p.name: file_hash(p) for p in local_paths}
        if path.exists():
            p = load(path, provenance)
            if p['parents'] != parents:
                raise ValueError('Local sources changed')
        else:
            # Only fitted local log parameters enter server initialization.
            restart = int(key.rsplit('r', 1)[1])
            state = fl.initialize_personal([p['local']['log_parameters'] for p in local],
                cfg['learned']['groups'], restart, sh['weights'])
            p = dict(original.start_payload(seed, key, provenance), parents=parents,
                     client_ids=[c['client'] for c in members], state=state)
            original.save(path, p, 'initialized')
        if p['status'] == 'complete':
            return dict(seed=seed, key=key, status='already_complete', convergence=p['state']['status'])
        endpoints = [fl.Client([original.training(c, seed, flat, lp['data'])[0]], c['geometry_m'],
                              original.base.settings(flat)) for c, lp in zip(members, local)]
        state = p['state']
        while state['status'] == 'running':
            for i, endpoint in enumerate(endpoints):
                if str(i) in state['pending']:
                    continue
                center = state['centers'][state['assignments'][i]]
                fit = endpoint.fit(state['log_parameters'][i], center, sh['weights'], sh['local_round_max_nfev'])
                state['pending'][str(i)] = fit
                original.save(path, p, f'round{state["round"] + 1}_client{i}')
            fits = [state['pending'][str(i)] for i in range(len(endpoints))]
            if not all(f['available'] for f in fits):
                original.save(path, p, 'unavailable_update')
                raise ValueError('Unavailable local update; no fallback')
            state = fl.finish_personal_round(state, [fl.client_round_message(f) for f in fits], sh['weights'], sh,
                                             all(f['optimizer_converged'] for f in fits))
            state['history'][-1]['local_updates'] = [{k: f[k] for k in
                ['optimizer_converged', 'nfev', 'optimality', 'near_bound']} for f in fits]
            p['state'] = state
            original.save(path, p, f'round{state["round"]}')
        p['status'] = 'complete'
        original.save(path, p, state['status'])
    return dict(seed=seed, key=key, status='complete', convergence=state['status'])


def group_job(seed, key, cfg, provenance, phase, condition):
    if key.startswith('LearnedGroups_'):
        return learned_job(seed, key, cfg, provenance)
    path = original.path(seed, 'group', key)
    members = original.blocks(original.fleet_for(cfg['flat'], seed))[key]
    for c in members:
        load(original.path(seed, 'local', c['client']), provenance)
    if phase == 'development' and condition == 'complement':
        source = OUT / f'{SOURCE_PREFIX}_seed{seed}_group_{key}.json'
        parents = {original.path(seed, 'local', c['client']).name: file_hash(original.path(seed, 'local', c['client']))
                   for c in members}
        with original.task_lock(path):
            return inherit(path, source, provenance, parents)
    if path.exists():
        load(path, provenance)
    return original.group_task(seed, key, cfg['flat'], provenance)


def choose_restart(candidates):
    """Accept converged training objectives only; stable tie-break by restart key."""
    valid = [(p['state']['history'][-1]['objective'], key) for key, p in candidates.items()
             if p['status'] == 'complete' and p['state']['status'] == 'converged']
    if not valid:
        raise ValueError('No converged learned-group restart; all attempts preserved')
    objective, key = min(valid)
    return dict(selected=key, training_objective=float(objective),
        candidates={k:dict(status=p['state']['status'], rounds=p['state']['round'],
                          training_objective=p['state']['history'][-1]['objective']) for k,p in candidates.items()})


def selection_job(seed, cfg, provenance, strict_runtime=True):
    path = original.path(seed, 'selection', 'LearnedGroups')
    paths = {f'LearnedGroups_r{r}':original.path(seed, 'group', f'LearnedGroups_r{r}')
             for r in cfg['learned']['restart_seeds']}
    candidates = {k:load(v, provenance, strict_runtime) for k,v in paths.items()}
    parents = {p.name:file_hash(p) for p in paths.values()}
    chosen = choose_restart(candidates)
    with original.task_lock(path):
        if path.exists():
            p = load(path, provenance, strict_runtime)
            if p['parents'] != parents or p['choice'] != chosen:
                raise ValueError('Selection changed after freezing')
        else:
            p = dict(original.start_payload(seed, 'LearnedGroups', provenance), parents=parents,
                     status='complete', choice=chosen)
            original.save(path, p, 'selected_by_training_objective')
    return p


def parameters(seed, c, cfg, provenance, strict_runtime=True):
    load(original.path(seed, 'local', c['client']), provenance, strict_runtime)
    pars, statuses, parents = original.fitted(seed, c, cfg['flat'], provenance, strict_runtime)
    selection = load(original.path(seed, 'selection', 'LearnedGroups'), provenance, strict_runtime)
    group_path = original.path(seed, 'group', selection['choice']['selected'])
    learned = load(group_path, provenance, strict_runtime)
    index = learned['client_ids'].index(c['client'])
    pars['LearnedGroups'] = np.exp(learned['state']['log_parameters'][index]).tolist()
    statuses['LearnedGroups'] = learned['state']['status']
    parents.update({group_path.name:file_hash(group_path),
                    original.path(seed, 'selection', 'LearnedGroups').name:file_hash(original.path(seed, 'selection', 'LearnedGroups'))})
    for name in list(parents):
        load(OUT / name, provenance, strict_runtime)
    return pars, statuses, parents


def evaluate(seed, c, speed, probe, cfg, pars, statuses, phase, condition):
    baseline_pars = {k:v for k,v in pars.items() if k != 'LearnedGroups'}
    cache = ROOT / 'results/data' / SOURCE_PREFIX / str(seed) / f'{c["client"]}_v{speed}_{probe}.npz'
    if phase == 'development' and condition == 'complement' and cache.exists():
        old = storage.read_checkpoint(OUT / f'{SOURCE_PREFIX}_seed{seed}_eval_{c["client"]}.json')['cases'][f'v{speed}_{probe}']
        with np.load(cache, allow_pickle=False) as saved:
            arrays = dict(saved)
        record = copy.deepcopy(old)
    else:
        record, arrays = original.evaluate(c, seed, speed, probe, cfg['flat'], baseline_pars, statuses)
    if phase == 'development' and condition == 'complement':
        old = storage.read_checkpoint(OUT / f'{SOURCE_PREFIX}_seed{seed}_eval_{c["client"]}.json')['cases'][f'v{speed}_{probe}']
        # Includes clean plant, sensor noise and every frozen baseline prediction.
        if storage.trace_hash(arrays) != old['trace_sha256']:
            raise ValueError('Original evaluation hash differs; use the original numerical platform')
        for method in BASE_METHODS + ['Exact']:
            np.testing.assert_allclose(record['methods'][method]['yaw_rmse_deg_s'],
                                       old['methods'][method]['yaw_rmse_deg_s'], rtol=1e-12, atol=1e-14)
    # Preserve the original six-candidate numerical map before adding one model.
    predicted = original.base.batch_predict(pars['LearnedGroups'], [arrays['steering']], [speed], c['geometry_m'],
        cfg['flat']['sample_time'], cfg['flat']['integration_max_step'])[0,0]
    arrays['predicted_LearnedGroups'] = predicted
    record['methods']['LearnedGroups'] = dict(
        yaw_rmse_deg_s=float(np.rad2deg(np.sqrt(np.mean((predicted[:,1]-arrays['state'][:,1])**2)))),
        beta_rmse_deg=float(np.rad2deg(np.sqrt(np.mean((predicted[:,0]-arrays['state'][:,0])**2)))),
        optimizer_status=statuses['LearnedGroups'])
    record['trace_sha256'] = storage.trace_hash(arrays)
    return record, arrays


def eval_job(seed, key, cfg, provenance, phase, condition):
    c = original.client_for(cfg['flat'], seed, key)
    path = original.path(seed, 'eval', key)
    with original.task_lock(path):
        pars, statuses, parents = parameters(seed, c, cfg, provenance)
        p = load(path, provenance) if path.exists() else dict(original.start_payload(seed, key, provenance),
                                                             parents=parents, cases={})
        if p['parents'] != parents:
            raise ValueError('Fitted parameters changed after evaluation')
        if p['status'] == 'complete':
            return dict(seed=seed, key=key, status='already_complete')
        for speed in cfg['flat']['evaluation_speeds']:
            for probe in ['slow', 'broadband']:
                case = f'v{speed}_{probe}'
                if case in p['cases']:
                    continue
                record, arrays = evaluate(seed, c, speed, probe, cfg, pars, statuses, phase, condition)
                storage.save_arrays(original.RAW / str(seed) / f'{key}_{case}.npz', arrays)
                p['cases'][case] = record
                original.save(path, p, case)
        p['status'] = 'complete'
        original.save(path, p, 'complete')
    return dict(seed=seed, key=key, status='complete')


def execute(job, cfg, provenance):
    stage, phase, condition, seed, key = job
    with threadpool_limits(limits=1), scenario(cfg, phase, condition) as flat:
        scoped = dict(cfg, flat=flat)
        return {'local':local_job, 'group':group_job, 'eval':eval_job}[stage](seed, key, scoped, provenance, phase, condition)


def require_confirmation_lock(cfg, provenance):
    path = summary_path('confirmation', 'freeze')
    if not path.exists():
        raise ValueError('Confirmation is gated: analyze development and run freeze first')
    lock = storage.read_checkpoint(path)
    if lock['provenance'] != provenance:
        raise ValueError('Confirmation protocol/code/runtime changed after freezing')
    for name, sha in lock['development_sha256'].items():
        if file_hash(ROOT / name) != sha:
            raise ValueError('Frozen development evidence changed')
    return lock


def run(cfg, phase, workers):
    provenance = identity()
    if phase == 'confirmation':
        require_confirmation_lock(cfg, provenance)
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        for stage in ['local', 'group']:
            for condition in cfg['conditions']:
                jobs = []
                with scenario(cfg, phase, condition) as flat:
                    for fleet in fleets(cfg, phase):
                        keys = ([c['client'] for c in fleet['clients']] if stage == 'local'
                                else group_keys(flat, fleet['seed'], cfg))
                        jobs.extend((stage, phase, condition, fleet['seed'], key) for key in keys)
                futures = [pool.submit(execute, job, cfg, provenance) for job in jobs]
                for future in as_completed(futures):
                    print(json.dumps(future.result()), flush=True)
        # Every selection is frozen before any condition's test outputs are opened.
        for condition in cfg['conditions']:
            with scenario(cfg, phase, condition) as flat:
                for fleet in fleets(cfg, phase):
                    selection_job(fleet['seed'], dict(cfg, flat=flat), provenance)
        jobs = [('eval', phase, condition, f['seed'], key) for condition in cfg['conditions']
                for f in fleets(cfg, phase) for key in target_ids(cfg, phase, f['seed'], condition)]
        for future in as_completed([pool.submit(execute, job, cfg, provenance) for job in jobs]):
            print(json.dumps(future.result()), flush=True)
    sources = {}
    for condition in cfg['conditions']:
        for path in sorted(OUT.glob(f'{prefix(phase, condition)}_seed*.json')):
            p = load(path, provenance)
            if p['status'] != 'complete':
                raise ValueError('Execution checkpoint incomplete')
            sources[path.relative_to(ROOT).as_posix()] = file_hash(path)
    storage.atomic_json(summary_path(phase, 'execution'), dict(status='execution_complete', provenance=provenance,
        phase=phase, source_sha256=sources))
    print(json.dumps(dict(status='execution_complete', phase=phase,
        next='Commit and push JSON checkpoints, then run analyze.')), flush=True)


def paired_interval(values, cfg):
    values = np.asarray(values, float)
    rng = np.random.default_rng(cfg['screen']['bootstrap_seed'])
    draws = rng.integers(0, len(values), size=(cfg['screen']['bootstrap_repetitions'], len(values)))
    return np.quantile(values[draws].mean(axis=1), [.025, .975]).tolist()


def decision(fleet, paired, quality, cfg, phase):
    screen = cfg['screen']
    mean = fleet.mean()
    comparisons = {}
    minimum_fleets = screen['development_min_fleets_improved'] if phase == 'development' else screen['confirmation_min_fleets_improved']
    checks = [quality]
    for control in screen['primary_controls'] + screen['other_controls']:
        differences = fleet[control] - fleet.LearnedGroups
        gain = float(100 * (1 - mean.LearnedGroups / mean[control]))
        ci = paired_interval(differences, cfg)
        comparisons[control] = dict(gain_percent=gain, absolute_reduction_deg_s=float(differences.mean()),
            fleets_improved=int((differences > 0).sum()), fleet_bootstrap_ci_deg_s=ci)
        if control in screen['primary_controls']:
            checks.extend([gain >= screen['min_gain_percent'], int((differences > 0).sum()) >= minimum_fleets])
            if phase == 'confirmation':
                checks.append(ci[0] > 0)
        else:
            checks.append(gain > 0)
    coverage_differences = paired.same13 - paired.complement
    coverage_gain = float(100 * (1 - paired.complement.mean() / paired.same13.mean()))
    coverage_ci = paired_interval(coverage_differences, cfg)
    coverage = quality and coverage_gain > screen['coverage_min_gain_percent'] and int((coverage_differences > 0).sum()) >= minimum_fleets
    if phase == 'confirmation':
        coverage = coverage and coverage_ci[0] > 0
    compatibility = bool(all(checks))
    gap = float(mean.SharedPrior - mean.OracleFamily)
    return dict(status=('compatible_sharing_and_coverage_supported' if compatibility and coverage
                        else 'compatible_sharing_only' if compatibility else 'no_automatic_progression'),
        compatible_sharing_passed=compatibility, complementary_coverage_passed=bool(coverage),
        numerical_and_operating_checks_passed=bool(quality), comparisons=comparisons,
        coverage=dict(gain_percent=coverage_gain, absolute_reduction_deg_s=float(coverage_differences.mean()),
                      fleets_improved=int((coverage_differences > 0).sum()), fleet_bootstrap_ci_deg_s=coverage_ci),
        oracle_gain_fraction_retained=float((mean.SharedPrior-mean.LearnedGroups)/gap) if gap > 0 else None,
        interpretation='Fixed K=3; family labels are retrospective. OracleFamily is one privileged partition, not an optimal bound. Development intervals are descriptive, with three fleets.')


def analyze(cfg, phase):
    provenance = identity()
    execution = storage.read_checkpoint(summary_path(phase, 'execution'))
    if execution['provenance']['files'] != provenance['files']:
        raise ValueError('Execution protocol/code changed')
    for name, sha in execution['source_sha256'].items():
        if file_hash(ROOT / name) != sha:
            raise ValueError('Execution results changed before analysis')
    rows, assignments, optim, local_audit = [], [], [], []
    training_ok = True
    for condition in cfg['conditions']:
        with scenario(cfg, phase, condition) as flat:
            scoped = dict(cfg, flat=flat)
            for f in fleets(cfg, phase):
                seed = f['seed']
                for c in original.fleet_for(flat, seed)['clients']:
                    lp = load(original.path(seed, 'local', c['client']), provenance, False)
                    if condition == 'same13' and c['client'] in target_ids(cfg, phase, seed, condition):
                        matched = load(checkpoint_path(phase, 'complement', seed, 'local', c['client']), provenance, False)
                        if (lp['data']['trace_sha256'] != matched['data']['trace_sha256']
                                or lp['local'] != matched['local'] or lp['prior'] != matched['prior']):
                            raise ValueError('Paired target data or local fits changed between conditions')
                    training_ok = training_ok and lp['data']['operating_checks_passed']
                    for method in ['local', 'prior']:
                        fit = lp[method]
                        if not fit['available'] or not fit['optimizer_converged']:
                            raise ValueError('Selected local/nominal fit is not converged')
                        local_audit.append(dict(condition=condition, seed=seed, client=c['client'], method=method,
                            nfev=fit['nfev'], reused='reused_from' in lp, near_bound=json.dumps(fit['near_bound'])))
                selection = load(original.path(seed, 'selection', 'LearnedGroups'), provenance, False)
                # Recalculate selection strictly from saved training objectives.
                candidates = {f'LearnedGroups_r{r}':load(original.path(seed, 'group', f'LearnedGroups_r{r}'), provenance, False)
                              for r in cfg['learned']['restart_seeds']}
                if choose_restart(candidates) != selection['choice']:
                    raise ValueError('Saved restart selection does not match training-only rule')
                for key in group_keys(flat, seed, cfg):
                    p = load(original.path(seed, 'group', key), provenance, False)
                    s = p['state']
                    if p['status'] != 'complete' or s['pending']:
                        raise ValueError('Incomplete group optimization')
                    is_learned = key.startswith('LearnedGroups_')
                    selected = key == selection['choice']['selected'] if is_learned else True
                    optim.append(dict(condition=condition, seed=seed, group=key, selected=selected,
                        status=s['status'], rounds=s['round'], active_groups=len(set(s['assignments'])),
                        objective=s['history'][-1]['objective'], communication_float64=s['communication_float64']))
                selected = candidates[selection['choice']['selected']]
                truth = {c['client']:c for c in f['clients']}
                labels = selected['state']['assignments']
                families = [truth[k]['family'] for k in selected['client_ids']]
                speeds = [truth[k]['training_speed'] if condition == 'complement' else 13 for k in selected['client_ids']]
                for index, key in enumerate(selected['client_ids']):
                    assignments.append(dict(condition=condition, seed=seed, client=key, family=truth[key]['family'],
                        training_speed=speeds[index], group=labels[index], selected_restart=selection['choice']['selected'],
                        family_ari=adjusted_rand_score(families, labels), speed_ari=adjusted_rand_score(speeds, labels)))
                for key in target_ids(cfg, phase, seed, condition):
                    c = original.client_for(flat, seed, key)
                    _, _, parents = parameters(seed, c, scoped, provenance, False)
                    ep = load(original.path(seed, 'eval', key), provenance, False)
                    expected = {f'v{v}_{p}' for v in flat['evaluation_speeds'] for p in ['slow', 'broadband']}
                    if ep['status'] != 'complete' or ep['parents'] != parents or set(ep['cases']) != expected:
                        raise ValueError('Missing or stale evaluation')
                    for record in ep['cases'].values():
                        if set(record['methods']) != set(METHODS + ['Exact']):
                            raise ValueError('Incomplete method comparison')
                        if record['methods']['Exact']['yaw_rmse_deg_s'] > 1e-10:
                            raise ValueError('Exact-model numerical reference failed')
                        for method, metrics in record['methods'].items():
                            if (not np.isfinite([metrics['yaw_rmse_deg_s'], metrics['beta_rmse_deg']]).all()
                                    or min(metrics['yaw_rmse_deg_s'], metrics['beta_rmse_deg']) < 0):
                                raise ValueError('Nonfinite or negative evaluation error')
                            rows.append(dict(condition=condition, seed=seed, client=key, family=truth[key]['family'],
                                original_training_speed=truth[key]['training_speed'], training_speed=c['training_speed'],
                                speed=record['speed'], probe=record['probe'], region=record['region'],
                                operating_checks_passed=record['operating_checks_passed'], method=method, **metrics))
    cases = pd.DataFrame(rows)
    primary = cases[(cases.condition == 'complement') & (cases.region == 'peer_observed') & cases.method.isin(METHODS)]
    vehicles = primary.groupby(['seed', 'client', 'family', 'training_speed', 'method']).yaw_rmse_deg_s.mean().unstack('method')[METHODS].reset_index()
    if len(vehicles) != sum(len(f['clients']) for f in fleets(cfg, phase)):
        raise ValueError('Missing primary target vehicles')
    if not primary.groupby(['seed', 'client', 'method']).size().eq(4).all():
        raise ValueError('Unexpected primary test count')
    fleet = vehicles.groupby('seed')[METHODS].mean()
    pair_cases = cases[(cases.original_training_speed == 13) & cases.speed.isin([21,29]) & cases.method.isin(METHODS)]
    if not pair_cases.groupby(['condition','seed','client','method']).size().eq(4).all():
        raise ValueError('Unexpected paired-target test count')
    paired_vehicles = pair_cases.groupby(['seed','client','family','method','condition']).yaw_rmse_deg_s.mean().unstack('condition').reset_index()
    if paired_vehicles[cfg['conditions']].isna().any().any():
        raise ValueError('Missing condition in a paired comparison')
    unchanged = paired_vehicles[paired_vehicles.method.isin(['LocalOE','NominalPrior'])]
    np.testing.assert_allclose(unchanged.complement, unchanged.same13, rtol=1e-12, atol=1e-14)
    paired = paired_vehicles[paired_vehicles.method == 'LearnedGroups'].groupby('seed')[cfg['conditions']].mean()
    selected_optim = pd.DataFrame(optim).query('selected')
    quality = bool(training_ok and cases.operating_checks_passed.all() and selected_optim.status.eq('converged').all())
    verdict = decision(fleet, paired, quality, cfg, phase)
    frames = dict(cases=cases, vehicles=vehicles, fleets=fleet.reset_index(),
        families=vehicles.groupby(['seed','family'])[METHODS].mean().reset_index(),
        speeds=vehicles.groupby(['seed','training_speed'])[METHODS].mean().reset_index(),
        paired_vehicles=paired_vehicles, paired_fleets=paired.reset_index(),
        optimization=pd.DataFrame(optim), assignments=pd.DataFrame(assignments), local_audit=pd.DataFrame(local_audit),
        quantiles=vehicles[METHODS].quantile([.5,.9,.95,1]).rename_axis('quantile').reset_index())
    for name, frame in frames.items():
        frame.to_csv(OUT / f'{PREFIX}_{phase}_{name}.csv', index=False)
    means = fleet.mean()
    wins = {m:int((vehicles.LearnedGroups < vehicles[m]).sum()) for m in METHODS if m != 'LearnedGroups'}
    report = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Learned vehicle groups: {phase}</title><style>body{{font:17px/1.6 system-ui;color:#183342;background:#edf2f5}}main{{max-width:1100px;margin:auto;background:white;padding:30px}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:9px;border-bottom:1px solid #ddd;text-align:left}}.scroll{{overflow:auto}}pre{{white-space:pre-wrap}}</style><main>
<h1>Learned compatible priors · {phase}</h1>
<p>Fixed K=3; three training-only restarts. Each vehicle retains its own physical parameters. Family labels and test outputs do not enter fitting or restart selection.</p>
<p><b>Progression:</b> {html.escape(verdict['status'])}. Numerical and operating checks: {quality}.</p>
<h2>Primary: locally unobserved anchor speeds</h2>
<div class="scroll">{means.rename('Mean yaw RMSE [deg/s]').to_frame().to_html(border=0)}</div>
<p>Each vehicle contributes four clean-yaw records. Average within vehicle and fleet first. Independent replication units are fleets. Vehicle win counts for learned groups: {html.escape(json.dumps(wins))}.</p>
<div class="scroll">{fleet.to_html(border=0)}</div>
<h2>Coverage control: unchanged 13 m/s targets</h2>
<p>Both conditions contain the same physical fleet and 18 records of 12 seconds. Only the six original low-speed clients are paired targets. Their local data, LocalOE and NominalPrior fits are identical. Peers supply complementary speeds or only 13 m/s. The same test inputs at 21/29 m/s are used in both conditions.</p>
<div class="scroll">{paired.to_html(border=0)}</div>
<h2>Decision evidence</h2><pre>{html.escape(json.dumps(verdict,indent=2))}</pre>
<p>Confirmation requires at least 5% lower mean error than shared and nominal priors, improvement in at least 8/10 fleets, and positive fleet-paired 95% bootstrap intervals for both comparisons. LocalOE and MixedGroups must also improve in the pooled mean. Development uses a 2/3-fleet exploratory screen; its intervals are descriptive. Coverage support is evaluated separately.</p>
<h2>Family and speed strata</h2><div class="scroll">{frames['families'].to_html(index=False,border=0)}{frames['speeds'].to_html(index=False,border=0)}</div>
<h2>Vehicle error distribution</h2><div class="scroll">{frames['quantiles'].to_html(index=False,border=0)}</div>
<h2>Optimization and grouping diagnostics</h2><div class="scroll">{frames['optimization'].to_html(index=False,border=0)}</div>
<p>All restart outcomes remain in the table. Only converged restarts are eligible for selection by final training penalized objective. Empty/collapsed groups remain visible. Family and speed ARI are retrospective diagnostics; prediction determines success. OracleFamily is a privileged reference partition, not an optimal bound.</p>
<p>This is a matched tanh/relaxation identification experiment with known geometry and common true relaxation lengths. It does not establish nonlinear Pacejka mismatch tolerance, reliable recovery of weak friction, or closed-loop improvement.</p>
</main></html>'''
    report_path = REPORT / f'cfl_learned_groups_{phase}.html'
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding='utf-8', newline='\n')
    output_paths = [report_path, *sorted(OUT.glob(f'{PREFIX}_{phase}_*.csv'))]
    storage.atomic_json(summary_path(phase, 'analysis'), dict(status='analysis_complete', phase=phase,
        provenance=execution['provenance'], analysis_runtime=provenance['runtime'],
        execution_sha256=file_hash(summary_path(phase,'execution')), decision=verdict,
        means_deg_s=means.to_dict(), vehicle_wins=wins,
        output_sha256={p.relative_to(ROOT).as_posix():file_hash(p) for p in output_paths}))
    print(json.dumps(dict(status='analysis_complete', phase=phase, decision=verdict['status'], report=str(report_path))))


def freeze(cfg):
    provenance = identity()
    analysis_path = summary_path('development', 'analysis')
    p = storage.read_checkpoint(analysis_path)
    if p['provenance'] != provenance or not p['decision']['compatible_sharing_passed']:
        raise ValueError('Complete and review a passing development run before freezing confirmation')
    execution_path = summary_path('development', 'execution')
    if file_hash(execution_path) != p['execution_sha256']:
        raise ValueError('Development execution manifest changed')
    execution = storage.read_checkpoint(execution_path)
    sources = {**execution['source_sha256'], **p['output_sha256'],
               analysis_path.relative_to(ROOT).as_posix():file_hash(analysis_path),
               execution_path.relative_to(ROOT).as_posix():file_hash(execution_path)}
    for name, sha in sources.items():
        if file_hash(ROOT / name) != sha:
            raise ValueError('Development evidence changed after analysis')
    path = summary_path('confirmation', 'freeze')
    if path.exists():
        require_confirmation_lock(cfg, provenance)
        print(json.dumps(dict(status='confirmation_already_frozen')))
        return
    storage.atomic_json(path, dict(status='confirmation_frozen', provenance=provenance,
        development_sha256=sources, confirmation_seeds=cfg['confirmation_seeds']))
    print(json.dumps(dict(status='confirmation_frozen', next='Commit/push this lock before running confirmation.')))


def restore(cfg, phase):
    """Recreate ignored arrays and check saved hashes; never refit or rewrite JSON."""
    provenance = identity()
    count = 0
    before = {p:file_hash(p) for p in OUT.glob(f'{PREFIX}_{phase}_*.json')}
    for condition in cfg['conditions']:
        with scenario(cfg, phase, condition) as flat:
            scoped = dict(cfg, flat=flat)
            for fleet in fleets(cfg, phase):
                seed = fleet['seed']
                for c in original.fleet_for(flat, seed)['clients']:
                    path = original.path(seed, 'local', c['client'])
                    if not path.exists():
                        continue
                    p = load(path, provenance)
                    if 'data' in p:
                        original.training(c, seed, flat, p['data'])
                        count += 1
                for key in target_ids(cfg, phase, seed, condition):
                    path = original.path(seed, 'eval', key)
                    if not path.exists():
                        continue
                    p = load(path, provenance)
                    c = original.client_for(flat, seed, key)
                    pars, statuses, parents = parameters(seed, c, scoped, provenance)
                    if p['parents'] != parents:
                        raise ValueError('Frozen evaluation parents changed')
                    for case, record in p['cases'].items():
                        target = original.RAW / str(seed) / f'{key}_{case}.npz'
                        if target.exists():
                            with np.load(target, allow_pickle=False) as arrays_file:
                                arrays = dict(arrays_file)
                        else:
                            _, arrays = evaluate(seed, c, record['speed'], record['probe'], scoped, pars, statuses, phase, condition)
                        if storage.trace_hash(arrays) != record['trace_sha256']:
                            raise ValueError('Exact restore mismatch; use original numerical platform')
                        if not target.exists():
                            storage.save_arrays(target, arrays)
                        count += 1
    if before != {p:file_hash(p) for p in before}:
        raise ValueError('Restore changed a checkpoint')
    print(json.dumps(dict(status='exact_arrays_verified', traces=count, checkpoint_files_changed=False)))


def smoke(cfg):
    # Real nonlinear residual/Jacobian/fit, on short synthetic records only.
    original.smoke(cfg['simulation'])
    logs = np.log(np.array(cfg['simulation']['starts']))
    state = fl.initialize_personal(logs, cfg['learned']['groups'], cfg['learned']['restart_seeds'][0],
                                   cfg['simulation']['sharing']['weights'])
    messages = [dict(log_parameters=z, data_sse=1.) for z in state['log_parameters']]
    updated = fl.finish_personal_round(state, messages, cfg['simulation']['sharing']['weights'],
                                      cfg['simulation']['sharing'], True)
    assert len(updated['assignments']) == 3 and updated['round'] == 1
    print(json.dumps(dict(status='learned_groups_smoke_passed', production_outputs_modified=False)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['preflight', 'smoke', 'run', 'analyze', 'freeze', 'restore'])
    parser.add_argument('--phase', choices=['development', 'confirmation'], default='development')
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    cfg = preflight()
    if args.command == 'preflight':
        print(json.dumps(dict(status='preflight_passed', development_fleets=3, confirmation_fleets=10,
            conditions=cfg['conditions'], groups=3, restarts=3, reused_local_jobs=72,
            new_development_local_jobs=36, reused_development_group_jobs=21,
            new_development_group_jobs=39, development_test_records=1008,
            confirmation_requires_freeze=True, new_fits=0)))
    elif args.command == 'smoke':
        with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context('spawn')) as pool:
            pool.submit(smoke, cfg).result()
    elif args.command == 'run':
        run(cfg, args.phase, args.workers)
    elif args.command == 'analyze':
        analyze(cfg, args.phase)
    elif args.command == 'freeze':
        freeze(cfg)
    else:
        restore(cfg, args.phase)


if __name__ == '__main__':
    main()
