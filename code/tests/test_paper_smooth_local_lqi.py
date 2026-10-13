import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments'))
import paper_smooth_local_lqi as supplement


def test_frozen_protocol():
    cfg, ctl, _, lock = supplement.preflight()
    assert cfg['seeds'] == ctl['confirmation_seeds'] == list(range(411, 421))
    assert 'results/tables/paper_local_smoothness_selection.json' in lock


def test_parent_comparison_catches_metric_and_partition_changes():
    row = dict(client='c', method='Global', scenario='hard', selected_k=2, feasible=True)
    for col in ['tracking_rmse_deg_s','tracking_peak_deg_s','beta_rms_deg','beta_peak_deg',
                'steering_rms_deg','steering_peak_deg','steering_rate_peak_deg_s',
                'saturation_fraction','rho']:
        row[col] = .1
    frame = pd.DataFrame([row])
    supplement.check_parent(frame, frame.copy(), 1e-8)
    bad = frame.copy()
    bad.loc[0, 'tracking_rmse_deg_s'] += .01
    with pytest.raises(AssertionError):
        supplement.check_parent(frame, bad, 1e-8)
    bad = frame.copy()
    bad.loc[0, 'selected_k'] = 1
    with pytest.raises(ValueError):
        supplement.check_parent(frame, bad, 1e-8)


def test_synthesis_failure_is_not_a_valid_rollout():
    assert supplement.simulate_with_arrays(None, None,
        dict(time=np.arange(3), speed=np.ones(3), curvature=np.zeros(3)),
        dict(steering_limit_deg=12, steering_rate_limit_deg_s=120, sample_time=.01,
             speed_grid=[10,30])) is None


def test_failure_prevents_complete_case_gain(monkeypatch, tmp_path):
    monkeypatch.setattr(supplement, 'OUT', tmp_path)
    cfg = dict(seeds=[411,412], bootstrap_seed=1)
    monkeypatch.setattr(supplement, 'preflight', lambda:(cfg,None,None,{}))
    rows = []
    for method in ['SmoothLocal','Global','Local','ExactLPV','FederatedLearned']:
        for scenario in ['moderate','hard']:
            for client in range(30):
                rows.append(dict(client=client, method=method, scenario=scenario,
                    tracking_rmse_deg_s=None if method=='SmoothLocal' and client==0 else .1,
                    feasible=method!='SmoothLocal', rho=.9))
    monkeypatch.setattr(supplement, 'read_checkpoint', lambda seed:dict(rows=[dict(seed=seed,**r) for r in rows]))
    monkeypatch.setattr(supplement, 'digest', lambda p:'test')
    supplement.summarize()
    c = pd.read_csv(tmp_path / f'{supplement.PREFIX}_comparisons.csv')
    bad = c[c.baseline=='SmoothLocal']
    assert (bad.status=='incomplete_no_gain_estimate').all()
    assert bad.gain_pct.isna().all()
