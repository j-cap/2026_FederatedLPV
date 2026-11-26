"""Protect the scientific comparisons and outcome-independent case selection."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'paper'))
import prepare_personalized_id_figures as figures


def sample_cases():
    rows=[]
    for seed,factor in zip([1301,1302,1303],[1.,2.,4.]):
        for i in range(9):
            speed=[13,21,29][i//3]
            for method in figures.METHODS:
                # Fleet-specific benefit yields different mean-of-ratios and ratio-of-means.
                value=factor if method!='Complementary' else factor-[.1,.2,2.][seed-1301]
                for v in [v for v in [13,21,29] if v!=speed]:
                    for probe in ['slow','broadband']:
                        rows.append(dict(seed=seed,client=f'vehicle_{i:02d}',training_speed=speed,speed=v,
                            probe=probe,method=method,region='target_unseen_anchor',yaw_rmse_deg_s=value))
                rows.append(dict(seed=seed,client=f'vehicle_{i:02d}',training_speed=speed,speed=speed,
                    probe='slow',method=method,region='target_observed',yaw_rmse_deg_s=1e9))
    return pd.DataFrame(rows)


def test_primary_comparison_uses_paired_unseen_cases_and_ratio_of_means():
    t=figures.summaries(sample_cases())
    assert len(t['vehicles'])==27 and len(t['strata'])==9
    expected=100*(.1+.2+2.)/(1.+2.+4.)
    np.testing.assert_allclose(t['pooled'].relative_reduction_pct,expected)
    assert not np.isclose(expected,np.mean([10.,10.,50.]))
    assert t['fleets'].LocalOE.max()==4.


def test_case_selection_does_not_use_fl_outcomes_or_row_order():
    cases=sample_cases()
    rule=dict(training_speeds=[13,21],evaluation_speed=29,probe='broadband')
    first=figures.select_trajectories(cases,rule)
    changed=cases.copy()
    changed.loc[changed.method=='Complementary','yaw_rmse_deg_s']=np.arange(sum(changed.method=='Complementary'))*1e6
    second=figures.select_trajectories(changed.sample(frac=1,random_state=7),rule)
    keys=['seed','client','training_speed','speed','probe']
    pd.testing.assert_frame_equal(first[first.selected][keys].reset_index(drop=True),second[second.selected][keys].reset_index(drop=True))
    # Three tied median values in fleet 1302; smallest opaque client ID wins.
    assert list(first[first.selected].client)==['vehicle_00','vehicle_03']
    assert list(first[first.selected].seed)==[1302,1302]


def test_frozen_sources_reconstruct_all_final_method_cases():
    data=figures.checked_data()
    assert len(data['cases'])==1512
    chosen=figures.select_trajectories(data['cases'],data['protocol']['trajectory_selection'])
    assert len(chosen)==18 and sum(chosen.selected)==2
    # Guards against accidental use of the previous, unfinished SameSpeed estimates.
    t=figures.summaries(data['cases'])
    means=t['fleets'][figures.METHODS].mean()
    np.testing.assert_allclose(100*(1-means.Complementary/means.SameSpeedAfter),4.0996447972102,rtol=1e-10)
