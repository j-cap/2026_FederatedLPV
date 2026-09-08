import numpy as np
from experiment_9b_group_robustness import continuous_fleet,discrete_fleet,ratios


def test_discrete_fleet_respects_unbalanced_counts():
    fleet=discrete_fleet(2,[11,7,3])
    assert [sum(c.family==f for c in fleet) for f in ('nominal','heavy','handling')]==[11,7,3]


def test_continuous_fleet_is_reproducible_positive_and_not_discretely_labelled():
    a=continuous_fleet(3,20);b=continuous_fleet(3,20)
    assert all(c.family=='continuum' for c in a)
    assert np.allclose([ratios(c) for c in a],[ratios(c) for c in b])
    assert np.all(np.asarray([ratios(c) for c in a])>0)
    assert np.unique(np.round([ratios(c)[0] for c in a],6)).size==20
