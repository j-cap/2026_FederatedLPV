import numpy as np

from experiment_10c_personalization_transition import progressive_speeds


def test_progressive_speeds_are_nested_and_reach_full_grid():
    grid = np.arange(10, 32, 2)
    base = [28, 30]
    stages = [progressive_speeds(base, grid, stage) for stage in ("base", 5, 8, 11)]
    assert [len(stage) for stage in stages] == [2, 5, 8, 11]
    assert all(set(stages[j]).issubset(stages[j + 1]) for j in range(3))
    assert np.array_equal(stages[-1], grid)


def test_progressive_speeds_add_nearest_missing_point_first():
    grid = np.arange(10, 32, 2)
    selected = progressive_speeds([16, 18, 20], grid, 5)
    assert np.array_equal(selected, [12, 14, 16, 18, 20])
