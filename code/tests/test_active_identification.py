from experiment_9i_active_identification import choose


def test_choose_returns_lowest_score_candidate():
    assert choose({'a':3.,'b':1.,'c':2.})=='b'
