import numpy as np
from experiment_9h_calibration_frontier import prefix_records


def test_prefix_records_preserves_exact_transition_budget():
    records=[dict(state=np.zeros((26,2)),input=np.zeros(25),speed=np.ones(26)*v) for v in (10,20,30)]
    selected=prefix_records(records,.6)
    assert sum(len(r['input']) for r in selected)==60
    assert len(selected)==3 and len(selected[-1]['input'])==10
