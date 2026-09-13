import json
import numpy as np
from experiment_9g_finite_personalized import CONFIG,coefficient_fit


def test_coefficient_fit_returns_positive_manifold_parameters():
    cfg=json.loads(CONFIG.read_text());center=np.log([50.,55.,.6]);direction=np.array([[.1],[-.1],[.05]])
    records=[dict(state=np.zeros((11,2)),input=np.zeros(10),speed=np.ones(11)*20)]
    p,a,d=coefficient_fit(records,center,direction,cfg)
    assert p.shape==(3,) and a.shape==(1,)
    assert np.all(p>0) and d['success']
