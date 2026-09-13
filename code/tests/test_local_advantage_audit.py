import json
import numpy as np
from federated_lpv.fleet import sample_fleet
from experiment_9e_local_advantage_audit import CONFIG,drift_client,geometric_mean


def test_geometric_mean_is_positive_and_log_centered():
    x=np.array([[1.,2.,4.],[4.,8.,16.]])
    assert np.allclose(np.log(geometric_mean(x)),np.log(x).mean(0))


def test_drift_preserves_identity_and_applies_locked_scales():
    cfg=json.loads(CONFIG.read_text());c=sample_fleet(1,1)[0];d=drift_client(c,cfg['deployment_conditions']['load_shift'])
    assert (d.client_id,d.family)==(c.client_id,c.family)
    assert np.isclose(d.parameters.mass/c.parameters.mass,1.1)
    assert np.isclose(d.parameters.yaw_inertia/c.parameters.yaw_inertia,1.12)
