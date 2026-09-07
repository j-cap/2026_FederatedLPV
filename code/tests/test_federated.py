import numpy as np
from federated_lpv.output_error import prepare,predict,fit
from federated_lpv.federated import secure_sum,fit_federated


def records(p,speed):
    t=np.arange(151)*.01;r=dict(speed=np.full_like(t,speed),input=.02*np.sin(2*t[:-1]),state=np.zeros((151,2)))
    r['state'][1:]=predict(p,prepare([r]));return [r]


def test_secure_sum_preserves_aggregate():
    values=np.arange(20,dtype=float).reshape(5,4)/7
    np.testing.assert_allclose(secure_sum(values,3),values.sum(axis=0),atol=1e-12)


def test_federated_fit_matches_centralized_clean_solution():
    p=np.array([48.,56.,.58]);clients=[records(p,v) for v in (12,18,25)]
    central,_=fit([r for client in clients for r in client]);fed,info=fit_federated(clients)
    assert info['success'];np.testing.assert_allclose(fed,central,rtol=2e-5)
