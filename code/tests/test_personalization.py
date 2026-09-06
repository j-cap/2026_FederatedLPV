import numpy as np
from federated_lpv.output_error import fit,prepare,predict
from federated_lpv import personalization


def records():
    p=np.array([48.,56.,.58]);result=[]
    for speed in (12,16,20):
        t=np.arange(201)*.01
        r=dict(speed=np.full_like(t,speed),input=.02*np.sin(2*t[:-1]),state=np.zeros((201,2)))
        r['state'][1:]=predict(p,prepare([r]));result.append(r)
    return result


def test_zero_strength_preserves_local_and_large_strength_approaches_prior():
    data=records();prior=np.array([60.,45.,.7])
    local,_=fit(data);zero,_=fit(data,prior,0)
    np.testing.assert_array_equal(local,zero)
    strong,info=fit(data,prior,1e6)
    assert info['success']
    np.testing.assert_allclose(strong,prior,rtol=1e-5)


def test_validation_excludes_each_episode_and_rejects_wrong_prior_on_clean_data(monkeypatch):
    data=records();seen=[]
    def spy(training,*args):
        seen.append([id(r) for r in training])
        return fit(training,*args)
    monkeypatch.setattr(personalization,'fit',spy)
    strength,rows=personalization.select_strength(data,np.array([60.,45.,.7]),[0,1])
    assert strength==0
    for fold in range(3):
        for index in (2*fold,2*fold+1):
            assert id(data[fold]) not in seen[index]
            assert len(seen[index])==2
    assert len(rows)==6
