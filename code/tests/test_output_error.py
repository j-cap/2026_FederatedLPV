import sys
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from federated_lpv.output_error import transitions,prepare,predict,fit
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
import experiment_7a_local_identification as exp

def test_varying_speed_prediction_matches_independent_ode():
    p=np.array([53.,51.,.6]);v=np.linspace(12,14,201);u=.01*np.sin(np.arange(200)*.1)
    record=dict(speed=v,input=u,state=np.zeros((201,2)))
    prediction=predict(p,prepare([record]));state=np.zeros(2);reference=[]
    for k in range(200):
        def rhs(t,z):
            speed=v[k]+(v[k+1]-v[k])*t/.01;f,r,h=p
            front=f*(u[k]-(z[0]+1.2*z[1])/speed);rear=r*(-z[0]+1.6*z[1])/speed
            return [front+rear-speed*z[1],h*(1.2*front-1.6*rear)]
        state=solve_ivp(rhs,(0,.01),state,rtol=1e-11,atol=1e-13).y[:,-1]
        reference.append([state[0]/v[k+1],state[1]])
    np.testing.assert_allclose(prediction,reference,atol=2e-8)

def test_record_boundaries_reset_and_fit_recovers_known_ratios():
    p=np.array([48.,56.,.58]);records=[]
    for speed in (12,25):
        t=np.arange(301)*.01
        r=dict(speed=np.full_like(t,speed),input=.02*np.sin(2*t[:-1]),state=np.zeros((301,2)))
        r['state'][1:]=predict(p,prepare([r]));records.append(r)
    np.testing.assert_allclose(predict(p,prepare(records)),np.concatenate([r['state'][1:] for r in records]),atol=1e-13)
    estimate,info=fit(records)
    assert info['success'];np.testing.assert_allclose(estimate,p,rtol=1e-6)

def test_per_client_controller_dispatch_preserves_shared_controller_result():
    clients=exp.corrected.six.fleet(31,1,1);control=exp.controller([53,53,.6])
    t=np.arange(101)*.01;v=18+2*t;ref=.03*np.sin(2*t)
    a=exp.corrected.simulate(clients,control,v,ref,'vy')
    b=exp.corrected.simulate(clients,control,v,ref,'vy',client_controllers={c.client_id:control for c in clients})
    np.testing.assert_allclose(a[0],b[0],atol=1e-13);np.testing.assert_allclose(a[1],b[1],atol=1e-13)
