import sys
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
import experiment_6b_acceleration_validation as exp

def test_constant_speed_matches_legacy_closed_loop():
    clients=exp.six.fleet(121,1,1)
    controller=exp.load_controllers(21,1,(10,30))['M4']
    time=np.arange(0,1.01,.01);v=np.full_like(time,22.);ref=.04*np.sin(3*time)
    old=exp.simulate(clients,controller,v,ref,'legacy')
    for mode in ('vy','beta_noacc','beta_corrected'):
        new=exp.simulate(clients,controller,v,ref,mode)
        np.testing.assert_allclose(new[0],old[0],atol=1e-12)
        np.testing.assert_allclose(new[1],old[1],atol=1e-12)
        for key in old[2]:np.testing.assert_allclose(new[2][key],old[2][key],atol=1e-10)

def test_coordinate_derivative_and_rk4_against_independent_momentum_reference():
    clients=exp.six.fleet(121,1,1);p=exp.vehicle_arrays(clients)
    v0=16.;acc=3.;dt=.01;v1=v0+acc*dt
    beta=np.array([[.01,.05],[.02,-.02],[-.01,.03]])
    u=np.array([.02,.015,-.01]);z=beta.copy();z[:,0]*=v0
    direct=exp.rhs(z,u,v0,acc,p,'vy')
    expected=direct.copy();expected[:,0]=(direct[:,0]-acc*beta[:,0])/v0
    np.testing.assert_allclose(exp.rhs(beta,u,v0,acc,p,'beta_corrected'),expected,atol=1e-14)
    propagated=exp.advance(z,u,v0,v1,p,'vy',dt)
    for j,c in enumerate(clients):
        par=c.parameters
        def momentum(t,state):
            v=v0+acc*t;vy,r=state
            ff=.9*par.mass*9.81*par.rear_length/(par.front_length+par.rear_length)
            fr=.9*par.mass*9.81*par.front_length/(par.front_length+par.rear_length)
            f=ff*np.tanh(par.front_stiffness*(u[j]-(vy+par.front_length*r)/v)/ff)
            rear=fr*np.tanh(par.rear_stiffness*(-vy+par.rear_length*r)/v/fr)
            return [(f+rear)/par.mass-v*r,(par.front_length*f-par.rear_length*rear)/par.yaw_inertia]
        reference=solve_ivp(momentum,(0,dt),z[j],rtol=1e-12,atol=1e-14).y[:,-1]
        # Halving the step controls the remaining fourth-order integration error.
        half=exp.advance(z[j:j+1],u[j:j+1],v0,(v0+v1)/2,p[:,j:j+1],'vy',dt/2)
        half=exp.advance(half,u[j:j+1],(v0+v1)/2,v1,p[:,j:j+1],'vy',dt/2)[0]
        assert np.linalg.norm(half-reference)<np.linalg.norm(propagated[j]-reference)/10
        np.testing.assert_allclose(half,reference,atol=1e-8)

def test_slow_profiles_preserve_envelope_and_limit_speed_rate():
    time=np.arange(0,24.01,.01)
    for name in ('previous_s_curve','shifted_double_lane','constant_22'):
        v,_=exp.scenario(name,time/2);v=exp.six.remap(v,(10,30))
        assert v.min()>=10 and v.max()<=30
        assert max(abs(np.diff(v)/.01))<2
