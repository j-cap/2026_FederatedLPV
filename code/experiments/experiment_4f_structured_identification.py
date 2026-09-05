"""4F: identify positive bicycle parameter ratios from near-linear trajectories."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.linalg import expm
import experiment_4c_nonlinear_oracle_models as base
from experiment_4d_nonlinear_controllers import redesign_controllers, stability_audit, trajectory_metrics
from federated_lpv import sample_fleet, fit_oracle_architectures, design_oracle_controllers

ROOT=Path(__file__).resolve().parents[2]
LF,LR=1.2,1.6

def matrices(p,v):
    """p=[Cf/m, Cr/m, m/Iz]; geometry alone is known."""
    f,r,h=p; v=np.asarray(v)
    a=np.zeros(v.shape+(2,2)); b=np.zeros(v.shape+(2,))
    a[...,0,0]=-(f+r)/v; a[...,0,1]=(r*LR-f*LF)/v**2-1
    a[...,1,0]=h*(r*LR-f*LF); a[...,1,1]=-h*(f*LF**2+r*LR**2)/v
    b[...,0]=f/v; b[...,1]=h*f*LF
    return a,b

def step(p,x,u,v):
    a,b=matrices(p,v)
    def rhs(z): return np.einsum('nij,nj->ni',a,z)+b*u[:,None]
    k1=rhs(x); k2=rhs(x+base.DT*k1/2); k3=rhs(x+base.DT*k2/2); k4=rhs(x+base.DT*k3)
    return x+base.DT*(k1+2*k2+2*k3+k4)/6

@dataclass
class StructuredModel:
    specialization: str
    scheduling: str
    parameters: dict

    def matrix(self,family,speed):
        if self.scheduling=='constant': speed=20.
        if self.scheduling=='grid': speed=float(base.FIT_SPEEDS[np.argmin(abs(base.FIT_SPEEDS-speed))])
        p=self.parameters['global' if self.specialization=='global' else family]
        a,b=matrices(p,speed)
        aug=np.zeros((3,3)); aug[:2,:2]=a; aug[:2,2]=b
        d=expm(base.DT*aug)
        return d[:2,:2],d[:2,2:]

def fit_structured(records):
    """Use the fixed 4F objective, bounds, and two starts; no plant parameters."""
    params={}; diagnostics=[]
    for key in ('global',*base.FAMILIES):
        rows=records if key=='global' else [r for r in records if r['family']==key]
        x=np.concatenate([r['state'][:-1] for r in rows]); y=np.concatenate([r['state'][1:] for r in rows])
        u=np.concatenate([r['input'] for r in rows]); v=np.concatenate([r['speed'][:-1] for r in rows])
        def residual(logp): return ((step(np.exp(logp),x,u,v)-y)/np.deg2rad([.05,.5])).ravel()
        solutions=[least_squares(residual,np.log(p),bounds=(np.log([1.,1.,.05]),np.log([300.,300.,3.])),
                    ftol=1e-10,xtol=1e-10,gtol=1e-10) for p in ([40,40,.5],[80,60,1.])]
        result=min(solutions,key=lambda s:s.cost)
        if not all(s.success for s in solutions): raise RuntimeError('fit did not converge')
        p=np.exp(result.x); params[key]=p
        singular=np.linalg.svd(result.jac/np.linalg.norm(result.jac,axis=0),compute_uv=False)
        diagnostics.append(dict(group=key,cf_over_m=p[0],cr_over_m=p[1],m_over_iz=p[2],
            scaled_jacobian_condition=singular[0]/singular[-1],cost=result.cost,
            multistart_relative_difference=float(np.max(abs(np.exp(solutions[0].x)-np.exp(solutions[1].x))/p))))
        print('fit',diagnostics[-1],flush=True)
    models={m:StructuredModel('global' if m in ('M1','M3') else 'family',
        'constant' if m in ('M1','M2') else ('grid' if m=='M5' else 'lpv'),params) for m in base.METHODS}
    return models, diagnostics

def main():
    clients=sample_fleet(seed=1); time=np.arange(0,base.DURATION+base.DT,base.DT)
    physics=fit_oracle_architectures(clients,base.FIT_SPEEDS,base.DT)
    transfer=design_oracle_controllers(physics,base.DT,base.Q,base.R,base.CONTROL_SPEEDS,base.FIT_SPEEDS)
    records=[]
    for profile,maneuver in zip(base.TRAIN_PROFILES,('lane','sine')):
        v=base.smooth_profile(time,profile); ref=.5*base.reference_signal(time,maneuver)
        for c in clients:
            x,u=base.simulate_nonlinear(c,transfer['M4'],v,ref)
            records.append(dict(family=c.family,state=x,input=u,speed=v))
    models,diagnostics=fit_structured(records)
    controllers=redesign_controllers(models)
    audit=stability_audit(clients,{'structured':controllers})
    v=base.smooth_profile(time,base.TEST_PROFILE); rows=[]; summary=[]
    for severity,scale in base.SEVERITIES.items():
        ref=scale*base.reference_signal(time,'unseen_s_curve')
        for m in base.METHODS:
            selected=[]
            for c in clients:
                x,u=base.simulate_nonlinear(c,controllers[m],v,ref)
                row=dict(severity=severity,method=m,client=c.client_id,family=c.family,**trajectory_metrics(c,x,u,v,ref))
                rows.append(row); selected.append(row)
            summary.append(dict(severity=severity,method=m,tracking=float(np.mean([r['tracking_rmse_deg_s'] for r in selected])),
                worst_family=float(max(np.mean([r['tracking_rmse_deg_s'] for r in selected if r['family']==f]) for f in base.FAMILIES)),
                feasible=float(np.mean([r['feasible'] for r in selected])),
                rho=max(r['max_small_signal_spectral_radius'] for r in audit if r['method']==m)))
        print(severity,summary[-2],flush=True)
    for suffix,data in [('parameters',diagnostics),('clients',rows),('summary',summary),('stability',audit)]:
        base.write_csv(ROOT/f'results/tables/experiment_4f_{suffix}.csv',data)

if __name__=='__main__': main()
