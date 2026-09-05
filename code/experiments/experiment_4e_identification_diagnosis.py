"""4E: controlled identification ablations; no changes to historical 4C/4D."""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from federated_lpv import (sample_fleet, fit_oracle_architectures,
    design_oracle_controllers, discrete_bicycle_matrices, augmented_tracking_matrices)
import experiment_4c_nonlinear_oracle_models as base
from experiment_4d_nonlinear_controllers import redesign_controllers, trajectory_metrics

ROOT = Path(__file__).resolve().parents[2]

def features(records):
    xs, ys = [], []
    for row in records:
        z = np.column_stack((row['state'][:-1], row['input']))
        v = row['speed'][:-1]
        phi = np.column_stack((np.ones(len(v)), 1/v, 1/v**2))
        xs.append(np.einsum('ni,nj->nij', phi, z).reshape(len(v), -1))
        ys.append(row['state'][1:])
    return np.concatenate(xs), np.concatenate(ys)

def fit(records, stable_solver=False):
    models = base.fit_models(records)
    if stable_solver:
        for method in ('M3', 'M4'):
            model = models[method]
            for key in model.coefficients:
                rows = records if key == 'global' else [r for r in records if r['family'] == key]
                x, y = features(rows)
                scale = np.linalg.norm(x, axis=0)
                # Preserve exactly the original raw-coordinate ridge objective.
                xa = np.vstack((x/scale, np.sqrt(base.RIDGE)*np.diag(1/scale)))
                ya = np.vstack((y, np.zeros((x.shape[1], 2))))
                coef = (np.linalg.lstsq(xa, ya, rcond=None)[0]/scale[:, None]).T
                model.coefficients[key] = coef.reshape(2, 3, 3).transpose(1, 0, 2)
    return models

def main():
    clients = sample_fleet(seed=1)
    time = np.arange(0, base.DURATION+base.DT, base.DT)
    physics = fit_oracle_architectures(clients, base.FIT_SPEEDS, base.DT)
    transfer = design_oracle_controllers(physics, base.DT, base.Q, base.R,
                                       base.CONTROL_SPEEDS, base.FIT_SPEEDS)
    datasets = {'nonlinear_all': [], 'nonlinear_near': [], 'linear_all': []}
    for profile, maneuver in zip(base.TRAIN_PROFILES, ('lane', 'sine')):
        speed = base.smooth_profile(time, profile)
        for client in clients:
            matrices = [discrete_bicycle_matrices(float(v), client.parameters, base.DT) for v in speed[:-1]]
            for severity, amplitude in base.SEVERITIES.items():
                reference = amplitude*base.reference_signal(time, maneuver)
                state, u = base.simulate_nonlinear(client, transfer['M4'], speed, reference)
                row = dict(family=client.family, client_id=client.client_id, speed=speed, state=state, input=u)
                datasets['nonlinear_all'].append(row)
                if severity == 'near_linear': datasets['nonlinear_near'].append(row)
                state = np.zeros_like(state); u = np.zeros_like(u); integral = 0.
                for k, (a,b) in enumerate(matrices):
                    gain, pre = transfer['M4'].evaluate(client.family, float(speed[k]))
                    u[k] = -gain@np.r_[state[k], integral]+pre*reference[k]
                    state[k+1] = a@state[k]+b[:,0]*u[k]
                    integral += base.DT*(state[k,1]-reference[k])
                datasets['linear_all'].append(dict(family=client.family, client_id=client.client_id,
                    speed=speed, state=state, input=u))
        print('generated', profile, flush=True)
    conditioning=[]; summary=[]; detail=[]; curves=[]; individual=[]
    fine = np.linspace(12,28,161)
    true = {(c.client_id,j): discrete_bicycle_matrices(float(v),c.parameters,base.DT)
            for c in clients for j,v in enumerate(fine)}
    # Separate pooling mismatch from single-client identifiability and ridge bias.
    for client in clients:
        if not client.client_id.endswith('_00'): continue
        x,y=features([r for r in datasets['linear_all'] if r['client_id']==client.client_id])
        scale=np.linalg.norm(x,axis=0)
        for ridge in (0.,base.RIDGE):
            xa=np.vstack((x/scale,np.sqrt(ridge)*np.diag(1/scale)))
            ya=np.vstack((y,np.zeros((9,2))))
            raw=(np.linalg.lstsq(xa,ya,rcond=None)[0]/scale[:,None]).T
            coef=raw.reshape(2,3,3).transpose(1,0,2)
            errors=[]
            for j,v in enumerate(fine):
                theta=np.tensordot([1,1/v,1/v**2],coef,axes=1)
                a,b=true[client.client_id,j]
                errors.append(np.linalg.norm(theta[:,2:]-b)/np.linalg.norm(b))
            individual.append(dict(client=client.client_id,ridge=ridge,
                mean_relative_b=float(np.mean(errors)),
                scaled_condition=float(np.linalg.cond(x/scale))))
    for name, records in datasets.items():
        for key in ('global', *base.FAMILIES):
            x,y=features(records if key=='global' else [r for r in records if r['family']==key])
            s=np.linalg.svd(x,compute_uv=False); scaled=np.linalg.svd(x/np.linalg.norm(x,axis=0),compute_uv=False)
            conditioning.append(dict(dataset=name,group=key,condition_raw=float(s[0]/s[-1]),
                condition_scaled=float(scaled[0]/scaled[-1]),rank=int(np.linalg.matrix_rank(x))))
    cases=[('nonlinear_all',False),('nonlinear_all',True),('nonlinear_near',True),('linear_all',True)]
    speed=base.smooth_profile(time,base.TEST_PROFILE)
    reference=base.reference_signal(time,'unseen_s_curve')
    for dataset, solver in cases:
        label=dataset+('_svd' if solver else '_original')
        models=fit(datasets[dataset],solver)
        controllers=redesign_controllers(models)
        for method in base.METHODS:
            rows=[]
            for client in clients:
                radii=[]; ea=[]; eb=[]
                for j,v in enumerate(fine):
                    a,b=true[client.client_id,j]; ah,bh=models[method].matrix(client.family,float(v))
                    aa,bb=augmented_tracking_matrices(a,b,base.DT)
                    gain,pre=controllers[method].evaluate(client.family,float(v))
                    rho=float(max(abs(np.linalg.eigvals(aa-bb@gain[None,:]))))
                    radii.append(rho)
                    ea.append(float(np.linalg.norm(ah-a)/np.linalg.norm(a)))
                    eb.append(float(np.linalg.norm(bh-b)/np.linalg.norm(b)))
                    if method=='M4' and client.client_id.endswith('_00'):
                        curves.append(dict(case=label,client=client.client_id,speed=float(v),rho=rho,
                            error_a=ea[-1],error_b=eb[-1],k_beta=float(gain[0]),k_yaw=float(gain[1]),k_integral=float(gain[2]),prefilter=pre))
                state,u=base.simulate_nonlinear(client,controllers[method],speed,reference)
                met=trajectory_metrics(client,state,u,speed,reference)
                rows.append(dict(case=label,method=method,client=client.client_id,family=client.family,
                    rho=max(radii),worst_speed=float(fine[np.argmax(radii)]),
                    mean_relative_a=float(np.mean(ea)),mean_relative_b=float(np.mean(eb)),**met))
            detail.extend(rows)
            worst=max(rows,key=lambda r:r['rho'])
            summary.append(dict(case=label,method=method,rho=worst['rho'],worst_client=worst['client'],
                worst_speed=worst['worst_speed'],tracking=float(np.mean([r['tracking_rmse_deg_s'] for r in rows])),
                feasible=float(np.mean([r['feasible'] for r in rows])),
                mean_relative_a=float(np.mean([r['mean_relative_a'] for r in rows])),
                mean_relative_b=float(np.mean([r['mean_relative_b'] for r in rows]))))
        print(label, summary[-2], flush=True)
    out=ROOT/'results/tables'
    for suffix, rows in [('summary',summary),('conditioning',conditioning),('clients',detail),('curves',curves),('individual',individual)]:
        base.write_csv(out/f'experiment_4e_{suffix}.csv',rows)
    fig,axes=plt.subplots(1,3,figsize=(11,3.3),constrained_layout=True)
    for label in dict.fromkeys(r['case'] for r in summary):
        rows=[r for r in curves if r['case']==label and r['client']=='heavy_00']
        for ax,metric in zip(axes,('rho','error_b','k_beta')):
            ax.plot([r['speed'] for r in rows],[r[metric] for r in rows],label=label.replace('_',' '))
    for ax,title in zip(axes,('Frozen spectral radius','Relative B error','Sideslip feedback gain')):
        ax.set(xlabel='speed [m/s]',title=title); ax.grid(alpha=.2)
    axes[0].axhline(1,color='k',linestyle=':'); axes[1].set_yscale('log')
    axes[0].legend(fontsize=6)
    fig.savefig(ROOT/'results/figures/experiment_4e_diagnosis.pdf')
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__': main()
