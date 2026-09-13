"""9F: oracle rank gate for a shared LPV backbone plus local coordinates."""
from concurrent.futures import ProcessPoolExecutor
import argparse,gzip,hashlib,json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import experiment_9e_local_advantage_audit as ninee

ROOT=ninee.ROOT;OUT=ninee.OUT;base=ninee.base;corrected=ninee.corrected;nineb=ninee.nineb;ninea=ninee.ninea;nined=ninee.nined
CONFIG=ROOT/'code/config/experiment_9f.json'


def gain_vector(log_parameters,speeds):
    c=ninea.eightg.eightf.eighte.eightd.prior.eight.seven.seven.controller(np.exp(log_parameters))
    gains=np.vstack([np.array([np.interp(v,c.speeds,c.gains['global'][:,q]) for q in range(3)]) for v in speeds])
    return gains.ravel()


def gain_metric(center,cfg):
    step=cfg['gain_difference_step'];speeds=np.asarray(cfg['gain_speeds']);columns=[]
    for q in range(3):
        d=np.zeros(3);d[q]=step
        columns.append((gain_vector(center+d,speeds)-gain_vector(center-d,speeds))/(2*step))
    jac=np.column_stack(columns);scale=np.maximum(np.sqrt(np.mean(jac**2,axis=1)),1e-12);jac=jac/scale[:,None]
    weight=jac.T@jac/jac.shape[0];weight+=cfg['weight_floor']*np.eye(3)
    return weight


def learn_basis(log_parameters,weight,rank):
    center=np.mean(log_parameters,axis=0);delta=log_parameters-center
    eigenvalues,eigenvectors=np.linalg.eigh(weight);root=eigenvectors@np.diag(np.sqrt(eigenvalues))@eigenvectors.T;invroot=eigenvectors@np.diag(1/np.sqrt(eigenvalues))@eigenvectors.T
    _,singular,vh=np.linalg.svd(delta@root,full_matrices=False);directions=invroot@vh[:rank].T
    return center,directions,singular**2/np.sum(singular**2)


def reconstruct(log_parameters,center,directions,weight):
    if directions.shape[1]==0:return center.copy()
    gram=directions.T@weight@directions
    coefficients=np.linalg.solve(gram,directions.T@weight@(log_parameters-center))
    return center+directions@coefficients


def run_seed(seed):
    cfg=json.loads(CONFIG.read_text());train=nineb.discrete_fleet(seed,cfg['train_counts']);test=nineb.discrete_fleet(seed+10000,cfg['test_counts']);records,nominal=nined.collect_pool(test,seed+20000,cfg);local={};fits=[]
    for c in test:
        (count,p,cov,condition,d),_=nined.fit_policy(records[c.client_id],cfg);local[c.client_id]=p;fits.append(dict(seed=seed,client=c.client_id,family=c.family,success=d['success'],records=count,seconds=nined.calibration_seconds(count,cfg)))
    representations={};basis_rows=[]
    for family in base.FAMILIES:
        z=np.log([nineb.ratios(c) for c in train if c.family==family]);center=z.mean(0);identity=np.eye(3);control_weight=gain_metric(center,cfg)
        for kind,weight in [('Parameter',identity),('Control',control_weight)]:
            for rank in cfg['ranks']:
                mu,directions,explained=learn_basis(z,weight,rank);representations[family,kind,rank]=(mu,directions,weight)
                basis_rows.append(dict(seed=seed,family=family,kind=kind,rank=rank,shared_parameters=3+3*rank,local_coordinates=rank,training_variance_captured=float(explained[:rank].sum()),direction_1=np.nan if rank==0 else directions[0,0],direction_2=np.nan if rank==0 else directions[1,0],direction_3=np.nan if rank==0 else directions[2,0]))
    model_maps={'Local':local,'ExactIndividual':{c.client_id:nineb.ratios(c) for c in test}}
    metadata={'Local':(0,3),'ExactIndividual':(0,3)}
    for kind in ('Parameter','Control'):
        for rank in cfg['ranks']:
            name='ExactFamily' if kind=='Parameter' and rank==0 else f'{kind}Rank{rank}'
            if kind=='Control' and rank==0:continue
            model_maps[name]={}
            for c in test:
                center,directions,weight=representations[c.family,kind,rank];model_maps[name][c.client_id]=np.exp(reconstruct(np.log(nineb.ratios(c)),center,directions,weight))
            metadata[name]=(3+3*rank,rank)
    controls={m:{c.client_id:ninea.eightg.eightf.eighte.eightd.prior.eight.seven.seven.controller(models[c.client_id]) for c in test} for m,models in model_maps.items()};rows=[];diagnostics=[]
    for method,models in model_maps.items():
        for c in test:diagnostics.append(dict(seed=seed,method=method,client=c.client_id,family=c.family,shared_parameters=metadata[method][0],local_coordinates=metadata[method][1],parameter_error=float(np.linalg.norm(np.log(models[c.client_id]/nineb.ratios(c)))),gain_error=ninee.gain_error(controls[method][c.client_id],controls['ExactIndividual'][c.client_id])))
    t=np.arange(0,24+base.DT,base.DT)
    for scenario in cfg['test_scenarios']:
        v,ref=corrected.scenario(scenario,t/2);v=corrected.six.remap(v,(10,30))
        for method,control_map in controls.items():
            _,_,metrics=corrected.simulate(test,nominal,v,ref,'vy',client_controllers=control_map)
            for j,c in enumerate(test):rows.append(dict(seed=seed,method=method,scenario=scenario,client=c.client_id,family=c.family,shared_parameters=metadata[method][0],local_coordinates=metadata[method][1],**{q:float(x[j]) for q,x in metrics.items()}))
    for suffix,items in [('fits',fits),('bases',basis_rows),('diagnostics',diagnostics),('clients',rows)]:
        (OUT/f'experiment_9f_seed{seed}_{suffix}.csv.gz').write_bytes(gzip.compress(pd.DataFrame(items).to_csv(index=False).encode(),mtime=0))
    print(seed,flush=True)


def summarize():
    cfg=json.loads(CONFIG.read_text());raw=pd.concat([pd.read_csv(OUT/f'experiment_9f_seed{s}_clients.csv.gz') for s in cfg['seeds']]);diag=pd.concat([pd.read_csv(OUT/f'experiment_9f_seed{s}_diagnostics.csv.gz') for s in cfg['seeds']]);bases=pd.concat([pd.read_csv(OUT/f'experiment_9f_seed{s}_bases.csv.gz') for s in cfg['seeds']]);fits=pd.concat([pd.read_csv(OUT/f'experiment_9f_seed{s}_fits.csv.gz') for s in cfg['seeds']])
    units=raw.groupby(['seed','method','shared_parameters','local_coordinates']).agg(tracking=('tracking','mean'),feasible=('feasible','min')).reset_index();summary=units.groupby(['method','shared_parameters','local_coordinates']).agg(tracking=('tracking','mean'),tracking_std=('tracking','std'),feasible_rate=('feasible','mean')).reset_index();dsummary=diag.groupby(['method','shared_parameters','local_coordinates']).agg(parameter_error=('parameter_error','mean'),gain_error=('gain_error','mean')).reset_index();summary=summary.merge(dsummary,on=['method','shared_parameters','local_coordinates']);s=summary.set_index('method');family=s.loc['ExactFamily','tracking'];exact=s.loc['ExactIndividual','tracking'];local=s.loc['Local','tracking'];comparisons=[]
    for method,row in s.iterrows():
        closed=100*(family-row.tracking)/(family-exact) if family!=exact else np.nan
        comparisons.append(dict(method=method,tracking_vs_local_pct=100*(row.tracking/local-1),tracking_vs_exact_pct=100*(row.tracking/exact-1),exact_family_gap_closed_pct=closed))
    comparisons=pd.DataFrame(comparisons);rank1=bool(s.loc['ControlRank1','tracking']<=local);rank2=bool(s.loc['ControlRank2','tracking']<=1.01*exact and s.loc['ControlRank2','local_coordinates']<3);rank3_error=float(max(s.loc['ParameterRank3','parameter_error'],s.loc['ControlRank3','parameter_error']));conclusions=dict(provenance={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [CONFIG,ROOT/'code/experiments/experiment_9f_personalized_manifold.py']},fits=len(fits),all_fits_converged=bool(fits.success.all()),closed_loop_evaluations=len(raw),all_feasible=bool(raw.feasible.all()),rank1_gate=rank1,rank2_gate=rank2,rank3_reconstruction_error=rank3_error)
    summary.to_csv(OUT/'experiment_9f_summary.csv',index=False);comparisons.to_csv(OUT/'experiment_9f_comparisons.csv',index=False);bases.groupby(['family','kind','rank']).agg(training_variance_captured=('training_variance_captured','mean')).reset_index().to_csv(OUT/'experiment_9f_basis_summary.csv',index=False);units.to_csv(OUT/'experiment_9f_seed_summary.csv',index=False);(OUT/'experiment_9f_conclusions.json').write_text(json.dumps(conclusions,indent=2)+'\n')
    order=['ExactFamily','ParameterRank1','ControlRank1','ParameterRank2','ControlRank2','Local','ExactIndividual'];view=s.loc[order];fig,ax=plt.subplots(figsize=(8.4,4.1));colors=['.65','#8ebad9','#2878b5','#8ebad9','#2878b5','#d9822b','#3c9d5d'];ax.bar(np.arange(len(order)),view.tracking,color=colors);ax.set(xticks=np.arange(len(order)),xticklabels=['Family','Param. r1','Control r1','Param. r2','Control r2','Local fit','Exact indiv.'],ylabel='Tracking RMSE [rad/s]');ax.tick_params(axis='x',rotation=20);ax.grid(axis='y',alpha=.2);fig.tight_layout();fig.savefig(ROOT/'results/figures/experiment_9f_personalized_manifold.pdf');plt.close(fig)
    print(json.dumps(conclusions,indent=2));print(summary.sort_values(['local_coordinates','method']).to_string(index=False));print(comparisons.to_string(index=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=5);p.add_argument('--summarize-only',action='store_true');a=p.parse_args();cfg=json.loads(CONFIG.read_text())
    if not a.summarize_only:
        with ProcessPoolExecutor(max_workers=a.workers) as pool:list(pool.map(run_seed,cfg['seeds']))
    summarize()
