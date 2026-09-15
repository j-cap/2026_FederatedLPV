"""9I: one-step fleet-informed active identification for controller design."""
from concurrent.futures import ProcessPoolExecutor
from federated_lpv.fleet import VehicleClient
from federated_lpv.vehicle import family_centers
from federated_lpv.output_error import fit,prepare,predict
from federated_lpv.privacy import normalize_log_parameters,transform
from federated_lpv.uncertainty_clustering import log_parameter_covariance,normalized_covariance
import argparse,gzip,hashlib,json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score
import experiment_9h_calibration_frontier as nineh
import experiment_9c_uncertainty_groups as ninec

ROOT=nineh.ROOT;OUT=nineh.OUT;base=nineh.base;corrected=nineh.corrected;nineb=nineh.nineb;ninea=nineh.ninea;nined=nineh.nined;nineg=nineh.nineg
CONFIG=ROOT/'code/config/experiment_9i.json'


def maneuver_record(client,control,spec,duration,phase,noise,rng):
    n=round(duration/base.DT);t=np.arange(n+1)*base.DT;speed=np.full_like(t,spec['speed'],dtype=float);reference=np.deg2rad(3)*(1-np.exp(-t/.18))*np.sin(2*np.pi*spec['frequency']*t+phase);x,u,m=corrected.simulate([client],control,speed,reference,'vy')
    if not m['feasible'].all():raise RuntimeError('infeasible calibration maneuver')
    return dict(state=x[:,0]+rng.normal(size=x[:,0].shape)*np.deg2rad(noise),input=u[:,0],speed=speed)


def collect(clients,seed,cfg):
    nominal=ninea.eightg.eightf.eighte.eightd.prior.eight.seven.seven.controller(np.array([80000/1500,80000/1500,1500/2500]));rng=np.random.default_rng(seed+900000);order=rng.permutation(len(clients));initial_speeds=[10,20,30];initial={};candidates={}
    for j,c in enumerate(clients):
        v=initial_speeds[int(order[j]%3)];spec={'speed':v,'frequency':.35};initial[c.client_id]=[maneuver_record(c,nominal,spec,cfg['initial_record_duration'],.7*q,cfg['noise_deg'],rng) for q in range(cfg['initial_records'])];candidates[c.client_id]={s['name']:maneuver_record(c,nominal,s,cfg['candidate_duration'],1.1,cfg['noise_deg'],rng) for s in cfg['candidate_maneuvers']}
    return initial,candidates,nominal


def template_records(cfg):
    c=VehicleClient('template','nominal',family_centers()['nominal']);nominal=ninea.eightg.eightf.eighte.eightd.prior.eight.seven.seven.controller(np.array([80000/1500,80000/1500,1500/2500]));rng=np.random.default_rng(1)
    return {s['name']:maneuver_record(c,nominal,s,cfg['candidate_duration'],1.1,[0,0],rng) for s in cfg['candidate_maneuvers']}


def coefficient_covariance(records,center,directions,a,cfg):
    data=prepare(records);scales=np.deg2rad(np.asarray(cfg['noise_deg']));h=1e-5
    def residual(q):return ((predict(np.exp(center+directions@q),data)-data['y'])/scales).ravel()
    jac=np.column_stack([(residual(a+h*np.eye(len(a))[j])-residual(a-h*np.eye(len(a))[j]))/(2*h) for j in range(len(a))]);information=jac.T@jac+len(jac)*cfg['coefficient_prior_strength']/cfg['coefficient_prior_scale']**2*np.eye(len(a));return np.linalg.inv(information)


def local_scores(initial,p,templates,cfg,control_aware):
    center,half=transform(np.asarray(cfg['bounds']));scale=np.diag(1/(2*half));weight=nineh.nineg.ninef.gain_metric(np.log(p),cfg) if control_aware else scale.T@scale;scores={}
    for name,record in templates.items():cov,_=log_parameter_covariance(initial+[record],p,cfg['noise_deg']);scores[name]=float(np.trace(weight@cov))
    return scores


def fleet_scores(initial,center,directions,a,templates,cfg,control_aware):
    weight=nineh.nineg.ninef.gain_metric(center+directions@a,cfg) if control_aware else np.eye(3);scores={}
    for name,record in templates.items():cov=coefficient_covariance(initial+[record],center,directions,a,cfg);scores[name]=float(np.trace(directions.T@weight@directions@cov))
    return scores


def choose(scores):return min(scores,key=scores.get)


def run_seed(seed):
    cfg=json.loads(CONFIG.read_text());train=nineb.discrete_fleet(seed,cfg['train_counts']);test=nineb.discrete_fleet(seed+10000,cfg['test_counts']);train_records,_=nined.collect_pool(train,seed+70000,cfg);initial,candidates,nominal=collect(test,seed,cfg);templates=template_records(cfg);bounds=np.asarray(cfg['bounds']);train_p=[];train_cov=[];fits=[]
    for c in train:
        p,d=fit(train_records[c.client_id]);cov,_=log_parameter_covariance(train_records[c.client_id],p,cfg['noise_deg']);train_p.append(p);train_cov.append(normalized_covariance(cov,bounds));fits.append(dict(seed=seed,scope='train',method='Backbone',client=c.client_id,success=d['success']))
    train_p=np.asarray(train_p);z=normalize_log_parameters(train_p,bounds);mixture,mixture_candidates=ninec.select_mixture(z,np.asarray(train_cov),cfg,seed);labels=np.argmax(mixture['responsibility'],axis=1);truth=np.array([base.FAMILIES.index(c.family) for c in train]);backbones=nineg.build_backbones(train_p,labels,mixture['k'],{**cfg,'ranks':[cfg['rank']]});methods=['LocalFixed','LocalParameterActive','LocalControlActive','FleetFixed','FleetParameterActive','FleetControlActive','OracleBestFleet'];models={m:{} for m in methods};choices=[]
    for c in test:
        init=initial[c.client_id];p0,d=fit(init);local_parameter=choose(local_scores(init,p0,templates,cfg,False));local_control=choose(local_scores(init,p0,templates,cfg,True));local_choice={'LocalFixed':cfg['fixed_candidate'],'LocalParameterActive':local_parameter,'LocalControlActive':local_control}
        for method,name in local_choice.items():p,d=fit(init+[candidates[c.client_id][name]]);models[method][c.client_id]=p;fits.append(dict(seed=seed,scope='test',method=method,client=c.client_id,success=d['success']));choices.append(dict(seed=seed,client=c.client_id,family=c.family,method=method,choice=name))
        _,group,p_init,a_init,d=nineh.select_personalized(init,backbones,cfg['rank'],cfg);center,directions=backbones[group,cfg['rank']];fleet_parameter=choose(fleet_scores(init,center,directions,a_init,templates,cfg,False));fleet_control=choose(fleet_scores(init,center,directions,a_init,templates,cfg,True));fleet_choice={'FleetFixed':cfg['fixed_candidate'],'FleetParameterActive':fleet_parameter,'FleetControlActive':fleet_control}
        exact_control=ninea.eightg.eightf.eighte.eightd.prior.eight.seven.seven.controller(nineb.ratios(c));oracle=[]
        for name in candidates[c.client_id]:
            p,a,d=nineg.coefficient_fit(init+[candidates[c.client_id][name]],center,directions,cfg);ctrl=ninea.eightg.eightf.eighte.eightd.prior.eight.seven.seven.controller(p);oracle.append((nineh.nineg.ninef.ninee.gain_error(ctrl,exact_control),name,p,d))
        _,oracle_name,oracle_p,oracle_d=min(oracle,key=lambda x:x[0]);models['OracleBestFleet'][c.client_id]=oracle_p;fits.append(dict(seed=seed,scope='test',method='OracleBestFleet',client=c.client_id,success=oracle_d['success']));choices.append(dict(seed=seed,client=c.client_id,family=c.family,method='OracleBestFleet',choice=oracle_name))
        for method,name in fleet_choice.items():p,a,d=nineg.coefficient_fit(init+[candidates[c.client_id][name]],center,directions,cfg);models[method][c.client_id]=p;fits.append(dict(seed=seed,scope='test',method=method,client=c.client_id,success=d['success']));choices.append(dict(seed=seed,client=c.client_id,family=c.family,method=method,choice=name))
    controls={m:{c.client_id:ninea.eightg.eightf.eighte.eightd.prior.eight.seven.seven.controller(models[m][c.client_id]) for c in test} for m in methods};rows=[];diagnostics=[]
    for method in methods:
        for c in test:diagnostics.append(dict(seed=seed,method=method,client=c.client_id,family=c.family,parameter_error=float(np.linalg.norm(np.log(models[method][c.client_id]/nineb.ratios(c)))),gain_error=nineh.nineg.ninef.ninee.gain_error(controls[method][c.client_id],ninea.eightg.eightf.eighte.eightd.prior.eight.seven.seven.controller(nineb.ratios(c)))))
    t=np.arange(0,24+base.DT,base.DT)
    for scenario in cfg['test_scenarios']:
        v,ref=corrected.scenario(scenario,t/2);v=corrected.six.remap(v,(10,30))
        for method,control_map in controls.items():
            _,_,metrics=corrected.simulate(test,nominal,v,ref,'vy',client_controllers=control_map)
            for j,c in enumerate(test):rows.append(dict(seed=seed,method=method,scenario=scenario,client=c.client_id,family=c.family,**{q:float(x[j]) for q,x in metrics.items()}))
    clusters=[dict(seed=seed,selected_k=mixture['k'],train_ari=adjusted_rand_score(truth,labels),candidate_k=x['k'],bic=x['bic'],minimum_size=x['minimum_size'],admissible=x['admissible']) for x in mixture_candidates]
    for suffix,items in [('fits',fits),('choices',choices),('diagnostics',diagnostics),('clusters',clusters),('clients',rows)]:
        (OUT/f'experiment_9i_seed{seed}_{suffix}.csv.gz').write_bytes(gzip.compress(pd.DataFrame(items).to_csv(index=False).encode(),mtime=0))
    print(seed,flush=True)


def summarize():
    cfg=json.loads(CONFIG.read_text());raw=pd.concat([pd.read_csv(OUT/f'experiment_9i_seed{s}_clients.csv.gz') for s in cfg['seeds']]);diag=pd.concat([pd.read_csv(OUT/f'experiment_9i_seed{s}_diagnostics.csv.gz') for s in cfg['seeds']]);fits=pd.concat([pd.read_csv(OUT/f'experiment_9i_seed{s}_fits.csv.gz') for s in cfg['seeds']]);choices=pd.concat([pd.read_csv(OUT/f'experiment_9i_seed{s}_choices.csv.gz') for s in cfg['seeds']]);clusters=pd.concat([pd.read_csv(OUT/f'experiment_9i_seed{s}_clusters.csv.gz') for s in cfg['seeds']]);units=raw.groupby(['seed','method']).agg(tracking=('tracking','mean'),feasible=('feasible','min')).reset_index();summary=units.groupby('method').agg(tracking=('tracking','mean'),tracking_std=('tracking','std'),feasible_rate=('feasible','mean')).reset_index().merge(diag.groupby('method').agg(parameter_error=('parameter_error','mean'),gain_error=('gain_error','mean')).reset_index(),on='method');s=summary.set_index('method');comparisons=pd.DataFrame([dict(method=m,improvement_vs_local_fixed_pct=100*(1-r.tracking/s.loc['LocalFixed','tracking']),improvement_vs_own_fixed_pct=100*(1-r.tracking/s.loc['FleetFixed' if m.startswith('Fleet') else 'LocalFixed','tracking'])) for m,r in s.iterrows()]);choice_summary=choices.groupby(['method','choice']).size().rename('clients').reset_index();selected=clusters[clusters.candidate_k==clusters.selected_k].groupby('seed').first();fleet_control=bool(s.loc['FleetControlActive','tracking']<s.loc['FleetFixed','tracking'] and s.loc['FleetControlActive','tracking']<s.loc['LocalControlActive','tracking']);conclusions=dict(provenance={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [CONFIG,ROOT/'code/experiments/experiment_9i_active_identification.py']},fits=len(fits),all_fits_converged=bool(fits.success.all()),closed_loop_evaluations=len(raw),all_feasible=bool(raw.feasible.all()),selected_k_mean=float(selected.selected_k.mean()),train_ari=float(selected.train_ari.mean()),fleet_control_active_gate=fleet_control)
    summary.to_csv(OUT/'experiment_9i_summary.csv',index=False);comparisons.to_csv(OUT/'experiment_9i_comparisons.csv',index=False);choice_summary.to_csv(OUT/'experiment_9i_choices.csv',index=False);units.to_csv(OUT/'experiment_9i_seed_summary.csv',index=False);(OUT/'experiment_9i_conclusions.json').write_text(json.dumps(conclusions,indent=2)+'\n');order=['LocalFixed','LocalParameterActive','LocalControlActive','FleetFixed','FleetParameterActive','FleetControlActive','OracleBestFleet'];view=s.loc[order];fig,ax=plt.subplots(figsize=(8.7,4.2));ax.bar(np.arange(len(order)),view.tracking,color=['.65','#d9a066','#d9822b','.45','#8ebad9','#2878b5','#3c9d5d']);ax.set(xticks=np.arange(len(order)),xticklabels=['Local fixed','Local param.','Local control','Fleet fixed','Fleet param.','Fleet control','Oracle fleet'],ylabel='Tracking RMSE [rad/s]');ax.tick_params(axis='x',rotation=22);ax.grid(axis='y',alpha=.2);fig.tight_layout();fig.savefig(ROOT/'results/figures/experiment_9i_active_identification.pdf');plt.close(fig);print(json.dumps(conclusions,indent=2));print(summary.to_string(index=False));print(comparisons.to_string(index=False));print(choice_summary.to_string(index=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=5);p.add_argument('--summarize-only',action='store_true');a=p.parse_args();cfg=json.loads(CONFIG.read_text())
    if not a.summarize_only:
        with ProcessPoolExecutor(max_workers=a.workers) as pool:list(pool.map(run_seed,cfg['seeds']))
    summarize()
