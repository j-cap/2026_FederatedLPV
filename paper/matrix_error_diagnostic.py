"""Post-hoc metric sensitivity diagnostic using saved maps; no fitting or simulation."""
from pathlib import Path
import sys
import json
import hashlib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'code/src'),str(ROOT/'code/experiments')]
from federated_lpv import sample_fleet
from experiment_10a_basis_coverage_audit import collect_targets
from experiment_10b_oracle_complementary_gate import predict
OUT=ROOT/'results/tables'
ENTRIES=['A11','A12','A21','A22','B1','B2']


def load(p):
    return json.loads(p.read_text())


def digest(p):
    b=p.read_bytes()
    if p.suffix in {'.json','.csv','.py'}: b=b.replace(b'\r\n',b'\n')
    return hashlib.sha256(b).hexdigest()


def balanced_error(error, scales):
    return np.sqrt(np.mean((np.asarray(error)/np.asarray(scales))**2,axis=-1))


def main():
    basis=load(ROOT/'code/config/experiment_10a.json')
    selection=load(OUT/'paper_local_smoothness_selection.json')
    names=set(selection['source_sha256'])|set(selection['development_sha256'])
    for n,h in {**selection['source_sha256'],**selection['development_sha256']}.items():
        assert digest(ROOT/n)==h,n
    # Fix entry scales from saved development references, before reading primary maps.
    references=[]
    for seed in selection['development_seeds']:
        d=load(OUT/f'paper_local_smoothness_development{seed}.json')
        references.extend(np.asarray(i['reference']) for i in d['frozen_inputs'])
    reference=np.concatenate(references)
    scales=np.sqrt(np.mean(reference**2,axis=0))
    assert scales.shape==(6,) and (scales>0).all()
    rows=[]; max_parent=0.
    for seed in range(371,381):
        pp=OUT/f'paper_review_validation_seed{seed}.json'
        sp=OUT/f'paper_local_smoothness_primary{seed}.json'
        names.update([pp.relative_to(ROOT).as_posix(),sp.relative_to(ROOT).as_posix()])
        parent,smooth=load(pp),load(sp)
        assert smooth['selection_sha256']==digest(OUT/'paper_local_smoothness_selection.json')
        for n,h in parent['source_sha256'].items(): assert digest(ROOT/n.replace('\\','/'))==h,n
        models={i['client']:np.asarray(i['coefficients']) for i in smooth['models']}
        for index,client in enumerate(sample_fleet(seed)):
            item=parent['frozen_inputs'][index]
            assert client.client_id==item['client']
            _,truth=collect_targets(client,.005,basis)
            grid=np.asarray(basis['speed_grid'])
            maps=dict(Global=np.asarray(parent['frozen_models']['global_model']),
                FederatedLearned=np.asarray(parent['frozen_models']['groups'][parent['frozen_models']['assignments'][index]]),
                SmoothLocal=models[client.client_id])
            for method,theta in maps.items():
                error=predict(theta,grid,basis)-truth
                original=np.linalg.norm(error,axis=1)/np.linalg.norm(truth,axis=1)
                archived=smooth['errors'] if method=='SmoothLocal' else parent['errors']
                expected=[r['prediction_error'] for r in archived if r['client']==client.client_id and r['method']==method]
                np.testing.assert_allclose(original,expected,rtol=0,atol=1e-8)
                max_parent=max(max_parent,float(np.max(np.abs(original-expected))))
                balanced=balanced_error(error,scales)
                for j,v in enumerate(grid):
                    row=dict(seed=seed,client=client.client_id,method=method,speed=float(v),
                        region='seen' if v in item['speeds'] else 'unseen',original=float(original[j]),balanced=float(balanced[j]))
                    row.update({f'{name}_error':float(error[j,k]) for k,name in enumerate(ENTRIES)})
                    rows.append(row)
    raw=pd.DataFrame(rows)
    units=[]
    for region in ['full','seen','unseen']:
        view=raw if region=='full' else raw[raw.region==region]
        for (seed,method),g in view.groupby(['seed','method']):
            row=dict(seed=seed,method=method,region=region,original=g.original.mean(),balanced=g.balanced.mean())
            for j,e in enumerate(ENTRIES):
                row[f'{e}_rmse']=float(np.sqrt(np.mean(g[f'{e}_error']**2)))
                row[f'{e}_scaled_mae']=float(np.mean(np.abs(g[f'{e}_error']))/scales[j])
            units.append(row)
    units=pd.DataFrame(units);comparisons=[];rng=np.random.default_rng(20261013)
    for region in ['full','seen','unseen']:
        view=units[units.region==region]
        for metric in ['original','balanced']+[f'{e}_scaled_mae' for e in ENTRIES]:
            p=view.pivot(index='seed',columns='method',values=metric)
            for baseline in ['Global','SmoothLocal']:
                gains=100*(1-p.FederatedLearned/p[baseline])
                ci=np.quantile(rng.choice(gains,(20000,len(gains))).mean(axis=1),[.025,.975])
                comparisons.append(dict(region=region,metric=metric,baseline=baseline,gain_pct=float(gains.mean()),
                    ci_low=float(ci[0]),ci_high=float(ci[1]),wins=int((gains>1e-10).sum())))
    raw.to_csv(OUT/'paper_matrix_diagnostic_clients.csv',index=False)
    units.to_csv(OUT/'paper_matrix_diagnostic_fleets.csv',index=False)
    pd.DataFrame(comparisons).to_csv(OUT/'paper_matrix_diagnostic_comparisons.csv',index=False)
    names.update(['results/tables/paper_local_smoothness_selection.json','paper/matrix_error_diagnostic.py'])
    manifest=dict(status='completed_no_fits_no_simulations',development_seeds=selection['development_seeds'],
        primary_seeds=list(range(371,381)),entries=ENTRIES,scales=scales.tolist(),
        scale_rule='entry RMS of saved exact Jacobians across development clients and all eleven speeds',
        balanced_rule='mean over client-speed pairs of sqrt((1/6)*sum_j(((estimated_j-reference_j)/scale_j)^2))',
        maximum_original_error_difference=max_parent,source_sha256={n:digest(ROOT/n) for n in sorted(names)},
        output_sha256={n:digest(OUT/n) for n in ['paper_matrix_diagnostic_clients.csv','paper_matrix_diagnostic_fleets.csv','paper_matrix_diagnostic_comparisons.csv']})
    (OUT/'paper_matrix_diagnostic_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(pd.DataFrame(comparisons).query("metric == 'balanced' or (metric.str.endswith('scaled_mae') and region == 'unseen' and baseline == 'Global')",engine='python').to_string(index=False))
    print('Development entry scales:',scales,'Original error difference:',max_parent)


if __name__=='__main__': main()
