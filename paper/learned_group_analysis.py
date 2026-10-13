"""Descriptive analysis of frozen primary groups; no fitting or simulations."""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'code/src'),str(ROOT/'code/experiments')]
from federated_lpv import sample_fleet
from experiment_10a_basis_coverage_audit import raw_basis
from experiment_10b_oracle_complementary_gate import predict
from paper_review_validation import digest
OUT=ROOT/'results/tables'


def main():
    basis=json.loads((ROOT/'code/config/experiment_10a.json').read_text())
    groups=[];clients=[];pairs=[];fleet=[];sources={}
    for seed in range(371,381):
        path=OUT/f'paper_review_validation_seed{seed}.json'
        p=json.loads(path.read_text()); sources[path.relative_to(ROOT).as_posix()]=digest(path)
        for name,sha in p['source_sha256'].items():
            assert digest(ROOT/name.replace('\\','/'))==sha,name
        plants=sample_fleet(seed); assignments=np.asarray(p['frozen_models']['assignments'])
        assert len(plants)==len(assignments)==len(p['frozen_inputs'])==30
        targets=np.concatenate([np.asarray(i['targets']) for i in p['frozen_inputs']])
        scales=targets.std(axis=0)
        scales=np.maximum(scales,.05*np.median(scales))
        saved_errors=pd.DataFrame(p['errors'])
        for g in range(p['selected_k']):
            ix=np.flatnonzero(assignments==g)
            speeds=np.concatenate([p['frozen_inputs'][i]['speeds'] for i in ix])
            phi=raw_basis(speeds,7,basis)
            singular=np.linalg.svd(phi,compute_uv=False)
            members=[plants[i].client_id for i in ix]
            archived=next(a for a in p['groups'] if a['group']==g)
            assert set(members)==set(archived['members'])
            assert np.linalg.matrix_rank(phi)==archived['rank']==7
            np.testing.assert_allclose(singular[0]/singular[-1],archived['design_condition'],rtol=1e-10)
            row=dict(seed=seed,selected_k=p['selected_k'],group=g,clients=len(ix),rank=7,
                condition=float(singular[0]/singular[-1]),unique_speeds=len(set(speeds)))
            row.update({f:sum(plants[i].family==f for i in ix) for f in ['nominal','heavy','handling']})
            for block in range(4):
                row[f'block{block}']=sum(p['frozen_inputs'][i]['speeds']==[[10,12,14],[16,18,20],[22,24,26],[28,30]][block] for i in ix)
            groups.append(row)
        for i,c in enumerate(plants):
            inp=p['frozen_inputs'][i];assert c.client_id==inp['client']
            y=np.asarray(inp['targets']); speed=np.asarray(inp['speeds'])
            model=np.asarray(p['frozen_models']['groups'][assignments[i]])
            global_map=np.asarray(p['frozen_models']['global_model'])
            score=float(np.sum(((y-predict(model,speed,basis))/scales)**2))
            global_score=float(np.sum(((y-predict(global_map,speed,basis))/scales)**2))
            e=saved_errors[(saved_errors.client==c.client_id)&(saved_errors.region=='unseen')].groupby('method').prediction_error.mean()
            clients.append(dict(seed=seed,client=c.client_id,family=c.family,group=int(assignments[i]),
                selected_k=p['selected_k'],observed_score=score,global_score=global_score,
                unseen_error_change=float(e.FederatedLearned-e.Global)))
        for a in ['nominal','heavy','handling']:
            for b in ['nominal','heavy','handling']:
                if a>b:continue
                eligible=[(i,j) for i in range(30) for j in range(i+1,30)
                    if sorted([plants[i].family,plants[j].family])==sorted([a,b])]
                pairs.append(dict(seed=seed,selected_k=p['selected_k'],family_a=a,family_b=b,
                    pair_count=len(eligible),coassignment=float(np.mean([assignments[i]==assignments[j] for i,j in eligible]))))
        candidates=pd.DataFrame(p['candidates']).sort_values('bic')
        fleet.append(dict(seed=seed,selected_k=p['selected_k'],bic_margin=float(candidates.iloc[1].bic-candidates.iloc[0].bic)))
    for name,rows in [('groups',groups),('clients',clients),('coassignment',pairs),('fleets',fleet)]:
        pd.DataFrame(rows).to_csv(OUT/f'paper_learned_groups_{name}.csv',index=False)
    sources['paper/learned_group_analysis.py']=digest(Path(__file__))
    payload=dict(status='verified_descriptive_no_fits',fleets=10,groups=len(groups),clients=len(clients),
        source_sha256=sources,outputs={n:digest(OUT/n) for n in [f'paper_learned_groups_{a}.csv' for a in ['groups','clients','coassignment','fleets']]})
    (OUT/'paper_learned_groups_manifest.json').write_text(json.dumps(payload,indent=2)+'\n')
    print(pd.DataFrame(groups).to_string(index=False))
    print(pd.DataFrame(pairs).query('selected_k==2').groupby(['family_a','family_b']).coassignment.agg(['mean','min','max']).to_string())
    print(pd.DataFrame(clients).query('selected_k==2').groupby('family').agg(unseen_change=('unseen_error_change','mean'),observed_score=('observed_score','mean'),global_score=('global_score','mean')).to_string())


if __name__=='__main__':main()
