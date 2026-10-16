"""Summarize replay checkpoints, preserving the original execution endpoints."""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'code/src'),str(ROOT/'code/experiments')]
import paper_smooth_replay_diagnosis as replay


def main():
    rows=[];hashes={};provenance=replay.lock()
    for seed in range(411,421):
        path=replay.OUT/f'{replay.PREFIX}_seed{seed}.json'
        saved=json.loads(path.read_text());assert saved['source_sha256']==provenance
        assert len(saved['rows'])==60
        hashes[path.relative_to(ROOT).as_posix()]=replay.frozen.digest(path)
        for r in saved['rows']:
            row=dict(seed=seed,**{k:v for k,v in r.items() if k not in ['replay_hashes','hash_matches','metrics']})
            row.update({f'hash_{k}':v for k,v in r.get('hash_matches',{}).items()})
            rows.append(row)
    frame=pd.DataFrame(rows)
    assert not frame.duplicated(['seed','client','scenario']).any()
    complete=frame[frame.status=='complete'].copy()
    complete['classification_changed']=complete.feasible!=complete.old_feasible
    summary=[]
    for scenario,g in complete.groupby('scenario'):
        summary.append(dict(scenario=scenario,rollouts=len(g),original_infeasible=int((~g.old_feasible).sum()),
            replay_infeasible=int((~g.feasible).sum()),classification_changes=int(g.classification_changed.sum()),
            original_rmse=float(g.groupby('seed').old_rmse.mean().mean()),
            replay_rmse=float(g.groupby('seed').rmse.mean().mean()),
            max_absolute_client_rmse_difference=float(g.rmse_difference.abs().max()),
            material_differences_over_1e_6=int((g.rmse_difference.abs()>1e-6).sum())))
    frame.to_csv(replay.OUT/f'{replay.PREFIX}_clients.csv',index=False)
    pd.DataFrame(summary).to_csv(replay.OUT/f'{replay.PREFIX}_summary.csv',index=False)
    payload=dict(status='checked_complete' if len(complete)==600 else 'incomplete',
        source_sha256=provenance,checkpoint_sha256=hashes,summary=summary,
        input_hash_match_counts={key:int(complete['hash_'+key].sum()) for key in ['time','speed','reference']},
        limitation='Original numeric input arrays and execution environment were not saved; an upstream input mismatch is established, its original magnitude and precise backend cause are not.',
        analysis_script_sha256=replay.frozen.digest(Path(__file__)))
    (replay.OUT/f'{replay.PREFIX}_audit.json').write_text(json.dumps(payload,indent=2)+'\n')
    print(pd.DataFrame(summary).to_string(index=False))
    print(payload['input_hash_match_counts'])


if __name__=='__main__':main()
