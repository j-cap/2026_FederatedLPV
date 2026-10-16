"""Verify original-input replay claims without fitting or rollouts."""
from pathlib import Path
import json
import argparse
import numpy as np
from check_smooth_lqi_integration import ROOT, OUT, digest


def load(p):return json.loads(p.read_text())


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--write',action='store_true');args=parser.parse_args()
    audit=load(OUT/'paper_smooth_original_input_replay_audit.json')
    assert audit['status']=='checked_complete'
    assert all(n==600 for n in audit['input_hash_match_counts'].values())
    for name,sha in {**audit['source_sha256'],**audit['checkpoint_sha256']}.items():assert digest(ROOT/name)==sha,name
    assert digest(ROOT/'paper/analyze_smooth_original_input_replay.py')==audit['analysis_script_sha256']
    count=0
    for seed in range(411,421):
        replay=load(OUT/f'paper_smooth_original_input_replay_seed{seed}.json')
        original=load(OUT/f'paper_smooth_local_lqi_seed{seed}.json')
        for row in replay['rows']:
            assert row['status']=='complete' and all(row['hash_matches'].values())
            old=next(r for r in original['rows'] if r['client']==row['client'] and r['scenario']==row['scenario'] and r['method']=='SmoothLocal')
            for key,value in row['metrics'].items():assert value==old[key],(seed,row['client'],key)
            count+=1
    original=load(OUT/'paper_smooth_replay_execution_reference.json')
    native=load(OUT/'paper_smooth_replay_diagnosis_seed411.json')
    difference=max(float(np.max(np.abs(np.asarray(original['scenarios'][n]['speed'])-np.asarray(sc['speed'])))) for n,sc in native['scenarios'].items())
    text='% Verified original-input cross-runtime replay.\n\\newcommand{\\ReplayExactCount}{'+str(count)+'}\n'
    text+='\\newcommand{\\ReplaySpeedDifference}{'+f'{difference/1e-15:.2f}'+'}\n'
    names=['results/tables/paper_smooth_original_input_replay_audit.json','results/tables/paper_smooth_replay_execution_reference.json',
           'results/tables/paper_smooth_replay_diagnosis_seed411.json','paper/analyze_smooth_original_input_replay.py']
    lock={n:digest(ROOT/n) for n in names}
    p=ROOT/'paper/original_replay_numbers.tex';l=ROOT/'paper/original_replay_evidence_lock.json'
    if args.write:p.write_text(text);l.write_text(json.dumps(lock,indent=2)+'\n')
    else:assert p.read_text()==text;assert load(l)==lock
    source=(ROOT/'paper/main.tex').read_text()
    assert '\\ReplayExactCount' in source and '\\ReplaySpeedDifference' in source
    print(json.dumps(dict(status='verified',trajectories=count,all_metric_values_identical=True)))


if __name__=='__main__':main()
