"""Check manuscript Smooth Local LQI claims against immutable execution evidence."""
from pathlib import Path
import argparse
import hashlib
import json
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/tables'
NUMBERS=ROOT/'paper/smooth_lqi_numbers.tex'
LOCK=ROOT/'paper/smooth_lqi_evidence_lock.json'


def digest(path):
    data=path.read_bytes()
    if path.suffix in {'.json','.csv','.py','.tex'}:
        data=data.replace(b'\r\n',b'\n')
    return hashlib.sha256(data).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--write',action='store_true')
    args=parser.parse_args()
    manifest=json.loads((OUT/'paper_smooth_local_lqi_manifest.json').read_text())
    for name,sha in manifest['source_sha256'].items():
        assert digest(ROOT/name)==sha,name
    for name,sha in {**manifest['outputs'],**manifest['checkpoints']}.items():
        assert digest(OUT/name)==sha,name
    audit=json.loads((OUT/'paper_smooth_local_lqi_audit.json').read_text())
    assert audit['execution_manifest_sha256']==digest(OUT/'paper_smooth_local_lqi_manifest.json')
    assert audit['status']=='saved_maps_and_summaries_verified'
    raw=pd.read_csv(OUT/'paper_smooth_local_lqi_clients.csv')
    d=pd.read_csv(OUT/'paper_smooth_local_lqi_map_audit.csv')
    for method in ['Local','Global','FederatedLearned','ExactLPV']:
        assert raw[raw.method==method].feasible.all(),method
    smooth=raw[raw.method=='SmoothLocal']
    assert len(smooth)==600 and (smooth.status=='complete').all()
    values={}
    for scenario,label in [('moderate','Moderate'),('hard','Hard')]:
        view=smooth[smooth.scenario==scenario]
        values[f'SmoothTrack{label}']=f'{view.groupby("seed").tracking_rmse_deg_s.mean().mean():.4f}'
        values[f'SmoothFailed{label}']=str(int((~view.feasible).sum()))
        values[f'SmoothMedian{label}']=f'{view.tracking_rmse_deg_s.median():.4f}'
    values['SmoothUnstable']=str(int((d.rho>=1).sum()))
    values['SmoothRadius']=f'{d.rho.max():.2f}'
    values['SmoothReplayDifference']=f'{audit["replay_seed411"]["max_absolute_rmse_difference"]:.2f}'
    text='% Generated from frozen Smooth Local LQI execution and audit.\n'
    text+=''.join('\\newcommand{\\'+k+'}{'+v+'}\n' for k,v in values.items())
    names=['results/tables/paper_smooth_local_lqi_manifest.json',
           'results/tables/paper_smooth_local_lqi_audit.json',
           'results/tables/paper_smooth_local_lqi_map_audit.csv',
           'results/tables/paper_smooth_local_lqi_replay_audit.csv',
           'paper/audit_smooth_local_lqi.py']
    lock={name:digest(ROOT/name) for name in names}
    if args.write:
        NUMBERS.write_text(text)
        LOCK.write_text(json.dumps(lock,indent=2)+'\n')
    else:
        assert json.loads(LOCK.read_text())==lock
        assert NUMBERS.read_text()==text
    maintext=(ROOT/'paper/main.tex').read_text()
    assert 'downstream control performance remains\nuntested' not in maintext
    for key in values:
        assert '\\'+key in maintext,key
    print(json.dumps(dict(status='verified',fleets=10,rollouts=600,claims=len(values))))


if __name__=='__main__':
    main()
