"""Lock manuscript metric/group diagnostic claims to saved evidence."""
from pathlib import Path
import argparse,json
import pandas as pd
from matrix_error_diagnostic import ROOT,OUT,digest,load


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--write',action='store_true');args=parser.parse_args()
    names=[]
    for prefix in ['paper_matrix_diagnostic','paper_learned_groups']:
        path=OUT/f'{prefix}_manifest.json';m=load(path);names.append(path.relative_to(ROOT).as_posix())
        for n,h in m['source_sha256'].items(): assert digest(ROOT/n)==h,n
        for n,h in m.get('output_sha256',m.get('outputs',{})).items(): assert digest(OUT/n)==h,n
    c=pd.read_csv(OUT/'paper_matrix_diagnostic_comparisons.csv')
    def value(region,metric,column):
        return c[(c.region==region)&(c.metric==metric)&(c.baseline=='Global')].iloc[0][column]
    values={}
    for region,label in [('full','Full'),('unseen','Unseen')]:
        for col,suffix in [('gain_pct','Gain'),('ci_low','Low'),('ci_high','High')]:
            values[f'Diag{label}{suffix}']=f'{value(region,"balanced",col):.2f}'
    values['DiagBOneLoss']=f'{-value("unseen","B1_scaled_mae","gain_pct"):.2f}'
    r=pd.read_csv(OUT/'paper_learned_groups_clients.csv');g=pd.read_csv(OUT/'paper_learned_groups_groups.csv')
    assert len(r)==300 and len(g)==18
    for seed,view in g[g.selected_k==2].groupby('seed'):
        assert sorted(view.heavy)==[0,10] and sorted(view.handling)==[0,10]
        assert (view.nominal>0).all()
        assert not ((view.heavy>0)&(view.handling>0)).any()
    values['DiagNominalWorse']=str(int(((r.family=='nominal')&(r.unseen_error_change>1e-12)).sum()))
    assert int(values['DiagNominalWorse'])==int((r.unseen_error_change>1e-12).sum())
    text='% Generated from frozen metric/group diagnostics.\n'+''.join('\\newcommand{\\'+k+'}{'+v+'}\n' for k,v in values.items())
    lock={n:digest(ROOT/n) for n in names}
    p=ROOT/'paper/diagnostic_numbers.tex';l=ROOT/'paper/diagnostic_evidence_lock.json'
    if args.write:p.write_text(text);l.write_text(json.dumps(lock,indent=2)+'\n')
    else:assert p.read_text()==text;assert load(l)==lock
    source=(ROOT/'paper/main.tex').read_text()
    for key in values: assert '\\'+key in source,key
    print(json.dumps(dict(status='verified',claims=len(values),fleets=10)))


if __name__=='__main__':main()
