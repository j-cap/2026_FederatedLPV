"""Read-only independent recomputation of saved metric summaries and uncertainty."""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parent))
from matrix_error_diagnostic import ROOT,OUT,digest,ENTRIES,load


def main():
    manifest=load(OUT/'paper_matrix_diagnostic_manifest.json')
    for name,sha in manifest['source_sha256'].items(): assert digest(ROOT/name)==sha,name
    for name,sha in manifest['output_sha256'].items(): assert digest(OUT/name)==sha,name
    references=np.concatenate([np.asarray(i['reference']) for seed in manifest['development_seeds']
        for i in load(OUT/f'paper_local_smoothness_development{seed}.json')['frozen_inputs']])
    scales=np.sqrt(np.mean(references**2,axis=0))
    np.testing.assert_allclose(scales,manifest['scales'],rtol=0,atol=1e-14)
    raw=pd.read_csv(OUT/'paper_matrix_diagnostic_clients.csv')
    assert len(raw)==9900 and not raw.duplicated(['seed','client','method','speed']).any()
    errors=raw[[e+'_error' for e in ENTRIES]].to_numpy()
    np.testing.assert_allclose(np.sqrt(np.sum((errors/scales)**2,axis=1)/6),raw.balanced,rtol=0,atol=1e-12)
    units=pd.read_csv(OUT/'paper_matrix_diagnostic_fleets.csv')
    for row in units.to_dict('records'):
        g=raw[(raw.seed==row['seed'])&(raw.method==row['method'])]
        if row['region']!='full': g=g[g.region==row['region']]
        for metric in ['original','balanced']: assert abs(g[metric].mean()-row[metric])<1e-12
        for j,e in enumerate(ENTRIES):
            assert abs(np.mean(np.abs(g[e+'_error']))/scales[j]-row[e+'_scaled_mae'])<1e-12
            assert abs(np.sqrt(np.mean(g[e+'_error']**2))-row[e+'_rmse'])<1e-12
    comparisons=pd.read_csv(OUT/'paper_matrix_diagnostic_comparisons.csv')
    rng=np.random.default_rng(20261013)
    for row in comparisons.to_dict('records'):
        p=units[units.region==row['region']].pivot(index='seed',columns='method',values=row['metric'])
        gains=100*(1-p.FederatedLearned/p[row['baseline']])
        ci=np.quantile(rng.choice(gains,(20000,10)).mean(axis=1),[.025,.975])
        np.testing.assert_allclose([gains.mean(),*ci],[row['gain_pct'],row['ci_low'],row['ci_high']],rtol=0,atol=1e-10)
    payload=dict(status='verified',rows=len(raw),fleet_summaries=len(units),comparisons=len(comparisons),
        operation='read_only_recomputation_no_fits_no_simulations',
        execution_manifest_sha256=digest(OUT/'paper_matrix_diagnostic_manifest.json'),
        audit_script_sha256=digest(Path(__file__)))
    (OUT/'paper_matrix_diagnostic_audit.json').write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps(payload))


if __name__=='__main__': main()
