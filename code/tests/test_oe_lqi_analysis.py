"""Check descriptive aggregation without rerunning any scientific trajectory."""
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'code/src'),str(ROOT/'code/experiments')]
import paper_oe_lqi_analysis as analysis


def fixture():
    rows=[];schedules=[]
    for seed in [0,1]:
        for i in range(30):
            for method in analysis.run.METHODS:
                schedules.append(dict(method=method,maximum_radius=.95))
                for scenario in analysis.run.SCENARIOS:
                    value=([.5,9.][seed] if method=='Federated' else [1.,10.][seed])
                    rows.append(dict(seed=seed,client=f'c{i}',family='nominal',method=method,scenario=scenario,
                        observed_speeds=[10.,12.,14.],feasible=True,status='complete',
                        metrics=dict(tracking_rmse_deg_s=value,beta_violation_samples=0)))
    return rows,schedules


class AnalysisTests(unittest.TestCase):
    def test_percent_is_mean_of_paired_fleet_reductions(self):
        rows,schedules=fixture(); result=analysis.summarize(rows,schedules)
        pair=next(p for p in result['pairs'] if p['baseline']=='Global' and p['scenario']=='moderate')
        self.assertAlmostEqual(pair['paired_fleet_percent']['mean'],30.)
        self.assertAlmostEqual(pair['paired_fleet_absolute_deg_s']['mean'],.75)
        self.assertEqual(pair['fleet_wins'],2);self.assertEqual(pair['client_wins'],60)

    def test_numerical_failure_is_counted_and_incomplete_pair_is_explicit(self):
        rows,schedules=fixture()
        bad=next(r for r in rows if (r['seed'],r['method'],r['scenario'])==(1,'Global','hard'))
        bad.update(status='rollout_failed',feasible=False,metrics=None)
        # A finite infeasible response stays in the mean and in finite pairs.
        finite=next(r for r in rows if (r['seed'],r['method'],r['scenario'])==(0,'Global','hard'))
        finite['feasible']=False;finite['metrics']['beta_violation_samples']=1
        result=analysis.summarize(rows,schedules)
        item=next(r for r in result['summary'] if (r['method'],r['scenario'])==('Global','hard'))
        self.assertEqual(item['finite'],59);self.assertEqual(item['numerical_failures'],1)
        self.assertEqual(item['sideslip_violations'],1);self.assertEqual(item['feasible'],58)
        self.assertEqual(item['full_fleets'],1)
        pair=next(p for p in result['pairs'] if p['baseline']=='Global' and p['scenario']=='hard')
        self.assertEqual(pair['complete_fleet_pairs'],1)
        self.assertEqual(pair['finite_client_pairs'],59)
        self.assertEqual(pair['both_feasible_client_pairs'],58)
        self.assertAlmostEqual(pair['paired_fleet_percent']['mean'],50.)


if __name__=='__main__':unittest.main()
