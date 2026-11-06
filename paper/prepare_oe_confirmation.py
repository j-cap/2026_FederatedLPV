"""Build Section V numbers and separate figure panels from frozen OE results.

Run: python paper/prepare_oe_confirmation.py
No trajectories, local estimation, clustering, or tuning are run. Historical
figures and frozen experiment files are not changed. Requires numpy, scipy,
matplotlib, and pandas (the existing experiment environment).
"""
from pathlib import Path
import hashlib
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'code/src'), str(ROOT / 'code/experiments')]
from experiment_10a_basis_coverage_audit import raw_basis

TABLES = ROOT / 'results/tables'
FIGURES = ROOT / 'results/figures'
PREFIX = 'paper_oe_lpv_confirmation'
METHODS = ['RestrictedLocal', 'SmoothLocal', 'Global', 'Federated']
LABELS = dict(zip(METHODS, ['Restricted Local', 'Smooth Local', 'Global', 'Grouped FL']))
COLORS = dict(zip(METHODS, ['#777777', '#8b4c96', '#de7e24', '#167a9e']))
STYLES = dict(zip(METHODS, ['--', ':', '-.', '-']))
BLOCKS = [[10, 12, 14], [16, 18, 20], [22, 24, 26], [28, 30]]
SOURCES = {}


def digest(path):
    data = path.read_bytes()
    if path.suffix in ['.json', '.py', '.yml']:
        data = data.replace(b'\r\n', b'\n')
    return hashlib.sha256(data).hexdigest()


def read(path):
    SOURCES[path.relative_to(ROOT).as_posix()] = digest(path)
    return json.loads(path.read_text(encoding='utf-8'))


def checked_inputs():
    analysis = read(TABLES / f'{PREFIX}_analysis.json')
    diagnosis = read(TABLES / f'{PREFIX}_diagnosis.json')
    assert diagnosis['analysis_sha256'] == digest(TABLES / f'{PREFIX}_analysis.json')
    protocol = read(ROOT / 'code/config/paper_oe_lpv_confirmation.json')
    basis = read(ROOT / 'code/config/experiment_10a.json')
    integration = read(ROOT / 'code/config/paper_oe_lpv_integration.json')
    for relative in ['code/config/experiment_10a.json', 'code/experiments/experiment_10a_basis_coverage_audit.py']:
        actual = digest(ROOT / relative)
        assert actual == integration['source_sha256'][relative]
        SOURCES[relative] = actual
    acquisition = read(ROOT / 'code/config/paper_trajectory_calibration.json')
    results, cells = [], []
    for seed in protocol['seeds']:
        for kind in ['result', 'acquisition']:
            path = TABLES / f'{PREFIX}_{kind}_seed{seed}.json'
            envelope = read(path)
            assert digest(path) == analysis['checkpoint_sha256'][path.relative_to(ROOT).as_posix()]
            payload = envelope['payload']
            canonical = json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
            assert hashlib.sha256(canonical).hexdigest() == envelope['payload_sha256']
            assert payload['seed'] == seed
            if kind == 'result':
                assert not payload['blocked_conditions']
                results.append(payload)
            else:
                assert results[-1]['acquisition_sha256'] == digest(path)
                cells.extend(payload['cells'])
    assert len(results) == 10 and len(cells) == 830
    for cell in cells:
        assert cell['models']['ClosedOE4']['fit_valid']
        for phase in ['training', 'validation']:
            p = cell[phase]
            assert p['settling_pass']
            for key, limit in [('max_beta_deg','max_beta_deg'),('max_ay','max_ay'),
                               ('max_steering_deg','steering_limit_deg'),('max_rate_deg_s','steering_rate_deg_s')]:
                assert p[key] <= acquisition[limit] + 1e-8
    # Re-evaluate saved coefficients against saved references; no model fitting.
    for result in results:
        grid = np.asarray(result['grid'])
        designs = {order: raw_basis(grid, order, basis) for order in [3, 7]}
        for condition, fit in result['conditions'].items():
            models = fit['models']
            for group in fit['groups']:
                speeds = np.concatenate([result['clients'][i]['speeds'] for i in group['members']])
                assert np.linalg.matrix_rank(raw_basis(speeds, 7, basis)) == group['rank']
                audited = next(g for g in diagnosis['groups'] if g['seed'] == result['seed'] and
                               g['target_condition'] == condition and g['group'] == group['group'])
                for key in group: assert group[key] == audited[key]
            for method, errors in fit['errors'].items():
                for i, error in enumerate(errors):
                    if method in ['RestrictedLocal', 'SmoothLocal', 'ExactLPV']:
                        coefficient = models[method][i]
                    elif method == 'Global':
                        coefficient = models[method]
                    else:
                        key = 'family_assignments' if method == 'FamilyLabels' else 'assignments'
                        coefficient = models[method][models[key][i]]
                    coefficient = np.asarray(coefficient)
                    prediction = designs[len(coefficient)] @ coefficient
                    truth = np.asarray(result['clients'][i]['reference'])
                    delta = prediction - truth
                    for name, sl in [('matrix', slice(None)), ('a', slice(0, 4)), ('b', slice(4, 6))]:
                        expected = np.linalg.norm(delta[:, sl], axis=1) / np.linalg.norm(truth[:, sl], axis=1)
                        np.testing.assert_allclose(expected, error[name], atol=1e-11, rtol=1e-8)
                    np.testing.assert_allclose(abs(delta), error['entry_absolute'], atol=1e-11, rtol=1e-8)
    return results, cells, analysis, diagnosis, protocol


def client_values(result, method, region='unseen', metric='matrix', condition='ClosedOE4'):
    grid = np.asarray(result['grid'])
    values = []
    for client, error in zip(result['clients'], result['conditions'][condition]['errors'][method]):
        seen = np.isin(grid, client['speeds'])
        mask = ~seen if region == 'unseen' else seen if region == 'seen' else np.ones(len(grid), dtype=bool)
        values.append(np.mean(np.asarray(error[metric])[mask], axis=0))
    return np.asarray(values)


def summarize(results, cells, analysis, diagnosis, protocol):
    summary, macros = {}, {}
    def number(key, value, precision=2):
        macros['OE' + key] = f'{value:.{precision}f}'
    for method in METHODS:
        summary[method] = {}
        for region in ['full', 'seen', 'unseen']:
            values = np.array([client_values(r, method, region).mean() for r in results])
            summary[method][region] = (100 * values).tolist()
            number(method + region.title(), 100 * values.mean())
        for metric in ['a', 'b']:
            value = 100 * np.mean([client_values(r, method, metric=metric).mean() for r in results])
            summary[method][metric] = value
            number(method + metric.upper(), value)
        rho = np.concatenate([np.asarray(e['spectral_radius']) for r in results for e in r['conditions']['ClosedOE4']['errors'][method]])
        summary[method]['unstable'] = int(np.sum(rho >= 1))
        number(method + 'Unstable', np.sum(rho >= 1), 0)
    intervals = {}
    fed = np.asarray(summary['Federated']['unseen'])
    for baseline in ['SmoothLocal', 'Global']:
        base = np.asarray(summary[baseline]['unseen'])
        difference = base - fed
        margin = t.ppf(1 - protocol['family_alpha'] / 4, 9) * difference.std(ddof=1) / np.sqrt(10)
        values = dict(mean_difference_pp=difference.mean(), lower_pp=difference.mean()-margin,
                      upper_pp=difference.mean()+margin)
        saved = next(x for x in analysis['primary_intervals'] if x['baseline'] == baseline)
        for key, value in values.items():
            np.testing.assert_allclose(value, saved[key], atol=1e-12)
        intervals[baseline] = {k:float(v) for k,v in values.items()}
        for key, value in zip(['Difference', 'Low', 'High'], values.values()):
            number(baseline + key, value)
        number(baseline + 'Gain', np.mean(100 * (1-fed/base)), 1)
        delta = np.concatenate([client_values(r, 'Federated')-client_values(r, baseline) for r in results])
        counts = dict(better=int(np.sum(delta < -1e-12)), ties=int(np.sum(abs(delta) <= 1e-12)), worse=int(np.sum(delta > 1e-12)))
        summary[baseline]['client_counts'] = counts
        for key, value in counts.items(): number(baseline + key.title(), value, 0)
    groups = [g for g in diagnosis['groups'] if g['target_condition'] == 'ClosedOE4']
    assert len(groups) == 30 and all(g['rank'] == 7 for g in groups)
    assert all(r['conditions']['ClosedOE4']['selected_k'] == 3 for r in results)
    for key, value in [('GroupConditionMin', min(g['condition'] for g in groups)),
                       ('GroupConditionMax', max(g['condition'] for g in groups))]: number(key, value)
    aris = [r['conditions']['ClosedOE4']['family_ari'] for r in results]
    number('FamilyARIMin', min(aris), 3)
    number('ExactFamilies', sum(np.isclose(a, 1) for a in aris), 0)
    for label, condition in [('Local', 'ClosedOE4'), ('CleanLocal', 'ClosedCleanOLS')]:
        errors = [100*c['models'][condition]['matrix_error'] for c in cells]
        number(label + 'Median', np.median(errors), 3)
        number(label + 'Pninetyfive', np.percentile(errors, 95), 3)
    number('LimiterPercent', 100*np.mean([c['tracking_validation']['limiter_fraction'] for c in cells]))
    number('BestSmoothError', diagnosis['best_smooth_mean_pct'])
    number('BestSmoothGain', diagnosis['best_smooth_paired_gain_pct'], 1)
    coefficient_delta = max(r['conditions']['ClosedOE4']['equivalence']['coefficient_max_abs'] for r in results)
    prediction_delta = max(r['conditions']['ClosedOE4']['equivalence']['prediction_max_abs'] for r in results)
    summary['equivalence'] = dict(coefficient_max_abs=coefficient_delta, prediction_max_abs=prediction_delta)
    for label, value in [('CoefficientDelta', coefficient_delta), ('PredictionDelta', prediction_delta)]:
        exponent = int(np.floor(np.log10(value)))
        macros['OE'+label] = f'{value/10.**exponent:.2f}\\times10^{{{exponent}}}'
    partial = [g for g in groups if len(g['unique_speeds']) < 11]
    deficient = [g for g in diagnosis['groups'] if g['rank'] < 7]
    assert len(partial) == len(deficient) == 1
    number('PartialGroupUnseen', partial[0]['unseen_pct'])
    number('DeficientGroupUnseen', deficient[0]['unseen_pct'])
    failed = next(r for r in results if r['seed'] == deficient[0]['seed'])
    number('CleanFailureUnseen', 100 * client_values(failed, 'Federated', condition='ClosedCleanOLS').mean())
    family_errors = [100 * client_values(r, 'FamilyLabels').mean() for r in results]
    for label,value in zip(['FamilyWorst','FamilySecondWorst'],sorted(family_errors,reverse=True)[:2]): number(label,value)
    summary['family_unseen_pct'] = family_errors
    number('MaximumRho', diagnosis['primary_federated_maximum_rho'], 4)
    summary['entry_mae_unseen'] = {m:np.mean([client_values(r,m,metric='entry_absolute').mean(axis=0) for r in results],axis=0).tolist() for m in METHODS}
    for baseline in ['SmoothLocal', 'Global']:
        assert np.all(np.asarray(summary['entry_mae_unseen']['Federated']) < summary['entry_mae_unseen'][baseline])
    summary['primary_intervals'] = intervals
    summary['family_ari'] = aris
    assert analysis['primary_confirmation_supported'] and all(v['lower_pp'] > 0 for v in intervals.values())
    return summary, macros


def style():
    plt.rcParams.update({'font.family':'serif', 'font.serif':['DejaVu Serif'], 'font.size':8,
                         'axes.labelsize':8, 'xtick.labelsize':7, 'ytick.labelsize':7,
                         'legend.fontsize':6.5, 'lines.linewidth':1.2, 'pdf.fonttype':42})


def save(fig, stem):
    path = FIGURES / (stem + '.pdf')
    fig.savefig(path, metadata={'CreationDate':None, 'ModDate':None})
    fig.savefig(FIGURES / (stem + '.png'), dpi=180)
    plt.close(fig)
    return path


def figures(results, diagnosis, summary):
    style()
    files = []
    grid = np.asarray(results[0]['grid'])
    # Same bootstrap fleet draws for every method, block, and speed.
    draws = np.random.default_rng(20261102).integers(0,10,size=(20000,10))
    for block, observed in enumerate(BLOCKS):
        fig, ax = plt.subplots(figsize=(3.48,2.25),layout='constrained')
        ax.axvspan(min(observed)-.35,max(observed)+.35,color='#dbe9d4',alpha=.7,linewidth=0)
        inset = ax.inset_axes([.53,.55,.44,.42])
        inset.axvspan(min(observed)-.35,max(observed)+.35,color='#dbe9d4',alpha=.7,linewidth=0)
        for method in METHODS:
            values = np.array([np.mean([e['matrix'] for c,e in zip(r['clients'],r['conditions']['ClosedOE4']['errors'][method]) if c['speeds']==observed],axis=0) for r in results])*100
            assert values.shape == (10,11)
            mean = values.mean(axis=0)
            low,high = np.quantile(values[draws].mean(axis=1),[.025,.975],axis=0)
            for target in [ax]+([] if method=='RestrictedLocal' else [inset]):
                target.plot(grid,mean,color=COLORS[method],linestyle=STYLES[method],label=LABELS[method])
                target.fill_between(grid,low,high,color=COLORS[method],alpha=.12,linewidth=0)
        ax.set(xlabel='Speed [m/s]',ylabel='Relative matrix error [%]',xlim=(10,30),yscale='log',ylim=(.1,300),xticks=[10,15,20,25,30])
        ax.grid(axis='y',alpha=.18)
        inset.set(xlim=(10,30),ylim=(0,10),xticks=[10,20,30],yticks=[0,5,10])
        inset.tick_params(labelsize=6,pad=1)
        for v in observed: ax.plot(v,.12,'|',color='#467942',markersize=5)
        if block==0: ax.legend(loc='upper left',frameon=False,prop={'style':'italic','size':6})
        files.append(save(fig,'paper_oe_coverage_'+'_'.join(map(str,observed))+'_panel'))
    fig,ax = plt.subplots(figsize=(3.48,2.65),layout='constrained')
    x=np.arange(1,11)
    for j in range(10):
        ax.plot([x[j]]*3,[summary[m]['unseen'][j] for m in ['Global','SmoothLocal','Federated']],color='#bbbbbb',linewidth=.8,zorder=1)
    for m,marker in [('Global','o'),('SmoothLocal','^'),('Federated','s')]:
        ax.plot(x,summary[m]['unseen'],linestyle='none',marker=marker,color=COLORS[m],markersize=4,label=LABELS[m])
    ax.set(xlabel='Independent evaluation fleet',ylabel='Unobserved-speed error [%]',xticks=x,ylim=(0,6))
    ax.grid(axis='y',alpha=.18);ax.legend(loc='upper center',ncol=3,frameon=False,prop={'style':'italic','size':6.5})
    files.append(save(fig,'paper_oe_paired_fleet_panel'))
    fig,ax = plt.subplots(figsize=(3.48,2.65),layout='constrained')
    for method in ['Global','SmoothLocal']:
        values=100*np.concatenate([client_values(r,'Federated')-client_values(r,method) for r in results])
        values.sort()
        label=f"vs {LABELS[method]} ({summary[method]['client_counts']['better']}/300 better)"
        ax.step(values,np.arange(1,len(values)+1)/len(values),where='post',color=COLORS[method],label=label)
    ax.axvline(0,color='#666666',linestyle='--',linewidth=.8)
    ax.set(xlabel='Grouped FL minus baseline [percentage points]',ylabel='Fraction of clients',ylim=(0,1.02))
    ax.grid(axis='y',alpha=.18);ax.legend(loc='lower right',frameon=False,prop={'style':'italic','size':6.5})
    files.append(save(fig,'paper_oe_client_tradeoff_panel'))
    fig,ax = plt.subplots(figsize=(3.48,2.65),layout='constrained')
    groups=[g for g in diagnosis['groups'] if g['target_condition']=='ClosedOE4']
    complete=[g for g in groups if len(g['unique_speeds'])==11]
    partial=[g for g in groups if len(g['unique_speeds'])<11]
    deficient=[g for g in diagnosis['groups'] if g['rank']<7]
    assert len(complete)==29 and len(partial)==len(deficient)==1
    rows=BLOCKS+[grid.tolist(),grid.tolist(),partial[0]['unique_speeds'],deficient[0]['unique_speeds']]
    labels=['Local 10–14','Local 16–20','Local 22–26','Local 28–30','Fleet union','OE groups (29/30)','OE group (1/30)','Clean-state example']
    ranks=[3,3,3,2,7,7,7,6]
    for j,speeds in enumerate(rows):
        for k,v in enumerate(grid):
            color=('#9d9d9d' if j<4 else '#167a9e' if j<7 else '#b67443') if v in speeds else 'white'
            ax.add_patch(plt.Rectangle((k-.43,j-.34),.86,.68,facecolor=color,edgecolor='#cccccc',linewidth=.4))
        ax.text(11,j,str(ranks[j]),ha='center',va='center',fontsize=7)
    ax.text(11,-.75,'Rank',ha='center',fontsize=7)
    ax.set(xlim=(-.6,11.5),ylim=(7.7,-1.05),yticks=range(8),yticklabels=labels,xticks=range(11),xticklabels=grid,xlabel='Observed grid speed [m/s]')
    ax.tick_params(length=0,labelsize=6.5)
    for spine in ax.spines.values(): spine.set_visible(False)
    files.append(save(fig,'paper_oe_scheduling_information_panel'))
    return files


def main():
    results,cells,analysis,diagnosis,protocol=checked_inputs()
    summary,macros=summarize(results,cells,analysis,diagnosis,protocol)
    FIGURES.mkdir(parents=True,exist_ok=True)
    files=figures(results,diagnosis,summary)
    numbers=ROOT/'paper/oe_confirmation_numbers.tex'
    numbers.write_text('% Generated by paper/prepare_oe_confirmation.py; do not edit.\n'+''.join(f'\\newcommand{{\\{k}}}{{{v}}}\n' for k,v in macros.items()),encoding='utf-8',newline='\n')
    files.append(numbers)
    manifest=dict(operation='verified_saved_results_only_no_fitting_or_simulation',generator_sha256=digest(Path(__file__)),
        source_sha256=SOURCES,output_sha256={p.relative_to(ROOT).as_posix():digest(p) for p in files},
        conventions=dict(endpoint='equal-speed within client, equal-client within fleet, equal-fleet',
                         coverage_bands='pointwise percentile bootstrap over 10 fleets, 20000 paired draws, seed 20261102; descriptive',
                         panels='separate PDFs, no embedded panel letters or titles',fleet_labels='1..10 map to frozen seeds 1311..1320'),summary=summary)
    (TABLES/'paper_oe_manuscript_manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(dict(status='verified_and_generated',fleets=len(results),local_fits=len(cells),panels=7,summary=summary),indent=2))


if __name__=='__main__':
    main()
