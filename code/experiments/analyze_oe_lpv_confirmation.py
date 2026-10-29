"""Independent coefficient/reference audit and standalone reviewed confirmation report.

No trajectory simulation, OE optimization, clustering search or retuning.
"""
from pathlib import Path
import html
import io
import json
import sys
from collections import Counter
import numpy as np
import pandas as pd
from scipy.stats import t
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_oe_lpv_confirmation as run
lpv, parent = run.lpv, run.parent
REPORT = run.ROOT / 'report/oe_lpv_confirmation_review.html'
PREFIX = run.PREFIX
COLORS = dict(RestrictedLocal='#9c667d', SmoothLocal='#cb862c', Global='#5077aa', Federated='#168273')
LABELS = dict(RestrictedLocal='Restricted local', SmoothLocal='Smooth local', Global='Global', Federated='Federated')


def audit():
    protocol, cfg, basis, mixture, acq_cfg, _, _, _, lock = run.preflight()
    source_path = run.OUT / f'{PREFIX}_analysis.json'
    source = parent.load(source_path)
    assert source['provenance'] == lock
    results, acquisitions, groups, family_groups, local, clients, unstable = [], [], [], [], [], [], []
    method_count = sensitivity_count = 0
    largest_pooled_delta = 0.
    for seed in protocol['seeds']:
        for kind in ['acquisition', 'result']:
            path = run.path_for(seed, kind)
            assert parent.tracked_unchanged(path)
            assert parent.file_hash(path) == source['checkpoint_sha256'][path.relative_to(run.ROOT).as_posix()]
        result = run.read(run.path_for(seed), seed, lock)
        acquired = run.read(run.path_for(seed, 'acquisition'), seed, lock)
        assert result['acquisition_sha256'] == parent.file_hash(run.path_for(seed, 'acquisition'))
        assert result['status'] == 'execution_complete' and not result['blocked_conditions']
        assert result['clients'] == acquired['clients'] and not run.validate_cells(acquired)
        assert result['grid'] == basis['speed_grid']
        # Recreate sampling/coverage and equilibrium Jacobians, never driving trajectories.
        reconstructed = run.make_clients(seed, acq_cfg, basis, mixture)
        for expected, saved in zip(reconstructed, result['clients']):
            for key in ['client', 'family', 'parameters', 'speeds']:
                assert expected[key] == saved[key]
            np.testing.assert_allclose(expected['reference'], saved['reference'], atol=1e-10, rtol=1e-8)
        for cell in acquired['cells']:
            truth = np.array(result['clients'][cell['client_index']]['reference'][result['grid'].index(cell['speed'])])
            np.testing.assert_array_equal(truth, cell['reference'])
            assert cell['acquisition_seed'] == parent.acquisition_seed(seed, cell['client_index'], cell['speed'])
            optimizer = cell['closed_optimizer']
            assert len(optimizer['attempts']) == 4 and optimizer['converged']
            eligible = [a for a in optimizer['attempts'] if a['converged']]
            np.testing.assert_allclose(optimizer['weighted_mse'], min(a['weighted_mse'] for a in eligible), atol=1e-12)
            np.testing.assert_array_equal(optimizer['target'], cell['models']['ClosedOE4']['target'])
            for condition in cfg['conditions']:
                model = cell['models'][condition]
                error = np.linalg.norm(np.array(model['target'])-truth)/np.linalg.norm(truth)
                np.testing.assert_allclose(error, model['matrix_error'], atol=1e-12)
            assert all(lpv.bounded(cell[k], acq_cfg) and cell[k]['settling_pass'] for k in ['training','validation'])
            local.append(dict(seed=seed, client=cell['client'], speed=cell['speed'],
                oe_error_pct=100*cell['models']['ClosedOE4']['matrix_error'],
                clean_error_pct=100*cell['models']['ClosedCleanOLS']['matrix_error'],
                selected_start=optimizer['selected_start'], jacobian_rank=optimizer['jacobian_rank'],
                jacobian_condition=optimizer['jacobian_condition'],
                converged_starts=sum(a['converged'] for a in optimizer['attempts']),
                nominal_converged=optimizer['attempts'][-1]['converged'],
                validation_limiter_pct=100*cell['tracking_validation']['limiter_fraction'],
                validation_yaw_rmse_deg_s=cell['tracking_validation']['yaw_rmse_deg_s']))
        for condition, fitted in result['conditions'].items():
            speeds, targets, diagnostic = lpv.assemble_inputs(acquired, acquired, condition, acq_cfg)
            assert diagnostic == fitted['local_identification']
            for actual, saved in zip(targets, result['inputs'][condition]):
                np.testing.assert_array_equal(actual, saved)
            errors, sensitivity = lpv.evaluate(fitted, result['clients'], speeds, basis)
            for method in lpv.METHODS:
                for i, (actual, saved) in enumerate(zip(errors[method], fitted['errors'][method])):
                    for key in actual:
                        np.testing.assert_allclose(actual[key], saved[key], atol=1e-11, rtol=1e-8)
                    method_count += len(result['grid'])
                    mask = lpv.region_mask(result['grid'], speeds[i], 'unseen')
                    clients.append(dict(seed=seed, condition=condition, client=result['clients'][i]['client'],
                        family=result['clients'][i]['family'], method=method,
                        unseen_pct=100*float(np.mean(np.array(actual['matrix'])[mask])),
                        seen_pct=100*float(np.mean(np.array(actual['matrix'])[~mask]))))
                    for j, rho in enumerate(actual['spectral_radius']):
                        if rho >= 1:
                            unstable.append(dict(seed=seed, condition=condition, method=method,
                                client=result['clients'][i]['client'], speed=result['grid'][j], rho=rho,
                                seen=bool(~mask[j])))
            for strength, actual in sensitivity.items():
                np.testing.assert_allclose(actual, fitted['sensitivity_errors'][strength], atol=1e-11, rtol=1e-8)
                sensitivity_count += np.asarray(actual).size
            pooled = lpv.central_ridge(np.concatenate(speeds), np.vstack(targets), basis, mixture['ridge_strength'])
            np.testing.assert_allclose(pooled, fitted['models']['Global'], atol=1e-8, rtol=0)
            largest_pooled_delta = max(largest_pooled_delta,float(np.max(abs(pooled-np.asarray(fitted['models']['Global'])))))
            assert min(fitted['candidates'],key=lambda row: row['bic'])['k'] == fitted['selected_k']
            assert sorted(i for g in fitted['groups'] for i in g['members']) == list(range(30))
            for g in fitted['groups']:
                members=np.flatnonzero(np.asarray(fitted['models']['assignments'])==g['group'])
                assert members.tolist()==g['members']
                observed=np.concatenate([speeds[i] for i in members])
                info=lpv.information(observed,basis)
                for key in ['rank','full_rank','observations','unique_speeds']:
                    assert info[key]==g[key]
                if info['condition'] is not None:np.testing.assert_allclose(info['condition'],g['condition'],rtol=1e-8)
                central=lpv.central_ridge(observed,np.vstack([targets[i] for i in members]),basis,mixture['ridge_strength'])
                for method in ['Federated','CentralSameGroups']:
                    delta=float(np.max(abs(central-np.asarray(fitted['models'][method][g['group']]))))
                    largest_pooled_delta=max(largest_pooled_delta,delta)
                    np.testing.assert_allclose(central,fitted['models'][method][g['group']],atol=1e-8,rtol=0)
                average=float(np.mean([np.mean(np.array(errors['Federated'][i]['matrix'])[lpv.region_mask(result['grid'],speeds[i],'unseen')]) for i in members]))
                groups.append(dict(seed=seed,target_condition=condition,**g,
                    family_counts=dict(Counter(result['clients'][i]['family'] for i in members)),unseen_pct=100*average,
                    family_ari=fitted['family_ari'],block_ari=fitted['coverage_block_ari']))
            for family in sorted({c['family'] for c in result['clients']}):
                members=[i for i,c in enumerate(result['clients']) if c['family']==family]
                info=lpv.information(np.concatenate([speeds[i] for i in members]),basis)
                family_groups.append(dict(seed=seed,target_condition=condition,family=family,**info))
        results.append(result);acquisitions.append(acquired)
    units,sensitivity,speeds,comparisons=lpv.summarize(results,cfg)
    for name,frame in [('fleet_summary',units),('smooth_sensitivity',sensitivity),('speed_summary',speeds)]:
        saved=pd.read_csv(run.OUT/f'{PREFIX}_{name}.csv')
        keys=[key for key in ['seed','condition','method','region','speed','strength'] if key in frame]
        pd.testing.assert_frame_equal(saved.sort_values(keys).reset_index(drop=True),frame.sort_values(keys).reset_index(drop=True),check_exact=False,atol=1e-12,rtol=1e-9)
    np.testing.assert_allclose([c['improvement_pct'] for c in comparisons],[c['improvement_pct'] for c in source['descriptive_comparisons']],atol=1e-10)
    primary=units[(units.condition=='ClosedOE4')&(units.region=='unseen')].pivot(index='seed',columns='method',values='matrix')
    for interval in source['primary_intervals']:
        differences=100*(primary[interval['baseline']]-primary.Federated).to_numpy()
        margin=t.ppf(.9875,9)*np.std(differences,ddof=1)/np.sqrt(10)
        np.testing.assert_allclose([np.mean(differences),np.mean(differences)-margin,np.mean(differences)+margin],
            [interval[k] for k in ['mean_difference_pp','lower_pp','upper_pp']],atol=1e-12)
        np.testing.assert_allclose(differences,interval['differences_pp'],atol=1e-12)
    assert source['primary_confirmation_supported'] and len(results)==10
    assert method_count==46200 and sensitivity_count==46200
    return source,results,acquisitions,units,sensitivity,speeds,pd.DataFrame(clients),pd.DataFrame(local),groups,family_groups,unstable,largest_pooled_delta


def figure(draw, size=(9,4.2)):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with plt.rc_context({'font.size':10,'svg.hashsalt':'oe_confirmation_review','axes.spines.top':False,'axes.spines.right':False}):
        fig,ax=plt.subplots(figsize=size,layout='constrained');draw(ax)
        stream=io.StringIO();fig.savefig(stream,format='svg',metadata={'Date':None});plt.close(fig)
    value=stream.getvalue();return value[value.index('<svg'):]


def table(headers, rows):
    def fmt(v):
        if isinstance(v,(float,np.floating)):return f'{v:.3f}'
        return '—' if v is None else str(v)
    return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(str(h))+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(fmt(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table></div>'


def build_report(data):
    source,results,acquisitions,units,sensitivity,speeds,clients,local,groups,family_groups,unstable,pooled_delta=data
    primary=units[units.condition=='ClosedOE4']
    unseen=primary[primary.region=='unseen'].pivot(index='seed',columns='method',values='matrix')
    summary=primary.groupby(['method','region']).matrix.mean()*100
    cp=clients[clients.condition=='ClosedOE4']
    client_pivot=cp.pivot(index=['seed','client'],columns='method',values='unseen_pct')
    primary_groups=[g for g in groups if g['target_condition']=='ClosedOE4']
    sweep=sensitivity[(sensitivity.condition=='ClosedOE4')&(sensitivity.region=='unseen')]
    best_strength=float(sweep.groupby('strength').matrix.mean().idxmin())
    best=sweep[sweep.strength==best_strength].set_index('seed').matrix
    sweep_wins=sum(unseen.loc[row.seed,'Federated']<row.matrix for row in sweep.itertuples())
    all_primary_rho=np.concatenate([np.array(e['spectral_radius']) for r in results for e in r['conditions']['ClosedOE4']['errors']['Federated']])
    sections=[]
    def section(identifier,title,text,plot='',caption=''):
        sections.append(f'<section id="{identifier}"><h2>{title}</h2>{text}'+(f'<figure>{plot}<figcaption>{caption}</figcaption></figure>' if plot else '')+'</section>')
    headline=f'''<div class="verdict"><strong>The prespecified identification claim is confirmed on all ten fresh fleets.</strong><p>Federated models achieve {summary['Federated','unseen']:.2f}% mean error at recipient-unobserved speeds. The improvement holds against both primary baselines in every fleet. This supports sharing complementary speed coverage under the frozen simulated acquisition protocol.</p></div><div class="cards"><div><b>830 / 830</b><span>valid local OE fits</span></div><div><b>10 / 10</b><span>fleet wins against both baselines</span></div><div><b>0 / 3,300</b><span>unstable primary fitted grid models</span></div></div>'''
    section('overview','1. What was tested',headline+'''<p>The practical problem is incomplete local speed coverage: each vehicle observes only two or three of eleven speeds. A fixed nominal 20 m/s LTI controller collects separate training and validation maneuvers at those speeds. Local four-start output-error (OE) identification estimates A and B. Vehicles then use restricted local regression, smooth local regression, global pooling, or learned-group pooling to obtain full-envelope models.</p><p>The ten reserved seeds 1311–1320 were run after the pipeline was frozen. Each fleet contains 30 vehicles and 83 observed client-speed cells. Family labels and full-envelope reference Jacobians are evaluation-only information for primary methods. No settings were selected from these confirmation results.</p>'''+table(['Method','Observed speeds [%]','Unobserved speeds [%]','Full envelope [%]'],[[LABELS[m],summary[m,'seen'],summary[m,'unseen'],summary[m,'full']] for m in lpv.PRIMARY])+'''<p>Each client's speeds are averaged within the stated region, followed by equal client weights and equal fleet weights. “Unobserved” is relative to the recipient vehicle; another member of its group may have data at that speed. The metric is relative Frobenius error of the six A/B entries, not tracking error.</p>''')
    def contrasts(ax):
        for j,interval in enumerate(source['primary_intervals']):
            ax.scatter(interval['differences_pp'],np.full(10,j)+np.linspace(-.13,.13,10),s=24,alpha=.6,color='#5077aa')
            ax.errorbar(interval['mean_difference_pp'],j,xerr=[[interval['mean_difference_pp']-interval['lower_pp']],[interval['upper_pp']-interval['mean_difference_pp']]],fmt='D',color='#168273',capsize=6,linewidth=3)
        ax.axvline(0,color='#ac4d45',linestyle='--');ax.set(yticks=[0,1],yticklabels=['Smooth local − Federated','Global − Federated'],xlabel='Paired error reduction [percentage points]');ax.grid(axis='x',alpha=.2)
    section('inference','2. The primary confirmation decision',table(['Baseline','Mean reduction [pp]','Adjusted interval [pp]','Fleet wins'],[[LABELS[v['baseline']],v['mean_difference_pp'],f"[{v['lower_pp']:.3f}, {v['upper_pp']:.3f}]",f"{v['wins']}/10"] for v in source['primary_intervals']])+'''<p>Both lower bounds are positive, all ten primary fleets are valid, and all fixed-partition centralized checks pass. The frozen decision criterion is therefore met. Each interval is a Bonferroni-adjusted 97.5% two-sided paired Student-t interval; the two comparisons have nominal simultaneous family coverage of at least 95% under the stated assumptions. Ten independent fleet differences are the inference units, not 300 clients or thousands of correlated speed evaluations.</p><p>The mean of the paired relative improvements is <strong>48.05% versus Smooth Local</strong> and <strong>54.54% versus Global</strong>. These percentages differ from ratios of aggregate means. The intervals remain approximate with ten fleets; the plot exposes every fleet difference.</p>''',figure(contrasts),'Dots: ten individual fleet differences. Diamonds and bars: paired means and the prespecified adjusted confidence intervals. Positive values favor Federated.')
    def fleets(ax):
        for m in lpv.PRIMARY:ax.plot(unseen.index,100*unseen[m],marker='o',label=LABELS[m],color=COLORS[m])
        ax.set(yscale='log',xticks=unseen.index,xlabel='Independent fleet seed',ylabel='Mean unobserved-speed matrix error [%]');ax.legend(ncol=2);ax.grid(alpha=.2)
    section('fleets','3. Consistency and the harder fleets','''<p>Federated unseen-speed error ranges from 1.53% to 3.18%. Fleets 1315, 1318 and 1320 have the largest federated errors, yet still improve over both primary baselines. Fleet 1318 provides the smallest improvement over Smooth Local: 0.64 percentage points. Reporting this variation avoids suggesting that the gain is identical across fleets.</p>'''+table(['Fleet','Restricted [%]','Smooth [%]','Global [%]','Federated [%]'],[[seed,*[100*unseen.loc[seed,m] for m in lpv.PRIMARY]] for seed in unseen.index]),figure(fleets),'Logarithmic vertical axis retains the much larger Restricted Local errors without compressing the three stronger methods.')
    def tradeoff(ax):
        for m in lpv.PRIMARY:
            view=primary[primary.method==m].pivot(index='seed',columns='region',values='matrix')
            ax.scatter(100*view.seen,100*view.unseen,alpha=.6,label=LABELS[m],color=COLORS[m])
            ax.scatter(summary[m,'seen'],summary[m,'unseen'],marker='D',s=100,edgecolors='black',color=COLORS[m])
        ax.set(xscale='log',yscale='log',xlabel='Observed-speed error [%]',ylabel='Unobserved-speed error [%]');ax.legend();ax.grid(alpha=.2)
    section('tradeoff','4. Where sharing helps—and its cost','''<p>Federated error at observed speeds is <strong>1.09%</strong>, compared with <strong>0.33%</strong> for Smooth Local and 0.38% for Restricted Local. Group pooling sacrifices some vehicle-specific fit where local information is available. The gain is its prediction at speeds without local observations: 2.05%, compared with 3.94% and 20.44%, respectively.</p><p>The paper should claim improved identification across the speed envelope from incomplete local coverage. These results do not support uniform superiority at every speed or client.</p>''',figure(tradeoff),'Dots: individual fleets; diamonds: equal-fleet means. Both axes are logarithmic. Lower is better on each axis.')
    def speed_plot(ax):
        data=speeds[(speeds.condition=='ClosedOE4')&(speeds.region=='unseen')]
        for m in lpv.PRIMARY:
            v=data[data.method==m].groupby('speed').matrix
            ax.plot(v.mean().index,100*v.mean(),marker='o',label=LABELS[m],color=COLORS[m]);ax.fill_between(v.mean().index,100*v.min(),100*v.max(),alpha=.10,color=COLORS[m])
        ax.set(yscale='log',xticks=results[0]['grid'],xlabel='Speed [m/s]',ylabel='Unobserved-recipient error [%]');ax.legend(ncol=2);ax.grid(alpha=.2)
    section('speed','5. Speed dependence and matrix components',table(['Method','Unseen A error [%]','Unseen B error [%]'],[[LABELS[m],100*primary[(primary.method==m)&(primary.region=='unseen')].a.mean(),100*primary[(primary.method==m)&(primary.region=='unseen')].b.mean()] for m in lpv.PRIMARY])+'''<p>The benefit remains visible when A and B are measured separately. In particular, the steering-input matrix B improves from 10.98% for Smooth Local to 3.68% for Federated; the combined metric is not hiding this component behind the larger A entries.</p><p>The privileged ExactLPV fit using all reference matrices has a maximum error below 0.00002% on this grid. The seven-function span can therefore represent the reference dynamics very accurately here; this does not make those unavailable reference targets usable by a deployed client.</p>''',figure(speed_plot),'At each speed, average over recipients that did not observe it, then over fleets. Shading is the fleet minimum–maximum range, not a confidence interval. This speed-wise summary has different weights from the primary equal-client endpoint.')
    def ecdf(ax):
        for m in lpv.PRIMARY:
            values=np.sort(client_pivot[m]);ax.step(values,np.arange(1,len(values)+1)/len(values),where='post',label=LABELS[m],color=COLORS[m])
        ax.set(xscale='log',xlabel='Client mean unobserved-speed error [%]',ylabel='Fraction of 300 clients',ylim=(0,1));ax.legend();ax.grid(alpha=.2)
    tails=[]
    for m in lpv.PRIMARY:
        values=np.concatenate([np.array(e['matrix'])[lpv.region_mask(r['grid'],c['speeds'],'unseen')] for r in results for c,e in zip(r['clients'],r['conditions']['ClosedOE4']['errors'][m])])*100
        tails.append([LABELS[m],*np.percentile(values,[50,95,100])])
    section('clients','6. Client outcomes and tails',table(['Baseline','Clients improved by Federated','Clients without strict improvement'],[[LABELS[m],int((client_pivot.Federated<client_pivot[m]).sum()),int((client_pivot.Federated>=client_pivot[m]).sum())] for m in lpv.PRIMARY[:-1]])+'''<p>Federated improves 239/300 clients against Smooth Local and 269/300 against Global. The remaining clients matter: fleet-average success does not imply individual benefit. The worst primary Federated client–speed error is 10.21%, compared with 30.37% for Smooth Local and 11.74% for Global.</p>'''+table(['Method','Client-speed median [%]','Client-speed p95 [%]','Maximum [%]'],tails),figure(ecdf),'Empirical distribution of client means, descriptive only. The tail table instead pools all 2,470 unobserved client–speed cases, so its weights differ from the primary endpoint.')
    def sweep_plot(ax):
        for seed,v in sweep.groupby('seed'):ax.semilogx(v.strength,100*v.matrix,color='#cb862c',alpha=.25)
        means=sweep.groupby('strength').matrix.mean();ax.semilogx(means.index,100*means,marker='o',color='#cb862c',linewidth=2.5,label='Smooth local: mean (thin lines: fleets)')
        ax.axhline(summary['Federated','unseen'],color='#168273',linewidth=2,label='Federated mean');ax.axvline(1e6,color='#3b4853',linestyle='--',label='Frozen primary strength')
        ax.set(xlabel='Curvature penalty',ylabel='Mean unobserved-speed error [%]');ax.legend(fontsize=9);ax.grid(alpha=.2)
    section('sensitivity','7. Does the result depend on the smoothness choice?',f'''<p>Federated wins in <strong>{sweep_wins}/70</strong> fleet-by-strength comparisons across the seven prespecified settings. The best aggregate confirmation setting is {best_strength:g}, with {100*best.mean():.2f}% error versus {summary['SmoothLocal','unseen']:.2f}% at the frozen 10⁶ setting. Against that best aggregate setting, the mean paired federated improvement is {100*(1-unseen.Federated/best).mean():.2f}%.</p><p>This is a secondary sensitivity result, not a new baseline selection or an additional confirmatory test. The primary penalty remains unchanged.</p>''',figure(sweep_plot),'Every inherited smoothness setting is shown. No new strength was chosen for the primary comparison.')
    def coverage(ax):
        from matplotlib.colors import ListedColormap
        values=np.array([[v in g['unique_speeds'] for v in results[0]['grid']] for g in primary_groups])
        ax.imshow(values,aspect='auto',cmap=ListedColormap(['#edc9a0','#168273']),vmin=0,vmax=1)
        ax.set(xticks=range(11),xticklabels=results[0]['grid'],yticks=range(30),yticklabels=[f"{g['seed']} · G{g['group']} · n={g['clients']}" for g in primary_groups],xlabel='Speed [m/s]')
        ax.tick_params(axis='y',labelsize=8)
        for line in np.arange(2.5,30,3):ax.axhline(line,color='white',linewidth=2)
    section('groups','8. What the learned groups represent','''<p>Every primary fleet selects three groups. All 30 group designs have rank seven, and 29 cover all eleven grid speeds. Fleet 1312's heavy-vehicle group lacks 22–26 m/s observations but has observations on both sides; its condition number is 16.0 and its client-mean unseen error is 1.32%. Missing grid points are therefore not automatically a failure when interpolation is well supported.</p><p>Family agreement varies: adjusted Rand index (ARI) ranges from 0.518 to 1.000. Four fleets reproduce the planted families exactly. In other fleets, group membership mixes family labels. Labels are not used in fitting. This supports learned compatibility under the observed coverage, rather than a claim of universally recovering physical vehicle classes.</p>'''+table(['Fleet','Selected K','Family ARI','Coverage-block ARI'],[[r['seed'],r['conditions']['ClosedOE4']['selected_k'],r['conditions']['ClosedOE4']['family_ari'],r['conditions']['ClosedOE4']['coverage_block_ari']] for r in results]),figure(coverage,(9,8.8)),'Green: at least one group member observes the speed. Sand: no group member observes it. The plot describes group support, not each recipient’s own data.')
    clean=units[(units.condition=='ClosedCleanOLS')&(units.region=='unseen')].pivot(index='seed',columns='method',values='matrix')
    def clean_plot(ax):
        ax.plot(unseen.index,100*unseen.Federated,'o-',color='#168273',label='Noisy-state OE → Federated')
        ax.plot(clean.index,100*clean.Federated,'s-',color='#ac4d45',label='Exact-state OLS → Federated')
        ax.plot(clean.index,100*clean.SmoothLocal,'--',color='#cb862c',label='Exact-state OLS → Smooth local')
        ax.plot(clean.index,100*clean.Global,'--',color='#5077aa',label='Exact-state OLS → Global')
        ax.set(xticks=clean.index,xlabel='Fleet seed',ylabel='Mean unobserved-speed error [%]');ax.legend(fontsize=9);ax.grid(alpha=.2)
    section('clean','9. Clean-state diagnostic: a real structural limitation','''<p>Exact-state local OLS is more accurate locally: its median matrix error is 0.052%, compared with 0.247% for noisy-state OE. Yet its subsequent group regression is not uniformly better. Mean clean-state Federated unseen error is 2.67%, and it beats clean-state Smooth Local and Global in nine of ten fleets.</p><p><strong>Fleet 1315 is the exception.</strong> Its clean-state clustering creates a five-client group observing only 10–20 m/s. That group's design has rank six instead of seven, and its mean unseen error is 38.76%. The current method permits this group; minimum group size does not guarantee adequate scheduling information. The primary OE groups for the same fleet are full rank and cover all speeds.</p><p>Do not interpret the better primary aggregate as evidence that measurement noise improves identification. These two pipelines select different partitions, and finite-amplitude approximation remains in the clean-state fits. The diagnostic exposes a grouping/coverage weakness even when local target noise is small. This negative result remains in the report and the frozen method is unchanged.</p>''',figure(clean_plot),'The secondary clean-state failure is included without replacing the fleet or altering the grouping rule.')
    def failure_curve(ax):
        r=next(r for r in results if r['seed']==1315);f=r['conditions']['ClosedCleanOLS'];g=next(g for g in f['groups'] if g['rank']<7)
        for condition,color,label in [('ClosedCleanOLS','#ac4d45','Clean OLS → Federated'),('ClosedOE4','#168273','OE → Federated')]:
            values=np.array([r['conditions'][condition]['errors']['Federated'][i]['matrix'] for i in g['members']])*100
            ax.plot(r['grid'],values.mean(axis=0),'o-',color=color,label=label);ax.fill_between(r['grid'],values.min(axis=0),values.max(axis=0),color=color,alpha=.12)
        ax.axvspan(20,30,color='#edc9a0',alpha=.25,label='Outside this clean group’s coverage')
        ax.set(yscale='log',xticks=r['grid'],xlabel='Speed [m/s]',ylabel='Matrix error for the same five clients [%]');ax.legend(fontsize=9);ax.grid(alpha=.2)
    section('coverage-failure','10. Why family labels alone are not an upper bound','''<p>The privileged FamilyLabels diagnostic also has two severe failures: fleet 1311 has 44.19% mean unseen error, and fleet 1318 has 23.35%. The affected nominal and handling families, respectively, have no observations at 10–14 m/s. Their full-rank design condition numbers are approximately 3,225 and 3,399. Family-specific pooling can therefore extrapolate poorly despite correct physical labels.</p><p>Learned primary groups in these fleets cover the full grid and avoid those extreme errors, although they mix some family members. This is consistent with the importance of useful aggregate coverage; it does not isolate coverage as the sole causal factor. FamilyLabels is a privileged diagnostic, not a guaranteed best-achievable model.</p>''',figure(failure_curve),'Separate diagnostic from fleet 1315: same five recipients under clean-state and primary OE group fits. Shading shows the client range. Stable matrices can still have large prediction errors.')
    def local_plot(ax):
        grid=results[0]['grid'];values=[local[local.speed==v].oe_error_pct.to_numpy() for v in grid]
        ax.boxplot(values,tick_labels=grid,showfliers=True,flierprops={'markersize':3,'alpha':.5},medianprops={'color':'#168273'})
        ax.set(yscale='log',xlabel='Identification speed [m/s]',ylabel='Local OE matrix error [%]');ax.grid(axis='y',alpha=.2)
    section('local','11. Local identification and acquisition quality',f'''<p>All 830 cells yield valid selected OE fits and satisfy the existing accuracy diagnostic, applied bounds and settling requirements. Local OE error has median {local.oe_error_pct.median():.3f}%, p95 {local.oe_error_pct.quantile(.95):.3f}% and maximum {local.oe_error_pct.max():.3f}%. All selected fits have Jacobian rank ten. {int(local.converged_starts.sum())}/3,320 individual starts converge; the four-start procedure still selects a valid fit in every cell.</p><p>The observed maxima are sideslip 2.52°, lateral acceleration 4.92 m/s² and steering 2.65°. Rate limiting reaches the prescribed 30°/s bound. Mean validation limiter duty is {local.validation_limiter_pct.mean():.2f}%. All 830 cells still fail the historical zero-limiter criterion; the prospectively frozen amendment permits limiting because applied steering is recorded. These results establish usable simulated acquisition, not smooth ordinary driving or real-world validation.</p><p>Known applied steering is used in OE. Noise is present in measured states and feedback. Affine offsets and fitting-window initial states remain local nuisance quantities even though the complete maneuver starts at β = r = 0.</p>''',figure(local_plot),'Boxplots show the observed-speed cells only; whiskers follow the standard 1.5-IQR rule and all outliers are displayed. Logarithmic error axis.')
    rho_rows=[]
    for m in lpv.PRIMARY:
        rho=np.concatenate([np.array(e['spectral_radius']) for r in results for e in r['conditions']['ClosedOE4']['errors'][m]])
        rho_rows.append([LABELS[m],int(np.sum(rho>=1)),len(rho),float(max(rho))])
    section('stability','12. Stability diagnostic and limits',table(['Method','Cases with ρ(A) ≥ 1','Client-speed cases','Maximum ρ(A)'],rho_rows)+f'''<p>No primary Federated grid model is unstable; its largest spectral radius is {max(all_primary_rho):.6f}. Smooth Local has 272 unstable client–speed cases and Restricted Local has 136. Global has zero. Counts include repeated use of shared models, so they are not counts of statistically independent models.</p><p>The development boundary failure does not recur in these ten fleets. This is favorable empirical evidence at the eleven evaluated speeds, <strong>not a stability guarantee</strong> between speeds or outside the envelope. It also says nothing yet about stability or performance after controller synthesis. The rank-deficient clean-state group remains stable despite large errors, showing why spectral radius and accuracy must be inspected separately.</p>''')
    section('decision','13. What this means for the paper','''<div class="verdict"><strong>Use the confirmation results as the main evidence for OE-based federated LPV identification.</strong></div><p>The supported contribution is accurate full-envelope identification from incomplete local coverage: shared group models improve on both a strong smooth-local baseline and global pooling under a frozen acquisition and estimation protocol. Keep the observed-speed trade-off, client-level exceptions, and the secondary rank-deficient group visible.</p><p>The next manuscript pass should replace the earlier random-perturbation/OLS acquisition story with the implemented fixed-LTI closed-loop trajectories and local OE estimator. Present the ten independent fleets as confirmation and the three earlier fleets as development. The primary comparison, adjusted intervals, speed-dependent results and compact group-coverage diagnostic are natural paper figures or tables.</p><p>If a controller-performance claim remains central, it needs a separate matched evaluation with common held-out references, sensing and actuator limits. Model-error gains alone cannot establish it. No additional estimator tuning is required to report this identification result. A coverage-aware grouping constraint is a motivated follow-up, but changing it now would define a new method requiring new independent evaluation.</p>''')
    section('audit','14. Audit and reproducibility',f'''<p>Verified all twenty execution checkpoint hashes and nested provenance, every local target passed to scheduling, the sampled fleet parameters/coverage, and all 3,300 reference Jacobians reconstructed from the saved simulation parameters. Recomputed <strong>46,200</strong> saved method/client/speed errors and their A/B, entrywise and spectral-radius diagnostics, plus <strong>46,200</strong> smoothness-sensitivity errors. Reproduced fleet summaries, paired gains and the prespecified intervals.</p><p>Independent pooled augmented least-squares solves reproduce the saved Global and fixed-partition group models; largest recomputed coefficient difference is {pooled_delta:.3g}. This checks regression equivalence for the same learned partition, not independent clustering agreement.</p><p>No driving trajectories, OE fits or clustering searches were rerun. Raw trajectory hashes are preserved, but raw arrays were not restored for this coefficient-level audit. Saved acquisition diagnostics were checked for bounds and settling; they were not recomputed from raw time series.</p><pre>python code/experiments/paper_oe_lpv_confirmation.py analyze
python code/experiments/analyze_oe_lpv_confirmation.py</pre><p>The first command creates the frozen runner's numerical summary. The second creates this separate reviewed report and audit outputs. <a href="../docs/oe_lpv_confirmation.html">Frozen protocol</a> · <a href="oe_lpv_confirmation.html">Original numerical report</a> · <a href="../results/tables/{PREFIX}_diagnosis.json">Audit manifest</a></p>''')
    detail=table(['Fleet','Targets','Group','Members','Rank','Condition','Coverage','Family composition','Unseen error [%]'],[[g['seed'],g['target_condition'],g['group'],g['clients'],g['rank'],g['condition'],', '.join(f'{v:g}' for v in g['unique_speeds']),str(g['family_counts']),g['unseen_pct']] for g in groups])
    sections.append('<section><details><summary>All 60 learned groups: coverage, composition and errors</summary>'+detail+'</details></section>')
    nav=''.join(f'<a href="#{i}">{label}</a>' for i,label in [('overview','Result'),('inference','Inference'),('clients','Clients'),('groups','Groups'),('clean','Clean-state failure'),('decision','Paper implications'),('audit','Audit')])
    style='''body{margin:0;background:#eef3f6;color:#203345;font:16px/1.65 system-ui,-apple-system,sans-serif}header,main{max-width:1120px;margin:auto;padding:28px}header{padding-bottom:0}h1{font-size:34px;line-height:1.2;margin-bottom:10px}h2{font-size:24px;line-height:1.3}nav{display:flex;gap:16px;flex-wrap:wrap;margin:22px 0}a{color:#146b85}section{background:white;border-radius:12px;padding:26px;margin:22px 0;box-shadow:0 2px 8px #20334508}p{max-width:1000px}.verdict{border-left:5px solid #168273;padding:16px 20px;background:#eaf5f1}.verdict p{margin-bottom:0}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:15px;margin:22px 0}.cards div{background:#f1f5f7;padding:20px;border-radius:9px}.cards b{display:block;font-size:27px;color:#168273}.cards span{font-size:14px}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:9px 12px;border-bottom:1px solid #dde5e9;text-align:right}th{background:#f2f6f8}td:first-child,th:first-child{text-align:left}tr:hover td{background:#f5f9fa}figure{margin:22px 0 0}svg{width:100%;height:auto}figcaption{color:#526774;font-size:14px;margin:8px 10px}pre{background:#eef3f6;padding:16px;white-space:pre-wrap}summary{cursor:pointer;font-weight:700}footer{color:#526774;padding:20px 0}@media(max-width:650px){header,main{padding:14px}section{padding:18px}.cards{grid-template-columns:1fr}h1{font-size:28px}}@media print{body{background:white}section{box-shadow:none;break-inside:avoid}nav{display:none}}'''
    content='<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Independent OE–LPV confirmation · reviewed analysis</title><style>'+style+'</style></head><body><header><p>FEDERATED LPV · INDEPENDENT CONFIRMATION</p><h1>Sharing local models across the speed envelope</h1><p>Ten fresh fleets · 300 vehicles · 830 local OE fits · frozen protocol</p><nav>'+nav+'</nav></header><main>'+''.join(sections)+'<footer>Execution commit: 89c4925390576fd0aa038c5591fbad28633c2e0b. Primary and secondary results remain separate. No manuscript changes in this analysis.</footer></main></body></html>'
    REPORT.write_text('\n'.join(line.rstrip() for line in content.splitlines()),encoding='utf-8',newline='\n')
    diagnosis=dict(status='audited_confirmation',checkpoint_sha256=source['checkpoint_sha256'],
        analysis_sha256=parent.file_hash(run.OUT/f'{PREFIX}_analysis.json'),generator_sha256=parent.file_hash(__file__),
        matrix_errors_recomputed=46200,sensitivity_errors_recomputed=46200,reference_matrices_recomputed=3300,
        maximum_recomputed_pooled_coefficient_delta=pooled_delta,primary_confirmation_supported=source['primary_confirmation_supported'],
        primary_intervals=source['primary_intervals'],groups=groups,family_coverage=family_groups,unstable_cases=unstable,
        primary_unseen_pct={m:float(summary[m,'unseen']) for m in lpv.PRIMARY},
        primary_client_wins={m:int((client_pivot.Federated<client_pivot[m]).sum()) for m in lpv.PRIMARY[:-1]},
        primary_federated_maximum_rho=float(max(all_primary_rho)),best_smooth_strength_secondary=best_strength,
        best_smooth_mean_pct=float(100*best.mean()),best_smooth_paired_gain_pct=float(100*(1-unseen.Federated/best).mean()),
        smooth_sweep_fleet_strength_wins=int(sweep_wins),raw_trajectories_reconstructed=False,control_benefit_tested=False)
    parent.save_json(run.OUT/f'{PREFIX}_diagnosis.json',diagnosis)
    clients.to_csv(run.OUT/f'{PREFIX}_client_diagnostics.csv',index=False,lineterminator='\n')
    local.to_csv(run.OUT/f'{PREFIX}_local_diagnostics.csv',index=False,lineterminator='\n')
    print(json.dumps({k:v for k,v in diagnosis.items() if k not in ['checkpoint_sha256','groups','family_coverage','unstable_cases','primary_intervals']},indent=2))
    print(REPORT)


if __name__=='__main__':
    with threadpool_limits(limits=1):
        build_report(audit())
