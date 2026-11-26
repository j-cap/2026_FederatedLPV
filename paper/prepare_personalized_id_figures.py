"""Separate publication panels from frozen physical-OE results; no fitting.

python paper/prepare_personalized_id_figures.py --preflight
python paper/prepare_personalized_id_figures.py --generate
"""
from __future__ import annotations
import argparse
import hashlib
import html
import io
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'code/src'), str(ROOT/'code/experiments')]
import paper_federated_relaxation_peers_finish as run

PREFIX = 'paper_personalized_id'
CONFIG = ROOT/'paper/personalized_id_figures.json'
METHODS = ['LocalOE', 'NominalPrior', 'SameSpeedAfter', 'Complementary']
LABELS = dict(zip(METHODS, ['Local OE', 'Nominal prior', 'FL: same speed', 'FL: complementary']))
COLORS = dict(zip(METHODS, ['#68778a', '#b47922', '#8061a8', '#087d87']))
STYLES = dict(zip(METHODS, [':', '--', '-.', '-']))
SPEED_COLORS = {13:'#0072B2', 21:'#D55E00', 29:'#009E73'}
MARKERS = {1301:'o', 1302:'s', 1303:'^'}
METRICS = ['yaw_rmse_deg_s', 'measured_yaw_rmse_deg_s', 'beta_rmse_deg']


def digest(p):
    return hashlib.sha256(p.read_bytes().replace(b'\r\n', b'\n') if p.suffix in ['.py','.json','.csv','.html','.drawio'] else p.read_bytes()).hexdigest()


def checked_data():
    protocol = json.loads(CONFIG.read_text(encoding='utf-8'))
    audit_path = ROOT/protocol['audit_source']
    if digest(audit_path) != protocol['audit_sha256']: raise ValueError('Frozen consolidation audit changed')
    audit = run.base.storage.read_checkpoint(audit_path)
    assert audit['status'] == 'audit_passed' and audit['same_speed_converged'] == 9
    sources = dict(audit['source_hashes'])
    for name, expected in sources.items():
        if run.base.canonical_hash(ROOT/name) != expected: raise ValueError('Frozen source changed: '+name)
    sources[protocol['audit_source']] = digest(audit_path)
    sources[CONFIG.relative_to(ROOT).as_posix()] = digest(CONFIG)
    cfg, finish_protocol = run.preflight()
    bundles = {}; rows = []
    for seed in [1301,1302,1303]:
        original, complementary, previous = run.sources(seed, cfg, finish_protocol)
        finish = None
        if seed in finish_protocol['seeds']:
            finish = run.base.storage.read_checkpoint(run.path(seed))
            run.validate(finish, original, previous, cfg, finish_protocol, execution=False)
            assert finish['status'] == 'execution_complete'
        states = {int(v): (finish['optimization'] if finish and v == '13' else c['optimization'])
                  for v,c in previous['conditions'].items()}
        assert all(s['status'] == 'converged' for s in states.values())
        assert complementary['optimization']['status'] == 'converged'
        fleet = run.base.fleet_for(cfg, seed)
        metadata = {c['client']:c for c in fleet['clients']}
        for key, case in original['evaluation'].items():
            same = finish['evaluation'][key] if finish and key in finish['evaluation'] else previous['evaluation'][key]
            assert same['source_trace_sha256'] == case['trace_sha256']
            results = {m:case['methods'][m] for m in METHODS[:2]}
            results.update(SameSpeedAfter=same['metrics'], Complementary=complementary['evaluation'][key]['metrics'])
            for method, result in results.items():
                assert result['status'] == 'complete' and np.isfinite([result[k] for k in METRICS]).all()
                rows.append(dict(seed=seed, client=case['client'], training_speed=metadata[case['client']]['training_speed'],
                    speed=case['speed'], probe=case['probe'], region=run.peers.REGIONS[case['region']], method=method, **result))
        bundles[seed] = dict(original=original, complementary=complementary, states=states, fleet=fleet)
    cases = pd.DataFrame(rows)
    keys = ['seed','client','speed','probe','method']
    saved = pd.read_csv(run.OUT/f'{run.PREFIX}_cases.csv')
    saved = saved[saved.method.isin(METHODS)]
    pd.testing.assert_frame_equal(cases.sort_values(keys).reset_index(drop=True), saved.sort_values(keys).reset_index(drop=True),
                                  check_exact=False, rtol=1e-10, atol=1e-12)
    assert len(cases) == 1512 and not cases.duplicated(keys).any()
    return dict(protocol=protocol, cfg=cfg, sources=sources, bundles=bundles, cases=cases)


def select_trajectories(cases, rule):
    candidates = cases[(cases.method == 'NominalPrior') & (cases.speed == rule['evaluation_speed']) &
                       (cases.probe == rule['probe']) & cases.training_speed.isin(rule['training_speeds'])].copy()
    candidates['selected'] = False; candidates['distance_to_median'] = np.nan
    for speed in rule['training_speeds']:
        g = candidates[candidates.training_speed == speed]
        assert len(g) == 9
        candidates.loc[g.index, 'distance_to_median'] = abs(g.yaw_rmse_deg_s-g.yaw_rmse_deg_s.median())
        chosen = candidates.loc[g.index].sort_values(['distance_to_median','seed','client'], kind='stable').index[0]
        candidates.loc[chosen, 'selected'] = True
    return candidates.sort_values(['training_speed','distance_to_median','seed','client']).reset_index(drop=True)


def summaries(cases):
    primary = cases[cases.region == 'target_unseen_anchor']
    vehicles = primary.pivot_table(index=['seed','client','training_speed'], columns='method', values='yaw_rmse_deg_s')
    assert len(vehicles) == 27
    fleets = vehicles.groupby('seed').mean()
    strata = vehicles.groupby(['seed','training_speed']).mean()
    pooled = vehicles.groupby('training_speed').mean()
    for frame in [strata, pooled]:
        frame['absolute_reduction_deg_s'] = frame.SameSpeedAfter-frame.Complementary
        frame['relative_reduction_pct'] = 100*(1-frame.Complementary/frame.SameSpeedAfter)
    return dict(vehicles=vehicles.reset_index(), fleets=fleets.reset_index(), strata=strata.reset_index(), pooled=pooled.reset_index())


def set_style():
    plt.rcParams.update({'font.family':'serif','font.serif':['DejaVu Serif'],'font.size':8,
        'mathtext.fontset':'dejavuserif','axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,
        'legend.fontsize':7,'axes.linewidth':.6,'lines.linewidth':1.25,'pdf.fonttype':42,
        'ps.fonttype':42,'svg.fonttype':'path','svg.hashsalt':PREFIX,'savefig.facecolor':'white'})


def tidy(ax):
    ax.spines[['top','right']].set_visible(False)
    ax.tick_params(width=.6, length=3)
    ax.grid(axis='y', color='#dddddd', linewidth=.5)
    ax.set_axisbelow(True)


def panel(height=2.65):
    fig, ax = plt.subplots(figsize=(3.48,height), layout='constrained')
    tidy(ax)
    return fig, ax


def export(fig, name, out, gallery, caption):
    files = []
    for suffix in ['pdf','svg','png']:
        p = out/f'{PREFIX}_{name}_panel.{suffix}'
        metadata = {'CreationDate':None,'ModDate':None} if suffix == 'pdf' else {'Date':None} if suffix == 'svg' else None
        fig.savefig(p, dpi=220, metadata=metadata)
        if suffix=='svg':
            p.write_text('\n'.join(line.rstrip() for line in p.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8',newline='\n')
        files.append(p)
    buf = io.StringIO(); fig.savefig(buf, format='svg', metadata={'Date':None})
    svg = buf.getvalue(); gallery.append(dict(name=name, caption=caption, svg=svg[svg.index('<svg'):]))
    plt.close(fig)
    return files


def summary_panels(t, out, gallery):
    files = []
    fig, ax = panel()
    for k, method in enumerate(METHODS):
        for j, seed in enumerate([1301,1302,1303]):
            value = t['fleets'].set_index('seed').loc[seed,method]
            ax.scatter(k+(j-1)*.12, value, color=COLORS[method], marker=MARKERS[seed], s=30, zorder=3)
        ax.scatter(k, t['fleets'][method].mean(), marker='_', s=220, color='black', linewidths=1.6, zorder=4)
    ax.set(xticks=range(4), xticklabels=['Local\nOE','Nominal\nprior','FL: same\nspeed','FL:\ncomplementary'],
           ylabel='Mean yaw prediction RMSE [deg/s]', ylim=(0,None), xlim=(-.45,3.45))
    handles = [Line2D([],[],linestyle='',marker=MARKERS[s],color='#444444',label=f'Fleet {s}') for s in MARKERS]
    handles.append(Line2D([],[],linestyle='',marker='_',markersize=10,color='black',label='Pooled mean'))
    ax.legend(handles=handles, frameon=False, ncol=2, loc='upper right', fontsize=6.4)
    ax.set_ylim(0,max(t['fleets'][m].max() for m in METHODS)*1.3)
    files += export(fig,'fleet_error',out,gallery,'Mean clean yaw prediction error at target-unseen anchor speeds. Each fleet mean averages nine vehicles and four tests per vehicle. Black ticks show the pooled mean; symbols are the three development fleets. No confidence intervals are implied.')
    fig, ax = panel(2.8)
    top = 1.09*max(t['vehicles'].NominalPrior.max(),t['vehicles'].Complementary.max())
    ax.plot([0,top],[0,top],color='#555555',linewidth=.8,linestyle='--')
    for row in t['vehicles'].to_dict('records'):
        ax.scatter(row['NominalPrior'],row['Complementary'],color=SPEED_COLORS[row['training_speed']],
            marker=MARKERS[row['seed']],s=28,edgecolors='white',linewidths=.3,zorder=3)
    ax.set(xlim=(0,top),ylim=(0,top),xlabel='Nominal-prior yaw RMSE [deg/s]',ylabel='Complementary-FL yaw RMSE [deg/s]')
    ax.set_aspect('equal',adjustable='box')
    handles = [Line2D([],[],linestyle='',marker='o',color=SPEED_COLORS[v],label=f'{v} m/s') for v in SPEED_COLORS]
    ax.legend(handles=handles,title='Target training speed',frameon=False,fontsize=6.4,title_fontsize=6.5,loc='upper left')
    files += export(fig,'vehicle_comparison',out,gallery,'Each point is one vehicle averaged over the same four unseen-anchor tests. Below the identity line favors complementary-peer learning. Color gives training speed; marker shapes identify fleets as in the fleet panel. Axes have equal scales.')
    for metric, ylabel, name in [('absolute_reduction_deg_s','Yaw RMSE reduction [deg/s]','peer_absolute'),
                                ('relative_reduction_pct','Yaw RMSE reduction [%]','peer_relative')]:
        fig, ax = panel(2.4)
        ax.axhline(0,color='#444444',linewidth=.85)
        for j, seed in enumerate(MARKERS):
            g=t['strata'][t['strata'].seed==seed]
            ax.plot(np.arange(3)+(j-1)*.12,g[metric],linestyle='',marker=MARKERS[seed],color='#68778a',markersize=5,label=f'Fleet {seed}')
        ax.plot(range(3),t['pooled'][metric],linestyle='',marker='D',color=COLORS['Complementary'],markersize=5,label='Pooled comparison')
        ax.set(xticks=range(3),xticklabels=['13','21','29'],xlim=(-.4,2.4),xlabel='Target training speed [m/s]',ylabel=ylabel)
        ax.legend(frameon=False,fontsize=6.2,ncol=2,loc='upper right')
        low,high=ax.get_ylim();ax.set_ylim(low-(high-low)*.12,high+(high-low)*.3)
        files += export(fig,name,out,gallery,'Complementary versus same-speed peers at target-unseen anchors. Positive values favor complementary peers. Fleet symbols average three targets and twelve tests at each training speed. Pooled percentages are ratios of pooled mean errors, not means of individual percentages. There is no uncertainty band.')
    return files


def replay_case(row, data):
    seed=int(row['seed']);bundle=data['bundles'][seed];cfg=data['cfg']
    i,c=next((i,c) for i,c in enumerate(bundle['fleet']['clients']) if c['client']==row['client'])
    a,_=run.base.generate(c,seed,row['probe'],int(row['speed']),cfg)
    parameters = [bundle['original']['nominal_prior'][c['client']]['parameters'],
        np.exp(bundle['states'][int(row['training_speed'])]['log_parameters'][i]),
        np.exp(bundle['complementary']['optimization']['log_parameters'][i])]
    predicted = run.base.batch_predict(parameters,[a['steering']],[row['speed']],c['geometry_m'],cfg['sample_time'],cfg['integration_max_step'])[:,0]
    key=f'{c["client"]}_v{int(row["speed"])}_{row["probe"]}'
    differences=[]
    for method,y in zip(METHODS[1:],predicted):
        rms=lambda x:float(np.rad2deg(np.sqrt(np.mean(x*x))))
        actual=[rms(y[:,1]-a['state'][:,1]),rms(y[:,1]-a['measured_yaw']),rms(y[:,0]-a['state'][:,0])]
        saved=data['cases'][(data['cases'].seed==seed)&(data['cases'].client==c['client'])&
            (data['cases'].speed==row['speed'])&(data['cases'].probe==row['probe'])&(data['cases'].method==method)]
        expected=saved[METRICS].iloc[0].to_numpy(float)
        np.testing.assert_allclose(actual,expected,**data['protocol']['replay_tolerance'])
        differences.append(float(np.max(abs(np.array(actual)-expected))))
        a['predicted_'+method]=y
    info=dict(seed=seed,client=c['client'],training_speed=int(row['training_speed']),evaluation_speed=int(row['speed']),
        probe=row['probe'],source_case=key,original_source_trace_sha256=bundle['original']['evaluation'][key]['trace_sha256'],
        reconstructed_plot_trace_sha256=run.base.storage.trace_hash(a),max_metric_difference=max(differences),
        provenance='Numerical replay of frozen models on an existing test; no fitting and no claim of byte-exact original trace recovery.')
    return a,info


def trajectory_panels(traces, out, gallery):
    files=[]
    # Identical scales across both selected vehicles, with no cropped time window.
    y_limit=1.08*max(np.rad2deg(abs(a['state'][:,1])).max() for a,_ in traces)
    error_limit=1.08*max(np.rad2deg(abs(a['predicted_'+m][:,1]-a['state'][:,1])).max() for a,_ in traces for m in METHODS[1:])
    steering_limit=1.08*max(np.rad2deg(abs(a['steering'])).max() for a,_ in traces)
    for a,info in traces:
        training=info['training_speed']; v=info['evaluation_speed']; t=a['time']
        label=f'Fleet {info["seed"]}, {info["client"]}; training {training} m/s, test {v} m/s; {info["probe"]} probe.'
        for kind,height in [('steering',1.2),('yaw',1.75),('error',1.6)]:
            fig,ax=panel(height)
            if kind=='steering':
                ax.step(t,np.rad2deg(np.r_[a['steering'],a['steering'][-1]]),where='post',color='#222222',linewidth=1)
                ax.set(ylabel=r'$\delta$ [deg]',ylim=(-steering_limit,steering_limit))
            elif kind=='yaw':
                ax.plot(t,np.rad2deg(a['state'][:,1]),color='black',linewidth=1.5,label='Simulation truth')
                for m in METHODS[1:]: ax.plot(t,np.rad2deg(a['predicted_'+m][:,1]),color=COLORS[m],linestyle=STYLES[m],label=LABELS[m])
                ax.set(ylabel=r'$r$ [deg/s]',ylim=(-y_limit,y_limit))
                ax.legend(frameon=False,fontsize=5.8,ncol=2,loc='upper right')
                ax.set_ylim(-y_limit,1.6*y_limit)
            else:
                ax.axhline(0,color='#555555',linewidth=.6)
                for m in METHODS[1:]: ax.plot(t,np.rad2deg(a['predicted_'+m][:,1]-a['state'][:,1]),color=COLORS[m],linestyle=STYLES[m])
                ax.set(ylabel=r'$\hat r-r$ [deg/s]',ylim=(-error_limit,error_limit))
            ax.set(xlim=(0,t[-1]),xlabel='Time [s]')
            files+=export(fig,f'trace_train{training}_{kind}',out,gallery,label+' '+
                {'steering':'Prescribed applied steering; this is an open-loop prediction test.',
                 'yaw':'Clean simulation truth and frozen-model predictions. Near-overlap is preserved; no inset.',
                 'error':'Signed clean-yaw prediction errors. Both examples use the same error-axis limits.'}[kind])
    return files


def write_gallery(gallery,selection,replays,path):
    sections=''.join(f'<section><h2>{html.escape(g["name"].replace("_"," "))}</h2>{g["svg"]}<p>{html.escape(g["caption"])}</p></section>' for g in gallery)
    text=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Personalized identification: figure review</title>
<style>body{{background:#eef3f6;color:#173348;font:16px/1.6 system-ui;margin:0}}main{{max-width:1000px;margin:auto;padding:28px}}section{{background:white;padding:24px;margin:20px 0;border-radius:10px}}svg{{width:100%;max-width:650px;height:auto}}table{{border-collapse:collapse;font-size:13px}}td,th{{padding:8px;border-bottom:1px solid #ddd}}.scroll{{overflow:auto}}pre{{overflow:auto}}h1,h2{{line-height:1.25}}</style><main>
<h1>Figure review: personalized physical identification</h1><p>Separate, untitled vector panels for later LaTeX assembly. Three development fleets, 27 vehicles; clean prediction error at the two target-unseen anchor speeds is the primary metric. All compared personalized fits are converged. No statistical confirmation or control benefit is inferred.</p>
<p>This is a figure gallery, not the planned paper discussion document. Each PDF has no embedded panel letter or title. Colors and line styles remain consistent across panels.</p>
<section><h2>Trajectory selection and replay</h2><p>For target training speeds 13 and 21 m/s, use the already existing 29 m/s broadband test. Within each nine-vehicle stratum, select the vehicle nearest the median nominal-prior RMSE; break ties by seed and client. No FL error or improvement enters selection. Both records are shown in full with common axis limits. The selection rule was fixed for this figure package after the experiments, so these are descriptive examples, not independent validation.</p><div class="scroll">{selection[['seed','client','training_speed','speed','yaw_rmse_deg_s','distance_to_median','selected']].to_html(index=False,border=0,float_format=lambda x:f'{x:.6g}')}</div><p>Frozen-model numerical replays agree with the saved yaw, noisy-yaw and sideslip metrics. Original execution artifacts are unchanged.</p><div class="scroll">{pd.DataFrame(replays)[['seed','client','training_speed','evaluation_speed','max_metric_difference']].to_html(index=False,border=0)}</div></section>{sections}</main></html>'''
    path.write_text('\n'.join(line.rstrip() for line in text.splitlines())+'\n',encoding='utf-8',newline='\n')


def main():
    parser=argparse.ArgumentParser(__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--preflight',action='store_true');mode.add_argument('--generate',action='store_true')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'results/figures')
    args=parser.parse_args();data=checked_data()
    selection=select_trajectories(data['cases'],data['protocol']['trajectory_selection'])
    selected=selection[selection.selected]
    if args.preflight:
        print(json.dumps(dict(status='preflight_passed',method_case_rows=len(data['cases']),result_panels=10,new_fits=0,
            existing_test_replays=2,selected=selected[['seed','client','training_speed','speed','probe']].to_dict('records')),indent=2));return
    out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    default=out==(ROOT/'results/figures').resolve()
    table_dir=run.OUT if default else out
    report=ROOT/'report/personalized_id_figures.html' if default else out/'personalized_id_figures.html'
    set_style();t=summaries(data['cases']);gallery=[];files=summary_panels(t,out,gallery)
    traces=[]
    for row in selected.to_dict('records'):
        print(f'Replaying selected {row["seed"]}/{row["client"]}: train {row["training_speed"]}, test {row["speed"]}',flush=True)
        traces.append(replay_case(row,data))
    files+=trajectory_panels(traces,out,gallery)
    for name,frame in dict(t,selection=selection).items():
        target=table_dir/f'{PREFIX}_{name}.csv';frame.to_csv(target,index=False);files.append(target)
    replays=[info for _,info in traces];write_gallery(gallery,selection,replays,report);files.append(report)
    for name,sha in data['sources'].items():
        if run.base.canonical_hash(ROOT/name)!=sha: raise ValueError('Source changed while rendering: '+name)
    display=lambda p:p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else str(p)
    manifest=dict(status='generated_from_frozen_results',new_fits=0,new_test_conditions=0,existing_test_replays=2,
        result_panels=10,generator_sha256=digest(Path(__file__)),source_sha256=data['sources'],
        runtime=run.base.identity(data['cfg'])['runtime'],matplotlib_version=matplotlib.__version__,
        selection_rule=data['protocol']['trajectory_selection'],replays=replays,
        conventions=dict(width_inches=3.48,font_points=8,panel_titles=False,insets=False,
            uncertainty='Individual fleet points, no confidence intervals',percentage='Ratio of mean errors; positive favors complementary'),
        output_sha256={display(p):digest(p) for p in files if p.suffix in ['.pdf','.csv','.html']})
    path=table_dir/f'{PREFIX}_manifest.json';path.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(f'Exported 10 PDF/SVG/PNG panels. Gallery: {report}\nManifest: {path}')


if __name__=='__main__':main()
