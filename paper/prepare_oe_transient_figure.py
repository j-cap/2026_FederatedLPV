r"""Publication panels from committed transient traces; no simulation or fitting.

python paper\prepare_oe_transient_figure.py --generate
python paper\prepare_oe_transient_figure.py --check
Optional: --compile-preview (requires pdflatex and preview/subfig packages).
The paper includes the separate PDFs using figures/oe_transient_figure.tex.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT/'paper/oe_transient_figure_spec.json'
PREFIX = 'paper_oe_transient_compact'
LABELS = {'NominalLTI':'Nominal LTI', 'Global':'Global', 'Federated':'Federated'}
COLORS = {'NominalLTI':'#666666', 'Global':'#de7e24', 'Federated':'#167a9e'}
STYLES = {'NominalLTI':'--', 'Global':'-.', 'Federated':'-'}
CAPTION = (r'Fixed-speed yaw-rate tracking at three speeds unobserved in local identification. '
           r'Top: reference $r_{\mathrm{ref}}$. Middle and bottom: errors '
           r'$e_r=r-r_{\mathrm{ref}}$ for the same fixed heavy and handling recipients, respectively. '
           r'All six error panels use identical limits. The reference has the same nominal '
           r'lateral-acceleration demand at each speed. These individual examples complement the fleet-level comparison.')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    path = Path(path); data = path.read_bytes()
    if path.suffix in ['.json','.py','.tex','.html','.yml']:
        data = data.replace(b'\r\n',b'\n')
    return hashlib.sha256(data).hexdigest()


def original_digest(path):
    # Match the historical manifest convention exactly, including CSV/HTML bytes.
    path = Path(path); data = path.read_bytes()
    if path.suffix in ['.json','.py','.yml']:
        data = data.replace(b'\r\n',b'\n')
    return hashlib.sha256(data).hexdigest()


def validate_traces(frame, metrics, spec):
    keys = ['seed','client','method','speed_mps']
    expected = {(spec['seed'],c,m,float(v)) for c in spec['clients'].values()
                for m in spec['methods'] for v in spec['speeds_mps']}
    require(set(map(tuple,frame[keys].drop_duplicates().to_numpy())) == expected,
            'Missing or unexpected displayed case')
    refs, traces = {}, {}
    checks = []
    for family, client in spec['clients'].items():
        for speed in spec['speeds_mps']:
            for method in spec['methods']:
                part = frame[(frame.seed==spec['seed']) & (frame.client==client) &
                             (frame.method==method) & (frame.speed_mps==speed)].sort_values('time_s')
                require(len(part)==1201 and part.time_s.nunique()==1201, 'Incomplete or duplicate trajectory')
                require(set(part.family)=={family}, 'Recipient family mismatch')
                columns = ['time_s','reference_deg_s','yaw_rate_deg_s','yaw_error_deg_s']
                require(np.isfinite(part[columns].to_numpy()).all(), 'Nonfinite plotted trace')
                np.testing.assert_allclose(part.time_s.to_numpy(),np.arange(1201)*.01,atol=1e-12,rtol=0)
                np.testing.assert_allclose(part.yaw_rate_deg_s-part.reference_deg_s,
                                           part.yaw_error_deg_s,atol=1e-12,rtol=0)
                reference = part.reference_deg_s.to_numpy()
                if speed in refs:
                    np.testing.assert_array_equal(reference,refs[speed])
                else:
                    refs[speed] = reference
                result = metrics[(metrics.seed==spec['seed']) & (metrics.client==client) &
                                 (metrics.method==method) & (metrics.speed_mps==speed)]
                require(len(result)==1 and result.iloc[0]['status']=='complete', 'Selected result is not complete')
                row = result.iloc[0]
                rmse = float(np.sqrt(np.mean(part.yaw_error_deg_s.to_numpy()**2)))
                np.testing.assert_allclose(rmse,row.tracking_rmse_deg_s,atol=1e-12,rtol=0)
                checks.append(dict(family=family,client=client,speed_mps=speed,method=method,
                                   tracking_rmse_deg_s=rmse,feasible=bool(row.feasible)))
                traces[(family,speed,method)] = part
    return refs,traces,checks


def prepare():
    spec = load(CONFIG)
    path = ROOT/spec['source_manifest']
    require(digest(path)==spec['source_manifest_sha256'], 'Original benchmark manifest changed')
    manifest = load(path)
    require(manifest['cases']==5400 and manifest['exact_selected_replays']==18, 'Incomplete source benchmark')
    require(manifest['specification']['selection']==dict(seed=spec['seed'],clients=spec['clients']),
            'Previously fixed recipients changed')
    # Validate committed evidence directly. No regeneration of transcendental
    # reference functions: their bit patterns can differ across OS/libm builds.
    for name, expected in (manifest['sources'] | manifest['checkpoints'] | manifest['outputs']).items():
        require(original_digest(ROOT/name)==expected, f'Changed evidence: {name}')
    traces_path = ROOT/'results/tables/paper_oe_transient_benchmark_traces.csv'
    metrics_path = ROOT/'results/tables/paper_oe_transient_benchmark_metrics.csv'
    frame = pd.read_csv(traces_path,float_precision='round_trip')
    metrics = pd.read_csv(metrics_path,float_precision='round_trip')
    refs,traces,checks = validate_traces(frame,metrics,spec)
    sources = {p.relative_to(ROOT).as_posix():digest(p) for p in [CONFIG,Path(__file__),path,traces_path,metrics_path]}
    return spec,refs,traces,checks,sources


def make_panels(spec,refs,traces):
    plt.rcParams.update({'font.family':'serif','font.serif':['DejaVu Serif'],'font.size':8,
        'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,'axes.linewidth':.6,
        'lines.linewidth':1.05,'mathtext.fontset':'dejavuserif','pdf.fonttype':42,
        'ps.fonttype':42,'svg.fonttype':'none','svg.hashsalt':PREFIX})
    width = spec['panel_width_in']
    error_peak = max(float(abs(p.yaw_error_deg_s).max()) for p in traces.values())
    error_lim = float(np.ceil(1.05*error_peak/.02)*.02)
    ref_lim = float(np.ceil(1.05*max(np.max(abs(r)) for r in refs.values())/2)*2)
    panels = {}
    for speed in spec['speeds_mps']:
        for row in ['reference','heavy','handling']:
            bottom = row=='handling'
            height = spec['row_heights_in'][row]
            fig = plt.figure(figsize=(width,height))
            # Inches, not proportional margins: labels remain 8 pt at final size.
            left,right,lower,upper = .68,.09,(.36 if bottom else .08),.06
            ax = fig.add_axes([left/width,lower/height,(width-left-right)/width,(height-lower-upper)/height])
            ax.set_xlim(0,12);ax.set_xticks([0,4,8,12])
            ax.tick_params(axis='x',labelbottom=bottom,pad=2,length=2.5)
            ax.tick_params(axis='y',pad=2,length=2.5)
            ax.set_axisbelow(True);ax.grid(color='#dddddd',linewidth=.4)
            ax.spines[['top','right']].set_visible(False)
            ax.axhline(0,color='#aaaaaa',linewidth=.45)
            if bottom:
                ax.set_xlabel('Time [s]',labelpad=2)
            if row=='reference':
                ax.set_ylim(-ref_lim,ref_lim);ax.set_yticks([-10,0,10])
                ax.set_ylabel('Reference\n[deg/s]',labelpad=2)
                ax.plot(np.arange(1201)*.01,refs[speed],color='#222222',linewidth=1.05)
            else:
                ax.set_ylim(-error_lim,error_lim);ax.set_yticks([-.1,0,.1])
                ax.set_yticklabels(['−0.10','0','0.10'])
                ax.set_ylabel(row.title()+': $e_r$\n[deg/s]',labelpad=2)
                for method in spec['methods']:
                    p=traces[(row,speed,method)]
                    ax.plot(p.time_s,p.yaw_error_deg_s,color=COLORS[method],linestyle=STYLES[method])
            panels[f'v{speed}_{row}'] = fig
    legend = plt.figure(figsize=(spec['figure_width_in'],.25))
    handles = [Line2D([],[],color=COLORS[m],linestyle=STYLES[m],linewidth=1.05,label=LABELS[m]) for m in spec['methods']]
    legend.legend(handles=handles,loc='center',ncol=3,frameon=False,fontsize=8,handlelength=2.8,columnspacing=2.4)
    panels['legend']=legend
    return panels,dict(error_limits_deg_s=[-error_lim,error_lim],reference_limits_deg_s=[-ref_lim,ref_lim])


def latex_fragment(spec):
    lines=[r'% Generated from frozen traces; keep titles and panel letters in LaTeX.',
           r'\begin{figure*}[t]',r'\centering']
    for i,v in enumerate(spec['speeds_mps']):
        lines += [rf'\subfloat[$v={v}\,\mathrm{{m/s}}$\label{{fig:oe-transient-v{v}}}]{{%',
                  rf'\begin{{minipage}}[t]{{{spec["panel_width_in"]:.2f}in}}',r'\centering']
        for row in ['reference','heavy','handling']:
            lines.append(rf'\includegraphics[width=\linewidth]{{{PREFIX}_v{v}_{row}_panel.pdf}}\par\nointerlineskip')
        lines += [r'\end{minipage}}'+(r'\hfill' if i<2 else '')]
    lines += [r'\par\vspace{1pt}',rf'\includegraphics[width=\textwidth]{{{PREFIX}_legend.pdf}}',
              r'\caption{'+CAPTION+'}',r'\label{fig:oe-transient}',r'\end{figure*}']
    return '\n'.join(lines)+'\n'


def write_text(path,text):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(text,encoding='utf-8',newline='\n')


def generate(context,base):
    spec,refs,traces,checks,sources=context
    figures=base/'results/figures';figures.mkdir(parents=True,exist_ok=True)
    files=[];svgs={}
    panels,limits=make_panels(spec,refs,traces)
    for name,fig in panels.items():
        stem=PREFIX+'_'+name+('' if name=='legend' else '_panel')
        path=figures/(stem+'.pdf')
        fig.savefig(path,metadata={'CreationDate':None,'ModDate':None});files.append(path)
        fig.savefig(figures/(stem+'.png'),dpi=300)
        buffer=io.StringIO();fig.savefig(buffer,format='svg',metadata={'Date':None})
        svgs[name]=buffer.getvalue()[buffer.getvalue().index('<svg'):]
        plt.close(fig)
    fragment=base/'paper/figures/oe_transient_figure.tex'
    write_text(fragment,latex_fragment(spec));files.append(fragment)
    preview=base/'paper/oe_transient_figure_preview.tex'
    write_text(preview,r'''\documentclass[conference]{IEEEtran}
\usepackage{graphicx}
\usepackage[caption=false,font=footnotesize]{subfig}
\captionsetup[subfloat]{farskip=0pt,captionskip=2pt}
\usepackage[active,tightpage,floats]{preview}
\setlength\PreviewBorder{3pt}
\graphicspath{{../results/figures/}}
\begin{document}
\input{figures/oe_transient_figure}
\end{document}
''');files.append(preview)
    cells=[]
    for row in ['reference','heavy','handling']:
        cells += [f'<div>{svgs[f"v{v}_{row}"]}</div>' for v in spec['speeds_mps']]
    heads=''.join(f'<div class="speed">({chr(97+i)}) {v} m/s</div>' for i,v in enumerate(spec['speeds_mps']))
    report=base/'report'/f'{PREFIX}.html'
    write_text(report,f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Compact transient tracking figure</title>
<style>body{{font:16px/1.5 system-ui;max-width:1100px;margin:32px auto;padding:0 20px;color:#17232d}}
.figure{{max-width:1000px;margin:24px auto}}.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:0 8px}}
.speed{{text-align:center;font-size:14px}}svg{{width:100%;height:auto;display:block}}.legend{{margin:6px auto}}
.caption{{font-size:14px}}code{{background:#eef2f5;padding:2px 4px}}details{{margin-top:18px}}table{{border-collapse:collapse;font-size:13px}}td,th{{padding:4px 10px;border-bottom:1px solid #ddd}}</style>
<h1>Tracking across unobserved speeds</h1>
<p>Publication layout: three speed columns, shared reference row, then tracking errors for both fixed recipients.
All six error axes are identical. No smoothing, time-window selection, normalization or new simulation.</p>
<div class="figure"><div class="grid">{heads}{''.join(cells)}</div><div class="legend">{svgs['legend']}</div>
<p class="caption"><b>Figure preview.</b> Fixed-speed yaw-rate tracking at three speeds unobserved in local identification.
Top: reference. Middle and bottom: e_r = r − r_ref for the fixed heavy and handling recipients.
The nominal lateral demand is identical across speeds. These individual examples complement the fleet-level comparison.</p></div>
<p>The full response can be read as r = r_ref + e_r. Separate error axes reveal small differences without hiding the reference scale.
The trials have zero longitudinal acceleration. Steering traces remain in the full benchmark report because they add little visual separation here.</p>
<p>Recipients: fleet {spec['seed']}, heavy_05 and handling_07. They were fixed before this benchmark; no reselection was made.
The handling example is not evidence of a uniform family-wide benefit. Retain the family-level qualification in the manuscript.</p>
<p>ISO-7401-inspired yaw-reference tracking adaptation, not an ISO-compliant steering-input test.
Noise-free state feedback and the existing nonlinear tanh-tire plant are used.</p>
<details><summary>Selected-case full-duration RMSE [deg/s]</summary>{pd.DataFrame(checks).to_html(index=False,float_format=lambda v:f'{v:.6f}')}</details>
<h2>LaTeX integration</h2><p>The nine data PDFs and shared legend are separate files. Panel letters and speed labels come from LaTeX.
The figure uses 2.30-inch panels at native size (8 pt axis labels, 7 pt ticks), with a total two-column width of approximately 7.16 inches.</p>
<p>When integrating the figure, add <code>\\input{{figures/oe_transient_figure}}</code> to paper/main.tex.
The file uses the manuscript's existing graphicspath and subfig package. The manuscript has not been changed.</p>
<p>Regenerate with <code>python paper\\prepare_oe_transient_figure.py --generate</code>, then verify with
<code>python paper\\prepare_oe_transient_figure.py --check</code>. Add <code>--compile-preview</code> to generate when a LaTeX installation is available.</p>
<p>Input checks use the committed trace/metric/checkpoint hashes and recompute displayed RMSE from the trace CSV.
They do not regenerate the floating-point sine waveform, so plotting is portable without weakening the original simulation checks.</p></html>''')
    files.append(report)
    return files,limits


def compile_preview(base):
    require(shutil.which('pdflatex') is not None,'pdflatex is required only for --compile-preview')
    paper=base/'paper';build=paper/'build/oe_transient_preview';build.mkdir(parents=True,exist_ok=True)
    # Isolated output directories can still resolve the project's IEEE class.
    env=os.environ.copy()
    env['TEXINPUTS']=str(ROOT/'paper')+os.pathsep+env.get('TEXINPUTS','')+os.pathsep
    for _ in range(2):
        result=subprocess.run(['pdflatex','-interaction=nonstopmode','-halt-on-error',
            '-output-directory',str(build),'oe_transient_figure_preview.tex'],cwd=paper,env=env,
            text=True,encoding='utf-8',errors='replace',stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        require(result.returncode==0,'LaTeX preview failed; inspect '+str(build/'oe_transient_figure_preview.log')+'\n'+result.stdout[-2000:])
    target=base/'results/figures'/f'{PREFIX}_preview.pdf'
    shutil.copyfile(build/'oe_transient_figure_preview.pdf',target)
    return target


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    for option in ['preflight','generate','check']:
        mode.add_argument('--'+option,action='store_true')
    parser.add_argument('--compile-preview',action='store_true')
    parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args()
    require(not args.compile_preview or args.generate,'--compile-preview requires --generate')
    context=prepare();spec,refs,traces,checks,sources=context
    base=args.output_dir.resolve() if args.output_dir else ROOT
    manifest_path=base/'results/tables'/f'{PREFIX}_manifest.json'
    if args.preflight:
        result=dict(status='preflight_passed',verified_traces=len(checks),new_simulations=0)
    elif args.check:
        manifest=load(manifest_path)
        require(manifest['sources']==sources and manifest['trace_checks']==checks,'Stale source/trace audit')
        for name,expected in manifest['outputs'].items():
            require(digest(base/name)==expected,f'Changed output: {name}')
        result=dict(status='verified',panels=9,shared_legends=1,new_simulations=0)
    else:
        files,limits=generate(context,base)
        if args.compile_preview:
            files.append(compile_preview(base))
        manifest=dict(version=1,sources=sources,specification=spec,trace_checks=checks,limits=limits,
            renderer=dict(matplotlib=matplotlib.__version__),
            outputs={p.relative_to(base).as_posix():digest(p) for p in files})
        write_text(manifest_path,json.dumps(manifest,indent=2,allow_nan=False)+'\n')
        result=dict(status='generated',panels=9,shared_legends=1,new_simulations=0,
            report=str(base/'report'/f'{PREFIX}.html'),latex=str(base/'paper/figures/oe_transient_figure.tex'))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
