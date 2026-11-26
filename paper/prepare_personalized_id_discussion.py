"""Review frozen figure outputs and build the offline paper discussion for Markus.

python paper/prepare_personalized_id_discussion.py
No fitting, simulation, source-figure edits or manuscript edits.
"""
from __future__ import annotations
import hashlib
import html
import io
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import fitz
import prepare_personalized_id_figures as figures

ROOT=figures.ROOT
REPORT=ROOT/'report/personalized_id_paper_discussion.html'
AUDIT=ROOT/'results/tables/paper_personalized_id_figure_review.json'


def namespace_svg(svg, prefix, label):
    """Keep inline font and clip-path IDs unique when combining saved panels."""
    ids=re.findall(r'\bid="([^"]+)"',svg)
    assert len(ids)==len(set(ids))
    for old in ids:
        new=prefix+'_'+old
        svg=svg.replace(f'id="{old}"',f'id="{new}"').replace(f'url(#{old})',f'url(#{new})').replace(f'href="#{old}"',f'href="#{new}"')
    svg=svg.replace('<svg ',f'<svg role="img" aria-label="{html.escape(label,quote=True)}" ',1)
    ET.fromstring(svg)
    assert '<script' not in svg and '<foreignObject' not in svg
    return svg


def equation(lines, name):
    fig=plt.figure(figsize=(10,.60*len(lines)+.20));ax=fig.add_axes([0,0,1,1]);ax.axis('off')
    artists=[]
    for i,line in enumerate(lines):
        artists.append(ax.text(.015,1-(i+.65)/(len(lines)+.3),'$'+line+'$',fontsize=13.5,va='center'))
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    assert all(fig.bbox.contains(*point) for a in artists
               for point in a.get_window_extent(renderer).get_points()),name
    b=io.StringIO();fig.savefig(b,format='svg',metadata={'Date':None});plt.close(fig)
    svg=b.getvalue();return '<div class="equation">'+namespace_svg(svg[svg.index('<svg'):],name,'Equation '+name)+'</div>'


def checked_inputs():
    data=figures.checked_data()
    path=ROOT/'results/tables/paper_personalized_id_manifest.json'
    manifest=json.loads(path.read_text(encoding='utf-8'))
    assert manifest['status']=='generated_from_frozen_results' and manifest['result_panels']==10
    assert manifest['new_fits']==manifest['new_test_conditions']==0 and manifest['existing_test_replays']==2
    assert manifest['generator_sha256']==figures.digest(Path(figures.__file__))
    assert manifest['source_sha256']==data['sources']
    for name,expected in manifest['output_sha256'].items():
        assert figures.digest(ROOT/name)==expected, name
    tables=figures.summaries(data['cases'])
    tables['selection']=figures.select_trajectories(data['cases'],data['protocol']['trajectory_selection'])
    for name,frame in tables.items():
        pd.testing.assert_frame_equal(frame,pd.read_csv(ROOT/f'results/tables/paper_personalized_id_{name}.csv'),
                                      check_exact=False,check_names=False,rtol=1e-10,atol=1e-12)
    selected=tables['selection'][tables['selection'].selected]
    for row,entry in zip(selected.to_dict('records'),manifest['replays']):
        assert (row['seed'],row['client'],row['training_speed'],row['speed'],row['probe'])==(
            entry['seed'],entry['client'],entry['training_speed'],entry['evaluation_speed'],entry['probe'])
        assert entry['max_metric_difference']<=1e-10
    checks=[]
    for name in manifest['output_sha256']:
        if name.endswith('.pdf'):
            with fitz.open(ROOT/name) as doc:
                assert len(doc)==1 and abs(doc[0].rect.width-3.48*72)<.01
                fonts=doc[0].get_fonts();assert fonts and all(f[2]!='Type3' for f in fonts)
                assert all(doc.extract_font(f[0])[3] for f in fonts)
                assert not doc[0].get_images()
                checks.append(dict(file=name,pages=1,width_inches=doc[0].rect.width/72,vector=True,type3_fonts=False,fonts_embedded=True))
    assert len(checks)==10
    gallery=(ROOT/'report/personalized_id_figures.html').read_text(encoding='utf-8')
    panels=dict(re.findall(r'<section><h2>([^<]+)</h2>\s*(<svg\b.*?</svg>)\s*<p>',gallery,re.S))
    panels={k.replace(' ','_'):v for k,v in panels.items()}
    assert len(panels)==10
    for name in panels: ET.fromstring(panels[name])
    assert manifest['output_sha256']['results/figures/paper_personalized_id_trace_train13_steering_panel.pdf']==manifest['output_sha256']['results/figures/paper_personalized_id_trace_train21_steering_panel.pdf']
    paths=dict(data['sources'],**manifest['output_sha256'])
    for p in [path,Path(figures.__file__),ROOT/'paper/figures/personalized_id/personalized_id_overview.drawio',
              ROOT/'paper/figures/personalized_id/personalized_id_overview_preview.svg']:
        paths[p.relative_to(ROOT).as_posix()]=figures.digest(p)
    return data,tables,manifest,panels,checks,paths


def render(data,t,manifest,panels):
    means=t['fleets'][figures.METHODS].mean()
    gain=lambda method,baseline:100*(1-means[method]/means[baseline])
    metrics=pd.DataFrame({'Method':[figures.LABELS[m] for m in figures.METHODS],
        'Mean yaw RMSE [deg/s]':[means[m] for m in figures.METHODS],
        'Reduction versus Local OE [%]':[gain(m,'LocalOE') for m in figures.METHODS]})
    table=lambda df:'<div class="scroll">'+df.to_html(index=False,border=0,float_format=lambda v:f'{v:.6g}')+'</div>'
    def panel(name,caption):
        return '<figure>'+namespace_svg(panels[name],'p_'+name,caption)+'<figcaption>'+html.escape(caption)+'</figcaption></figure>'
    overview=(ROOT/'paper/figures/personalized_id/personalized_id_overview_preview.svg').read_text(encoding='utf-8')
    overview=namespace_svg(overview[overview.index('<svg'):],'overview','Personalized physical identification and matched peer-speed comparison')
    plt.rcParams.update({'font.family':'serif','font.serif':['DejaVu Serif'],'mathtext.fontset':'dejavuserif','svg.fonttype':'path','svg.hashsalt':'personalized-id-discussion'})
    state=equation([r'\mathbf{x}_i=(\beta_i,r_i,q_{f,i},q_{r,i})^{\mathsf{T}},\qquad q_{f,i}=F_{yf,i}/m_i,\quad q_{r,i}=F_{yr,i}/m_i',
        r'\boldsymbol{\theta}_i=(p_{1,i},p_{2,i},p_{3,i},\mu_i,\sigma_{f,i},\sigma_{r,i})^{\mathsf{T}}',
        r'p_{1,i}=C_{f,i}/m_i,\qquad p_{2,i}=C_{r,i}/m_i,\qquad p_{3,i}=m_i/I_{z,i}'],'state')
    plant=equation([r'\dot\beta=(q_f+q_r)/v-r,\qquad \dot r=p_3(l_fq_f-l_rq_r)',
        r'\dot q_f=(v/\sigma_f)(q_f^\star-q_f),\qquad \dot q_r=(v/\sigma_r)(q_r^\star-q_r)',
        r'q_f^\star=a_f\tanh(p_1\alpha_f/a_f),\qquad q_r^\star=a_r\tanh(p_2\alpha_r/a_r)',
        r'\alpha_f=\delta-\beta-l_fr/v,\qquad \alpha_r=-\beta+l_rr/v',
        r'a_f=\mu g l_r/(l_f+l_r),\qquad a_r=\mu g l_f/(l_f+l_r)'],'plant')
    loss=equation([r'y_{i,k}=r_{i,k}+\epsilon_{i,k},\qquad \epsilon_{i,k}\sim\mathcal{N}(0,s_r^2)',
        r'\mathbf{z}_i=\log\boldsymbol{\theta}_i,\qquad L_i(\mathbf{z}_i)=\sum_{k=0}^{N}\left(\frac{\hat{r}_{i,k}(\exp\mathbf{z}_i)-y_{i,k}}{s_r}\right)^2'],'loss')
    joint=equation([r'\min_{\{\mathbf{z}_i\in\mathcal{Z}\}_{i=1}^{M},\,\boldsymbol{\eta}}\ \sum_{i=1}^{M}\left[L_i(\mathbf{z}_i)+\|\mathbf{W}(\mathbf{z}_i-\boldsymbol{\eta})\|_2^2\right]',
        r'\mathbf{W}=\operatorname{diag}(10,10,10,0,4,4)'],'objective')
    updates=equation([r'\mathbf{z}_i^{(t+1)}\ \approx\ \operatorname{arg\,min}_{\mathbf{z}\in\mathcal{Z}}\left[L_i(\mathbf{z})+\|\mathbf{W}(\mathbf{z}-\boldsymbol{\eta}^{(t)})\|_2^2\right]',
        r'\boldsymbol{\eta}^{(t+1)}=\frac{1}{M}\sum_{i=1}^{M}\mathbf{z}_i^{(t+1)}'],'updates')
    metric=equation([r'E_{i,v,p}=\sqrt{\frac{1}{N+1}\sum_{k=0}^{N}\left(\hat{r}_{i,k}(v,p)-r_{i,k}(v,p)\right)^2}',
        r'\Delta E=\overline{E}_{\mathrm{same}}-\overline{E}_{\mathrm{comp}},\qquad G=100\left(1-\frac{\overline{E}_{\mathrm{comp}}}{\overline{E}_{\mathrm{same}}}\right)'],'metric')
    selected=[]
    for row in t['selection'][t['selection'].selected].to_dict('records'):
        g=data['cases'];g=g[(g.seed==row['seed'])&(g.client==row['client'])&(g.speed==row['speed'])&(g.probe==row['probe'])]
        m=g.set_index('method').yaw_rmse_deg_s
        selected.append(dict(Fleet=row['seed'],Vehicle=row['client'],Training_speed=row['training_speed'],
            Nominal_prior=m.NominalPrior,Same_speed=m.SameSpeedAfter,Complementary=m.Complementary,
            Reduction_vs_nominal_pct=100*(1-m.Complementary/m.NominalPrior)))
    fleet=t['fleets'].copy()
    fleet['FL gain versus nominal [%]']=100*(1-fleet.Complementary/fleet.NominalPrior)
    wins=int((t['vehicles'].Complementary<t['vehicles'].NominalPrior).sum())
    html_doc=f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Personalized physical identification - discussion draft for Markus</title>
<style>
:root{{--ink:#19313f;--muted:#536671;--teal:#087d87;--gold:#b47922;--line:#dce5e9}}
*{{box-sizing:border-box}}html{{scroll-behavior:smooth}}body{{margin:0;color:var(--ink);background:#eef3f6;font:17px/1.65 system-ui,sans-serif}}main{{max-width:1120px;margin:auto;padding:30px 26px 65px}}header{{padding:20px 4px}}h1{{font-size:clamp(28px,4vw,43px);line-height:1.15;max-width:1000px;letter-spacing:-.02em}}h2{{font-size:25px;line-height:1.25;margin-top:0}}h3{{font-size:19px;line-height:1.35}}.eyebrow{{font-size:13px;letter-spacing:.08em;text-transform:uppercase;color:var(--teal);font-weight:700}}.lead{{font-size:20px;max-width:92ch}}nav{{display:flex;flex-wrap:wrap;gap:9px;margin:20px 0}}nav a{{background:white;border:1px solid var(--line);border-radius:20px;padding:5px 14px;text-decoration:none;color:var(--ink);font-size:14px}}section{{background:white;padding:28px;margin:22px 0;border-radius:12px;border:1px solid var(--line);scroll-margin-top:15px}}p{{margin:14px 0}}.claim{{border-left:5px solid var(--teal);padding:10px 20px;background:#edf7f6}}.question{{border-left:4px solid var(--gold);padding:10px 18px;background:#faf5ea}}.muted,figcaption,.small{{color:var(--muted);font-size:14px}}.grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px;align-items:start}}.cards{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}}.card{{border:1px solid var(--line);padding:14px;border-radius:8px}}.card strong{{display:block;color:var(--teal);font-size:27px;line-height:1.2}}figure{{margin:12px 0;min-width:0}}figure>svg,.overview>svg{{width:100%;height:auto;display:block}}figcaption{{margin-top:8px;line-height:1.5}}.shared-input{{max-width:510px;margin:15px auto}}.equation{{overflow-x:auto;margin:18px 0;background:#fafbfc;border-radius:6px;padding:8px}}.equation svg{{width:100%;min-width:570px;height:auto;display:block}}.scroll{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;font-size:14px;line-height:1.5}}td,th{{padding:10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}}th{{font-weight:650;background:#f7f9fa}}td:first-child{{font-weight:500}}ul,ol{{padding-left:24px}}li{{margin:8px 0}}details{{margin:16px 0;border-top:1px solid var(--line);padding-top:12px}}summary{{cursor:pointer;font-weight:600}}a{{color:var(--teal)}}code{{font-size:14px}}pre{{overflow:auto;background:#edf3f6;padding:15px;font-size:13px}}.tag{{display:inline-block;padding:3px 9px;border-radius:5px;background:#edf7f6;font-size:13px;color:var(--teal)}}
@media(max-width:740px){{main{{padding:18px 12px 45px}}section{{padding:20px 16px}}.grid,.cards{{grid-template-columns:1fr}}body{{font-size:16px}}h2{{font-size:22px}}}}
@media print{{body{{background:white;font-size:10pt}}main{{padding:0;max-width:none}}header{{padding:0}}nav{{display:none}}section{{border:0;padding:15px 0;margin:0;border-radius:0}}h1{{font-size:24pt}}h2{{font-size:16pt}}figure,.claim,.card{{break-inside:avoid}}.equation svg{{min-width:0}}details:not([open]){{display:none}}.grid{{gap:12px}}a{{color:inherit}}}}
</style></head><body><main>
<header><div class="eyebrow">Scientific discussion draft · Jakob / Markus · 26 November 2026</div>
<h1>Personalized Federated Identification of Vehicle Lateral Dynamics from Limited Local Data</h1>
<p class="lead">Can a vehicle improve predictions beyond its locally observed speed by learning a physical parameter prior from other vehicles, and how much does this add beyond a fixed nominal prior?</p>
<p class="muted">Working paper framing based on the completed development experiments and reviewed figure outputs. This document supports a scope decision; it is not a submission-ready manuscript or a claim of established methodological novelty.</p>
<nav aria-label="Sections"><a href="#problem">Problem</a><a href="#model">Physical model</a><a href="#method">Learning method</a><a href="#design">Experiment</a><a href="#results">Results</a><a href="#discussion">Discussion</a><a href="#review">Figure review</a></nav></header>
<section id="problem"><h2>1. From the practical problem to a testable question</h2>
<p>A lateral-dynamics model should predict how a particular vehicle responds to steering across the speeds at which it operates. Vehicle parameters differ, while collecting informative identification records across that entire range can require additional testing. We therefore study a restricted-data scenario: each vehicle supplies one short record at one speed, and its model is evaluated at other speeds.</p>
<p>Physical structure already provides a substantial part of the answer. A continuous-time vehicle model contains an explicit speed dependence, so missing local speeds do not automatically imply that the model cannot predict there. The remaining question is whether a noisy, short local record determines the physical parameters accurately enough, and whether other vehicles supply useful information beyond an engineering nominal prior.</p>
<div class="question"><b>Desired capability.</b> Improve predictions for a vehicle with limited local data while preserving its own dynamics, keeping measured trajectories local, and avoiding additional identification records for that target.</div>
<p>Our proposed solution uses personalized physical models and a shared learned regularization center. The essential comparison is against a local estimator with a fixed nominal prior. A second, matched comparison changes only the operating speeds of peers to determine whether complementary coverage contributes further.</p>
<div class="claim"><b>Claim supported by the present pilots.</b> In the tested limited-data regime, personalized fleet-learned regularization modestly improves mean yaw prediction accuracy beyond a fixed nominal prior. Complementary peer speeds provide an additional benefit concentrated in low-speed targets.</div>
<p class="small">Limited local data are an explicit scenario assumption here, not an established restriction of all practical vehicles. The current results address identification; a meaningful closed-loop improvement has not been established.</p>
<figure class="overview">{overview}<figcaption>Proposed overview. Each vehicle identifies a separate physical model; the server updates a shared log-parameter center. The lower-left counts describe peers only, for a target trained at 13 m/s. This is the same-geometry preview of the editable draw.io source; a native draw.io PDF export remains a final typesetting task.</figcaption></figure>
</section>
<section id="model"><h2>2. Physical structure provides the speed dependence</h2>
<p>The current plant is a four-state bicycle model with tanh tire forces and first-order front/rear force relaxation. For vehicle i, β is sideslip, r is yaw rate, and q<sub>f</sub>, q<sub>r</sub> are lateral tire forces divided by vehicle mass. The six positive parameters contain stiffness-to-mass ratios, mass-to-inertia ratio, friction, and two relaxation lengths:</p>{state}
<p>C<sub>f</sub> and C<sub>r</sub> are axle cornering stiffnesses, m is mass, I<sub>z</sub> is yaw inertia, μ is the friction coefficient, and σ<sub>f</sub>, σ<sub>r</sub> are relaxation lengths. The known geometry l<sub>f</sub>, l<sub>r</sub> gives the center-of-gravity distances to the axles. With steering δ, fixed positive speed v, and gravity g=9.81 m/s², the implemented dynamics are, omitting the vehicle index:</p>{plant}
<p>Here α<sub>f</sub>, α<sub>r</sub> are the model's small-angle slip expressions, a<sub>f</sub>, a<sub>r</sub> are force-per-mass saturation levels, and q<sub>f</sub><sup>★</sup>, q<sub>r</sub><sup>★</sup> are equilibrium tire responses. Relaxation supplies the two additional dynamic states. The model identifies physical ratios, not absolute mass, inertia and force scale separately.</p>
<p>Only yaw rate is observed during identification; steering and speed are known. Sideslip and force states are simulated internally. Every record begins from the known zero state in this simulation study. No state reconstruction or unknown-initial-state estimation is part of the present experiment.</p>
<p>This gives a speed-dependent physical model through a compact parameter vector. It changes the earlier paper emphasis: the current experiment learns parameters of these equations, rather than independent matrix entries or a flexible polynomial LPV basis. Both identification and test records have fixed speed; variable-speed and longitudinal dynamics are outside this study.</p>
</section>
<section id="method"><h2>3. Share a prior, keep a model for each vehicle</h2>
<p>For a record with N applied inputs and N+1 yaw samples, let y<sub>i,k</sub> be the measured yaw rate and let the model prediction r̂<sub>i,k</sub> be obtained by integrating the model using the known input history. The prediction is not corrected with future measurements. Define the log-parameter vector z<sub>i</sub> and the yaw output-error loss:</p>{loss}
<p>The noise standard deviation is s<sub>r</sub>=0.03 deg/s, converted to radians per second in the implementation. Log parameters use the fixed numerical unit convention of the code. The set ℤ contains the declared log-parameter bounds. A quadratic penalty uses a shared center η and fixed diagonal weights W:</p>{joint}
<p>The first three weights regularize the dynamic ratios, and the last two regularize relaxation lengths. The zero fourth weight leaves friction unpenalized. W is a fixed design choice for this study; it was not selected from these test errors. Calling η a prior describes its regularization role, not a calibrated probability distribution.</p>
<p>Starting from local multistart OE estimates, the implementation alternates bounded local nonlinear least-squares updates and a server mean:</p>{updates}
<p>The mean is a valid server minimizer because all vehicles use the same W. Its friction coordinate is immaterial to the penalty. Local solves and the overall nonlinear problem are not guaranteed to reach a global optimum. The reported models satisfy the declared numerical stopping tests.</p>
<div class="scroll"><table><tr><th>Method</th><th>What the vehicle learns</th><th>Purpose in the comparison</th></tr>
<tr><td>Local OE</td><td>Local parameters from yaw residuals only.</td><td>Measures the effect of adding regularization.</td></tr>
<tr><td>Nominal prior</td><td>Local parameters with the same W and a fixed engineering center η<sub>nom</sub>.</td><td>Tests whether peer learning adds value beyond a strong local baseline.</td></tr>
<tr><td>FL: same speed</td><td>Personalized parameters with a learned center; all peers operate at the target's speed.</td><td>Tests sharing without additional peer-speed coverage.</td></tr>
<tr><td>FL: complementary</td><td>Personalized parameters with a learned center; peer records span the three anchor speeds.</td><td>Tests the additional effect of complementary peer operation.</td></tr></table></div>
<details><summary>What is actually federated?</summary><p>Each local endpoint owns its yaw/input record and solves its own nonlinear problem. Per round, it uploads six log parameters and one scalar data loss, and receives six center coordinates: 13 float64 values, or 104 numerical payload bytes per client-round, before network headers and excluding initialization. The scalar losses support objective and convergence accounting. The experiment emulates this message interface on one machine. It establishes neither a deployed distributed implementation nor a privacy guarantee.</p></details>
</section>
<section id="design"><h2>4. A matched experiment separates three effects</h2>
<p>The three development fleets each contain nine vehicle realizations drawn from nominal, heavy and handling families. Each family contributes one target at 13, 21 and 29 m/s. Family labels and generating parameters are used for simulation/evaluation, not supplied to the learner. Every target provides one 12 s record, sampled at 0.01 s: 1200 steering intervals and 1201 yaw samples. Integration uses RK4 with 0.001 s internal steps.</p>
<p>The prescribed steering is a tapered multisine with a 0.35° peak. Training frequencies are 0.35, 0.7 and 1.2 Hz; evaluation uses distinct slow and broadband probes with different frequencies and phases. Steering is known without measurement noise; only the yaw observations are noisy. These are gentle synthetic identification maneuvers, not recorded road-driving trajectories.</p>
<p>For a target at 13 m/s, the same-speed condition places all eight peers at 13 m/s. The complementary condition keeps two peers at 13 m/s and places three at 21 m/s and three at 29 m/s. The corresponding permutation applies to the other target speeds. The target record, peer identities, input signals, paired noise samples, data duration and fitting settings are held fixed. Only six peers' operating speeds change. Each fitted fleet condition uses nine records, totaling 108 s; no target gains its own other-speed training data.</p>
<p>Primary evaluation averages clean yaw prediction RMSE over the other two anchor speeds, using both probes. The indices v and p below identify evaluation speed and probe. Errors are computed in radians per second and reported in degrees per second:</p>{metric}
<p>There are four primary cases per vehicle, 36 per fleet, and 108 across 27 vehicles. Means first average matched cases and are equally weighted across vehicles/fleets in this balanced design. Percentage gains are ratios of mean errors. The 108 trajectories do not constitute 108 independent fleet replications. Additional speeds 9, 17, 25 and 33 m/s are available as secondary diagnostics.</p>
<p>All nine same-speed fits and all three complementary fits now meet the original stopping criteria. The two previously incomplete same-speed fits ended at rounds 32 and 46; completing them changed the pooled peer-speed benefit from 4.30% to 4.10%.</p>
</section>
<section id="results"><h2>5. Results: a large prior effect and a modest sharing increment</h2>
<div class="cards"><div class="card"><strong>{gain('NominalPrior','LocalOE'):.1f}%</strong>Nominal-prior reduction versus unregularized local OE</div><div class="card"><strong>{gain('Complementary','NominalPrior'):.1f}%</strong>Additional mean reduction versus the fixed nominal prior</div><div class="card"><strong>{gain('Complementary','SameSpeedAfter'):.1f}%</strong>Additional mean reduction versus same-speed peers</div></div>
<p>Regularization accounts for most of the improvement over local OE. Learning the center from complementary peers reduces the mean error further, but the difference is small in absolute units: {means.NominalPrior-means.Complementary:.6f} deg/s versus the fixed nominal prior. The method ranking should therefore be read together with the fleet and vehicle variation.</p>
<div class="grid">{panel('fleet_error','Figure 2(a). Mean clean yaw prediction errors on the matched target-unseen anchor tests. The three symbols show fleet results; black ticks show the pooled mean. These are descriptive development results, without confidence intervals.')}{panel('vehicle_comparison','Figure 2(b). One point per vehicle, averaged over four unseen-anchor tests. Below the identity line favors complementary FL over the nominal prior. Colors identify training speed; circles, squares and triangles identify fleets 1301, 1302 and 1303.')}</div>
{table(metrics)}
<p>The learned complementary prior improves {wins}/27 vehicle means versus the nominal prior. It is worse at the fleet level in fleet 1301 and better in fleets 1302 and 1303. Thus the pooled gain does not establish consistent individual or fleet-level dominance.</p>
<details><summary>Fleet-level values and sensitivity to the population</summary>{table(fleet)}</details>
<h3>Complementary peer speeds mainly help low-speed targets</h3>
<p>Compared with same-speed peers, complementary peers reduce mean error by {gain('Complementary','SameSpeedAfter'):.2f}%, or {means.SameSpeedAfter-means.Complementary:.6f} deg/s. The effect depends on the target's training speed: a 6.79% improvement at 13 m/s, a 0.13% deterioration at 21 m/s, and a 1.63% deterioration at 29 m/s. In the low-speed stratum, fleet-specific gains range from 0.65% to 19.07%.</p>
<div class="grid">{panel('peer_absolute','Figure 3(a). Absolute reduction in mean yaw RMSE from changing same-speed peers to complementary-speed peers. Positive values favor complementary peers. Symbols identify fleets; the diamond compares pooled means.')}{panel('peer_relative','Figure 3(b). The same paired comparison in percent. The pooled percentage is computed from pooled errors, rather than averaging individual percentages. The zero line distinguishes improvements from deterioration.')}</div>
<p>This supports a conditional benefit from the tested peer-speed intervention. Changing speed also changes the yaw response amplitude, parameter sensitivity and output signal-to-noise ratio for the same steering. The experiment does not isolate geometric coverage independently of those effects.</p>
<h3>What the error differences look like in time</h3>
<p>Both examples below use the existing 29 m/s broadband test. For each training-speed stratum, the target was selected by proximity to the median nominal-prior error; no FL gain entered selection. This is a transparent descriptive choice made after the experiments, not an independent confirmation. A median baseline error does not imply a median FL improvement.</p>
<div class="shared-input">{panel('trace_train13_steering','Figure 4, shared input. Both examples use exactly the same applied steering at the same test speed. The two committed steering PDFs are identical, so only one is shown here.')}</div>
<div class="grid"><div><h3>Target trained at 13 m/s</h3><p class="small">Fleet 1302 · vehicle_03 · test 29 m/s</p>{panel('trace_train13_yaw','Clean simulation yaw and frozen-model predictions. The curves nearly overlap at the physical response scale.')}{panel('trace_train13_error','Signed errors reveal the improved prediction in this example. The error axis has the same limits as the 21 m/s example.')}</div><div><h3>Target trained at 21 m/s</h3><p class="small">Fleet 1303 · vehicle_07 · test 29 m/s</p>{panel('trace_train21_yaw','The same steering and test speed for a different vehicle and local training speed.')}{panel('trace_train21_error','This selected example slightly favors the nominal prior. The similar curves and common scale preserve the small magnitude of the difference.')}</div></div>
<p class="small">Selected-test RMSE values below are in deg/s. Positive reductions favor complementary FL.</p>
{table(pd.DataFrame(selected).rename(columns={'Training_speed':'Training speed [m/s]', 'Nominal_prior':'Nominal prior', 'Same_speed':'Same-speed FL', 'Complementary':'Complementary FL', 'Reduction_vs_nominal_pct':'Reduction vs nominal [%]'}))}
<p>For these individual tests, complementary FL reduces error by 56.76% versus the nominal prior in the 13 m/s example, and increases it by 11.16% in the 21 m/s example. These values illustrate variation; they must not replace the aggregate 7.0% comparison. The curves are open-loop predictions under prescribed steering, not closed-loop tracking responses.</p>
</section>
<section id="discussion"><h2>6. What the paper can claim, and what remains open</h2>
<p>The result is consistent with the following interpretation: physical structure makes prediction beyond a local speed possible, a prior substantially improves short-record estimation, and a fleet-learned prior can add a modest population-dependent improvement. Complementary operating speeds supply a further benefit for some low-speed targets. This interpretation is supported by matched comparisons; it is not a proof of parameter recovery or a causal decomposition of the total gain.</p>
<h3>Proposed paper contributions</h3><ol>
<li><b>Formulation:</b> personalized yaw-only physical identification with data-local nonlinear OE and a shared log-parameter regularizer.</li>
<li><b>Evaluation design:</b> comparisons that separately assess regularization, learning the prior, and changing peer operating speeds while preserving target data and peer identities.</li>
<li><b>Empirical finding:</b> a qualified benefit over a strong fixed-prior baseline, with explicit conditions under which complementary peers help and cases where they do not.</li></ol>
<p>The formulation should not be presented as a new general FL algorithm on the strength of these experiments. Its distinction from existing personalized physical identification and regularized multi-task learning requires a focused literature assessment before settling a novelty claim.</p>
<h3>Limits that materially affect interpretation</h3><ul>
<li><b>Three reused development fleets.</b> This is exploratory evidence; there is no independent confirmation for the new physical-prior story.</li>
<li><b>Favorable shared structure.</b> All vehicles use known geometry 1.2/1.6 m and true relaxation lengths 0.4/0.7 m. The nominal prior uses 0.6/0.6 m. A common true dynamic property creates an opportunity for sharing; this does not establish transfer between vehicles with different relaxation lengths.</li>
<li><b>Matched physical model class.</b> Both the plant and estimator use the same four-state tanh/relaxation structure. This is not a demonstration under unmodeled dynamics or measured-road data.</li>
<li><b>One prescribed nominal prior and weight matrix.</b> Beating this baseline does not imply superiority to every locally selected physical prior or regularization strategy.</li>
<li><b>Known zero initial state and fixed-speed tests.</b> General operation with unknown initial force/sideslip states is not tested here.</li>
<li><b>Prediction is the endpoint.</b> The earlier four-state controller/observer pilot did not establish better yaw tracking; the present peer comparison adds no new closed-loop evidence.</li></ul>
<details><summary>Earlier Global, grouped and control results</summary><p>The initial limited-speed pilot found poor prediction from a single shared Global model, and the tested grouped variant did not outperform the single learned prior. Those experiments motivate the personalized formulation, but they are not added to the final four-method figure as though all optimization protocols were equally closed. The earlier four-state prediction/control pilot also used a different protocol and should not be pooled with these errors. See the preserved <a href="federated_relaxation_coverage_review.html">limited-speed audit</a> and <a href="tire_relaxation_pilot_review.html">four-state control audit</a> for their evidence and limits.</p></details>
<h3>Decisions for the discussion with Markus</h3><ol>
<li>Is a conditional identification benefit beyond a strong physical prior substantial enough to be the paper's central contribution?</li>
<li>Is the assumed local-data restriction credible for the intended use case, and how should we state it without implying a demonstrated real-world constraint?</li>
<li>Should the intended paper be identification-focused, or should a meaningful closed-loop benefit remain a requirement before pursuing a control-paper submission?</li>
<li>Which genuinely new element distinguishes the formulation or evidence from established regularized/personalized identification?</li></ol>
<p class="question"><b>Recommended scope decision.</b> Use this document to agree on the question and required level of evidence before preparing further experiments or rewriting the manuscript. The current evidence supports a qualified identification story; it does not yet support a strong claim that FL materially improves vehicle control.</p>
</section>
<section id="review"><h2>7. Figure review and proposed paper structure</h2>
<p><span class="tag">Ready for scientific discussion</span> All ten PDF panels pass source/output hash checks, reconstruct their saved summaries, and use vector graphics with embedded fonts at 3.48-inch width. Visual review found no clipping, unreadable math, or unintended insets. The low-speed and higher-speed examples use common axes and show both benefit and deterioration.</p>
<div class="scroll"><table><tr><th>Figure</th><th>Decision</th></tr>
<tr><td>Overview</td><td>Use the new personalized-prior design. Native draw.io PDF export and final math spacing remain for publication assembly.</td></tr>
<tr><td>Fleet and vehicle comparison</td><td>Keep as the main evidence. Captions must name clean yaw prediction error and the target-unseen anchor tests.</td></tr>
<tr><td>Absolute and relative peer-speed effects</td><td>Keep both for discussion. For a tight paper layout, these can be paired with short subcaptions; the absolute magnitude must remain available.</td></tr>
<tr><td>Trajectory examples</td><td>Assemble one shared steering panel with two yaw/error columns. Preserve the error axes and state explicitly that case selection uses baseline error, not FL improvement.</td></tr></table></div>
<p>The separate panels are suitable for later LaTeX subfigures. No changes to the plotting code or regenerated figures are required for this discussion draft.</p>
<div class="scroll"><table><tr><th>Proposed section</th><th>Reader's take-home</th></tr>
<tr><td>I. Introduction and related work</td><td>Physical structure already generalizes across speeds; the remaining question is the incremental value of fleet information under limited local data.</td></tr>
<tr><td>II. Physical model and identification problem</td><td>What is known, measured, estimated, and assumed about speed and initial state.</td></tr>
<tr><td>III. Personalized federated identification</td><td>The local yaw loss, shared prior, alternating updates, and fixed-prior baseline.</td></tr>
<tr><td>IV. Matched evaluation</td><td>Which information each method receives and how target-unseen prediction is evaluated.</td></tr>
<tr><td>V. Results</td><td>Most benefit comes from regularization; learned sharing and complementary speeds add conditional increments.</td></tr>
<tr><td>VI. Discussion and conclusion</td><td>The supported claim, practical assumptions, and unresolved control relevance.</td></tr></table></div>
<details><summary>Provenance and reproducibility</summary><p>Figure results: commit 242fb93. Code and editable overview: commit 8511815. The source manifest verifies the final numerical continuation, original baselines, figure outputs, selection rules and two saved-model replays. On the user's execution runtime, both displayed replays reproduce the saved metrics with zero recorded difference. This review performs no new fitting, simulation or figure edits.</p>
<p>The figure-review audit is saved as <code>results/tables/paper_personalized_id_figure_review.json</code>. The generator uses NumPy, Pandas, Matplotlib and PyMuPDF (for PDF inspection). Rebuild this document after validating the committed inputs:</p><pre>python paper/prepare_personalized_id_discussion.py</pre>
<p>Figures and equations are embedded; the document can be read offline. Links to the original audits and editable diagram require the surrounding repository. The formal manuscript is unchanged.</p>
<p><a href="personalized_id_figures.html">Original figure gallery</a> · <a href="federated_relaxation_consolidation.html">Full consolidation audit</a> · <a href="../paper/figures/personalized_id/personalized_id_overview.drawio">Editable overview</a></p></details>
</section><footer class="small">Discussion draft prepared from frozen development evidence. No claim of a closed-loop benefit or independent confirmation is made.</footer>
</main></body></html>'''
    REPORT.write_text('\n'.join(line.rstrip() for line in html_doc.splitlines())+'\n',encoding='utf-8',newline='\n')
    return dict(means_deg_s=means.to_dict(),vehicle_wins_vs_nominal=wins,selected_examples=selected)


def main():
    data,t,manifest,panels,checks,sources=checked_inputs()
    info=render(data,t,manifest,panels)
    for name,expected in sources.items():assert figures.digest(ROOT/name)==expected,name
    output=REPORT.read_text(encoding='utf-8')
    ids=re.findall(r'\bid="([^"]+)"',output);assert len(ids)==len(set(ids))
    assert output.count('<figure')==10 and '<script' not in output
    audit=dict(status='figure_review_passed',new_fits=0,new_simulations=0,source_artifacts_modified=False,
        pdf_checks=checks,summary_and_selection_tables_reconstructed=True,shared_steering_pdfs_identical=True,
        source_sha256=sources,generator_sha256=figures.digest(Path(__file__)),discussion_sha256=figures.digest(REPORT),
        overview_native_pdf_present=(ROOT/'results/figures/paper_personalized_id_overview.pdf').exists(),
        figure_replay_metadata=manifest['replays'],**info)
    figures.run.base.storage.atomic_json(AUDIT,audit)
    print(json.dumps(dict(status=audit['status'],pdf_panels=len(checks),report=str(REPORT)),indent=2))


if __name__=='__main__':main()
