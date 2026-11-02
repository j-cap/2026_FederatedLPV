"""Read-only audit of the frozen pilot; write an offline HTML discussion report.

No acquisition, local fit, selection, or historical result is changed. Run:
python code/experiments/analyze_trajectory_pilot.py
"""
from pathlib import Path
import json
import subprocess
import numpy as np
import pandas as pd
import paper_trajectory_calibration as pilot

ROOT = pilot.ROOT
HTML = ROOT / "report/trajectory_calibration_pilot.html"
PREFIX = pilot.PREFIX


def main():
    cfg, parent, _, _, provenance = pilot.preflight()
    paths = pilot.pilot_paths(cfg)
    results = [pilot.read_checkpoint(path, provenance) for path in paths]
    selection = pilot.load(pilot.SELECTION)
    assert selection["provenance"] == provenance
    for path, expected in selection["pilot_sha256"].items():
        assert pilot.file_hash(ROOT/path) == expected, path
    rows = pilot.local_rows(results)
    saved = pd.read_csv(pilot.TABLES/f"{PREFIX}_pilot_cells.csv")
    pd.testing.assert_frame_equal(rows, saved, check_dtype=False, rtol=1e-12, atol=1e-12)
    profiles, chosen = pilot.pilot_decision(rows, cfg)
    assert profiles == selection["profiles"] and chosen == selection["selected_profile"]
    physical, details = [], []
    for r in results:
        for cell in r["cells"]:
            truth = np.asarray(r["clients"][cell["client_index"]]["reference"])[parent["speed_grid"].index(cell["speed"])]
            for phase in ["training", "validation"]:
                physical.append(cell[phase])
            for case, values in cell["cases"].items():
                error = float(np.linalg.norm(np.asarray(values["target"])-truth)/np.linalg.norm(truth))
                assert np.isclose(error, values["matrix_error"], rtol=1e-12, atol=1e-14)
                details.append(dict(profile=r["profile"], seed=r["seed"], client=cell["client"],
                    family=cell["client"].rsplit("_",1)[0], speed=cell["speed"], case=case,
                    matrix=100*error, rollout=values["rollout_nrmse"], persistence=values["persistence_nrmse"],
                    condition=values["scaled_condition"], sigma=values["scaled_sigma_min"],
                    diverged=values["rollout_diverged"], truth=truth.tolist(), target=values["target"],
                    amplitude=min(cfg["max_excitation_deg"], cfg["profiles"][r["profile"]]*(20/cell["speed"])**2)))
    summary = []
    for (profile,case), v in rows.groupby(["profile","case"],sort=False):
        summary.append(dict(profile=profile,case=case,n=len(v),
            median_pct=100*v.matrix_error.median(),p95_pct=100*v.matrix_error.quantile(.95),
            matrix_pass=int((v.matrix_error <= .1).sum()),
            rollout_pass=int((v.rollout_nrmse <= .5).sum()),
            persistence_wins=int((v.rollout_nrmse < v.persistence_nrmse).sum()),
            diverged=int(v.rollout_diverged.sum()),rollout_p95=float(v.rollout_nrmse.quantile(.95))))
    sensor = rows[(rows.profile == "strong") & (rows.case == "TrajectoryAllSensor")].copy()
    sensor["matrix_pass"] = sensor.matrix_error <= .1
    speed = sensor.groupby("speed").agg(n=("matrix_error","size"),
        median_pct=("matrix_error",lambda x:100*x.median()),
        p95_pct=("matrix_error",lambda x:100*x.quantile(.95)),passing=("matrix_pass","sum"),
        rollout_median=("rollout_nrmse","median")).reset_index()
    maxima = {key:max(p[key] for p in physical) for key in
        ["max_beta_deg","max_ay","max_jerk","max_steering_deg","max_rate_deg_s","quiet_yaw_error","quiet_drift"]}
    commit = subprocess.check_output(["git","log","-1","--format=%h","--",str(pilot.SELECTION.relative_to(ROOT))],cwd=ROOT,text=True).strip()
    diagnostic = dict(status="frozen_pilot_audited",result_commit=commit,
        audit=dict(checkpoints=len(results),cell_condition_rows=len(rows),matrix_errors_recomputed=len(details),
            checkpoint_hashes=True,source_locks=True,csv_reproduced=True,selection_reproduced=True),
        selected_profile=chosen,profile_summary=profiles,condition_summary=summary,
        strong_by_speed=speed.to_dict("records"),physical_maxima=maxima,
        decision="Do not start confirmation. Diagnose noisy local identification and low-speed excitation.",
        limitations=["Three pilot fleets, descriptive summaries; cells are not independent fleets.",
            "No federated/global/Smooth Local comparison was run in the scientific pilot.",
            "Physical plausibility is assessed within the simulated plant and chosen limits, not on-road validation.",
            "Validation is a separate waveform but the same speed and amplitude envelope; it does not test all state directions.",
            "Conditioning/rank thresholds do not ensure adequate information relative to measurement noise.",
            "No raw trajectories were regenerated and no alternative estimator was fitted for this report."],
        source_sha256=provenance["source_sha256"],checkpoint_sha256=selection["pilot_sha256"])
    pilot.save_json(pilot.TABLES/f"{PREFIX}_pilot_diagnosis.json",diagnostic)
    speed.to_csv(pilot.TABLES/f"{PREFIX}_pilot_diagnosis_by_speed.csv",index=False)
    data = dict(details=details,summary=summary,profiles=profiles,maxima=maxima,commit=commit)
    html = TEMPLATE.replace("__DATA__",json.dumps(data,allow_nan=False,separators=(",",":")).replace("<","\\u003c"))
    HTML.write_text(html,encoding="utf-8",newline="\n")
    print(json.dumps(dict(report=str(HTML),**diagnostic["audit"],selected_profile=chosen),indent=2))


TEMPLATE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Continuous calibration pilot — evidence and next decision</title>
<style>
:root{--ink:#162b3a;--muted:#586b79;--bg:#f4f6f8;--line:#dce3e9;--blue:#205cc5;--green:#14836d;--red:#bd3a36;--gold:#b07708}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.55 system-ui,sans-serif}main{max-width:1280px;margin:auto;padding:38px 26px 70px}h1{font-size:36px;line-height:1.17;margin:12px 0 18px;max-width:1000px}h2{font-size:24px;margin:0 0 12px}h3{font-size:18px;margin:0 0 8px}p{margin:8px 0 14px}.eyebrow{color:var(--muted);font-size:13px;letter-spacing:.08em;text-transform:uppercase}.lead{font-size:19px;max-width:1000px}.card{background:white;border:1px solid var(--line);border-radius:12px;padding:24px;margin:20px 0}.decision{border-left:6px solid var(--gold);background:#fffaf0}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}.metric{background:white;border:1px solid var(--line);border-radius:10px;padding:18px}.metric b{font-size:31px;display:block}.muted,small{color:var(--muted)}small{font-size:13px}.controls{display:flex;gap:18px;flex-wrap:wrap;align-items:end;margin:18px 0}label{font-size:13px;color:var(--muted)}select,button{display:block;background:white;border:1px solid #a8b7c3;border-radius:6px;padding:9px 12px;color:var(--ink);font:inherit}button{cursor:pointer}button:hover{background:#eef4ff}.two{display:grid;grid-template-columns:1fr 1fr;gap:22px}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:10px 12px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}th{background:#f7f9fb;color:var(--muted);font-weight:600}td.bad{color:var(--red);font-weight:650}.note{padding:14px 16px;background:#edf3f8;border-radius:6px;font-size:14px}.legend{display:flex;gap:14px;flex-wrap:wrap;font-size:13px;margin:8px 0}.legend span:before{content:'';display:inline-block;width:10px;height:10px;background:var(--c);border-radius:2px;margin-right:6px}.chart svg{width:100%;height:auto;display:block}.chart{min-width:0}svg text{font-family:system-ui,sans-serif;fill:var(--muted);font-size:12px}circle{cursor:crosshair}a{color:var(--blue)}code{font-size:13px;background:#eef2f5;padding:2px 4px;border-radius:3px}details{margin:14px 0}summary{cursor:pointer;font-weight:650}.math{overflow:auto;font-family:Cambria,serif;font-size:19px;padding:15px;background:#f7f9fb}footer{font-size:13px;color:var(--muted)}.tag{display:inline-block;border-radius:4px;padding:3px 7px;background:#fff0d2;color:#77520c;font-size:13px}tr.pick{cursor:pointer}tr.pick:hover{background:#eef4ff}.chosen{background:#eef4ff}#detail{margin-top:16px}
@media(max-width:850px){.grid,.two{grid-template-columns:1fr}main{padding:22px 14px}h1{font-size:29px}.card{padding:16px}}@media print{body{background:white}.controls,button{display:none}.card{break-inside:avoid}main{padding:0}h1{font-size:27px}}
</style></head><body><main>
<div class="eyebrow">Federated LPV · Continuous calibration · Pilot seeds 1301–1303</div>
<h1>The trajectories work in the simulator.<br>The noisy local matrix estimates need more work.</h1>
<p class="lead">All nine pilot jobs verified. Continuous motion, settling, and clean-data identification pass. With sensor noise, ordinary least squares still fails the matrix-accuracy requirement, especially at low speed. The ten-fleet sharing comparison has not started.</p>
<div class="grid"><div class="metric"><span class="muted">Maneuver / settling checks</span><b>747 / 747</b><small>client–speed–profile cells; training and validation both pass</small></div><div class="metric"><span class="muted">Noise-free accuracy screen</span><b>747 / 747</b><small>matrix accuracy, validation prediction and persistence comparison</small></div><div class="metric"><span class="muted">Strong profile, noisy matrices</span><b>183 / 249</b><small>73.5% pass; at least 95% was required</small></div></div>
<section class="card decision"><span class="tag">Decision: pause confirmation</span><h2 style="margin-top:10px">Keep the continuous-data direction. Diagnose local identification first.</h2><p>The pilot does not show a failure of federation: it contains no scientific group-fitting comparison. It identifies a problem earlier in the pipeline, when noisy state and steering observations are converted into local A/B matrices.</p><p><strong>Do not relax the screen or substitute noise-free results for the sensor condition.</strong> The clean and target-only-noise conditions are useful controls for the next diagnosis.</p></section>
<section class="card"><h2>1. What passed, and what stopped selection?</h2><div class="scroll"><table id="profiles"></table></div><p class="muted">Each profile covers 249 local fits across three fleets of 30 vehicles. The 95th percentile describes the distribution across those cells; it is not a confidence interval.</p><div class="two"><div><h3>Physical envelope observed</h3><table id="physical"></table></div><div><h3>How to read “passed”</h3><p>Every profile satisfies the declared steering, lateral-acceleration and sideslip limits, reaches the intended turn, and has full local regressor rank with acceptable scaled condition number.</p><p>Every sensor fit also meets the one-second prediction limit and beats holding the initial state constant. <strong>Only the matrix-error condition stops the sensor accuracy screen.</strong></p><p class="note">These are simulation checks under instrumented calibration assumptions. They do not certify road safety, ordinary-driving identification, or observer-based sensing.</p></div></div></section>
<section class="card"><h2>2. Separate the acquisition and noise effects</h2><div class="controls"><label>Steering profile<select id="profile"><option value="gentle">Gentle — 0.15° at 20 m/s</option><option value="standard">Standard — 0.30° at 20 m/s</option><option value="strong" selected>Strong — 0.60° at 20 m/s</option></select></label><label>Vehicle family<select id="family"><option value="all">All families</option><option>nominal</option><option>heavy</option><option>handling</option></select></label><label>Fleet<select id="fleet"><option value="all">All three fleets</option><option>1301</option><option>1302</option><option>1303</option></select></label><button id="reset">Reset filters</button></div><p id="filterCount" class="muted"></p><div class="scroll"><table id="conditions"></table></div><p class="note"><strong>Matched row count is not matched driving time.</strong> The 36 selected trajectory transitions are spread over the full 20-second excitation window. “All” uses 2,000 consecutive transitions. Reset36 also uses an analytical equilibrium, while the trajectory fits use an empirical quiet-segment center.</p><div class="legend" id="legend"></div><div class="chart" id="speedChart"></div><p class="muted">Lines show median matrix error at each observed speed, conditional on the filters. The logarithmic axis separates clean-data accuracy from the larger noisy errors.</p></section>
<section class="card"><h2>3. Why does stronger steering leave the same upper tail?</h2><p>The amplitude is capped at <strong>0.8°</strong>. At 10 and 12 m/s, both the standard and strong settings hit that cap and therefore generate identical physical steering and paired noise realizations. Raising the profile label cannot improve these cells.</p><div class="two"><div class="chart" id="amplitudeChart"></div><div><h3>Strong profile: where the 66 failures occur</h3><table id="failures"></table><p><strong>62 of 66 failures are at 10–14 m/s.</strong> Every 10 and 12 m/s sensor fit fails the 10% matrix-error limit. All 18–30 m/s cells pass.</p><p>The clean regressor remains full rank. However, its weakest standardized direction is substantially smaller at low speed. Full rank and a permissive condition-number limit do not establish robustness to measurement noise.</p></div></div></section>
<section class="card"><h2>4. Good trajectory prediction can hide a poor dynamics matrix</h2><p>For the strongest profile, all 249 noisy fits predict the separate chirp satisfactorily, even though 66 have excessive matrix error. A validation maneuver explores a restricted set of state/input combinations; errors in individual coefficients can compensate along that motion.</p><div class="chart" id="scatter"></div><p class="muted">Sensor condition, using the filters above. Hover over a point for client, speed and errors. The shaded area passes prediction but fails matrix accuracy. Both axes are logarithmic. Prediction uses a one-second free run from a common exact initial state; it does not test state-estimation error.</p><details><summary>Why measurement noise can bias this regression</summary><p>Ignoring centering and nonlinear residuals for this explanation, let measured states and steering be x̃<sub>k</sub> = x<sub>k</sub> + ν<sub>k</sub> and ũ<sub>k</sub> = u<sub>k</sub> + η<sub>k</sub>. The equation expressed in measured variables is</p><div class="math">x̃<sub>k+1</sub> = A x̃<sub>k</sub> + B ũ<sub>k</sub> + [ν<sub>k+1</sub> − Aν<sub>k</sub> − Bη<sub>k</sub>].</div><p>The regression residual contains the same current measurement errors as the regressors. Its correlation with them can bias ordinary least squares. The clean → target-only noise → coherent sensor contrast strongly supports this explanation, but the current pilot does not separately isolate state noise, steering noise, temporal noise reuse, and noisy centering.</p><p>See <a href="https://www.mathworks.com/help/ident/ug/picking-instrumental-variables-for-system-identification.html">MathWorks’ identification example on measurement noise, output-error models and instrumental variables</a>. It motivates alternatives; it is not a guarantee that a particular IV estimator will work with these weakly excited data.</p></details></section>
<section class="card"><h2>5. Inspect the individual fits</h2><p>Sensor fits sorted by decreasing matrix error. Click a row to compare its estimated matrix entries with the reference. Filters above apply here too.</p><button id="export">Download filtered sensor diagnostics (CSV)</button><div class="scroll" style="max-height:380px;margin-top:14px"><table id="cells"></table></div><div id="detail"></div></section>
<section class="card"><h2>6. The next focused step</h2><ol><li><strong>Keep this pilot frozen.</strong> Use its saved trajectories and targets to investigate noise sensitivity; the ten confirmation fleets remain untouched.</li><li><strong>Separate the noise sources on the same physical motion.</strong> Compare state-only, steering-only and centering effects. This identifies what an estimator must handle.</li><li><strong>Test one noise-aware local estimator against OLS.</strong> A carefully constructed instrumental-variable or output-error fit is a candidate. Verify instruments/noise assumptions and local matrix accuracy, not just one-step prediction.</li><li><strong>If low-speed directions remain weak, revise the input spectrum.</strong> Compare a small predefined low-speed bandwidth change within the same steering/rate envelope. The amplitude cap explains why simply increasing the current profile is ineffective at 10 and 12 m/s.</li><li><strong>Freeze a passing calibration/estimator combination, then run the fleet comparison.</strong> Reassess Global and Smooth Local alongside Federated on the same new targets.</li></ol><p class="note">No alternative estimator was fitted and no new trajectory campaign was run for this analysis. Recommendations are prospective. More samples alone do not establish that measurement-error bias is removed.</p></section>
<section class="card"><h2>Provenance and scope</h2><p id="provenance"></p><p>Audited: nine payload hashes and source locks; all 3,735 cell–condition matrix errors independently recomputed from saved targets and references; the saved CSV and the frozen selection reproduced. No historical artifact was rewritten.</p><p>There are three independent pilot fleets. Filtered cell summaries are descriptive and should not be treated as 747 independent experimental replications. No confidence interval for a sharing benefit is available here.</p><p>Rebuild: <code>python code/experiments/analyze_trajectory_pilot.py</code></p><p>Output: <code>report/trajectory_calibration_pilot.html</code>. This page contains its data and charts and works offline.</p></section>
<footer>Prepared for discussion of the continuous calibration pilot. Existing manuscript evidence remains unchanged.</footer>
</main><script>
const D=__DATA__;
const cases=['Reset36','Trajectory36Bridge','TrajectoryAllBridge','TrajectoryAllClean','TrajectoryAllSensor'];
const names={Reset36:'Reset · 36',Trajectory36Bridge:'Trajectory · 36 · target noise',TrajectoryAllBridge:'Trajectory · 2,000 · target noise',TrajectoryAllClean:'Trajectory · 2,000 · clean',TrajectoryAllSensor:'Trajectory · 2,000 · sensor noise'};
const colors={Reset36:'#788a99',Trajectory36Bridge:'#b07708',TrajectoryAllBridge:'#205cc5',TrajectoryAllClean:'#14836d',TrajectoryAllSensor:'#bd3a36'};
const $=id=>document.getElementById(id),fmt=(x,n=2)=>x==null?'diverged / unavailable':Number(x).toFixed(n);
const quant=(a,p)=>{a=[...a].filter(Number.isFinite).sort((x,y)=>x-y);if(!a.length)return null;const k=(a.length-1)*p,i=Math.floor(k);return a[i]+(a[Math.ceil(k)]-a[i])*(k-i)};
function table(id,heads,rows){$(id).innerHTML='<thead><tr>'+heads.map(h=>'<th>'+h+'</th>').join('')+'</tr></thead><tbody>'+rows.map(r=>'<tr>'+r.map(v=>'<td>'+v+'</td>').join('')+'</tr>').join('')+'</tbody>'}
function filtered(){return D.details.filter(d=>d.profile===$('profile').value&&($('family').value==='all'||d.family===$('family').value)&&($('fleet').value==='all'||d.seed===+$('fleet').value))}
function chartBase(w,h,xmin,xmax,ymin,ymax,logY=false,logX=false){const m={l:73,r:25,t:24,b:61};let f=(v,lo,hi,log)=>log?(Math.log10(v)-Math.log10(lo))/(Math.log10(hi)-Math.log10(lo)):(v-lo)/(hi-lo);return {w,h,m,x:v=>m.l+f(v,xmin,xmax,logX)*(w-m.l-m.r),y:v=>h-m.b-f(v,ymin,ymax,logY)*(h-m.t-m.b)}}
function axes(c,xt,yt,xlabel,ylabel){let s='';xt.forEach(v=>{s+=`<line x1="${c.x(v)}" x2="${c.x(v)}" y1="${c.m.t}" y2="${c.h-c.m.b}" stroke="#edf0f3"/><text x="${c.x(v)}" y="${c.h-c.m.b+22}" text-anchor="middle">${v}</text>`});yt.forEach(v=>{s+=`<line x1="${c.m.l}" x2="${c.w-c.m.r}" y1="${c.y(v)}" y2="${c.y(v)}" stroke="#e2e8ed"/><text x="${c.m.l-10}" y="${c.y(v)+4}" text-anchor="end">${v}</text>`});return s+`<text x="${c.w/2}" y="${c.h-10}" text-anchor="middle">${xlabel}</text><text transform="translate(17 ${c.h/2}) rotate(-90)" text-anchor="middle">${ylabel}</text>`}
function svg(c,s){return `<svg viewBox="0 0 ${c.w} ${c.h}" role="img">${s}</svg>`}
function line(c,data,color,label){const pts=data.map(d=>`${c.x(d[0])},${c.y(d[1])}`).join(' ');return `<polyline points="${pts}" fill="none" stroke="${color}" stroke-width="2.4"/>`+data.map(d=>`<circle cx="${c.x(d[0])}" cy="${c.y(d[1])}" r="3.5" fill="${color}"><title>${label}: ${d[0]} m/s, ${fmt(d[1],3)}</title></circle>`).join('')}
table('profiles',['Profile','Clean median error','Sensor median error','Sensor 95th percentile','Sensor screen','Selected'],D.profiles.map(p=>{let clean=D.summary.find(s=>s.profile===p.profile&&s.case==='TrajectoryAllClean');return [p.profile,fmt(clean.median_pct,3)+'%',fmt(p.sensor_matrix_error_median*100)+'%',fmt(p.sensor_matrix_error_p95*100)+'%',Math.round(p.sensor_accuracy_fraction*249)+'/249','No']}));
table('physical',['Quantity','Observed maximum','Declared limit'],[['Sideslip',fmt(D.maxima.max_beta_deg)+'°','5°'],['Lateral acceleration',fmt(D.maxima.max_ay)+' m/s²','6 m/s²'],['Steering angle',fmt(D.maxima.max_steering_deg)+'°','12°'],['Steering rate',fmt(D.maxima.max_rate_deg_s)+'°/s','30°/s'],['Lateral jerk',fmt(D.maxima.max_jerk)+' m/s³','Diagnostic only']]);
let strong=D.details.filter(d=>d.profile==='strong'&&d.case==='TrajectoryAllSensor');let fails=[10,12,14,16,18,20,22,24,26,28,30].map(v=>{let a=strong.filter(d=>d.speed===v);return [v+' m/s',a.length,a.filter(d=>d.matrix>10).length,fmt(quant(a.map(d=>d.matrix),.5))+'%']});table('failures',['Speed','Fits','Fail matrix','Median error'],fails);
let ac=chartBase(530,310,10,30,0,.85),as=axes(ac,[10,14,18,22,26,30],[0,.2,.4,.6,.8],'Speed [m/s]','Peak steering deviation [deg]');['gentle','standard','strong'].forEach((p,i)=>{let a=D.details.filter(d=>d.profile===p&&d.case==='TrajectoryAllSensor');as+=line(ac,[...new Set(a.map(d=>d.speed))].sort((a,b)=>a-b).map(v=>[v,a.find(d=>d.speed===v).amplitude]),['#788a99','#205cc5','#bd3a36'][i],p)});$('amplitudeChart').innerHTML=svg(ac,as)+'<div class="legend"><span style="--c:#788a99">Gentle</span><span style="--c:#205cc5">Standard</span><span style="--c:#bd3a36">Strong</span></div>';
$('legend').innerHTML=cases.map(k=>`<span style="--c:${colors[k]}">${names[k]}</span>`).join('');
let sensorRows=[];
function showDetail(index){const d=sensorRows[index],labels=['A₁₁','A₁₂','A₂₁','A₂₂','B₁','B₂'];$('detail').innerHTML=`<h3>${d.client} · fleet ${d.seed} · ${d.speed} m/s</h3><p>Matrix error ${fmt(d.matrix)}%; one-second normalized RMSE ${fmt(d.rollout,4)}. Entries have their native state/input units; absolute magnitudes are not interchangeable across entries.</p><div class="scroll"><table id="matrixDetail"></table></div>`;table('matrixDetail',['Entry','Reference','Estimated','Signed difference'],labels.map((n,i)=>[n,fmt(d.truth[i],6),fmt(d.target[i],6),fmt(d.target[i]-d.truth[i],6)]))}
function render(){let data=filtered();sensorRows=data.filter(d=>d.case==='TrajectoryAllSensor').sort((a,b)=>b.matrix-a.matrix);$('filterCount').textContent=`${sensorRows.length} client–speed cells per condition. These filters do not change the frozen selection.`;
table('conditions',['Condition','Median error','95th percentile','≤10% matrix','≤0.5 rollout','Diverged'],cases.map(k=>{let a=data.filter(d=>d.case===k);return [names[k],fmt(quant(a.map(d=>d.matrix),.5))+'%',fmt(quant(a.map(d=>d.matrix),.95))+'%',a.filter(d=>d.matrix<=10).length+'/'+a.length,a.filter(d=>d.rollout!==null&&d.rollout<=.5).length+'/'+a.length,a.filter(d=>d.diverged).length]}));
let c=chartBase(1120,380,10,30,.005,300,true),s=axes(c,[10,12,14,16,18,20,22,24,26,28,30],[.01,.1,1,10,100],'Observed speed [m/s]','Local relative matrix error [%]');s+=`<line x1="${c.m.l}" x2="${c.w-c.m.r}" y1="${c.y(10)}" y2="${c.y(10)}" stroke="#bd3a36" stroke-dasharray="6 5"/><text x="${c.w-c.m.r-5}" y="${c.y(10)-7}" text-anchor="end">Sensor matrix screen: 10%</text>`;cases.forEach(k=>{let a=data.filter(d=>d.case===k),speeds=[...new Set(a.map(d=>d.speed))].sort((a,b)=>a-b);s+=line(c,speeds.map(v=>[v,quant(a.filter(d=>d.speed===v).map(d=>d.matrix),.5)]),colors[k],names[k])});$('speedChart').innerHTML=svg(c,s);
let q=chartBase(1120,380,.005,1,1,100,true,true),z=`<rect x="${q.x(.005)}" y="${q.y(100)}" width="${q.x(.5)-q.x(.005)}" height="${q.y(10)-q.y(100)}" fill="#fff1e8"/>`+axes(q,[.01,.02,.05,.1,.2,.5,1],[1,2,5,10,20,50,100],'Independent 1 s prediction NRMSE','Matrix error [%]');z+=`<line x1="${q.m.l}" x2="${q.w-q.m.r}" y1="${q.y(10)}" y2="${q.y(10)}" stroke="#bd3a36" stroke-dasharray="5 5"/><line x1="${q.x(.5)}" x2="${q.x(.5)}" y1="${q.m.t}" y2="${q.h-q.m.b}" stroke="#bd3a36" stroke-dasharray="5 5"/>`;sensorRows.forEach(d=>{if(d.rollout!==null)z+=`<circle cx="${q.x(Math.max(.005,d.rollout))}" cy="${q.y(Math.max(1,d.matrix))}" r="4.5" opacity=".7" fill="${d.speed<=14?'#bd3a36':'#205cc5'}"><title>${d.client}, fleet ${d.seed}, ${d.speed} m/s; matrix ${fmt(d.matrix)}%; prediction ${fmt(d.rollout,4)}</title></circle>`});$('scatter').innerHTML=svg(q,z)+'<div class="legend"><span style="--c:#bd3a36">10–14 m/s</span><span style="--c:#205cc5">16–30 m/s</span></div>';
$('cells').innerHTML='<thead><tr><th>Client</th><th>Fleet</th><th>Speed</th><th>Matrix error</th><th>1 s NRMSE</th><th>Scaled condition</th></tr></thead><tbody>'+sensorRows.map((d,i)=>`<tr class="pick" data-index="${i}"><td>${d.client}</td><td>${d.seed}</td><td>${d.speed}</td><td class="${d.matrix>10?'bad':''}">${fmt(d.matrix)}%</td><td>${fmt(d.rollout,4)}</td><td>${fmt(d.condition)}</td></tr>`).join('')+'</tbody>';$('cells').querySelectorAll('tr.pick').forEach(row=>row.addEventListener('click',()=>showDetail(+row.dataset.index)));if(sensorRows.length)showDetail(0);}
['profile','family','fleet'].forEach(id=>$(id).addEventListener('change',render));$('reset').onclick=()=>{$('profile').value='strong';$('family').value='all';$('fleet').value='all';render()};$('export').onclick=()=>{let csv='profile,seed,client,speed,matrix_error_percent,rollout_nrmse,persistence_nrmse,condition\n'+sensorRows.map(d=>[d.profile,d.seed,d.client,d.speed,d.matrix,d.rollout,d.persistence,d.condition].join(',')).join('\n');let url=URL.createObjectURL(new Blob([csv],{type:'text/csv'})),a=document.createElement('a');a.href=url;a.download='filtered_sensor_diagnostics.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)};$('provenance').textContent=`Result commit ${D.commit}. Profiles: gentle, standard, strong. Calibration and validation: 44 seconds each; fitting window: 20 seconds at 100 Hz.`;render();
</script></body></html>'''


if __name__ == "__main__":
    main()
