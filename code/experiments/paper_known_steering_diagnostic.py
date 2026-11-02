"""Paired known-steering diagnostic on frozen continuous-calibration pilot arrays."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import html
import json
from multiprocessing import get_context
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/"code/src"),str(ROOT/"code/experiments")]
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import paper_trajectory_calibration as parent
from federated_lpv import sample_fleet
from federated_lpv.trajectory_calibration import (
    array_hash, fit_records, generate_trajectory, observations, prediction_diagnostics,
)

PREFIX = "paper_known_steering_diagnostic"
CONFIG = ROOT/f"code/config/{PREFIX}.json"
OUT = ROOT/"results/tables"
REPORT = ROOT/"report/known_steering_diagnostic.html"


def preflight():
    cfg = parent.load(CONFIG)
    old,_,_,_,provenance = parent.preflight()
    for path,sha in cfg["source_sha256"].items():
        if parent.file_hash(ROOT/path) != sha:
            raise ValueError(f"Frozen source changed: {path}")
    selection = parent.load(parent.SELECTION)
    if selection["provenance"] != provenance:
        raise ValueError("Parent selection provenance changed")
    for path,sha in selection["pilot_sha256"].items():
        if parent.file_hash(ROOT/path) != sha:
            raise ValueError(f"Parent pilot changed: {path}")
    assert cfg["seeds"] == old["pilot_seeds"] and cfg["profiles"] == list(old["profiles"])
    lock = dict(parent_provenance=provenance,parent_checkpoints=selection["pilot_sha256"],
        diagnostic_sources={CONFIG.relative_to(ROOT).as_posix():parent.file_hash(CONFIG),
            Path(__file__).relative_to(ROOT).as_posix():parent.file_hash(__file__)})
    return cfg,old,lock


def source_result(seed,profile,lock):
    return parent.read_checkpoint(parent.checkpoint_path("pilot",seed,profile),lock["parent_provenance"])


def known_observations(trace,cfg):
    """Only the recorded steering channel changes; source arrays are not mutated."""
    view = dict(observed_state=trace["observed_state"],observed_steering=trace["steering"])
    return observations(view,cfg,"sensor")


def verify_arrays(arrays,cell):
    if set(arrays) != set(cell["array_sha256"]):
        raise ValueError("Raw array set differs from frozen pilot")
    for key,sha in cell["array_sha256"].items():
        if array_hash(arrays[key]) != sha:
            raise ValueError(f"Raw array hash mismatch: {cell['client']}/{cell['speed']}/{key}")


def load_traces(source,cell):
    path = parent.raw_path("pilot",source["seed"],source["profile"],cell["client_index"],cell["speed"])
    if not path.exists():
        raise FileNotFoundError(f"Missing original arrays: {path}. Run the diagnostic restore command on the original machine.")
    with np.load(path,allow_pickle=False) as data:
        arrays = {k:data[k] for k in data.files}
    verify_arrays(arrays,cell)
    return tuple({k[len(prefix):]:v for k,v in arrays.items() if k.startswith(prefix)}
                 for prefix in ["train_","validation_"])


def score(target,diagnostics,validation,center,reference,cfg):
    result = dict(target=np.asarray(target).tolist(),**diagnostics)
    if diagnostics["fit_valid"]:
        result.update(prediction_diagnostics(target,validation,center,cfg))
    else:
        result.update(one_step_nrmse=None,rollout_nrmse=None,rollout_diverged=True,
                      rollout_state_rmse=None,persistence_nrmse=None)
    delta = np.asarray(target)-reference
    result.update(matrix_error=float(np.linalg.norm(delta)/np.linalg.norm(reference)),
        entry_signed_error=delta.tolist(),entry_absolute_error=np.abs(delta).tolist())
    prediction_pass = (result["rollout_nrmse"] is not None and
        result["rollout_nrmse"] <= cfg["pilot_sensor_rollout_nrmse"] and
        result["rollout_nrmse"] < result["persistence_nrmse"])
    result["accuracy_pass"] = bool(diagnostics["fit_valid"] and prediction_pass and
        result["matrix_error"] <= cfg["pilot_sensor_matrix_error"])
    return result


def compare_cell(training,validation,cell,reference,cfg,settings):
    sensor,old_center = observations(training,cfg,"sensor")
    known,new_center = known_observations(training,cfg)
    _,common_validation_center = known_observations(validation,cfg)
    _,old_validation_center = observations(validation,cfg,"sensor")
    frozen = cell["cases"]["TrajectoryAllSensor"]
    if array_hash(sensor) != frozen["record_sha256"]:
        raise ValueError("Original regression records do not match the frozen pilot")
    np.testing.assert_array_equal(sensor[:,[0,1,3,4]],known[:,[0,1,3,4]])
    np.testing.assert_array_equal(old_center[:2],new_center[:2])
    np.testing.assert_array_equal(old_validation_center[:2],common_validation_center[:2])
    np.testing.assert_array_equal(old_center,np.asarray(frozen["center"]))
    np.testing.assert_array_equal(old_validation_center,np.asarray(frozen["validation_center"]))
    old_target,old_diag = fit_records(sensor,old_center,cfg)
    np.testing.assert_allclose(old_target,frozen["target"],atol=settings["baseline_atol"],rtol=settings["baseline_rtol"])
    original_prediction = prediction_diagnostics(np.asarray(frozen["target"]),validation,old_validation_center,cfg)
    for key in ["one_step_nrmse","rollout_nrmse","persistence_nrmse"]:
        if original_prediction[key] is None or frozen[key] is None:
            assert original_prediction[key] is frozen[key]
        else:
            np.testing.assert_allclose(original_prediction[key],frozen[key],atol=settings["baseline_atol"],rtol=settings["baseline_rtol"])
    new_target,new_diag = fit_records(known,new_center,cfg)
    # Both predictions use exactly the same validation center and physical inputs.
    baseline = score(np.asarray(frozen["target"]),old_diag,validation,common_validation_center,reference,cfg)
    treatment = score(new_target,new_diag,validation,common_validation_center,reference,cfg)
    return dict(client=cell["client"],speed=cell["speed"],reference=reference.tolist(),
        state_records_unchanged=True,physical_arrays_verified=True,
        baseline_refit_max_abs=float(np.max(np.abs(old_target-np.asarray(frozen["target"])))),
        source_record_sha256=frozen["record_sha256"],known_record_sha256=array_hash(known),
        frozen_baseline_rollout_nrmse=frozen["rollout_nrmse"],
        old_center=old_center.tolist(),known_center=new_center.tolist(),
        common_validation_center=common_validation_center.tolist(),
        physical_pass=bool(cell["training"]["physical_pass"] and cell["validation"]["physical_pass"]),
        settling_pass=bool(cell["training"]["settling_pass"] and cell["validation"]["settling_pass"]),
        OriginalSensor=baseline,KnownSteering=treatment,
        matrix_change_pp=100*(treatment["matrix_error"]-baseline["matrix_error"]))


def path_for(seed,profile,directory=OUT):
    return directory/f"{PREFIX}_seed{seed}_{profile}.json"


def read_result(path,lock):
    envelope=parent.load(path);r=envelope["payload"]
    if envelope["payload_sha256"] != hashlib.sha256(parent.canonical(r)).hexdigest():
        raise ValueError(f"Corrupt diagnostic: {path}")
    if r["provenance"] != lock or r["status"] != "complete" or len(r["cells"]) != 83:
        raise ValueError(f"Stale or incomplete diagnostic: {path}")
    if Path(path).name != path_for(r["seed"],r["profile"]).name:
        raise ValueError("Checkpoint filename/identity mismatch")
    return r


def run_job(seed,profile):
    settings,cfg,lock = preflight()
    source=source_result(seed,profile,lock)
    rows=[]
    with threadpool_limits(limits=1):
        for cell in source["cells"]:
            training,validation=load_traces(source,cell)
            reference=np.asarray(source["clients"][cell["client_index"]]["reference"])[list(range(10,31,2)).index(cell["speed"])]
            rows.append(compare_cell(training,validation,cell,reference,cfg,settings))
    result=dict(status="complete",seed=seed,profile=profile,provenance=lock,cells=rows)
    parent.save_json(path_for(seed,profile),dict(payload_sha256=hashlib.sha256(parent.canonical(result)).hexdigest(),payload=result))
    return dict(seed=seed,profile=profile,cells=len(rows),status="checkpoint_saved")


def smoke_worker():
    settings,cfg,_=preflight()
    client=sample_fleet(cfg["smoke_seed"])[0]
    # Synthetic technical fixture only; no completed pilot case is fitted here.
    training=generate_trajectory(client.parameters,20.,.3,87520,cfg)
    validation=generate_trajectory(client.parameters,20.,.3,87520,cfg,validation=True)
    records,center=observations(training,cfg,"sensor")
    _,valcenter=observations(validation,cfg,"sensor")
    target,diag=fit_records(records,center,cfg)
    frozen=dict(target=target.tolist(),center=center.tolist(),validation_center=valcenter.tolist(),
        record_sha256=array_hash(records),**prediction_diagnostics(target,validation,valcenter,cfg))
    cell=dict(client="smoke",speed=20.,cases={"TrajectoryAllSensor":frozen},
        training=dict(physical_pass=True,settling_pass=True),validation=dict(physical_pass=True,settling_pass=True))
    result=compare_cell(training,validation,cell,np.array([.9,0.,0.,.9,.01,.1]),cfg,settings)
    assert result["state_records_unchanged"] and result["KnownSteering"]["fit_valid"]
    return dict(status="smoke_passed",saved_scientific_results=False)


def analyze():
    settings,cfg,lock=preflight()
    paths=[path_for(s,p) for s in settings["seeds"] for p in settings["profiles"]]
    if not all(p.exists() and parent.tracked_unchanged(p) for p in paths):
        raise ValueError("Complete, commit and push all nine diagnostic checkpoints before analysis")
    results=[read_result(p,lock) for p in paths]
    rows=[]
    for r in results:
        for cell in r["cells"]:
            for method in ["OriginalSensor","KnownSteering"]:
                values=cell[method]
                rows.append(dict(seed=r["seed"],profile=r["profile"],client=cell["client"],speed=cell["speed"],method=method,
                    matrix_error=values["matrix_error"],rollout_nrmse=values["rollout_nrmse"],
                    accuracy_pass=values["accuracy_pass"],fit_valid=values["fit_valid"],
                    physical_pass=cell["physical_pass"],settling_pass=cell["settling_pass"],
                    matrix_change_pp=cell["matrix_change_pp"],
                    **{f"mae_{n}":x for n,x in zip(["A11","A12","A21","A22","B1","B2"],values["entry_absolute_error"])}))
    frame=pd.DataFrame(rows)
    frame.to_csv(OUT/f"{PREFIX}_cells.csv",index=False)
    summary=frame.groupby(["profile","method"],sort=False).agg(n=("matrix_error","size"),
        median_pct=("matrix_error",lambda x:100*x.median()),p95_pct=("matrix_error",lambda x:100*x.quantile(.95)),
        passing=("accuracy_pass","sum"),rollout_p95=("rollout_nrmse",lambda x:x.quantile(.95))).reset_index()
    speed=frame.groupby(["profile","method","speed"]).agg(n=("matrix_error","size"),
        median_pct=("matrix_error",lambda x:100*x.median()),p95_pct=("matrix_error",lambda x:100*x.quantile(.95)),
        passing=("accuracy_pass","sum")).reset_index()
    fleet=frame.groupby(["profile","method","seed"]).matrix_error.mean().reset_index()
    summary.to_csv(OUT/f"{PREFIX}_summary.csv",index=False)
    speed.to_csv(OUT/f"{PREFIX}_by_speed.csv",index=False)
    fleet.to_csv(OUT/f"{PREFIX}_fleet_means.csv",index=False)
    passing=[]
    for profile in settings["profiles"]:
        v=frame[(frame.profile==profile)&(frame.method=="KnownSteering")]
        if v.accuracy_pass.mean()>=cfg["pilot_fraction"] and v.physical_pass.all() and v.settling_pass.all():
            passing.append(profile)
    audit=dict(status="analysis_complete",provenance=lock,
        checkpoint_sha256={p.relative_to(ROOT).as_posix():parent.file_hash(p) for p in paths},
        profiles_meeting_original_thresholds=passing,original_selection_unchanged=True,
        decision="Review post-pilot evidence and explicitly freeze a revised protocol before confirmation.",
        note="Three reused development fleets; descriptive paired diagnostic, no independent confirmation or sharing result.")
    parent.save_json(OUT/f"{PREFIX}_analysis.json",audit)
    write_report(summary,speed,fleet,frame,passing)
    print(summary.to_string(index=False))
    print(json.dumps(dict(profiles_meeting_thresholds=passing,report=str(REPORT))))


def write_report(summary,speed,fleet,frame,passing):
    def table(data):
        return data.to_html(index=False,float_format=lambda x:f"{x:.4f}",border=0)
    charts=[]
    for profile in ["gentle","standard","strong"]:
        parts=[]
        for tick in [.1,1,10,100]:
            y=240-(np.log10(tick)+1)*65
            parts.append(f'<line x1="55" x2="710" y1="{y}" y2="{y}" stroke="#ddd"/><text x="8" y="{y+4}">{tick}%</text>')
        for method,color in [("OriginalSensor","#bd3a36"),("KnownSteering","#205cc5")]:
            v=speed[(speed.profile==profile)&(speed.method==method)].sort_values("speed")
            points=" ".join(f'{55+(r.speed-10)*32.75},{240-(np.log10(max(.1,r.median_pct))+1)*65}' for r in v.itertuples())
            parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3"/>')
        for v in range(10,31,2):
            parts.append(f'<text x="{55+(v-10)*32.75}" y="265" text-anchor="middle">{v}</text>')
        charts.append(f'<h3>{profile.title()}: median local matrix error</h3><svg viewBox="0 0 740 290">'+"".join(parts)+'</svg>')
    changes=frame[frame.method=="KnownSteering"].groupby("profile").matrix_change_pp.agg(["mean","median","min","max"]).reset_index()
    REPORT.write_text('''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Known steering diagnostic</title><style>body{font:16px/1.55 system-ui;background:#f3f6f8;color:#192f40;margin:0}main{max-width:1100px;margin:auto;padding:32px}section{background:white;padding:24px;margin:18px 0;border-radius:10px}h1{line-height:1.2}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:8px;border-bottom:1px solid #ddd;text-align:right}th:first-child,td:first-child{text-align:left}.scroll{overflow:auto}svg{width:100%;max-width:740px}svg text{font:13px system-ui}code{background:#eef2f6}small{color:#506779}</style><main><h1>Does known steering resolve the local calibration problem?</h1><p>Paired diagnostic on the frozen pilot: same physical motion, same state noise, same OLS fit. Only recorded steering and its quiet-segment offset change.</p><section><h2>Outcome</h2><p>Profiles meeting the original local accuracy thresholds: <strong>'''+html.escape(", ".join(passing) or "none")+'''</strong>.</p><p>This is post-pilot development evidence. It does not automatically change the frozen selection or authorize the old confirmation runner.</p><div class="scroll">'''+table(summary)+'''</div></section><section><h2>Paired change in matrix error</h2><p>Known steering minus original sensor error, in percentage points. Negative is an improvement. Cell summaries are descriptive; only three independent fleets were reused.</p>'''+table(changes)+'''</section><section><h2>Speed dependence</h2><p><span style="color:#bd3a36">Red: original noisy steering</span> · <span style="color:#205cc5">Blue: known applied steering</span>. Speed [m/s] on the horizontal axis; logarithmic error axis.</p>'''+"".join(charts)+'''</section><section><h2>Detailed speed comparison</h2><div class="scroll">'''+table(speed)+'''</div></section><section><h2>Fleet-level mean errors</h2><p>Dimensionless relative errors, one value per fleet and condition. No confidence intervals are claimed from three reused pilot fleets.</p>'''+table(fleet)+'''</section><section><h2>What was controlled?</h2><ul><li>All source trajectory arrays verified against original hashes.</li><li>Original sensor fits reproduced within the declared numerical tolerance.</li><li>Current/next noisy state records and their quiet-segment centers unchanged exactly.</li><li>Known input is the applied steering after amplitude and rate limiting, not the unconstrained request.</li><li>Both models evaluated with the same validation center, known steering and exact initial state. The original validation score is also preserved in each checkpoint.</li><li>No new physical trajectories, alternative estimator, clustering, or controller experiment.</li></ul><p>Steering can only be treated as known here because the simulator applies that limited command directly. A model of actuator uncertainty would require a separate assumption.</p><p>Source: <code>results/tables/paper_known_steering_diagnostic_seed*.json</code>. Rebuild with <code>python code/experiments/paper_known_steering_diagnostic.py analyze</code>.</p></section></main></html>''',encoding="utf-8",newline="\n")


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command",choices=["preflight","smoke","restore","run","status","analyze"])
    ap.add_argument("--workers",type=int,default=5)
    args=ap.parse_args()
    if args.workers<1: ap.error("workers must be positive")
    settings,cfg,lock=preflight()
    jobs=[(s,p) for s in settings["seeds"] for p in settings["profiles"]]
    if args.command in {"preflight","status"}:
        present=total=complete=0
        for s,p in jobs:
            source=source_result(s,p,lock)
            for cell in source["cells"]:
                total+=1
                present+=parent.raw_path("pilot",s,p,cell["client_index"],cell["speed"]).exists()
            path=path_for(s,p)
            if path.exists(): read_result(path,lock);complete+=1
        print(json.dumps(dict(status="preflight_passed",jobs=len(jobs),complete=complete,
            original_raw_files_present=present,original_raw_files_required=total,
            action="run" if present==total else "restore original arrays before run"),indent=2))
    elif args.command=="smoke":
        with ProcessPoolExecutor(max_workers=1,mp_context=get_context("spawn")) as pool:
            print(json.dumps(pool.submit(smoke_worker).result()))
    elif args.command=="analyze": analyze()
    else:
        pending=[]
        for s,p in jobs:
            path=path_for(s,p)
            if args.command=="restore":
                source=source_result(s,p,lock)
                if any(not parent.raw_path("pilot",s,p,c["client_index"],c["speed"]).exists() for c in source["cells"]):
                    pending.append(parent.checkpoint_path("pilot",s,p))
            elif path.exists(): read_result(path,lock)
            else: pending.append((s,p))
        if args.command=="run":
            for s,p in pending:
                source=source_result(s,p,lock)
                if any(not parent.raw_path("pilot",s,p,c["client_index"],c["speed"]).exists() for c in source["cells"]):
                    raise FileNotFoundError("Missing original arrays. Run restore on the original machine; exact hashes are required.")
        print(json.dumps(dict(command=args.command,pending=len(pending),workers=args.workers)),flush=True)
        with ProcessPoolExecutor(max_workers=args.workers,mp_context=get_context("spawn")) as pool:
            futures=[pool.submit(parent.restore_job,p) for p in pending] if args.command=="restore" else [pool.submit(run_job,*j) for j in pending]
            for f in as_completed(futures): print(json.dumps(f.result()),flush=True)
        if args.command=="run": print("Commit and push diagnostic checkpoints, then run analyze.")


if __name__=="__main__": main()
