"""Staged continuous-calibration validation. See docs/paper_trajectory_calibration_plan.md."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from multiprocessing import get_context
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "code/src"), str(ROOT / "code/experiments")]
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from federated_lpv import VehicleParameters, sample_fleet
from federated_lpv.trajectory_calibration import (
    array_hash, fit_records, generate_trajectory, observations, physical_diagnostics,
    prediction_diagnostics, window_indices,
)
from experiment_10a_basis_coverage_audit import collect_targets, equilibrium
from experiment_10b_oracle_complementary_gate import predict, rk4_step
from experiment_10e_federated_latent_groups import label_free_categories
from paper_robustness_validation import client_model, fit_observed

PREFIX = "paper_trajectory_calibration"
CONFIG = ROOT / f"code/config/{PREFIX}.json"
LOCK = ROOT / f"code/config/{PREFIX}_sources.json"
TABLES = ROOT / "results/tables"
DATA = ROOT / "results/data" / PREFIX
FIGURES = ROOT / "results/figures"
SELECTION = TABLES / f"{PREFIX}_selection.json"
METHODS = ("RestrictedLocal", "SmoothLocal", "Global", "Federated")
CASES = ("Reset36", "Trajectory36Bridge", "TrajectoryAllBridge", "TrajectoryAllClean", "TrajectoryAllSensor")


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def file_hash(path):
    data = Path(path).read_bytes()
    if Path(path).suffix in {".json", ".py", ".yml"}:
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8", newline="\n")
    temporary.replace(path)


def preflight():
    cfg, lock = load(CONFIG), load(LOCK)
    for path, expected in lock.items():
        if file_hash(ROOT / path) != expected:
            raise ValueError(f"Frozen dependency changed: {path}")
    versions = {key: importlib.metadata.version(key) for key in cfg["package_versions"]}
    if versions != cfg["package_versions"] or f"{sys.version_info.major}.{sys.version_info.minor}" != cfg["python_minor"]:
        raise ValueError("Activate the repository's federated_lpv environment (environment.yml)")
    if len(set(cfg["pilot_seeds"]+cfg["confirmation_seeds"]+[cfg["smoke_seed"]])) != 14:
        raise ValueError("Pilot, confirmation, and smoke seeds must be distinct")
    parent, basis = load(ROOT / "code/config/experiment_10e.json"), load(ROOT / "code/config/experiment_10a.json")
    smooth = load(TABLES / "paper_local_smoothness_selection.json")
    if smooth["selected_strength"] != 1e6 or smooth["status"] != "development_frozen_before_primary":
        raise ValueError("Expected frozen Smooth Local selection")
    if cfg["sample_time"] != parent["sample_time"] or cfg["curvature_per_m"] != parent["curvature_per_m"]:
        raise ValueError("Sample time/curvature must match the original reference problem")
    provenance = dict(source_sha256={**lock, CONFIG.relative_to(ROOT).as_posix():file_hash(CONFIG),
        LOCK.relative_to(ROOT).as_posix():file_hash(LOCK),
        Path(__file__).relative_to(ROOT).as_posix():file_hash(__file__),
        "code/src/federated_lpv/trajectory_calibration.py":file_hash(ROOT / "code/src/federated_lpv/trajectory_calibration.py")},
        environment=dict(python_minor=cfg["python_minor"], packages=versions))
    return cfg, parent, basis, smooth["selected_strength"], provenance


def checkpoint_path(stage, seed, profile, directory=None):
    directory = TABLES if directory is None else directory
    return directory / f"{PREFIX}_{stage}_seed{seed}_{profile}.json"


def read_checkpoint(path, provenance):
    envelope = load(path)
    value = envelope["payload"]
    if hashlib.sha256(canonical(value)).hexdigest() != envelope["payload_sha256"]:
        raise ValueError(f"Corrupt checkpoint: {path}")
    if value["provenance"] != provenance or value["status"] != "execution_complete":
        raise ValueError(f"Stale checkpoint: {path}")
    if Path(path).name != checkpoint_path(value["stage"],value["seed"],value["profile"]).name:
        raise ValueError(f"Checkpoint identity does not match its filename: {path}")
    if len(value["clients"]) != value["client_count"] or len(value["cells"]) != sum(len(c["speeds"]) for c in value["clients"]):
        raise ValueError(f"Incomplete checkpoint: {path}")
    return value


def save_checkpoint(result, directory=None):
    path = checkpoint_path(result["stage"], result["seed"], result["profile"], directory)
    save_json(path, dict(payload_sha256=hashlib.sha256(canonical(result)).hexdigest(), payload=result))
    return path


def acquisition_seed(seed, index, speed):
    return int(np.random.SeedSequence([20261029, seed, index, round(speed*10)]).generate_state(1)[0])


def raw_path(stage, seed, profile, index, speed, directory=None):
    directory = DATA if directory is None else directory
    return directory / f"{stage}_{seed}_{profile}_{index}_{int(speed)}.npz"


def pack_traces(training, validation):
    return {**{f"train_{k}":v for k,v in training.items()}, **{f"validation_{k}":v for k,v in validation.items()}}


def write_raw(path, arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def reset_records(client, speed, seed, parent):
    center = equilibrium(client.parameters, speed, parent["curvature_per_m"])
    rng = np.random.default_rng(np.random.SeedSequence([seed, 77]))
    n = parent["samples_per_local_speed"]
    dx = rng.normal(size=(n,2))*parent["state_excitation_std"]
    du = rng.normal(size=n)*parent["steering_excitation_std"]
    noise = rng.normal(size=(n,2))*parent["next_state_noise_std"]
    states, inputs = dx+center[:2], du+center[2]
    next_states = np.asarray([rk4_step(client.parameters,x,u,speed,parent["sample_time"])
                              for x,u in zip(states,inputs)])
    return np.column_stack([states, inputs, next_states+noise]), center


def fit_cell(client, speed, seed, amplitude, cfg, parent):
    training = generate_trajectory(client.parameters,speed,amplitude,seed,cfg)
    validation = generate_trajectory(client.parameters,speed,amplitude,seed,cfg,validation=True)
    bridge, clean_center = observations(training,cfg,"bridge")
    clean, _ = observations(training,cfg,"clean")
    sensor, sensor_center = observations(training,cfg,"sensor")
    _, val_clean_center = observations(validation,cfg,"clean")
    _, val_sensor_center = observations(validation,cfg,"sensor")
    reset, true_center = reset_records(client,speed,seed,parent)
    subset = np.linspace(0,len(bridge)-1,parent["samples_per_local_speed"],dtype=int)
    cases = dict(Reset36=(reset,true_center,true_center),
        Trajectory36Bridge=(bridge[subset],clean_center,val_clean_center),
        TrajectoryAllBridge=(bridge,clean_center,val_clean_center),
        TrajectoryAllClean=(clean,clean_center,val_clean_center),
        TrajectoryAllSensor=(sensor,sensor_center,val_sensor_center))
    fitted = {}
    for case,(records,center,val_center) in cases.items():
        target, diagnostics = fit_records(records,center,cfg)
        if diagnostics["fit_valid"]:
            diagnostics.update(prediction_diagnostics(target,validation,val_center,cfg))
        else:
            diagnostics.update(one_step_nrmse=None,rollout_diverged=True,
                rollout_nrmse=None,rollout_state_rmse=None,persistence_nrmse=None)
        fitted[case] = dict(target=target.tolist(),center=center.tolist(),
            validation_center=val_center.tolist(), record_sha256=array_hash(records), **diagnostics)
    arrays = pack_traces(training,validation)
    # Reference-center access is recorded only as a diagnostic; trajectory fits above
    # received centers estimated from the quiet trace, not true_center.
    result = dict(speed=float(speed), acquisition_seed=seed,
        training=physical_diagnostics(training,speed,cfg),
        validation=physical_diagnostics(validation,speed,cfg),
        true_center=true_center.tolist(), clean_center_error=(clean_center-true_center).tolist(),
        sensor_center_error=(sensor_center-true_center).tolist(), cases=fitted,
        subset_indices=subset.tolist(), array_sha256={k:array_hash(v) for k,v in arrays.items()})
    return result, arrays


def run_job(stage, seed, profile, selection_hash=None, directory=TABLES, raw_directory=DATA, smoke=False):
    started = time.monotonic()
    cfg,parent,basis,smooth,provenance = preflight()
    clients = sample_fleet(seed,clients_per_family=2 if smoke else 10)
    tags = label_free_categories(clients,np.random.default_rng(seed+cfg["coverage_rng_offset"]),parent["coverage_blocks"])
    speeds = [np.asarray(parent["coverage_blocks"][tags[c.client_id]],dtype=float) for c in clients]
    if smoke:
        parent = {**parent,"candidate_clusters":[1,2],"mixture_restarts":2,"minimum_cluster_size":2}
    cells, frozen_clients = [], []
    targets = {case:[] for case in CASES}
    with threadpool_limits(limits=1):
        for i,(client,observed) in enumerate(zip(clients,speeds)):
            fitted = {case:[] for case in CASES}
            # References evaluated separately, never supplied to fitting functions.
            reference = collect_targets(client,cfg["curvature_per_m"],basis)[1]
            for speed in observed:
                cell,arrays = fit_cell(client,speed,acquisition_seed(seed,i,speed),cfg["profiles"][profile],cfg,parent)
                truth = np.asarray(reference[list(parent["speed_grid"]).index(speed)])
                for case, result in cell["cases"].items():
                    estimate = np.asarray(result["target"])
                    result["matrix_error"] = float(np.linalg.norm(estimate-truth)/np.linalg.norm(truth))
                    result["entry_absolute_error"] = np.abs(estimate-truth).tolist()
                    result["b_relative_error"] = float(np.linalg.norm(estimate[4:]-truth[4:])/np.linalg.norm(truth[4:]))
                    fitted[case].append(estimate)
                cell["client_index"],cell["client"] = i,client.client_id
                cells.append(cell)
                write_raw(raw_path(stage,seed,profile,i,speed,raw_directory),arrays)
            for case in CASES:
                targets[case].append(np.asarray(fitted[case]))
            frozen_clients.append(dict(client=client.client_id,family=client.family,
                parameters=asdict(client.parameters),speeds=observed.tolist(),reference=np.asarray(reference).tolist()))
        models, errors, blocked = {}, {}, {}
        if stage == "confirmation" or smoke:
            for case in CASES:
                invalid = [dict(client=c["client"],speed=c["speed"]) for c in cells if not c["cases"][case]["fit_valid"]]
                if invalid:
                    blocked[case] = dict(reason="invalid_local_fit",cells=invalid)
                    continue
                model = fit_observed(speeds,targets[case],seed,parent,basis,smooth)
                models[case] = model
                errors[case] = {}
                for method in METHODS:
                    values = []
                    for i,client in enumerate(frozen_clients):
                        truth = np.asarray(client["reference"])
                        predicted = predict(client_model(model["models"],method,i),parent["speed_grid"],basis)
                        err = np.linalg.norm(predicted-truth,axis=1)/np.linalg.norm(truth,axis=1)
                        if not np.isfinite(err).all():
                            raise FloatingPointError(f"Nonfinite LPV predictions: {case}/{method}")
                        values.append(err.tolist())
                    errors[case][method] = values
    result = dict(status="execution_complete",stage=stage,seed=seed,profile=profile,
        selection_sha256=selection_hash,smoke=smoke,provenance=provenance,
        client_count=len(clients),clients=frozen_clients,cells=cells,models=models,errors=errors,
        blocked_conditions=blocked,elapsed_s=time.monotonic()-started)
    save_checkpoint(result,directory)
    return dict(stage=stage,seed=seed,profile=profile,elapsed_s=round(result["elapsed_s"],1),
                blocked=list(blocked),status="checkpoint_saved")


def pilot_paths(cfg):
    return [checkpoint_path("pilot",s,p) for s in cfg["pilot_seeds"] for p in cfg["profiles"]]


def tracked_unchanged(path):
    relative = Path(path).relative_to(ROOT).as_posix()
    tracked = subprocess.run(["git","ls-files","--error-unmatch",relative],cwd=ROOT,capture_output=True)
    changed = subprocess.run(["git","diff","HEAD","--",relative],cwd=ROOT,capture_output=True)
    return tracked.returncode == 0 and changed.returncode == 0 and not changed.stdout


def selected_profile(cfg,provenance,require_committed=True):
    selection = load(SELECTION)
    if selection["provenance"] != provenance:
        raise ValueError("Pilot selection uses changed sources/configuration")
    for path, sha in selection["pilot_sha256"].items():
        if file_hash(ROOT/path) != sha:
            raise ValueError("Pilot changed after selection")
    if selection["selected_profile"] is None:
        raise ValueError("No pilot profile passed. Diagnose the pilot before confirmation.")
    if require_committed and not tracked_unchanged(SELECTION):
        raise ValueError("Commit and push the frozen pilot selection before confirmation")
    return selection["selected_profile"],file_hash(SELECTION)


def local_rows(results):
    rows = []
    for result in results:
        for cell in result["cells"]:
            for case,values in cell["cases"].items():
                row = dict(seed=result["seed"],profile=result["profile"],client=cell["client"],
                    speed=cell["speed"],case=case,
                    physical_pass=cell["training"]["physical_pass"] and cell["validation"]["physical_pass"],
                    settling_pass=cell["training"]["settling_pass"] and cell["validation"]["settling_pass"],
                    max_beta_deg=max(cell["training"]["max_beta_deg"],cell["validation"]["max_beta_deg"]),
                    max_ay=max(cell["training"]["max_ay"],cell["validation"]["max_ay"]),
                    max_jerk=max(cell["training"]["max_jerk"],cell["validation"]["max_jerk"]),
                    **{k:v for k,v in values.items() if isinstance(v,(int,float,bool)) or v is None})
                row.update({f"mae_{name}":e for name,e in zip(["A11","A12","A21","A22","B1","B2"],values["entry_absolute_error"])})
                rows.append(row)
    return pd.DataFrame(rows)


def pilot_decision(rows,cfg):
    """Select by local quality only; fleet sharing errors are not inputs."""
    summary = []
    for profile,amplitude in cfg["profiles"].items():
        view = rows[rows.profile == profile]
        clean = view[view.case == "TrajectoryAllClean"]
        sensor = view[view.case == "TrajectoryAllSensor"]
        info = (clean["rank"] == 3) & (clean.scaled_condition <= cfg["max_scaled_condition"])
        clean_ok = clean.fit_valid & (clean.matrix_error <= cfg["pilot_clean_matrix_error"]) & (clean.rollout_nrmse <= cfg["pilot_clean_rollout_nrmse"]) & (clean.rollout_nrmse < clean.persistence_nrmse)
        sensor_ok = sensor.fit_valid & (sensor.matrix_error <= cfg["pilot_sensor_matrix_error"]) & (sensor.rollout_nrmse <= cfg["pilot_sensor_rollout_nrmse"]) & (sensor.rollout_nrmse < sensor.persistence_nrmse)
        item = dict(profile=profile,amplitude_deg=amplitude,cells=len(clean),
            physical_fraction=float(clean.physical_pass.mean()),settling_fraction=float(clean.settling_pass.mean()),
            information_fraction=float(info.mean()),clean_accuracy_fraction=float(clean_ok.mean()),
            sensor_accuracy_fraction=float(sensor_ok.mean()),
            sensor_matrix_error_median=float(sensor.matrix_error.median()),
            sensor_matrix_error_p95=float(sensor.matrix_error.quantile(.95)),
            sensor_rollout_nrmse_p95=float(sensor.rollout_nrmse.quantile(.95)) if sensor.rollout_nrmse.notna().any() else None)
        item["passed"] = bool(item["physical_fraction"] == 1. and item["settling_fraction"] == 1.
            and min(item["information_fraction"],item["clean_accuracy_fraction"],item["sensor_accuracy_fraction"]) >= cfg["pilot_fraction"])
        summary.append(item)
    passed = sorted([s for s in summary if s["passed"]],key=lambda s:s["amplitude_deg"])
    return summary, passed[0]["profile"] if passed else None


def select_pilot():
    cfg,_,_,_,provenance = preflight()
    paths = pilot_paths(cfg)
    if not all(path.exists() for path in paths):
        raise ValueError("All nine pilot jobs must complete before selection")
    if not all(tracked_unchanged(p) for p in paths):
        raise ValueError("Commit and push pilot checkpoints before selection analysis")
    results = [read_checkpoint(p,provenance) for p in paths]
    rows = local_rows(results)
    summary,selected = pilot_decision(rows,cfg)
    selection = dict(status="pilot_selection_complete",selected_profile=selected,
        provenance=provenance,pilot_sha256={p.relative_to(ROOT).as_posix():file_hash(p) for p in paths},profiles=summary)
    if SELECTION.exists() and load(SELECTION) != selection:
        raise ValueError("Refusing to overwrite a different frozen selection; version the protocol")
    save_json(SELECTION,selection)
    rows.to_csv(TABLES/f"{PREFIX}_pilot_cells.csv",index=False)
    pd.DataFrame(summary).to_csv(TABLES/f"{PREFIX}_pilot_profiles.csv",index=False)
    calibration_figures(results,rows,"pilot",cfg)
    print(json.dumps(selection["profiles"],indent=2))
    print(json.dumps(dict(selected_profile=selection["selected_profile"],next="commit selection then run" if selected else "stop and diagnose pilot")))


def calibration_figures(results,rows,stage,cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIGURES.mkdir(parents=True,exist_ok=True)
    fig,axes = plt.subplots(1,2,figsize=(9,3.6),layout="constrained")
    cases = ["TrajectoryAllClean","TrajectoryAllSensor"]
    for case in cases:
        view = rows[rows.case == case]
        agg = view.groupby("profile",sort=False).agg(matrix=("matrix_error","median"),rollout=("rollout_nrmse","median"))
        axes[0].plot(agg.index,100*agg.matrix,"o-",label=case.replace("TrajectoryAll",""))
        axes[1].plot(agg.index,agg.rollout,"o-",label=case.replace("TrajectoryAll",""))
    axes[0].set_ylabel("Median local matrix error [%]")
    axes[1].set_ylabel("Median validation 1 s NRMSE")
    for ax in axes:
        ax.grid(alpha=.25);ax.legend()
    fig.savefig(FIGURES/f"{PREFIX}_{stage}_quality.pdf");plt.close(fig)
    # Predetermined first fleet and first cell in each coverage block, not best cases.
    first = sorted(results,key=lambda r:(r["seed"],r["profile"]))[0]
    if SELECTION.exists() and load(SELECTION).get("selected_profile"):
        first = next((r for r in results if r["seed"] == first["seed"] and r["profile"] == load(SELECTION)["selected_profile"]),first)
    chosen = []
    for block in ([10,12,14],[16,18,20],[22,24,26],[28,30]):
        chosen.append(next(c for c in first["cells"] if c["speed"] in block))
    fig,axes = plt.subplots(5,4,figsize=(13,11),layout="constrained")
    for j,cell in enumerate(chosen):
        path = raw_path(stage,first["seed"],first["profile"],cell["client_index"],cell["speed"])
        if not path.exists():
            plt.close(fig)
            raise FileNotFoundError(f"Restore raw arrays before plotting: {path}")
        with np.load(path) as data:
            for key,sha in cell["array_sha256"].items():
                if array_hash(data[key]) != sha:
                    raise ValueError(f"Raw array mismatch: {path}/{key}")
            t = data["train_time"]
            axes[0,j].plot(t[:-1],np.rad2deg(data["train_steering"]))
            axes[1,j].plot(t,np.rad2deg(data["train_state"][:,0]))
            axes[2,j].plot(t,np.rad2deg(data["train_state"][:,1]))
            axes[3,j].plot(t[:-1],data["train_ay"])
            axes[4,j].plot(data["train_position"][:,0],data["train_position"][:,1])
            axes[4,j].set_aspect("equal",adjustable="datalim")
        axes[0,j].set_title(f"{cell['client']}, {cell['speed']:g} m/s")
        for ax in axes[:4,j]:
            ax.axvline(cfg["settle_s"],color="gray",ls=":")
            ax.axvline(cfg["settle_s"]+cfg["hold_s"],color="gray",ls="--")
            ax.grid(alpha=.2)
        axes[3,j].set_xlabel("Time [s]");axes[4,j].set_xlabel("X [m]")
    for ax,label in zip(axes[:,0],["Steering [deg]","Sideslip [deg]","Yaw rate [deg/s]","Lateral acceleration [m/s²]","Y [m]"]):
        ax.set_ylabel(label)
    fig.suptitle(f"{stage}: fixed examples, {first['profile']} profile, seed {first['seed']}")
    fig.savefig(FIGURES/f"{PREFIX}_{stage}_trajectories.pdf");plt.close(fig)


def analyze_confirmation():
    cfg,parent,_,_,provenance = preflight()
    profile,selection_hash = selected_profile(cfg,provenance)
    paths = [checkpoint_path("confirmation",s,profile) for s in cfg["confirmation_seeds"]]
    if not all(p.exists() and tracked_unchanged(p) for p in paths):
        raise ValueError("Complete, commit and push all confirmation checkpoints before analysis")
    results = [read_checkpoint(p,provenance) for p in paths]
    if any(r["selection_sha256"] != selection_hash for r in results):
        raise ValueError("Confirmation used a different pilot selection")
    local = local_rows(results)
    local.to_csv(TABLES/f"{PREFIX}_confirmation_cells.csv",index=False)
    units,groups,failures = [],[],[]
    grid = np.asarray(parent["speed_grid"])
    for r in results:
        observed = np.asarray([np.isin(grid,c["speeds"]) for c in r["clients"]])
        for case,errors in r["errors"].items():
            for method,values in errors.items():
                values = np.asarray(values)
                for region,mask in [("full",np.ones_like(observed)),("observed",observed),("unobserved",~observed)]:
                    units.append(dict(seed=r["seed"],case=case,method=method,region=region,error=float(values[mask].mean())))
            model = r["models"][case]
            groups.append(dict(seed=r["seed"],case=case,selected_k=model["selected_k"],
                deficient_groups=sum(not g["full_rank"] for g in model["groups"])))
        failures.extend(dict(seed=r["seed"],case=c,**details) for c,details in r["blocked_conditions"].items())
    frame = pd.DataFrame(units,columns=["seed","case","method","region","error"])
    frame.to_csv(TABLES/f"{PREFIX}_seed_summary.csv",index=False)
    frame.groupby(["case","method","region"]).error.agg(["mean","std","count"]).reset_index().to_csv(TABLES/f"{PREFIX}_summary.csv",index=False)
    pd.DataFrame(groups).to_csv(TABLES/f"{PREFIX}_groups.csv",index=False)
    comparisons = []
    rng = np.random.default_rng(cfg["bootstrap_seed"])
    for case,region in [(c,r) for c in CASES for r in ["full","observed","unobserved"]]:
        view = frame[(frame.case == case) & (frame.region == region)]
        pivot = view.pivot(index="seed",columns="method",values="error").reindex(cfg["confirmation_seeds"])
        for baseline in ["Global","SmoothLocal"]:
            if "Federated" not in pivot or baseline not in pivot or pivot.isna().any().any():
                comparisons.append(dict(case=case,region=region,baseline=baseline,status="blocked_incomplete_fleets"))
                continue
            gain = 100*(1-pivot.Federated/pivot[baseline])
            boot = rng.choice(gain.to_numpy(),size=(cfg["bootstrap_resamples"],len(gain))).mean(axis=1)
            lo,hi = np.quantile(boot,[.025,.975])
            comparisons.append(dict(case=case,region=region,baseline=baseline,status="complete",
                mean_paired_gain=float(gain.mean()),ci_low=float(lo),ci_high=float(hi),fleets=len(gain)))
    pd.DataFrame(comparisons).to_csv(TABLES/f"{PREFIX}_paired_gains.csv",index=False)
    sensor = local[local.case == "TrajectoryAllSensor"]
    data_quality = dict(physical_fraction=float(sensor.physical_pass.mean()),
        settling_fraction=float(sensor.settling_pass.mean()),
        local_accuracy_fraction=float((sensor.fit_valid &
            (sensor.matrix_error <= cfg["pilot_sensor_matrix_error"]) &
            (sensor.rollout_nrmse <= cfg["pilot_sensor_rollout_nrmse"]) &
            (sensor.rollout_nrmse < sensor.persistence_nrmse)).mean()))
    main = [c for c in comparisons if c["case"] == "TrajectoryAllSensor" and c["region"] == "full"]
    supported = len(main) == 2 and all(c["status"] == "complete" and c["ci_low"] > 0 for c in main)
    adequate = (data_quality["physical_fraction"] == data_quality["settling_fraction"] == 1.
                and data_quality["local_accuracy_fraction"] >= cfg["pilot_fraction"])
    decision = "review_for_integration" if adequate and supported else (
        "diagnose_calibration" if not adequate else "sharing_benefit_not_confirmed_against_both_baselines")
    save_json(TABLES/f"{PREFIX}_analysis.json",dict(status="analysis_complete",provenance=provenance,
        selection_sha256=selection_hash,checkpoint_sha256={p.relative_to(ROOT).as_posix():file_hash(p) for p in paths},
        blocked_conditions=failures,physical_failures=int((~local[local.case == "TrajectoryAllSensor"].physical_pass).sum()),
        settling_failures=int((~local[local.case == "TrajectoryAllSensor"].settling_pass).sum()),
        data_quality=data_quality,decision=decision,comparisons=comparisons))
    calibration_figures(results,local,"confirmation",cfg)
    comparison_figure(frame,comparisons)
    print(pd.DataFrame(comparisons).to_string(index=False))
    print(json.dumps(dict(data_quality=data_quality,decision=decision),indent=2))


def comparison_figure(frame,comparisons):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(1,2,figsize=(11,4),layout="constrained")
    labels = ["Reset\n36","Trajectory\n36 bridge","Trajectory\nall bridge","Trajectory\nall clean","Trajectory\nall sensor"]
    for method in METHODS:
        view = frame[(frame.method == method) & (frame.region == "full")]
        means = view.groupby("case").error.mean().reindex(CASES)
        axes[0].plot(range(len(CASES)),100*means,"o-",label=method)
    axes[0].set(xticks=range(len(CASES)),xticklabels=labels,ylabel="Mean full-envelope matrix error [%]")
    axes[0].set_yscale("log");axes[0].legend(fontsize=8)
    for offset,baseline in [(-.1,"Global"),(.1,"SmoothLocal")]:
        for i,case in enumerate(CASES):
            item = next(c for c in comparisons if c["case"] == case and c["region"] == "full" and c["baseline"] == baseline)
            if item["status"] != "complete":
                continue
            color = "tab:blue" if baseline == "Global" else "tab:orange"
            axes[1].plot([i+offset]*2,[item["ci_low"],item["ci_high"]],color=color)
            axes[1].plot(i+offset,item["mean_paired_gain"],"o",color=color)
    for color,baseline in [("tab:blue","Global"),("tab:orange","SmoothLocal")]:
        axes[1].plot([],[],"o-",color=color,label=f"vs {baseline}")
    axes[1].axhline(0,color="gray",lw=.8)
    axes[1].set(xticks=range(len(CASES)),xticklabels=labels,ylabel="Paired Federated gain [%], 95% CI")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=.2)
    fig.savefig(FIGURES/f"{PREFIX}_confirmation_comparison.pdf");plt.close(fig)


def restore_job(path):
    cfg,_,_,_,provenance = preflight()
    result = read_checkpoint(path,provenance)
    for cell in result["cells"]:
        client = result["clients"][cell["client_index"]]
        parameters = VehicleParameters(**client["parameters"])
        arrays = pack_traces(*[generate_trajectory(parameters,cell["speed"],cfg["profiles"][result["profile"]],
            cell["acquisition_seed"],cfg,validation=validation) for validation in [False,True]])
        for key,sha in cell["array_sha256"].items():
            if array_hash(arrays[key]) != sha:
                raise ValueError(f"Exact replay failed: {path.name}/{cell['client']}/{key}; keep original arrays")
        destination = raw_path(result["stage"],result["seed"],result["profile"],cell["client_index"],cell["speed"])
        if destination.exists():
            with np.load(destination) as saved:
                if any(array_hash(saved[k]) != sha for k,sha in cell["array_sha256"].items()):
                    raise ValueError(f"Existing raw file differs; refusing overwrite: {destination}")
        else:
            write_raw(destination,arrays)
    return dict(checkpoint=path.name,status="raw_arrays_verified_or_restored")


def execute(stage,workers):
    cfg,_,_,_,provenance = preflight()
    profile,selection_hash = (None,None) if stage == "pilot" else selected_profile(cfg,provenance)
    jobs = [(s,p) for s in cfg["pilot_seeds"] for p in cfg["profiles"]] if stage == "pilot" else [(s,profile) for s in cfg["confirmation_seeds"]]
    pending = []
    for seed,name in jobs:
        path = checkpoint_path(stage,seed,name)
        if path.exists():
            saved = read_checkpoint(path,provenance)
            if saved["stage"] != stage or saved["seed"] != seed or saved["profile"] != name or saved["selection_sha256"] != selection_hash:
                raise ValueError(f"Checkpoint identity changed: {path}")
        else:
            pending.append((seed,name))
    print(json.dumps(dict(stage=stage,complete=len(jobs)-len(pending),pending=len(pending),workers=workers)),flush=True)
    with ProcessPoolExecutor(max_workers=workers,mp_context=get_context("spawn")) as pool:
        futures = [pool.submit(run_job,stage,s,p,selection_hash) for s,p in pending]
        for future in as_completed(futures):
            print(json.dumps(future.result()),flush=True)
    print("Execution complete. Commit and push checkpoints before select/analyze.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=["preflight","smoke","pilot","select","run","analyze","status","restore"])
    parser.add_argument("--workers",type=int,default=5)
    parser.add_argument("--stage",choices=["pilot","confirmation"],default="confirmation")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    cfg,_,_,_,provenance = preflight()
    if args.command == "preflight":
        print(json.dumps(dict(status="preflight_passed",pilot_jobs=9,confirmation_jobs=10,conditions=list(CASES),
            scope=cfg["scope"],source_files=len(provenance["source_sha256"])),indent=2))
    elif args.command == "smoke":
        with tempfile.TemporaryDirectory(prefix=PREFIX) as tmp:
            with ProcessPoolExecutor(max_workers=1,mp_context=get_context("spawn")) as pool:
                future = pool.submit(run_job,"pilot",cfg["smoke_seed"],"standard",
                    directory=Path(tmp),raw_directory=Path(tmp)/"raw",smoke=True)
                print(json.dumps(future.result()))
            path = checkpoint_path("pilot",cfg["smoke_seed"],"standard",Path(tmp))
            result = read_checkpoint(path,provenance)
            assert len(result["models"])+len(result["blocked_conditions"]) == len(CASES)
            assert result["smoke"] and len(result["clients"]) == 6
            print("Smoke passed: technical execution only; no pilot selection or scientific result saved.")
    elif args.command in {"pilot","run"}:
        execute("pilot" if args.command == "pilot" else "confirmation",args.workers)
    elif args.command == "select":
        select_pilot()
    elif args.command == "analyze":
        analyze_confirmation()
    elif args.command == "status":
        for stage,seeds in [("pilot",cfg["pilot_seeds"]),("confirmation",cfg["confirmation_seeds"])]:
            paths = list(TABLES.glob(f"{PREFIX}_{stage}_seed*.json"))
            for path in paths:
                read_checkpoint(path,provenance)
            print(json.dumps(dict(stage=stage,completed=len(paths),expected=len(seeds)*(len(cfg["profiles"]) if stage == "pilot" else 1))))
        if SELECTION.exists():
            print(json.dumps(dict(selected_profile=load(SELECTION)["selected_profile"])))
    else:
        paths = list(TABLES.glob(f"{PREFIX}_{args.stage}_seed*.json"))
        if not paths:
            raise ValueError("No completed checkpoints to restore")
        with ProcessPoolExecutor(max_workers=args.workers,mp_context=get_context("spawn")) as pool:
            for result in pool.map(restore_job,paths):
                print(json.dumps(result),flush=True)


if __name__ == "__main__":
    main()
