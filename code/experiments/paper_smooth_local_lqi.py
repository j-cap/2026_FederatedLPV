"""Frozen Smooth Local LQI supplement; original 10F evidence is read-only."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "code/src"), str(ROOT / "code/experiments")]
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from federated_lpv import sample_fleet
from federated_lpv.local_scheduling import curvature_factor, fit_smooth_local
import experiment_10f_control_relevance as parent
from experiment_10a_basis_coverage_audit import raw_basis
from experiment_10e_federated_latent_groups import label_free_categories
from paper_review_validation import digest

CONFIG = ROOT / "code/config/paper_smooth_local_lqi.json"
OUT = ROOT / "results/tables"
PREFIX = "paper_smooth_local_lqi"
# Frozen 10F simulator copied verbatim except returning restoration arrays.
interpolate_schedule = parent.interpolate_schedule
rk4_step = parent.rk4_step

def simulate_with_arrays(client, schedule, scenario, cfg):
    time, speed, curvature = (
        scenario["time"],
        scenario["speed"],
        scenario["curvature"],
    )
    reference = speed * curvature
    state = np.zeros((len(time), 2))
    integral = np.zeros(len(time))
    steering = np.zeros(len(time) - 1)
    raw_steering = np.zeros(len(time) - 1)
    limit = np.deg2rad(cfg["steering_limit_deg"])
    rate_step = np.deg2rad(cfg["steering_rate_limit_deg_s"]) * cfg["sample_time"]
    grid = np.asarray(cfg["speed_grid"], dtype=float)
    if schedule is None:
        return None
    for index in range(len(time) - 1):
        gain, prefilter = interpolate_schedule(schedule, grid, speed[index])
        raw = -gain @ np.r_[state[index], integral[index]] + prefilter * reference[index]
        previous = steering[index - 1] if index else 0.0
        command = np.clip(raw, previous - rate_step, previous + rate_step)
        command = float(np.clip(command, -limit, limit))
        raw_steering[index] = raw
        steering[index] = command
        state[index + 1] = rk4_step(
            state[index], command, float(speed[index]), client.parameters, cfg["sample_time"]
        )
        error = state[index, 1] - reference[index]
        candidate_integral = np.clip(
            integral[index] + cfg["sample_time"] * error,
            -cfg["integral_limit"],
            cfg["integral_limit"],
        )
        # Conditional integration prevents windup when saturation pushes in the
        # same direction as the unconstrained command.
        if np.isclose(command, raw) or np.sign(raw - command) != np.sign(-gain[2] * error):
            integral[index + 1] = candidate_integral
        else:
            integral[index + 1] = integral[index]
        if not np.isfinite(state[index + 1]).all():
            return None
    tracking = state[:, 1] - reference
    steering_rate = np.diff(np.r_[0.0, steering]) / cfg["sample_time"]
    saturated = np.abs(raw_steering - steering) > 1e-10
    metrics = {
        "tracking_rmse_deg_s": float(np.rad2deg(np.sqrt(np.mean(tracking**2)))),
        "tracking_peak_deg_s": float(np.rad2deg(np.max(np.abs(tracking)))),
        "beta_rms_deg": float(np.rad2deg(np.sqrt(np.mean(state[:, 0] ** 2)))),
        "beta_peak_deg": float(np.rad2deg(np.max(np.abs(state[:, 0])))),
        "steering_rms_deg": float(np.rad2deg(np.sqrt(np.mean(steering**2)))),
        "steering_peak_deg": float(np.rad2deg(np.max(np.abs(steering)))),
        "steering_rate_peak_deg_s": float(np.rad2deg(np.max(np.abs(steering_rate)))),
        "saturation_fraction": float(np.mean(saturated)),
    }
    metrics["feasible"] = bool(
        metrics["beta_peak_deg"] <= cfg["beta_limit_deg"]
        and np.isfinite(list(metrics.values())).all()
    )
    return metrics, dict(time=time, speed=speed, reference=reference, state=state, integral=integral, steering=steering, raw_steering=raw_steering)


def load(path):
    return json.loads(path.read_text())


def preflight():
    cfg = load(CONFIG)
    for name, sha in cfg["source_sha256"].items():
        if digest(ROOT / name) != sha:
            raise ValueError(f"Frozen source changed: {name}")
    ctl, basis = load(parent.CONFIG), load(parent.CONFIG_10A)
    selection = load(OUT / "paper_local_smoothness_selection.json")
    if ctl["confirmation_seeds"] != cfg["seeds"] or selection["selected_strength"] != 1e6:
        raise ValueError("Frozen confirmation or smoothness selection changed")
    lock = dict(cfg["source_sha256"])
    for path in (CONFIG, Path(__file__)):
        lock[path.relative_to(ROOT).as_posix()] = digest(path)
    return cfg, ctl, basis, lock


def array_hashes(arrays):
    return {k: hashlib.sha256(np.asarray(v, dtype="<f8").tobytes()).hexdigest()
            for k, v in arrays.items()}


def check_parent(actual, expected, tolerance):
    keys = ["client", "method", "scenario"]
    a, b = actual.set_index(keys).sort_index(), expected.set_index(keys).sort_index()
    if not a.index.equals(b.index):
        raise ValueError("Parent record keys differ")
    for col in ["tracking_rmse_deg_s", "tracking_peak_deg_s", "beta_rms_deg",
                "beta_peak_deg", "steering_rms_deg", "steering_peak_deg",
                "steering_rate_peak_deg_s", "saturation_fraction", "rho"]:
        np.testing.assert_allclose(a[col], b[col], atol=tolerance, rtol=0,
                                   err_msg=f"Parent mismatch: {col}")
    for col in ["selected_k", "feasible"]:
        if not np.array_equal(a[col], b[col]):
            raise ValueError(f"Parent mismatch: {col}")


def run_seed(seed, smoke=False):
    cfg, ctl, basis, lock = preflight()
    with threadpool_limits(limits=1):
        clients, models, k = parent.build_fleet_models(seed, ctl, basis)
        # Reproduce the exact 10F RNG stream, without additional random draws.
        rng = np.random.default_rng(seed + 600000)
        categories = label_free_categories(clients, rng, ctl["coverage_blocks"])
        inputs = []
        factor = curvature_factor()
        for client in clients:
            speeds = np.asarray(ctl["coverage_blocks"][categories[client.client_id]], float)
            targets = np.asarray([parent.estimate_matrix(client, v, ctl["samples_per_local_speed"], ctl, rng)
                                  for v in speeds])
            smooth = fit_smooth_local(raw_basis(speeds, 7, basis), targets, factor, 1e6)
            models[client.client_id]["SmoothLocal"] = smooth
            inputs.append(dict(client=client.client_id, speeds=speeds.tolist(), targets=targets.tolist()))
        weights = load(parent.SELECTION)["selected"]
        used = clients[:1] if smoke else clients
        original = parent.evaluate_prebuilt(seed, weights, "confirmation", used, models, k, ctl, basis)
        expected = pd.read_csv(OUT / f"experiment_10f_confirmation_seed{seed}.csv.gz")
        expected = expected[expected.client.isin([c.client_id for c in used])]
        check_parent(original, expected, cfg["parent_tolerance"])
        rows, saved = [], []
        for client in used:
            item = models[client.client_id]
            schedule = parent.design_schedule(item["SmoothLocal"], np.asarray(ctl["speed_grid"]), weights, basis)
            rho = parent.stability_radius(schedule, item["ExactLPV"], ctl, basis)
            records = []
            for name, scenario in parent.scenarios(ctl).items():
                result = simulate_with_arrays(client, schedule, scenario, ctl)
                metrics, arrays = result if result is not None else (None, None)
                if smoke:
                    reference_metrics = parent.simulate_tracking(client, schedule, scenario, ctl)
                    if metrics != reference_metrics:
                        raise ValueError("Trajectory-returning simulator differs from frozen 10F")
                if metrics is not None and not np.isfinite(list(metrics.values())).all():
                    metrics, arrays, result = None, None, None
                status = "synthesis_failed" if schedule is None else "rollout_failed" if result is None else "complete"
                row = dict(seed=seed, client=client.client_id, method="SmoothLocal", scenario=name,
                           selected_k=k, status=status, rho=float(rho) if np.isfinite(rho) else None,
                           feasible=False if metrics is None else metrics["feasible"],
                           tracking_rmse_deg_s=None if metrics is None else metrics["tracking_rmse_deg_s"])
                if metrics is not None:
                    row.update(metrics)
                rows.append(row)
                records.append(dict(scenario=name, status=status,
                                    hashes=None if arrays is None else array_hashes(arrays)))
            saved.append(dict(client=client.client_id, model=item["SmoothLocal"].tolist(),
                              exact_model=item["ExactLPV"].tolist(),
                              schedule=None if schedule is None else [a.tolist() for a in schedule],
                              records=records))
    original_rows = json.loads(original.to_json(orient="records", double_precision=15))
    return dict(seed=seed, status="parent_verified", smoke=smoke, source_sha256=lock,
                selected_k=k, weights=weights, inputs=inputs, clients=saved,
                rows=original_rows + rows)


def save(result, directory):
    path = directory / f"{PREFIX}_seed{result['seed']}.json"
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temp.replace(path)
    print(f"Seed {result['seed']}: parent verified, checkpoint saved", flush=True)


def read_checkpoint(seed):
    result = load(OUT / f"{PREFIX}_seed{seed}.json")
    lock = preflight()[3]
    if result["source_sha256"] != lock or result["smoke"] or result["seed"] != seed or result["status"] != "parent_verified":
        raise ValueError(f"Stale or invalid checkpoint: {seed}")
    return result


def worker(seed):
    if (OUT / f"{PREFIX}_seed{seed}.json").exists():
        read_checkpoint(seed)
        print(f"Seed {seed}: resumed", flush=True)
    else:
        save(run_seed(seed), OUT)


def summarize():
    cfg = preflight()[0]
    raw = pd.DataFrame([row for s in cfg["seeds"] for row in read_checkpoint(s)["rows"]])
    units, comparisons = [], []
    for (seed, scenario, method), frame in raw.groupby(["seed", "scenario", "method"]):
        complete = len(frame) == 30 and frame.tracking_rmse_deg_s.notna().all() and np.isfinite(frame.tracking_rmse_deg_s).all()
        units.append(dict(seed=seed, scenario=scenario, method=method,
                          tracking=frame.tracking_rmse_deg_s.mean() if complete else np.nan,
                          failed_clients=int((frame.tracking_rmse_deg_s.isna() | ~np.isfinite(frame.tracking_rmse_deg_s)).sum()),
                          infeasible_clients=int((~frame.feasible).sum()), rho=frame.rho.max()))
    units = pd.DataFrame(units)
    rng = np.random.default_rng(cfg["bootstrap_seed"])
    for scenario in ("moderate", "hard"):
        p = units[units.scenario == scenario].pivot(index="seed", columns="method", values="tracking")
        for baseline in ("SmoothLocal", "Global", "Local", "ExactLPV"):
            valid = np.isfinite(p["FederatedLearned"]) & np.isfinite(p[baseline]) & (p[baseline] > 0)
            row = dict(scenario=scenario, method="FederatedLearned", baseline=baseline,
                       complete_fleets=int(valid.sum()), status="complete" if valid.all() else "incomplete_no_gain_estimate")
            # Do not silently discard failing fleets from comparison.
            if valid.all():
                gain = 100*(1-p.FederatedLearned/p[baseline])
                absolute = p[baseline]-p.FederatedLearned
                ci = np.quantile(rng.choice(gain, (20000, len(gain))).mean(axis=1), [.025,.975])
                row.update(gain_pct=float(gain.mean()), ci_low=float(ci[0]), ci_high=float(ci[1]),
                           absolute_deg_s=float(absolute.mean()), wins=int((absolute > 1e-12).sum()))
            comparisons.append(row)
    raw.to_csv(OUT / f"{PREFIX}_clients.csv", index=False)
    units.to_csv(OUT / f"{PREFIX}_seed_summary.csv", index=False)
    pd.DataFrame(comparisons).to_csv(OUT / f"{PREFIX}_comparisons.csv", index=False)
    manifest = dict(status="complete", source_sha256=preflight()[3],
                    outputs={f"{PREFIX}_{name}.csv":digest(OUT / f"{PREFIX}_{name}.csv")
                             for name in ("clients", "seed_summary", "comparisons")},
                    checkpoints={f"{PREFIX}_seed{s}.json":digest(OUT / f"{PREFIX}_seed{s}.json") for s in cfg["seeds"]})
    (OUT / f"{PREFIX}_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print("Ten-fleet summary saved; failures retained explicitly.")


def restore(seed, directory, result=None):
    result = read_checkpoint(seed) if result is None else result
    ctl = preflight()[1]
    clients = {c.client_id:c for c in sample_fleet(seed)}
    directory.mkdir(parents=True, exist_ok=True)
    for item in result["clients"]:
        schedule = None if item["schedule"] is None else tuple(np.asarray(a) for a in item["schedule"])
        for rec in item["records"]:
            if rec["hashes"] is None:
                continue
            metrics, arrays = simulate_with_arrays(clients[item["client"]], schedule, parent.scenarios(ctl)[rec["scenario"]], ctl)
            if array_hashes(arrays) != rec["hashes"]:
                raise ValueError("Restored trajectory hash mismatch")
            path = directory / f"seed{seed}_{item['client']}_{rec['scenario']}.npz"
            np.savez_compressed(path, **arrays)
            with np.load(path) as disk:
                if array_hashes(dict(disk)) != rec["hashes"]:
                    raise ValueError("Saved trajectory hash mismatch")
    print(f"Trajectories restored and hashes verified: {directory}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["preflight", "smoke", "run", "status", "summarize", "restore-data"])
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--seed", type=int, default=411)
    args = parser.parse_args()
    cfg = preflight()[0]
    if args.workers < 1 or args.seed not in cfg["seeds"]:
        parser.error("Positive workers and a confirmation seed required")
    if args.action == "preflight":
        print("Preflight passed: frozen 10F and Smooth Local inputs verified")
    elif args.action == "smoke":
        with tempfile.TemporaryDirectory() as temp:
            result = run_seed(args.seed, smoke=True)
            save(result, Path(temp))
            restore(args.seed, Path(temp) / "trajectories", result=result)
            print("Smoke passed: one client, both maneuvers, all parent methods; restoration hashes verified")
    elif args.action == "run":
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(worker, cfg["seeds"]))
        summarize()
    elif args.action == "summarize":
        summarize()
    elif args.action == "restore-data":
        restore(args.seed, ROOT / "results/data" / PREFIX)
    else:
        done = []
        for seed in cfg["seeds"]:
            if (OUT / f"{PREFIX}_seed{seed}.json").exists():
                read_checkpoint(seed)
                done.append(seed)
        print(f"Verified checkpoints: {len(done)}/10 {done}")


if __name__ == "__main__":
    main()
