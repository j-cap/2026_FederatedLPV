"""Reconcile the frozen point 5 outputs and save descriptive diagnostics.

Run after the execution checkpoint and saved-model audit are pushed.
No calibration, model fitting, model selection, or controller rollout is run.
The original checkpoints, summary tables, manifest, and audit remain read-only.
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code/experiments"))
import numpy as np
import pandas as pd
import paper_robustness_validation as study


def main():
    cfg, parent, _, _, provenance = study.preflight()
    results, missing = study.scan(study.OUT, cfg, provenance)
    if missing:
        raise ValueError("Analysis requires all 70 production checkpoints")
    manifest_path = study.OUT / f"{study.PREFIX}_manifest.json"
    audit_path = study.OUT / f"{study.PREFIX}_audit.json"
    manifest, audit = study.load(manifest_path), study.load(audit_path)
    if (manifest["status"] != "complete_execution"
            or manifest["provenance"] != provenance
            or manifest["fleets"] != 10 or manifest["jobs"] != 70):
        raise ValueError("Unexpected execution manifest")
    locked = manifest["checkpoint_sha256"]
    if len(locked) != 77:
        raise ValueError("Expected 70 checkpoints and seven summary tables")
    for name, sha in locked.items():
        if study.digest(study.OUT / name) != sha:
            raise ValueError(f"Frozen output changed: {name}")
    if (audit["status"] != "audited_saved_models_no_refitting"
            or audit["jobs"] != 70 or audit["provenance"] != provenance
            or audit["execution_manifest_sha256"] != study.digest(manifest_path)):
        raise ValueError("A matching saved-model audit is required")

    # Independently reduce the saved client-speed errors, including eligibility.
    grid = np.asarray(parent["speed_grid"])
    seed_rows, family_rows, compositions, calibration = [], [], [], []
    baselines = {r["seed"]: r for r in results if r["case"] == "reference"}
    cells = records = center_paired_cells = 0
    for result in results:
        inputs = result["frozen_inputs"]
        families = np.asarray([item["family"] for item in inputs])
        observed = np.asarray([np.isin(grid, item["speeds"]) for item in inputs])
        fleet_unseen = np.broadcast_to(
            ~np.isin(grid, result["global_information"]["unique_speeds"]), observed.shape)
        regions = {"full": np.ones_like(observed), "seen": observed,
                   "unseen": ~observed, "fleet_unseen": fleet_unseen}
        for method in cfg["methods"]:
            errors = np.asarray(result["errors"][method])
            for region, mask in regions.items():
                if not mask.any():
                    continue
                seed_rows.append(dict(seed=result["seed"], case=result["case"],
                    method=method, region=region, prediction_error=float(np.mean(errors[mask])),
                    eligible_pairs=int(mask.sum())))
                for family in sorted(set(families)):
                    family_mask = mask & (families == family)[:, None]
                    if family_mask.any():
                        family_rows.append(dict(seed=result["seed"], case=result["case"],
                            method=method, region=region, family=family,
                            error_pct=float(100*np.mean(errors[family_mask]))))
        base = baselines[result["seed"]]
        for item, original in zip(inputs, base["frozen_inputs"]):
            if (item["client"], item["family"], item["parameters"]) != (
                    original["client"], original["family"], original["parameters"]):
                raise ValueError("Physical fleets differ across paired conditions")
            if len(item["speeds"]) != len(original["speeds"]):
                raise ValueError("A recipient's speed count changed")
            for cell, ref_cell in zip(item["cells"], original["cells"]):
                if cell["records"] != 36:
                    raise ValueError("Calibration budget differs from 36 records/speed")
                cells += 1
                records += cell["records"]
                if result["case"] == "center_bias":
                    if cell["record_sha256"] != ref_cell["record_sha256"]:
                        raise ValueError("Center bias changed measured records")
                    center_paired_cells += 1
        assignments = np.asarray(result["models"]["assignments"])
        if len(result["groups"]) != result["selected_k"]:
            raise ValueError("Selected group count differs")
        if result["selected_bic"] != min(c["bic"] for c in result["candidates"]):
            raise ValueError("Selected candidate does not minimize saved BIC")
        for info in result["groups"]:
            counts = {family: int(((assignments == info["group"]) & (families == family)).sum())
                      for family in sorted(set(families))}
            if sum(counts.values()) != info["clients"] or info["clients"] < parent["minimum_cluster_size"]:
                raise ValueError("Learned group membership/size differs")
            compositions.append(dict(seed=result["seed"], case=result["case"],
                group=info["group"], rank=info["rank"], clients=info["clients"], families=counts))
        calibration_cells = [cell for item in inputs for cell in item["cells"]]
        calibration.append(dict(seed=result["seed"], case=result["case"],
            median_condition=float(np.median([c["regressor_condition"] for c in calibration_cells])),
            maximum_condition=max(c["regressor_condition"] for c in calibration_cells),
            mean_residual_rms=float(np.mean([c["residual_rms"] for c in calibration_cells])),
            minimum_rank=min(c["regressor_rank"] for c in calibration_cells)))

    seed_table = pd.DataFrame(seed_rows)
    saved_seed = pd.read_csv(study.OUT / f"{study.PREFIX}_seed_summary.csv")
    key = ["seed", "case", "method", "region"]
    pd.testing.assert_frame_equal(seed_table.sort_values(key).reset_index(drop=True),
        saved_seed[seed_table.columns].sort_values(key).reset_index(drop=True),
        check_exact=False, rtol=1e-12, atol=1e-14)
    summary = seed_table.groupby(["case", "method", "region"]).prediction_error.agg(["mean", "std"]).reset_index()
    saved_summary = pd.read_csv(study.OUT / f"{study.PREFIX}_summary.csv")
    pd.testing.assert_frame_equal(summary, saved_summary, check_exact=False, rtol=1e-12, atol=1e-14)

    # Replay table/bootstrap generation in a temporary directory only. This
    # verifies the frozen CI values and diagnostics without rewriting results.
    table_names = ["seed_summary", "summary", "groups", "diagnostics", "comparisons",
                   "sensitivity_deltas", "sensitivity_comparisons"]
    with tempfile.TemporaryDirectory(prefix="lpv_robustness_summary_check_") as temporary:
        directory = Path(temporary)
        for result in results:
            path = study.checkpoint_path(study.OUT, result["seed"], result["case"])
            shutil.copyfile(path, directory / path.name)
        study.summarize(directory, cfg, parent, provenance, announce=False)
        for name in table_names:
            filename = f"{study.PREFIX}_{name}.csv"
            pd.testing.assert_frame_equal(pd.read_csv(directory / filename),
                pd.read_csv(study.OUT / filename), check_exact=False, rtol=1e-12, atol=1e-14)

    diag = pd.read_csv(study.OUT / f"{study.PREFIX}_diagnostics.csv")
    family_summary = pd.DataFrame(family_rows).groupby(["case", "method", "family", "region"]).error_pct.agg(["mean", "std"]).reset_index()
    calibration_summary = pd.DataFrame(calibration).groupby("case").agg(
        mean_median_condition=("median_condition", "mean"),
        maximum_condition=("maximum_condition", "max"),
        mean_residual_rms=("mean_residual_rms", "mean"), minimum_rank=("minimum_rank", "min")).reset_index()
    group_counts = []
    for case, rows in diag.groupby("case"):
        group_counts.append(dict(case=case, selected_k_counts={str(k): int(n)
            for k, n in rows.selected_k.value_counts().sort_index().items()},
            deficient_selected_groups=int(rows.deficient_selected_groups.sum()),
            fallback_clients=int(rows.fallback_clients.sum()),
            deficient_global_fleets=int(rows.fallback_global_deficient.sum())))
    heavy = [r for r in family_summary.to_dict("records")
             if r["family"] == cfg["restricted_family"] and r["case"] in {"reference", "group_coverage"}]
    output = dict(status="frozen_results_reconciled_no_refitting", fleets=10, jobs=70,
        execution_manifest_sha256=study.digest(manifest_path), saved_model_audit_sha256=study.digest(audit_path),
        analysis_source_sha256=study.digest(Path(__file__)),
        independent_seed_summary_rows=len(seed_table), replayed_summary_tables=table_names,
        calibration_cells=cells, calibration_records=records, identical_center_bias_record_cells=center_paired_cells,
        group_counts=group_counts,
        error_summary=[{**r, "mean_pct": 100*r["mean"], "std_pct": 100*r["std"]} for r in summary.to_dict("records")],
        heavy_family_diagnostics=heavy, learned_group_compositions=compositions,
        calibration_diagnostics=calibration_summary.to_dict("records"),
        interpretation="Ten paired fleets are the statistical units. Frozen bootstrap intervals are pointwise descriptive. Family/composition diagnostics are post-fit descriptive checks; no new inferential comparisons, fitting, or controller rollouts.")
    for name, sha in locked.items():
        if study.digest(study.OUT / name) != sha:
            raise ValueError("Analysis modified a frozen execution output")
    if study.digest(audit_path) != output["saved_model_audit_sha256"]:
        raise ValueError("Analysis modified the saved-model audit")
    path = study.OUT / f"{study.PREFIX}_analysis.json"
    path.write_text(json.dumps(output, indent=2, allow_nan=False)+"\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: output[k] for k in ["status", "fleets", "jobs", "independent_seed_summary_rows",
        "calibration_cells", "calibration_records", "identical_center_bias_record_cells"]}))
    print("Saved", path.relative_to(ROOT))


if __name__ == "__main__":
    main()
