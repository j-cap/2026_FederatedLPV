"""Verify audited robustness evidence and manuscript numbers without refitting."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/tables"
PREFIX = "paper_robustness_validation"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    raw = path.read_bytes()
    if path.suffix in {".py", ".json", ".csv"}:
        raw = raw.replace(b"\r\n", b"\n")
    return hashlib.sha256(raw).hexdigest()


def rows(name):
    with (OUT / f"{PREFIX}_{name}.csv").open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def evidence():
    lock = load(ROOT / "paper/robustness_evidence_lock.json")
    for name, sha in lock["source_sha256"].items():
        if digest(ROOT / name) != sha:
            raise ValueError(f"Frozen robustness evidence changed: {name}")
    manifest = load(OUT / f"{PREFIX}_manifest.json")
    if manifest["status"] != "complete_execution" or (manifest["fleets"], manifest["jobs"]) != (10, 70):
        raise ValueError("Complete ten-fleet, seven-condition execution required")
    if len(manifest["checkpoint_sha256"]) != 77:
        raise ValueError("Expected 70 checkpoints and seven summary tables")
    for name, sha in manifest["provenance"]["source_sha256"].items():
        if digest(ROOT / name) != sha:
            raise ValueError(f"Frozen experiment source changed: {name}")
    for name, sha in manifest["checkpoint_sha256"].items():
        if digest(OUT / name) != sha:
            raise ValueError(f"Frozen execution output changed: {name}")
    audit = load(OUT / f"{PREFIX}_audit.json")
    if (audit["status"] != "audited_saved_models_no_refitting" or audit["jobs"] != 70
            or audit["provenance"] != manifest["provenance"]
            or audit["execution_manifest_sha256"] != digest(OUT / f"{PREFIX}_manifest.json")
            or audit["maximum_grouped_stationarity_residual"] > 1e-5):
        raise ValueError("Matching successful saved-model audit required")
    analysis = load(OUT / f"{PREFIX}_analysis.json")
    if (analysis["status"] != "frozen_results_reconciled_no_refitting"
            or analysis["jobs"] != 70 or analysis["fleets"] != 10
            or analysis["execution_manifest_sha256"] != digest(OUT / f"{PREFIX}_manifest.json")
            or analysis["saved_model_audit_sha256"] != digest(OUT / f"{PREFIX}_audit.json")
            or analysis["analysis_source_sha256"] != digest(ROOT / "paper/analyze_robustness_validation.py")):
        raise ValueError("Matching frozen-table reconciliation required")
    return analysis


def numerical_claims(analysis):
    means = {(r["case"], r["method"], r["region"]): float(r["mean"]) for r in rows("summary")}
    diagnostics = rows("diagnostics")
    groups = rows("groups")
    comparisons = rows("comparisons")
    for r in diagnostics:
        expected_rank = 6 if r["case"] == "fleet_coverage" else 7
        if int(r["global_rank"]) != expected_rank:
            raise ValueError("Reported fleet coverage rank differs")
        if r["case"] in {"weak_excitation", "target_noise"} and r["selected_k"] != "1":
            raise ValueError("Reported single-group selection differs")
        expected_fallback = 30 if r["case"] == "fleet_coverage" else 0
        if int(r["fallback_clients"]) != expected_fallback:
            raise ValueError("Reported fallback activation differs")
    for r in groups:
        expected_rank = 6 if r["case"] == "fleet_coverage" else 7
        if r["case"] == "group_coverage" and r["partition"] == "oracle" and r["family"] == "heavy":
            expected_rank = 3
            if int(float(r["clients"])) != 10:
                raise ValueError("Restricted oracle-family client count differs")
        if int(r["rank"]) != expected_rank:
            raise ValueError("Reported group scheduling rank differs")
    for r in analysis["learned_group_compositions"]:
        if r["case"] == "group_coverage" and r["families"]["heavy"]:
            if sum(count > 0 for count in r["families"].values()) < 2:
                raise ValueError("Reported cross-family pooling differs")
    if any(r["minimum_rank"] != 3 for r in analysis["calibration_diagnostics"]):
        raise ValueError("Reported local transition rank differs")
    for case in ["weak_excitation", "target_noise", "regressor_noise", "center_bias"]:
        if means[case, "Federated", "full"] >= means[case, "SmoothLocal", "full"]:
            raise ValueError("Reported calibration advantage over Smooth Local differs")
    for case in ["regressor_noise", "center_bias"]:
        for region in ["full", "unseen"]:
            row = next(r for r in comparisons if r["case"] == case and r["region"] == region
                       and r["method"] == "Federated" and r["baseline"] == "Global")
            if not float(row["ci_low"]) <= 0 <= float(row["ci_high"]):
                raise ValueError("Reported interval including zero differs")
    cases = dict(reference="Ref", group_coverage="Group", fleet_coverage="Fleet",
                 weak_excitation="Weak", target_noise="Noise", regressor_noise="Regressor", center_bias="Bias")
    numbers = {}
    for case, tag in cases.items():
        for method, short in [("Federated", "Fed"), ("Global", "Global"), ("SmoothLocal", "Smooth")]:
            numbers[f"Robust{tag}{short}"] = f"{100*means[case, method, 'full']:.3f}"
        view = [r for r in diagnostics if r["case"] == case]
        if len(view) != 10 or set(r["selected_k"] for r in view)-{"1", "2"}:
            raise ValueError("Unexpected selection counts in robustness table")
        numbers[f"Robust{tag}KOne"] = str(sum(r["selected_k"] == "1" for r in view))
    for method, short in [("Federated", "Fed"), ("SmoothLocal", "Smooth")]:
        numbers[f"RobustFleet{short}Outside"] = f"{100*means['fleet_coverage', method, 'fleet_unseen']:.3f}"
    for case, short in [("reference", "Ref"), ("group_coverage", "Group")]:
        value = next(r["mean"] for r in analysis["heavy_family_diagnostics"]
                     if r["case"] == case and r["method"] == "OracleFamily" and r["region"] == "full")
        numbers[f"RobustHeavy{short}Oracle"] = f"{value:.3f}"
    comparison = next(r for r in comparisons if r["case"] == "group_coverage"
                      and r["region"] == "full" and r["method"] == "Federated" and r["baseline"] == "Global")
    for field, tag in [("paired_gain_pct", "Gain"), ("ci_low", "Low"), ("ci_high", "High")]:
        numbers[f"RobustGroupGlobal{tag}"] = f"{float(comparison[field]):.2f}"
    return numbers


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--write", action="store_true", help="Regenerate only the manuscript-number include")
    args = parser.parse_args()
    numbers = numerical_claims(evidence())
    expected = "% Generated from frozen audited outputs by check_robustness_integration.py.\n" + "".join(
        f"\\newcommand{{\\{name}}}{{{value}}}\n" for name, value in sorted(numbers.items()))
    path = ROOT / "paper/robustness_numbers.tex"
    if args.write:
        path.write_text(expected, encoding="utf-8", newline="\n")
    if path.read_text(encoding="utf-8").replace("\r\n", "\n") != expected:
        raise ValueError("Robustness manuscript numbers differ from frozen evidence")
    tex = (ROOT / "paper/main.tex").read_text(encoding="utf-8")
    used = set(re.findall(r"\\(Robust[A-Za-z]+)", tex))
    if used != set(numbers) or "\\input{robustness_numbers}" not in tex:
        raise ValueError(f"Robustness macro mismatch: missing={set(numbers)-used}, unknown={used-set(numbers)}")
    if "\\label{tab:robustness}" not in tex or "\\label{sec:robustness-protocol}" not in tex:
        raise ValueError("Robustness table or protocol is missing")
    print(json.dumps(dict(status="verified", fleets=10, jobs=70, numerical_claims=len(numbers),
                          operation="no_fits_or_simulations")))


if __name__ == "__main__":
    main()
