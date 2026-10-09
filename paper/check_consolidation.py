"""Verify and extract existing evidence; never fit models or run simulations."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/tables"
BASE = "aef68883d4caa4c97d06cfa244b13ba1671cb9a5"


def read(name):
    with (OUT / f"experiment_{name}.csv").open(newline="") as stream:
        return list(csv.DictReader(stream))


def one(rows, **match):
    found = [row for row in rows if all(row[key] == str(value) for key, value in match.items())]
    if len(found) != 1:
        raise ValueError(f"Expected one evidence row: {match}, found {len(found)}")
    return found[0]


def close(actual, expected):
    if not math.isclose(float(actual), float(expected), rel_tol=1e-11, abs_tol=1e-12):
        raise ValueError(f"Evidence mismatch: {actual} != {expected}")


def paired_check(rows, comparison, metric, **context):
    target = {
        row["seed"]: float(row[metric])
        for row in rows
        if row["method"] == comparison["method"]
        and all(row[key] == value for key, value in context.items())
    }
    reference = {
        row["seed"]: float(row[metric])
        for row in rows
        if row["method"] == comparison["baseline"]
        and all(row[key] == value for key, value in context.items())
    }
    if target.keys() != reference.keys() or len(target) != 10:
        raise ValueError("Expected ten paired fleet units")
    close(mean(100 * (1 - target[seed] / reference[seed]) for seed in target),
          comparison["improvement_pct"])
    if "absolute_improvement_deg_s" in comparison:
        close(mean(reference[seed] - target[seed] for seed in target),
              comparison["absolute_improvement_deg_s"])
    return len(target)


def main():
    source_names = [
        "10a_selection", "10a_summary", "10a_coverage_summary", "10b_summary",
        "10b_comparisons", "10c_comparisons", "10e_summary", "10e_comparisons",
        "10e_seed_summary", "10e_diagnostics", "10f_summary", "10f_comparisons",
        "10f_seed_summary",
    ]
    tables = {name: read(name) for name in source_names}
    macros, claims = {}, {}

    def number(macro, row, column, places=2, multiplier=1):
        value = float(row[column]) * multiplier
        macros[macro] = f"{value:.{places}f}"
        claims[macro] = {"value": value, "column": column, "source_row": row}

    for order, macro, places in [(3, "BasisLowError", 4), (7, "BasisHighError", 4)]:
        number(macro, one(tables["10a_summary"], curvature="0.005", order=order),
               "error_mean", places, 100)
    if one(tables["10a_selection"], curvature="0.0")["selected_order"] != "3":
        raise ValueError("Straight-line negative control changed")
    if one(tables["10a_selection"], curvature="0.005")["selected_order"] != "7":
        raise ValueError("Selected richer basis changed")
    union = one(tables["10a_coverage_summary"], curvature="0.005",
                scope="aggregate_envelope", order=7)
    if union["rank_min"] != "7" or union["rank_max"] != "7":
        raise ValueError("Aggregate scheduling rank changed")
    for block, rank in enumerate([3, 3, 3, 2]):
        row = one(tables["10a_coverage_summary"], curvature="0.005",
                  scope=f"local_block_{block}", order=7)
        if int(row["rank_min"]) != rank or int(row["rank_max"]) != rank:
            raise ValueError("Local scheduling rank changed")
    number("UnionCondition", union, "condition_max", 3)
    for method, macro in [("FamilyPool", "OracleUnseenError"),
                          ("Local", "LocalUnseenError"), ("Global", "GlobalUnseenError")]:
        number(macro, one(tables["10b_summary"], region="unseen", method=method),
               "prediction_error", 3, 100)
    for stage in range(4):
        row = one(tables["10c_comparisons"], stage=stage, method="FamilyPersonalized",
                  baseline="FamilyPool", metric="recovery")
        if float(row["improvement_pct"]) >= 0:
            raise ValueError("Personalization recovery boundary changed")

    pairs = 0
    for method, macro in [("Local", "IdentLocalError"), ("Global", "IdentGlobalError"),
                          ("CentralLearned", "IdentCentralError"),
                          ("FederatedLearned", "IdentFedError")]:
        number(macro, one(tables["10e_summary"], protocol="label_free", method=method),
               "prediction_error", 3, 100)
    for baseline, macro in [("Local", "IdentLocalGain"), ("Global", "IdentGlobalGain")]:
        row = one(tables["10e_comparisons"], protocol="label_free", method="FederatedLearned",
                  baseline=baseline, metric="prediction_error")
        pairs += paired_check(tables["10e_seed_summary"], row, "prediction_error",
                              protocol="label_free")
        number(macro, row, "improvement_pct")
        if baseline == "Global":
            number("IdentGlobalLow", row, "ci_low")
            number("IdentGlobalHigh", row, "ci_high")
    diagnostics = tables["10e_diagnostics"]
    if len(diagnostics) != 20 or any(row["same_partition"] != "True" for row in diagnostics):
        raise ValueError("Federated equivalence jobs changed")
    label = [row for row in diagnostics if row["protocol"] == "label_free"]
    if sorted(int(row["selected_k"]) for row in label) != [1, 1] + [2] * 8:
        raise ValueError("Label-free group-count distribution changed")
    macros["SearchTraffic"] = f"{mean(float(row['total_megabytes']) for row in label):.3f}"
    macros["CoefficientError"] = f"{max(float(row['coefficient_error']) for row in diagnostics) / 1e-14:.2f}"
    macros["BicError"] = f"{max(float(row['candidate_bic_error']) for row in diagnostics) / 1e-9:.2f}"

    for scenario, suffix in [("moderate", "Moderate"), ("hard", "Hard")]:
        for method, short in [("Local", "Local"), ("Global", "Global"),
                              ("FederatedLearned", "Fed"), ("ExactLPV", "Exact")]:
            number(f"Track{short}{suffix}", one(tables["10f_summary"], scenario=scenario,
                                                method=method), "tracking", 4)
        for baseline in ["Local", "Global"]:
            row = one(tables["10f_comparisons"], scenario=scenario, method="FederatedLearned",
                      baseline=baseline, metric="tracking")
            pairs += paired_check(tables["10f_seed_summary"], row, "tracking", scenario=scenario)
            number(f"Control{baseline}{suffix}", row, "improvement_pct")
            if baseline == "Global":
                number(f"ControlGlobal{suffix}Low", row, "ci_low")
                number(f"ControlGlobal{suffix}High", row, "ci_high")
                number(f"ControlAbsolute{suffix}", row, "absolute_improvement_deg_s", 6)
    fed = [row for row in tables["10f_seed_summary"] if row["method"] == "FederatedLearned"]
    if len(fed) != 20 or any(row["feasible"] != "True" for row in fed):
        raise ValueError("Federated control feasibility changed")
    macros["ControlRadius"] = f"{max(float(row['rho']) for row in fed):.4f}"

    sources = [f"results/tables/experiment_{name}.csv" for name in source_names]
    for prefix in ["10a", "10e", "10f"]:
        path = OUT / f"experiment_{prefix}_conclusions.json"
        conclusion = json.loads(path.read_text())
        sources.append(str(path.relative_to(ROOT)))
        for name, expected in conclusion["provenance"].items():
            actual = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError(f"Historical provenance mismatch: {name}")
            sources.append(name)
    sources += [
        "code/experiments/experiment_10b_oracle_complementary_gate.py",
        "code/experiments/experiment_10d_learned_compatible_groups.py",
        "code/config/experiment_10b.json", "code/config/experiment_10c.json",
        "code/config/experiment_10d.json",
        "results/figures/experiment_10e_federated_latent_groups.pdf",
    ]
    for name in set(sources):
        original = subprocess.check_output(["git", "show", f"{BASE}:{name}"], cwd=ROOT)
        if (ROOT / name).read_bytes() != original:
            raise ValueError(f"Evidence differs from consolidation baseline: {name}")
    archive = ["main.tex", "ALGORITHM_AND_TRUST_MODEL.md", "METHOD_EVIDENCE_AUDIT.md",
               "NOVELTY_AND_OVERLAP.md", "UNITS_AND_METRICS_AUDIT.md"]
    for name in archive:
        original = subprocess.check_output(["git", "show", f"{BASE}:paper/{name}"], cwd=ROOT)
        if (ROOT / "paper/archive/9m_cold_start" / name).read_bytes() != original:
            raise ValueError(f"Archived draft is not byte-identical: {name}")

    tex = (ROOT / "paper/main.tex").read_text()
    import re
    used = set(re.findall(r"\\([A-Z][A-Za-z]+)", tex)) & set(macros)
    if used != set(macros):
        raise ValueError(f"Unused evidence macros: {set(macros) - used}")
    (ROOT / "paper/evidence_numbers.tex").write_text(
        "% Generated from frozen evidence by paper/check_consolidation.py.\n"
        + "".join(f"\\newcommand{{\\{name}}}{{{value}}}\n" for name, value in sorted(macros.items()))
    )
    manifest = {
        "decision_date": "2026-10-09", "evidence_baseline_commit": BASE,
        "operation": "existing_evidence_consolidation_no_fitting_or_simulations",
        "primary_claim": "full-envelope LPV identification from incomplete local coverage",
        "state_assumption": "available beta and yaw rate for identification and feedback",
        "benchmark_boundary": "controlled equilibrium-centered perturbations; exact current regressors; noisy next states",
        "primary_metric": "whole-speed-envelope relative matrix error, not unseen-speed-only error",
        "not_validated": ["ordinary sequential state-record-only identification",
                          "noisy state regressors", "frozen transfer to wholly held-out recipients",
                          "formal privacy", "scheduled stability", "calibration-time saving percentage"],
        "paired_fleet_contrast_checks": pairs,
        "archived_files_byte_identical": len(archive), "all_checks_pass": True,
        "evidence_macros": macros, "extracted_claims": claims,
        "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                          for name in sorted(set(sources))},
    }
    (OUT / "consolidation_identification_evidence.json").write_text(
        json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": "verified", "paired_fleet_contrast_checks": pairs,
                      "evidence_sources": len(set(sources)), "archived_files": len(archive),
                      "manuscript_numbers": len(macros)}))


if __name__ == "__main__":
    main()
