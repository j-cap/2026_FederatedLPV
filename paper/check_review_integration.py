"""Verify audited review evidence and generate manuscript numbers. No fits/runs."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import re
from statistics import mean
import subprocess

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/tables"
AUDIT_BASE = "b3024323bad22b5b2077983d502c5f1d70004a15"


def canonical(data):
    return data.replace(b"\r\n", b"\n")


def digest(path):
    return hashlib.sha256(canonical(path.read_bytes())).hexdigest()


def rows(name):
    with (OUT / name).open(newline="") as stream:
        return list(csv.DictReader(stream))


def one(table, **match):
    result = [r for r in table if all(r[k] == str(v) for k, v in match.items())]
    if len(result) != 1:
        raise ValueError(f"Expected one row for {match}")
    return result[0]


def main():
    audit_path = OUT / "paper_review_validation_independent_audit.json"
    baseline = subprocess.check_output(["git", "show", f"{AUDIT_BASE}:{audit_path.relative_to(ROOT).as_posix()}"], cwd=ROOT)
    if canonical(audit_path.read_bytes()) != canonical(baseline):
        raise ValueError("The independently audited result has changed")
    audit = json.loads(audit_path.read_text())
    if audit["status"] != "independently_audited" or audit["fleets"] != 10:
        raise ValueError("Complete independent audit required")
    for path, sha in audit["source_sha256"].items():
        if digest(ROOT / path) != sha:
            raise ValueError(f"Audited source changed: {path}")
    summary = rows("paper_review_validation_summary.csv")
    comparisons = rows("paper_review_validation_comparisons.csv")
    units = rows("paper_review_validation_seed_summary.csv")
    groups = rows("paper_review_validation_groups.csv")
    manifest = json.loads((OUT / "paper_review_validation_manifest.json").read_text())
    for path, sha in manifest["source_sha256"].items():
        path = path.replace("\\", "/")
        local = ROOT / path
        actual = (hashlib.sha256(local.read_bytes()).hexdigest()
                  if local.suffix == ".gz" else digest(local))
        if actual != sha:
            raise ValueError(f"Frozen fitting source changed: {path}")
    macros, claims = {}, {}

    def number(name, value, places, source):
        macros[name] = f"{value:.{places}f}"
        claims[name] = {"value": value, "source": source}

    for region, suffix in [("full", "Full"), ("seen", "Seen"), ("unseen", "Unseen")]:
        row = one(summary, region=region, method="LocalRegularized")
        number(f"ReviewLocal{suffix}Error", 100 * float(row["mean"]), 3, row)
    paired = 0
    for region, suffix in [("full", "Full"), ("unseen", "Unseen")]:
        row = one(comparisons, region=region, method="FederatedLearned", baseline="LocalRegularized")
        target = {int(r["seed"]): float(r["prediction_error"]) for r in units
                  if r["region"] == region and r["method"] == "FederatedLearned"}
        reference = {int(r["seed"]): float(r["prediction_error"]) for r in units
                     if r["region"] == region and r["method"] == "LocalRegularized"}
        if set(target) != set(range(371, 381)) or set(reference) != set(target):
            raise ValueError("Ten original paired fleet units required")
        gains = [100 * (1 - target[s] / reference[s]) for s in sorted(target)]
        if not math.isclose(mean(gains), float(row["improvement_pct"]), abs_tol=1e-11):
            raise ValueError("Paired improvement changed")
        if sum(g > 0 for g in gains) != int(row["wins"]) or int(row["wins"]) != 10:
            raise ValueError("Fleet win count changed")
        paired += 10
        for column, ending in [("improvement_pct", "Gain"), ("ci_low", "Low"), ("ci_high", "High")]:
            number(f"Review{suffix}{ending}", float(row[column]), 2, row)
    delta = 100 * (float(one(summary, region="unseen", method="Local")["mean"])
                   - float(one(summary, region="unseen", method="LocalRegularized")["mean"]))
    number("ReviewUnseenLocalChange", delta, 5, "100*(Local.mean-LocalRegularized.mean), region=unseen")
    if len(groups) != audit["groups"] or any(int(r["rank"]) != 7 for r in groups):
        raise ValueError("Selected group rank changed")
    speeds = list(range(10, 31, 2))
    if any(json.loads(r["unique_speeds"]) != speeds for r in groups):
        raise ValueError("Selected group coverage changed")
    number("ReviewGroups", len(groups), 0, "paper_review_validation_groups.csv")
    number("ReviewConditionMin", min(float(r["design_condition"]) for r in groups), 2,
           "minimum stacked scheduling design condition")
    number("ReviewConditionMax", max(float(r["design_condition"]) for r in groups), 2,
           "maximum stacked scheduling design condition")
    tex = (ROOT / "paper/main.tex").read_text()
    used = set(re.findall(r"\\(Review[A-Za-z]+)", tex))
    if used != set(macros):
        raise ValueError(f"Review macro mismatch: {used.symmetric_difference(macros)}")
    text = re.sub(r"\s+", " ", tex)
    if any(p in text for p in ["has not yet been evaluated in the primary", "remain to be checked by a matched", "remaining matched checks concern"]):
        raise ValueError("Stale pending-evidence statement")
    if "matched check" not in text or "independent confirmation" not in text:
        raise ValueError("Matched-versus-confirmation distinction missing")
    (ROOT / "paper/review_evidence_numbers.tex").write_text(
        "% Generated from independently audited matched results.\n" +
        "".join(f"\\newcommand{{\\{name}}}{{{value}}}\n" for name, value in sorted(macros.items())))
    record = dict(status="verified", operation="manuscript_evidence_integration_no_fitting_or_simulations",
                  original_result_commit=audit["result_commit"], independent_audit_commit=AUDIT_BASE,
                  fleets=10, groups=len(groups), paired_fleet_checks=paired,
                  numerical_claims=claims, macros=macros,
                  source_sha256={**audit["source_sha256"], audit_path.relative_to(ROOT).as_posix(): digest(audit_path)})
    (OUT / "paper_review_manuscript_evidence.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"status": "verified", "fleets": 10, "groups": len(groups),
                      "paired_fleet_checks": paired, "manuscript_numbers": len(macros)}))


if __name__ == "__main__":
    main()
