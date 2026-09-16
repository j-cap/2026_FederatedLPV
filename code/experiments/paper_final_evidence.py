"""Build one canonical, provenance-rich evidence table for the IV paper."""
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "tables"

def main():
    k = pd.read_csv(OUT / "experiment_9k_comparisons.csv")
    m = pd.read_csv(OUT / "experiment_9m_statistics.csv")
    l = pd.read_csv(OUT / "experiment_9l_comparisons.csv")
    comm = pd.read_csv(OUT / "paper_communication_audit.csv")
    rows = []
    for r in k[k.method.isin(["Local", "Central", "Fed100", "Fed50", "Fed20"])].itertuples():
        rows.append(dict(study="9K", budget_s=r.budget, method=r.method,
            tracking_rad_s=r.tracking, feasible_rate=r.feasible_rate,
            comparison="Local", effect_pct=-r.vs_local_pct,
            effect_definition="reduction; ratio of aggregate means", evidence_role="blind confirmation"))
    for r in m.itertuples():
        rows.append(dict(study="9M", budget_s=r.budget, method=r.method,
            tracking_rad_s=None, feasible_rate=None, comparison=r.reference,
            effect_pct=r.mean_improvement_pct,
            effect_definition="mean paired fleet-wise reduction",
            ci95_low_pct=r.ci95_low_pct, ci95_high_pct=r.ci95_high_pct,
            wins=r.wins, fleets=r.fleets, evidence_role="retrospective mechanism ablation"))
    for r in l[(l.budget == .75) & l.method.isin(["Uniform20", "Persistent20", "Dropout50", "FamilySkew20", "OperatingSkew20"])].itertuples():
        rows.append(dict(study="9L", budget_s=r.budget, method=r.method,
            tracking_rad_s=r.tracking, feasible_rate=r.feasible_rate,
            comparison="Local", effect_pct=-r.vs_local_pct,
            effect_definition="reduction; ratio of aggregate means", evidence_role="development coverage stress test"))
    pd.DataFrame(rows).to_csv(OUT / "paper_final_results.csv", index=False)
    primary = k[(k.budget == .75) & (k.method == "Fed20")].iloc[0]
    paired = m[(m.budget == .75) & (m.label == "cold_start_superiority")].iloc[0]
    manifest = {
        "primary_claim": {
            "method": "Fed20Rank1", "budget_s": .75,
            "tracking_reduction_vs_local_aggregate_ratio_pct": float(-primary.vs_local_pct),
            "paired_fleet_mean_improvement_pct": float(paired.mean_improvement_pct),
            "paired_fleet_bootstrap_ci95_pct": [float(paired.ci95_low_pct), float(paired.ci95_high_pct)],
            "wins": int(paired.wins), "fleets": int(paired.fleets), "feasible_rate": float(primary.feasible_rate)
        },
        "communication_confirmation": comm[comm.phase.eq("confirmation")].to_dict("records"),
        "provenance": {
            "tracking": "experiment_9k_comparisons.csv",
            "paired_statistics": "experiment_9m_statistics.csv",
            "coverage": "experiment_9l_comparisons.csv",
            "communication": "paper_communication_audit.csv"
        }
    }
    (OUT / "paper_final_evidence.json").write_text(json.dumps(manifest, indent=2) + "\n")

if __name__ == "__main__":
    main()
