"""Plot the matched Smooth Local supplement; historical figures stay intact."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import prepare_identification_figures as figures
from local_baseline_validation import OUT, ROOT, PREFIX, load, digest, verify_hashes


def generate():
    manifest = load(OUT / f"{PREFIX}_manifest.json")
    audit = load(OUT / f"{PREFIX}_audit.json")
    if audit["status"] != "saved_models_audited_without_refitting" or audit["manifest_sha256"] != digest(OUT / f"{PREFIX}_manifest.json"):
        raise ValueError("Audited execution required")
    verify_hashes(audit["source_sha256"])
    verify_hashes(manifest["output_sha256"])
    raw = pd.read_csv(OUT / f"{PREFIX}_errors.csv")
    cfg = load(ROOT / "code/config/experiment_10e.json")
    blocks = {}
    for seed in cfg["seeds"]:
        parent = load(OUT / f"paper_review_validation_seed{seed}.json")
        blocks.update({(seed, item["client"]): cfg["coverage_blocks"].index(item["speeds"]) for item in parent["frozen_inputs"]})
    raw["block"] = [blocks[(s, c)] for s, c in zip(raw.seed, raw.client)]
    figures.COLORS["SmoothLocal"] = "#8b4c96"
    figures.LABELS["SmoothLocal"] = "Smooth Local"
    figures.STYLES["SmoothLocal"] = ":"
    data = {"raw": raw, "cfg": cfg}
    figures.set_style()
    statistics = figures.coverage_statistics(data)
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 3.65), sharex=True, sharey=True, layout="constrained")
    for block, ax in enumerate(axes.flat):
        figures.coverage_panel(ax, data, statistics, block, f"({chr(97+block)}) ",
                               xlabel=block >= 2, ylabel=block % 2 == 0)
    axes[0, 0].legend(frameon=False, fontsize=7, loc="upper left")
    files = figures.save(fig, "paper_identification_coverage_v3", ROOT / "results/figures")
    record = dict(status="generated_from_audited_matched_models", operation="plotting_only",
         generator_sha256=digest(Path(__file__)), execution_manifest_sha256=digest(OUT / f"{PREFIX}_manifest.json"),
         audit_sha256=digest(OUT / f"{PREFIX}_audit.json"),
         output_sha256={f.relative_to(ROOT).as_posix(): digest(f) for f in files if f.suffix == ".pdf"})
    (OUT / f"{PREFIX}_figure_manifest.json").write_text(json.dumps(record, indent=2)+"\n")
    print("Exported coverage v3 PDF/PNG from verified matched records")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--generate", action="store_true", required=True)
    parser.parse_args()
    generate()
