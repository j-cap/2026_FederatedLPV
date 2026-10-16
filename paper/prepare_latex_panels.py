"""Export Figures 2--4 as separate, untitled panels for LaTeX subfigures.

Plotting only: existing combined figures and historical manifests stay intact.
Run --preflight first, then --generate. --output-dir supports preview exports.
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import prepare_identification_figures as figures
from local_baseline_validation import PREFIX, OUT, ROOT, load, digest, verify_hashes


def checked_data():
    data = figures.load_and_check()
    manifest_path = OUT / f"{PREFIX}_manifest.json"
    audit_path = OUT / f"{PREFIX}_audit.json"
    manifest, audit = load(manifest_path), load(audit_path)
    if (audit["status"] != "saved_models_audited_without_refitting"
            or audit["manifest_sha256"] != digest(manifest_path)):
        raise ValueError("Audited Smooth Local execution required")
    verify_hashes(audit["source_sha256"])
    verify_hashes(manifest["output_sha256"])
    raw = pd.read_csv(OUT / f"{PREFIX}_errors.csv")
    blocks = {}
    for seed in data["cfg"]["seeds"]:
        parent = load(OUT / f"paper_review_validation_seed{seed}.json")
        blocks.update({(seed, item["client"]): data["cfg"]["coverage_blocks"].index(item["speeds"])
                       for item in parent["frozen_inputs"]})
    raw["block"] = [blocks[(s, c)] for s, c in zip(raw.seed, raw.client)]
    data["raw"] = raw
    sources = dict(data["lock"]["source_sha256"])
    sources.update(manifest["output_sha256"])
    sources.update(audit["source_sha256"])
    for path in [manifest_path, audit_path, figures.LOCK]:
        sources[path.relative_to(ROOT).as_posix()] = digest(path)
    return data, sources


def coverage_inset(ax, data, statistics, block):
    inset = ax.inset_axes([.52, .53, .46, .44])
    observed = data["cfg"]["coverage_blocks"][block]
    inset.axvspan(min(observed)-.35, max(observed)+.35,
                  color="#dbe9d4", alpha=.65, linewidth=0)
    for method in ["Global", "FederatedLearned", "SmoothLocal"]:
        mean, low, high = statistics[(block, method)]
        inset.plot(data["cfg"]["speed_grid"], mean,
                   color=figures.COLORS[method], linestyle=figures.STYLES[method])
        inset.fill_between(data["cfg"]["speed_grid"], low, high,
                           color=figures.COLORS[method], alpha=.13, linewidth=0)
    inset.set(xlim=(10, 30), ylim=(0, 10), xticks=[10, 20, 30], yticks=[0, 5, 10])
    inset.tick_params(labelsize=6, pad=1)
    inset.grid(axis="y", alpha=.2)
    return inset


def generate(data, output):
    output.mkdir(parents=True, exist_ok=True)
    figures.set_style()
    files = []
    for name, draw in [
        ("paper_basis_validation_panel", figures.basis_panel),
        ("paper_scheduling_information_panel", figures.information_panel),
        ("paper_paired_fleet_panel", figures.fleet_panel),
        ("paper_client_tradeoff_panel", figures.client_panel),
    ]:
        fig, ax = plt.subplots(figsize=(3.48, 2.65), layout="constrained")
        draw(ax, data)
        ax.set_title("")
        files.extend(figures.save(fig, name, output))
    figures.COLORS["SmoothLocal"] = "#8b4c96"
    figures.LABELS["SmoothLocal"] = "Smooth Local"
    figures.STYLES["SmoothLocal"] = ":"
    statistics = figures.coverage_statistics(data)
    for block in range(4):
        fig, ax = plt.subplots(figsize=(3.48, 2.25), layout="constrained")
        figures.coverage_panel(ax, data, statistics, block)
        ax.set_title("")
        coverage_inset(ax, data, statistics, block)
        if block == 0:
            ax.legend(frameon=False, fontsize=6.5, loc="upper left")
        speeds = "_".join(map(str, data["cfg"]["coverage_blocks"][block]))
        files.extend(figures.save(fig, f"paper_coverage_{speeds}_panel", output))
    return files


def main():
    parser = argparse.ArgumentParser(__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--generate", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/figures")
    args = parser.parse_args()
    data, sources = checked_data()
    if args.preflight:
        print(json.dumps(dict(status="preflight_passed", panels=8, checks=data["checks"]), indent=2))
        return
    output = args.output_dir.resolve()
    files = generate(data, output)
    default_output = (ROOT / "results/figures").resolve()
    manifest_path = (OUT if output == default_output else output) / "paper_latex_panels_manifest.json"
    record = dict(status="generated_from_verified_frozen_evidence",
                  operation="plotting_only_no_fitting_or_simulation",
                  generator_sha256=digest(Path(__file__)),
                  rendering_helper_sha256=digest(Path(figures.__file__)),
                  source_sha256=sources, checks=data["checks"],
                  conventions=dict(panel_titles="none; labels and subcaptions belong to LaTeX",
                                   coverage_insets="linear 0--10 percent; Global, Grouped, Smooth Local",
                                   coverage_bands="unchanged pointwise fleet bootstrap; 20000 draws"),
                  output_sha256={figures.display_path(p): digest(p) for p in files if p.suffix == ".pdf"})
    manifest_path.write_text(json.dumps(record, indent=2)+"\n", encoding="utf-8")
    print(f"Exported 8 PDF panels and 8 PNG previews to {output}; manifest: {manifest_path}")


if __name__ == "__main__":
    main()
