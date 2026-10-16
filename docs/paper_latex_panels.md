# Separate manuscript panels for LaTeX

From the repository root in the `federated_lpv` Anaconda environment:

```bat
python paper\prepare_latex_panels.py --preflight
python paper\prepare_latex_panels.py --generate
```

This reads frozen records and verifies their provenance. It performs no fitting
or simulation and does not overwrite the combined figures, historical manifests,
or the manuscript. Each panel has its own axes and exports PDF plus PNG, without
an embedded title or panel letter. LaTeX will supply the letters and subcaptions.
The coverage panels retain the Smooth Local comparator, log axes, observed-speed
marks, bootstrap bands, and linear 0--10% insets from the current Figure 3.

| Current figure | Panel PDF in `results/figures/` | Intended LaTeX subcaption |
| --- | --- | --- |
| 2(a) | `paper_basis_validation_panel.pdf` | Development-only basis validation |
| 2(b) | `paper_scheduling_information_panel.pdf` | Scheduling information ($L=7$) |
| 3(a) | `paper_coverage_10_12_14_panel.pdf` | Local grid: 10, 12, 14 m/s |
| 3(b) | `paper_coverage_16_18_20_panel.pdf` | Local grid: 16, 18, 20 m/s |
| 3(c) | `paper_coverage_22_24_26_panel.pdf` | Local grid: 22, 24, 26 m/s |
| 3(d) | `paper_coverage_28_30_panel.pdf` | Local grid: 28, 30 m/s |
| 4(a) | `paper_paired_fleet_panel.pdf` | Paired primary results |
| 4(b) | `paper_client_tradeoff_panel.pdf` | Client tradeoff (descriptive) |

PDFs use approximately one-column width so the fonts retain their intended size
when paired across two columns. Every coverage panel has axis labels so it can
also stand alone. Panel order and composition will be handled in LaTeX after
reviewing the generated files. The inset scale explanation belongs in the
LaTeX caption rather than a second embedded plot title.

Commit and push the eight PDFs and new provenance manifest:

```bat
git add results/figures/*_panel.pdf results/tables/paper_latex_panels_manifest.json
git commit -m "Generate separate manuscript figure panels"
git push
```

PNG previews are regenerable and need not be committed. The manifest records
PDF hashes, rendering-code hashes, verified input hashes, and scientific checks.
For previews outside the repository, use `--output-dir PATH`; its manifest stays
in that output directory.
