# Active manuscript: full-envelope LPV identification

The current contribution is **full-envelope identification from incomplete local operating coverage**, assuming available two-state lateral observations. The scope decision and evidence boundaries are in [the consolidation](../docs/identification_consolidation.md).

`main.tex` uses the 10A-10E identification chain and 10F as supporting LQI validation. It explicitly describes controlled equilibrium-centered perturbation data, whole-envelope matrix error, client participation in fitting, modest control headroom and the absence of a general stability or privacy guarantee. It is a working draft, not a submission-ready deployment claim.

Build and check from the repository root:

```bash
python paper/check_consolidation.py
python paper/analyze_identification_records.py
python paper/check_review_integration.py
python paper/check_local_baseline_integration.py
python paper/check_robustness_integration.py
make -C paper
```

The PDF is `2026_FederatedLPV_identification_draft.pdf`. The unmodified CTAN
IEEEtran 1.8b class is included and conference formatting is required.

The [review revision and run instructions](../docs/manuscript_review_revision.md)
describe the resolved text issues, the existing-record region analysis, and the
selected-group/regularized-Local supplement. The full ten-fleet results are
complete and independently audited; see [the analysis](../docs/manuscript_review_results_analysis.md).
The matched baseline and selected-group findings are integrated in the main
table, results and discussion. `check_review_integration.py` verifies the audited
inputs and generates `review_evidence_numbers.tex`; it performs no fitting or
simulation. Audit saved coefficients without fitting or rollouts using
`python paper/audit_review_validation.py` when an independent audit is needed.

IEEE IV review point 1 is addressed through explicit comparison with IFCA,
Clustered SysID, FedSysID, and IC-SYSID, and a contribution statement focused on
locally deficient LPV scheduling coverage. The [revision note](../docs/manuscript_review_point1_novelty.md)
records the supported distinctions, sources, and frozen-input recovery. Review
point 2 is addressed through the prior-preservation audit and a distinct
development-tuned Smooth Local comparator on the same frozen primary
targets. Its protocol, results, limitations, and Windows commands are in
[the point 2 revision note](../docs/manuscript_review_point2_local_baseline.md).
Review point 3, as requested in the current pass, makes the local-baseline
near-equivalence explicit and motivates the distinct local-prior test.
The centered-ridge correction and nullspace equations distinguish ideal
equality from the implemented finite-ridge case; the comparison reuses the
completed Smooth Local supplement. See [the point 3 revision note](../docs/manuscript_review_point3_prior_equivalence.md).
Review point 4 clarifies simulator-centered, independently reset calibration
records and the absolute-state scheduled-LQI implementation, including
augmentation, prefilter, limiting, conditional integration, and the scope of
the frozen-speed stability check. See [the point 4 revision note](../docs/manuscript_review_point4_workflow.md).
Review point 5 is integrated through a focused coverage/calibration robustness
protocol, Table IV, and a results subsection with explicit failure boundaries.
The abstract and conclusion qualify the benefit by pooled coverage and
calibration quality. Review point 6 is addressed by the editorial and formatting
pass described in [the revision note](../docs/manuscript_review_point6_editorial.md).
The author block remains a working placeholder pending the final author list
and affiliations.
The focused experiment is complete:
all 70 jobs across ten paired fleets pass the saved-model audit and frozen-table
reconciliation. See [the analysis and manuscript implications](../docs/manuscript_review_point5_robustness_analysis.md)
and [the original protocol and Windows commands](../docs/manuscript_review_point5_robustness_protocol.md).
It identifies fleet-wide coverage failure and loss of the grouping advantage
under weaker calibration signal-to-noise. The fresh validation cohort is reported
separately from the original primary results. `check_robustness_integration.py`
verifies the frozen execution/audit/analysis chain and all 35 manuscript numbers.
Use `--write` only to regenerate `robustness_numbers.tex` from that evidence.

The preceding personalized cold-start manuscript and its audits are preserved in [archive/9m_cold_start](archive/9m_cold_start). They describe a different claim and method. Their quantitative headlines and assumptions do not apply to this draft. The chronological development report remains the full experimental record.

The author's overview PDF is integrated as Figure 1, with an introductory
reference and a caption stating the basis-rank and calibration assumptions.
It is saved as `results/figures/paper_method_overview.pdf`.
The committed diagnostics are integrated as Figures 2--4: development basis
validation and scheduling information, speed-wise identification error by local
block, and paired fleet gains with descriptive client tradeoffs. Captions and
results text distinguish the development and primary data, define fleet-level
uncertainty, and identify clients as nested within fleets. The figures reuse
frozen evidence and add no simulations or independent confirmation runs.
See [figure generation and the PaperBanana overview brief](../docs/manuscript_figure_generation.md).
Run `python paper/prepare_identification_figures.py --preflight` to verify the
frozen inputs. If figure regeneration is needed, run
`python paper/prepare_identification_figures.py --generate` to export the basis,
information, fleet/client, and revised speed-error panels from frozen records.
Figure 3 now includes Smooth Local in a separate v3 export, generated with
`python paper/plot_local_baseline.py --generate`. The historical v2 figure and
its locks are preserved.

The frozen Smooth Local LQI supplement is integrated in Table V and Section VI.
The table pairs executed tracking RMSEs with sideslip-violation counts; the text
reports frozen-point instability, descriptive client medians, and the numerical
replay limitation. No prior or controller retuning was performed. The conclusion
distinguishes matrix accuracy from controller suitability for the tested prior.
Run `python paper/check_smooth_lqi_integration.py` to verify the nine generated
claims and their execution/audit hashes. `--write` regenerates the numbers and
integration lock without fitting or simulating. See
[the analysis](../docs/paper_smooth_local_lqi_analysis.md) for the full audit scope.
