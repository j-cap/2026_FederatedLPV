# Review point 5: manuscript integration

The focused study is integrated into the active IEEE conference manuscript.
It uses the already pushed execution, saved-model audit, and analysis
checkpoints. No calibration, model fitting, representation selection, prior
tuning, controller tuning, or rollout is added in this revision.

## Placement and argument

- The benchmark ends with a focused robustness protocol. It defines the fresh
  ten-fleet cohort, seven paired interventions, constant recipient budgets,
  noise/bias magnitudes, label separation, fixed prior, and rank-only fallback.
- The identification results include a compact table with Federated, Global,
  Smooth Local, and the number of fleets selecting one group. The table reports
  all seven conditions, including fleet-wide coverage failure.
- The accompanying results separate a ten-client rank-three known-family
  diagnostic from the full-rank learned partitions. Their response is an
  observed cross-family pooling behavior, not a BIC rank guarantee or an
  empirically successful fallback.
- Fleet-wide rank-six failure is reported on both the fixed full grid and
  the five speeds absent from the fleet. The text explains why Global fallback
  cannot recover the missing direction and why the fixed Smooth Local prior
  performs better in this condition.
- Calibration sensitivity is linked to group selection. Lower signal-to-noise
  selects one group in all ten fleets, so Federated becomes Global. The noisy
  regressor and biased-center results do not establish a grouping advantage
  over Global. All four calibration perturbations retain a sharing advantage
  over the fixed Smooth Local prior, conditional on this benchmark.
- The abstract and conclusion carry these limitations. Primary calibration
  assumptions are distinguished from their controlled perturbations. The LQI
  evidence retains its original calibration/feedback scope.

The old brief known-family failure illustration is replaced by the quantitative
focused analysis to avoid repeating the same coverage argument across different
cohorts. Existing primary errors, baseline evidence, figures, and controller
results remain unchanged. The new 3.058% reference error belongs to seeds
1201--1210; it does not replace the primary 3.249% result.

## Evidence and verification

`paper/robustness_evidence_lock.json` locks the completed execution manifest,
saved-model audit, descriptive analysis, and its source. The checker follows
the original manifest through all 70 checkpoints and seven summary tables,
verifies the source and audit chain, and checks the manuscript's 35 numerical
macros against the frozen summaries. Regeneration writes only the separate
`paper/robustness_numbers.tex` include.

Run from the repository root:

```bat
python paper\check_consolidation.py
python paper\check_review_integration.py
python paper\check_local_baseline_integration.py
python paper\check_robustness_integration.py
```

Compile with the existing `paper/Makefile`, or compile `paper/main.tex` three
times using pdfLaTeX from its directory. The revised PDF uses the existing
conference class, typography, figures, and bibliography.

Point 5 is addressed at the controlled identification level. Claims about
rescuing an isolated deficient selected subgroup with full-rank Global,
ordinary driving, state estimators, real fleets, or tracking under the new
calibration perturbations are not added. Point 6 remains the separate
editorial/formatting review.

Verification: all four manuscript-evidence checks pass, including the 35 new
robustness macros. The compiled draft has eight pages. The new table and
affected pages are rendered and inspected for readable typography, correct
float order, mathematical notation, column fit, and consistent scope. No frozen
scientific output is regenerated or edited.
