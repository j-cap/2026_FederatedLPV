# IEEE-style manuscript revision and focused validation

Date: 9 October 2026. Parent manuscript: `b572036`.

The manuscript remains a controlled, available-state study of full-envelope
identification from incomplete local speed coverage. The basis, plant,
primary error, original seeds, and historical results are unchanged.

## Review resolution

| Review issue | Change | Status |
|---|---|---|
| Novelty and IFAC overlap | Introduction distinguishes missing scheduling information from local LPV interpolation, clustered learning, and earlier uncertainty-aware vehicle identification. Adds primary LPV references. | Text revised; no claim of a new clustering or regression primitive. |
| Undefined endpoint | Specifies matrix-entry order, physical units, relative norm, eleven-speed grid, averaging order, paired gains and fleet bootstrap. | Complete. |
| Unobserved speeds obscured | Reanalyses the original 19,800 label-free records; adds observed/unobserved columns and speed/block curves with fleet-bootstrap bands. | Complete without fitting or simulation. |
| Conditional rank argument | Adds a fixed-partition proposition and justification; separates this from successful group discovery. | Text complete; selected-group diagnostics await reconstruction. |
| Incomplete method | Gives local transition regression, QR basis convention, ridge constants, assignment scaling, initialization, group-size rejection, stopping and BIC conventions. | Complete. |
| Incomplete benchmark | Adds vehicle centers/scatter, tire equations, perturbation/noise levels, timing, seed partitions and control settings. | Complete. |
| Local baseline fairness | States $L=3$ Local versus $L=7$ shared models beside the results. Discloses that the richer locally regularized comparator was tested only in the earlier oracle experiment. | Matched primary comparison prepared, not yet executed. |
| Development-report presentation | Uses vendored IEEEtran, removes internal experiment names from scientific results, replaces the old three-panel figure, and compresses personalization and repeated limitations. | Complete. |

## Evidence already available

`python paper/analyze_identification_records.py` reads only the committed
`experiment_10e_seed*_label_free_clients.csv.gz` files and verifies their fleet
means against the historical summary before generating new tables and a figure.
It does not alter historical tables or figures.

| Method | Full-envelope error [%] | Observed-speed [%] | Unobserved-speed [%] |
|---|---:|---:|---:|
| Restricted Local | 49.562 | 1.930 | 65.568 |
| Global | 4.250 | 3.840 | 4.388 |
| Federated groups | 3.249 | 2.423 | 3.527 |

Local is best at its own calibration speeds. Group sharing primarily gives it
access to useful predictions outside that region. The new region summaries
average the eligible client-speed pairs within each fleet. The original
full-envelope paired gains and confidence interval remain the primary result;
new bootstrap draws do not replace the historical interval.
Grouped error is higher than Global at unobserved speeds for 71 of the 300
client instances (absolute error-difference tolerance $10^{-12}$). This is
a descriptive client count, not 300 independent statistical units.

## One focused supplementary run

Entry point: `code/experiments/paper_review_validation.py`.
Configuration: `code/config/paper_review_validation.json`.

This is a matched supplementary check on the original ten label-free fleets
(371--380), not new independent confirmation evidence. It addresses exactly:

1. Rank, smallest singular value, condition and speed coverage of each selected
   group, including groups that fail the rank requirement.
2. The existing 10B `LocalRegularized` method on the same primary 10E observations:
   $L=7$ ridge toward a local $L=3$ prior, leave-one-speed-out regularization
   selection, and the original middle-candidate fallback for two speeds.

The original group assignments and learned coefficients were not saved in the
10E summaries. The runner therefore restores the original noisy one-step
transitions using the same RNG sequence, refits the same federated procedure,
and checks Local, Global, grouped fleet errors and selected group counts against
the parent results. It also compares all per-client/per-speed parent errors
when the committed records are present. A mismatch stops the run. It never
silently relabels a changed fit as restoration.

Physical parameters are used by the calibration simulator and exact Jacobian
evaluation. The fitting interface receives only observed speeds and estimated
matrices; neither family labels nor evaluation matrices enter selection.
No tracking rollouts, new controller costs, new state estimation, new basis
selection, or ordinary-driving claims are part of this supplement.

The source lock covers original configuration, calibration, basis, grouping,
plant/fleet code, summaries and saved primary records. Each completed fleet is
written immediately to `results/tables/paper_review_validation_seed<seed>.json`.
These durable checkpoints include reconstructed matrix targets, fitted maps,
assignments, new baseline strengths, group diagnostics, errors and provenance.
The run resumes unchanged checkpoints; incompatible checkpoints cause an error.
Final summarization requires all ten verified fleets.

### Commands from the repository root

Activate the existing project environment. If dependencies need installing:

```bash
python -m pip install -e './code[dev]'
```

Preflight and a full one-fleet smoke check:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python code/experiments/paper_review_validation.py --preflight
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python code/experiments/paper_review_validation.py --smoke
```

Run the ten-fleet supplement:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python code/experiments/paper_review_validation.py --run --workers 5
```

Rerunning this command resumes verified checkpoints. If all checkpoints exist
and only the summary needs regenerating:

```bash
python code/experiments/paper_review_validation.py --summarize-only
```

The complete result includes `paper_review_validation_summary.csv`,
`paper_review_validation_comparisons.csv`, `paper_review_validation_groups.csv`,
and `paper_review_validation_manifest.json`, plus the ten frozen JSON checkpoints.
Raw historical records and tables are never rewritten. Smoke output is temporary
and is not included in the manuscript or production summaries.

Push the execution checkpoint before further analysis:

```bash
git add results/tables/paper_review_validation_*.json results/tables/paper_review_validation_*.csv
git commit -m "Add matched manuscript-review validation results"
git push origin experiment/10hq-preparation
```

The manuscript explicitly leaves these two checks unresolved until the complete
result is audited. No automatic manuscript claim is changed after execution.

## Preparation verification

All 174 repository tests pass, including guards for changed historical sources,
false rank conclusions from repeated speeds, and mismatched parent predictions.
Preflight passes and the full original seed-371 smoke check reproduces the
parent errors and selected group count. The smoke does not write production
results. The revised five-page conference PDF is compiled and visually checked.
The full ten-fleet supplementary run has not been executed.

## IEEE class provenance

`paper/IEEEtran.cls` is the unmodified CTAN IEEEtran 1.8b class downloaded from
<https://mirrors.ctan.org/macros/latex/contrib/IEEEtran/IEEEtran.cls>.
Its embedded copyright and license are retained. The article fallback has been
removed so a missing conference class causes a visible build failure.
