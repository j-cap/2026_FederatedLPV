# Review point 5: coverage and calibration robustness results

The focused experiment is complete and audited. It supports a conditional
identification claim: complementary coverage can remain useful under the tested
calibration perturbations, but grouping cannot recover scheduling directions
absent from the entire fleet. The advantage over Global also depends on whether
the finite-data evidence supports multiple groups. These limits belong in the
paper alongside the positive results.

Execution checkpoint: `84ec66d` on `main`. The protocol and runner were pushed
before execution in `6a03b1a`. The separate saved-model audit was pushed in
`39ac612`. This assessment does not refit models or change the manuscript,
historical evidence, frozen execution summaries, or selection settings.

## Completeness and verification

- All 70 jobs are present: ten paired fleets, seeds 1201--1210, seven conditions.
- Each condition retains 30 clients and 83 observed client-speed cells per
  fleet, with 36 independent transitions per cell. The saved reconstruction
  inputs account for 5,810 cells and 209,160 transitions across the study.
- All 77 manifest-locked checkpoint/table hashes and source/environment locks
  pass. All 70 saved models pass the reference, prediction, scheduling-rank,
  fallback-rule, and stationarity audit without refitting. Maximum normalized
  stationarity residual is 7.58e-7, below the predeclared 1e-5 threshold.
- Independent reductions of all 1,320 seed/method/region rows match the saved
  summaries. Replaying all seven table/bootstrap outputs in a temporary
  directory reproduces the saved outputs to numerical tolerance.
- All 830 center-bias cell hashes match the paired reference measured records.
  Physical fleet parameters and per-recipient record budgets match across
  conditions. Exact raw-record restoration passes for seed 1201/reference
  (83 cells), using frozen centers and parameters without matrix fitting.

The statistical units are ten fleets, not 300 vehicles or the individual
transitions. Errors below are mean relative six-entry matrix errors on the
fixed eleven-speed evaluation grid, expressed in percent. The reported
intervals are the frozen 20,000-resample paired-fleet bootstrap intervals,
with a pointwise descriptive interpretation.
This is a fresh validation cohort. Its 3.058% reference Federated error does
not replace the 3.249% primary result from the manuscript's original fleets.

## Full-grid results

| Condition | Federated | Global | Smooth Local | Restricted Local | OracleFamily | RankFallback | Fleets with K=1 / K=2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Reference | 3.058 | 4.214 | 7.221 | 48.644 | 1.817 | 3.058 | 3 / 7 |
| Heavy-family coverage restricted | 3.879 | 4.584 | 7.779 | 39.670 | 24.272 | 3.879 | 5 / 5 |
| Fleet coverage restricted | 33.423 | 33.255 | 6.645 | 28.845 | 33.634 | 33.255 | 5 / 5 |
| Excitation standard deviations halved | 4.324 | 4.324 | 13.179 | 86.246 | 2.608 | 4.324 | 10 / 0 |
| Next-state noise standard deviations doubled | 4.326 | 4.326 | 13.286 | 86.692 | 2.625 | 4.326 | 10 / 0 |
| Current-state/input measurement noise | 4.039 | 4.259 | 9.707 | 62.416 | 2.055 | 4.039 | 7 / 3 |
| Assumed calibration-center bias | 4.810 | 4.906 | 9.734 | 61.664 | 3.519 | 4.810 | 9 / 1 |

The Restricted Local row is retained for completeness, but the distinct
development-tuned Smooth Local prior is the meaningful strong recipient-only
comparison. Neither prior nor grouping settings were retuned on these fleets.

## Group deficiency with fleet-wide coverage preserved

Restricting all ten heavy-family clients to two or three speeds in
{10,12,14} m/s leaves their known-family scheduling design at rank three.
There are ten clients in this group, exceeding the minimum size of five.
Thus a size guard alone does not ensure seven-function scheduling information.
The full fleet still covers all eleven speeds and has rank seven.

The heavy-family OracleFamily error rises from 1.860% to 69.225% on the full
grid. The other two oracle-family maps and recipient-only maps are unchanged,
as their data are unchanged. This makes the deficient-family diagnostic
specific to the coverage intervention. The overall OracleFamily error rises
from 1.817% to 24.272%. Known labels are consequently not a performance upper
bound when their partition excludes complementary scheduling information.

All 15 selected learned groups across the ten stressed fleets retain rank
seven. In every fleet, all ten heavy clients share their selected group with
clients from at least one other family. The method selects one fleet-wide group
in five fleets and two groups in five. This is an observed pooling response,
not an explicit rank constraint or a guarantee that BIC preserves rank.

Federated full-grid error increases by 0.821 percentage points relative to
reference (95% interval [0.234,1.484], nine deteriorating fleets). It still
improves on Global by a mean paired relative 15.49% [5.03,26.76] and on Smooth
Local by 49.73% [43.16,56.74]. Recipient-unobserved errors are 4.339%, 5.001%,
and 9.844%, respectively.

No deficient learned group is selected in this condition, so RankFallback is
inactive. These results show the failure of a coverage-deficient known-family
partition and the behavior of the learned partition. They do not demonstrate
that the fallback repairs an isolated deficient learned group while Global
remains full rank.

## Fleet deficiency and the limit of a Global fallback

Restricting the fleet to six speeds in {10,12,14,16,18,20} m/s yields rank six
for Global, all 15 selected learned groups, and all 30 oracle-family groups.
Every recipient triggers RankFallback, and the Global-deficiency flag is true
in all ten fleets. Ridge solves still return coefficients, but the records do
not supply the missing scheduling direction.

Federated error rises to 33.423%, a paired increase of 30.365 percentage
points [29.867,30.839], with deterioration in all ten fleets. Global and
OracleFamily also reach about 33%. The Global fallback gives 33.255% and
cannot restore the absent information. Its full-grid paired relative gain
over Federated is 0.50% [-0.32,1.56], which does not establish a positive gain.

On the five speeds absent from the entire fleet (22--30 m/s), errors are
69.303% for Federated, 67.862% for Global/RankFallback, and 10.437% for Smooth
Local. Recipient-unobserved errors are 43.582%, 43.016%, and 8.330%.
Smooth Local's fixed curvature prior therefore strongly outperforms the
grouped method in this predeclared failure condition. This is useful evidence
for the value of a prior when data cannot identify the chosen representation.
It does not make that prior's extrapolation universally valid.

Smooth Local's full-grid error is slightly lower than under reference
coverage. Both coverage interventions move the fitted data to different
speed locations. Neither that change nor the improvement of Restricted Local
proves that less coverage is beneficial in general. Recipient-unobserved
regions also differ across these interventions. The fixed full-grid endpoint
is the main cross-condition comparison.

## Calibration sensitivity and selection behavior

| Perturbation | Federated full-grid increase (percentage points) | 95% paired interval | Deteriorating fleets |
| --- | ---: | ---: | ---: |
| Weaker excitation | 1.266 | [0.741,1.747] | 10 / 10 |
| Doubled next-state noise | 1.268 | [0.754,1.752] | 10 / 10 |
| Noisy current regressors | 0.981 | [0.449,1.497] | 10 / 10 |
| Biased assumed centers | 1.752 | [1.264,2.226] | 10 / 10 |

Halving excitation or doubling target-noise standard deviations makes BIC
choose K=1 in every fleet. Federated and Global are then exactly the same
saved model, with full-grid errors near 4.32%. Relative gains over Smooth
Local remain about 67%, but there is no grouping advantage over Global.
All K=2 candidates are feasible; their saved BIC exceeds K=1 by at least
45.49 (weaker excitation) or 47.61 (doubled noise). Thus selection of K=1
cannot be attributed to a missing two-group candidate.

These two interventions reduce calibration signal-to-noise in related ways.
Their paired draws and nearly linear local dynamics help explain their similar
results. They should not be presented as independent replications of the same
claim. The result supports a statement that selection becomes more pooled
under these perturbations, not that it has identified a homogeneous fleet.

With noisy regressors, K=2 is selected in three fleets. The full-grid paired
relative gain over Global is 5.29% [-0.00044,12.88], and the unobserved gain is
3.15% [-2.28,9.62]. Neither interval excludes zero. With biased centers,
K=2 is selected only once; its BIC margin is just 0.0175. The mean full-grid
gain is 1.99% [0,5.97], driven by that one fleet, with nine exact ties.
There is insufficient evidence here for a persistent grouping advantage
under either perturbation.

All 5,810 local transition designs retain rank three. The largest reported
regressor condition number is 14.50 in the fixed state/input coordinates.
The four calibration interventions therefore degrade estimation/selection
without causing a local transition-rank failure. Scheduling rank remains
seven in every Global/selected group outside the fleet-coverage condition.
Local transition information and scheduling information must be discussed
separately.

## Manuscript implications

The evidence is sufficient to integrate the focused robustness study with
explicit limitations. It supports the following changes:

1. Add a compact robustness table reporting Federated, Global, Smooth Local,
   selected-K frequency, and coverage rank for all seven conditions. Keep the
   negative fleet-coverage result visible.
2. Explain that minimum client count does not ensure full scheduling rank.
   Use the ten-client, rank-three family diagnostic as the concrete example.
3. State that the learned partition preserved rank in the tested family
   restriction by pooling across families. Do not attribute that behavior to
   the fallback or claim an unconditional rank guarantee from BIC.
4. State that a Global fallback requires adequate fleet coverage. Treat
   RankFallback as a diagnostic safeguard; these results do not justify a
   guaranteed improvement or controller-safety claim.
5. Qualify the gain over Global by calibration quality: the tested reduction
   in signal-to-noise eliminated the selected grouping advantage. Gains over
   the fixed Smooth Local prior persist in the four calibration conditions,
   but reverse strongly under fleet-wide deficiency.

No extra simulation is needed to substantiate these scoped statements.
A claim specifically about rescuing a deficient *selected* subgroup with a
full-rank Global model would require another predeclared test, because that
situation did not occur in this run. It is unnecessary if the manuscript
limits itself to rank monitoring and the observed failure boundary.

The study evaluates controlled available-state matrix identification only.
It adds no observer experiment, ordinary-driving calibration, real-fleet
validation, LQI rollout, or scheduled-stability evidence. Calibration noise
must not be described as online sensor-noise robustness of the controller.
Point 5's experiment and assessment are complete. The subsequent manuscript
integration is documented in [the revision note](manuscript_review_point5_integration.md).
Point 6 remains the editorial/formatting pass.

## Reproduction and sources

From the repository root in the pinned `federated_lpv` environment:

```bat
python code\experiments\paper_robustness_validation.py --status
python code\experiments\paper_robustness_validation.py --audit
python paper\analyze_robustness_validation.py
```

The first command is read-only. The second rewrites only the separate audit
record. The third verifies the frozen tables and writes only the separate
analysis JSON. It performs independent error reductions and replays summary
generation in a temporary directory, without any fitting or selection.

Tested exact restoration of ignored measured records:

```bat
python code\experiments\paper_robustness_validation.py --restore-data --seed 1201 --case reference
```

The command verifies all 83 committed cell hashes and writes an ignored NPZ,
without modifying checkpoints, summaries, or the execution manifest.

Sources: `results/tables/paper_robustness_validation_manifest.json`, the 70
checkpoints named in that manifest, the seven locked CSV summaries, the separate
`paper_robustness_validation_audit.json`, and the derived
`paper_robustness_validation_analysis.json`. The existing protocol specifies
condition magnitudes, pairing, fixed estimators, and endpoints.
