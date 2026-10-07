# Experiment 10H-L: validated restart selection and global fallback

10H-K is complete. This next stage tests whether its average two-model benefit
can become a reproducible selection procedure with fewer harmed clients. The
protocol is committed before fitting; 601–605 and their outer clients remain
opened development data. No blind confirmation or closed-loop claim is made.

## Fixed comparison and fitting

- Inherit 10H-K's measured arrays, client split, eight positive coupled plant
  coordinates, coefficient multipliers [0.1,10], two separate Q scales 0.01/1,
  public R/Q shape, quiet-bias correction and likelihood initialization.
  Verify measured-array hashes. No engineering bound or covariance is tuned.
- Increase to six deterministic split directions using the unchanged seed
  offset 2,700,000 and amplitude 0.3 around the context's learned K=1 plant.
  All contexts use the same six direction vectors for each fleet. A restart
  index identifies an initialization strategy, not a common vehicle family.
- Two inner folds are blocked by entire clients: sorted training-client rank
  modulo two defines validation membership. Each fold fits ten clients and
  validates on the other ten. All speed records of a client stay together.
- Fit each inner K=1 model from three nominal/random starts, choose the lowest
  training likelihood among stationary feasible starts, then fit six K=2
  starts. Keep 80 restricted-stage and 800 joint iterations, exact derivatives
  and numerical limits inherited from 10H-K. No larger budget after seeing
  outer results. Save every optimizer result, including failures.
- On all twenty training clients, reuse the audited 10H-J K=1 and first three
  10H-K K=2 restarts; recompute objectives and KKT audits and verify original
  initial splits/source/output hashes. Fit only the three additional full
  starts. Thus prior work remains immutable and all six full models are available.
- A selectable direction must have successful, stationary, feasible fits in
  both folds and the full training refit. Numerical eligibility uses training
  data only, not outer performance. If none qualifies, the primary deployment
  uses K=1 and records the failure; failures are never silently discarded.

## Calibration and deployment rules

Every new client supplies the first 100 samples (1 s) of its first listed
record for calibration. No other record influences its model selection.
The same budget and future windows apply to every comparison.

For each global/group candidate, compute fixed-R normalized forecasts at
horizons 5,20,50 from common prefix origins 20..50 inclusive (31 origins).
Filtered states use outputs strictly before each origin; forecast propagation
has no intervening output corrections. Later outputs within this calibration
prefix score these predictions; all are observed before the deployment cutoff.
The mean of the three horizon errors is the candidate's prefix score.

- Likelihood MAP: compute the unchanged working-mixture membership from the
  first-record 100-sample prefix and choose its maximum weight.
- Forecast group choice: choose the lower prefix forecast score of the two
  group plants. Ties choose the lower component index.
- Forecast choice with global fallback: deploy that group only if its prefix
  score is strictly less than (1-margin) times the global prefix score;
  otherwise deploy K=1. Predeclared margin candidates: 0%,10%,25%.

These are empirical decision rules, not calibrated posterior probabilities
or no-harm guarantees. Decisions are frozen across all subsequent speed
records; each chosen model runs its own KF/plant. No gain/matrix interpolation.

## Training-only cross-validation selection

For every fold, direction and margin, deploy the above fallback rule on inner
validation clients, then evaluate only their post-calibration future outputs.
Score horizons 5,20,50 using the same origins 100..150 inclusive (51 origins).
For each client and horizon, aggregate errors across that client's records,
then compute log(E_candidate/E_fold_global). The selector averages these
values equally across clients, horizons and folds. All E use fixed public R;
1e-12 is only a numerical floor for the logarithm, not a tuned noise prior.

Equal-client log ratios address the distinction exposed in 10H-K between
average absolute fleet errors and relative individual-client losses. They do
not impose a worst-client constraint. Jointly select direction and margin
by the lowest inner score. Ties within 1e-12 prefer a larger fallback margin,
then a lower direction index. Refit transfer uses that same direction around
the full-training K=1 model; it does not assume fold/final parameters coincide.
Report all candidate inner scores and transfer behavior. Outer test outputs
never rank restarts, margins or covariance panels.

## Outer comparison and ablations

At each fixed Q, compare these predeclared deployments on the same windows:

1. Global K=1.
2. Six-start training-likelihood winner with prefix likelihood MAP.
3. Inner-validation selected restart with prefix likelihood MAP.
4. The same selected restart with prefix forecast group choice.
5. Primary: selected restart and margin with prefix forecast choice + fallback.
6. Archived three-start 10H-K winner, recalibrated on the same 100-sample prefix.
7. Public nominal reference.

All forecasts score horizons 1,5,20,50 from origins 100..150 inclusive.
Input-only simulations start from zero at record start but score only samples
100..199. State filtering may use past measurements; future targets cannot
affect assignments or forecasts. Simulation is diagnostic and does not select
hyperparameters. Report per-client and fleet errors, paired gains vs global
and likelihood-only baselines, improved/degraded client counts, median and
worst ratios, absolute error tail, and fallback counts.

For all six full restarts, evaluate prefix-MAP future predictions on the same
outer windows as a descriptive repeatability audit. Their outer scores cannot
change selection. Report restart partitions after label alignment, parameter
spread and per-client prediction-error variation. Apply selected-model suffix
residual diagnostics without cross-record lag pairs. No IID p-values or
calibrated confidence intervals are inferred.

## Interpretation and safeguards

The experiment is useful if longer-horizon benefit persists while prediction
variability or unfavorable client outcomes decrease. Report every fleet and
both Q panels, including adverse transfer from inner selection to full refit.
Do not invent a physical-readiness pass gate afterward or conceal boundary,
covariance or numerical failures. Even a successful fallback is an empirical
prediction result, not a safety certificate or KF/controller guarantee.

Only simulator-generated measured arrays and their client associations leave
the simulator boundary. No labels, client physical truths, latent states or
true matrices enter fitting, selection or audit. Historical artifacts remain
unchanged. Seeds 591–600 and 611–620 remain sealed. Code and results are
committed locally; pushing is a separate action.
