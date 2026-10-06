# Experiment 10H-J: bound sensitivity and single-model validation

This protocol is fixed before executing new fits. The development fleets and
held-out clients have already been examined in 10H-H/I. This is an exploratory
diagnostic with a predeclared design, not blind confirmation.

## Questions

1. Does widening the heuristic coefficient box produce an interior, repeatable
   optimum, or does the optimizer keep following its limits?
2. Are parameter profiles informative after releasing the original limits?
3. Do one-step improvements persist at longer prediction horizons?
4. Do unexplained serial and input correlations remain?

## Controlled design

- Opened seeds 601–605; the same measured records, 200 samples and client split
  as 10H-H/I. Compare measured-array hashes with 10H-I before fitting.
- Preserve all eight positive coordinates and exact physical coupling.
- Nested coefficient multipliers: [0.35, 3], [0.2, 5], [0.1, 10]. These are
  diagnostic search domains, not validated engineering intervals.
- Two separate panels at fixed public-base-Q scales 0.01 and 1. The first is a
  controlled historical setting, not a newly validated output-only tuning rule.
  The second is the setting previously selected in all five 10H-I fleets.
  Public R, Q shape, bias preprocessing and working likelihood remain fixed.
- Identical deterministic nominal/random starts across boxes; three staged
  restarts, 80 iterations per restricted stage and 500 for joint refinement.
  Choose a restart by training likelihood only. No bound or Q is selected.
- Report all independent KKT residuals, feasibility, exact coupling, full/face
  curvature, information conditioning, effective/physical parameters, implied
  relaxation lengths, actuator time constant and normalized stiffnesses.
- For every widest-box model: eight nuisance-reoptimized local profiles at
  log offsets ±0.08 and ±0.16, plus center. Also fix b_r/b_r_nom at
  0.1, 0.15, 0.2, 0.25, 0.35, 0.5 and 1 and reoptimize all other coordinates.
  Mark infeasible targets explicitly; do not clip them. These are training
  working-objective profiles, not calibrated confidence intervals.

## Output validation

- Score every predeclared model on held-out clients after training selection.
- Fixed-R sensor-normalized MSE and individual output RMSE.
- Forecast horizons 1, 5, 20 and 50 samples (0.01, 0.05, 0.2 and 0.5 s).
  Use a 20-sample observation burn-in and the same forecast origins at each
  horizon. Initialize from the causal filtered state before an origin, then
  propagate commands without intervening output corrections.
- Full zero-initialized open-loop output simulation as a separate diagnostic.
- Innovation mean/covariance/NIS; autocorrelation and correlation with past
  steering commands at lags 1–20, without joining record boundaries. Report
  both pooled and within-record-centered correlations to expose constant-bias
  sensitivity. Correlations are descriptive; no IID significance claims.

## Interpretation and safeguards

Successful constrained optimization does not require an interior optimum.
The numerical checks retain KKT ≤1e-4, feasibility ≤1e-8, physical log-coupling
error ≤1e-12, restart parameter CV ≤0.08 and objective spread ≤0.5%. Check that
best training objectives do not increase under nested feasible sets beyond
1e-8. A violated numerical check must remain visible and limits interpretation.

There is no new physical-readiness pass/fail threshold based on zero bound
activity, profile percentages, or prediction gain. Historical gates remain
unchanged. An interior solution alone does not establish physical plausibility,
and a boundary solution alone does not establish optimizer failure. Persistent
boundary motion, shallow profiles or degraded longer-horizon predictions are
evidence against accepting the pooled coefficients as calibrated vehicle
parameters. They cannot isolate heterogeneity from covariance/bias mismatch.

Do not read labels or true vehicle parameters for learning, bounds, validation
selection or interpretation of this experiment. Confirmation seeds 591–600 and
611–620 remain sealed. Before selecting actual engineering priors, define the
intended vehicle envelope and obtain independent public specifications or
measurements. Toolbox defaults are examples, not fleet-wide admissible ranges.
