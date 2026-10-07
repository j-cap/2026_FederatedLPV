# Experiment 10H-K: jointly learned models and client memberships

This protocol is committed before fitting. Seeds 601–605 and their held-out
clients are already opened development data; this is not blind confirmation.

## Question and controls

Does a two-component structured plant explain future measured outputs better
than the single pooled plant, while giving reproducible client partitions?

- Reuse the audited 10H-J widest-box K=1 solutions and their three-restart
  evidence. Verify measured-array hashes and recompute the K=1 objective with
  the new client-level evaluator before fitting K=2.
- Keep eight coupled positive coordinates per plant, coefficient box [0.1,10],
  public R and Q shape, quiet-start bias preprocessing, record initialization
  and likelihood convention unchanged. The box is a diagnostic search domain,
  not an engineering prior. Q scales 0.01 and 1 are separate fixed panels.
- K=2 has 16 plant coordinates and one mixing weight, bounded to [0.01,0.99]
  to keep log weights finite. Report occupancy/collapse; this limit does not
  force any client to belong to either component. No new covariance tuning.
- Three deterministic symmetric random log-coordinate splits of amplitude 0.3
  around the learned K=1 plant, projected onto the coupled feasible polytope.
  Seed offset 2,700,000; normalize each direction by its largest absolute entry.
  No clustering at the public nominal model and no truth-based initialization.
- Use joint SLSQP with exact total gradients. Restricted actuator, relaxation,
  mechanics stages move both models; the final joint stage also learns mixing
  weight. Membership weights are recomputed from current models at every
  objective evaluation. Budgets: 80 per restricted stage, 800 joint iterations,
  ftol 1e-12. Select the restart by training working likelihood only.

## Objective

For client i, D_ig sums e' S^-1 e + logdet(S)-logdet(R) over every sample of
every record belonging to that client. Minimize

    L_K = -2/(3 N) sum_i log(sum_g pi_g exp(-D_ig/2)),

where N is the total number of record-time samples. Use stable log-sum-exp.
For K=1 this is exactly the previous objective. Duplicating the K=1 plant in
both components gives the same objective, providing an available nested
reference. These are Gaussian working weights, not calibrated class
probabilities: the covariance/residual problems in 10H-I/J remain relevant.
Client identity is only a measured-record association, never a class label.

## Causal validation

- For each unseen held-out client, use only the first 50 samples (0.5 s) of its
  first listed speed record to compute membership under the learned models
  and training mixing weights. Freeze that membership across all its records.
  Other records' outputs cannot affect membership.
- Primary deployment: select the maximum-weight component and run its own KF
  and plant forecasts. Secondary diagnostic: average the separately computed
  output forecasts with the frozen weights. Never average KF/controller gains
  or select a component using full held-out errors.
- Forecast horizons 1/5/20/50 samples use the same origins 50...150 inclusive
  in each 200-sample record: 101 origins per record. States use outputs strictly
  before each origin; known commands enter propagation, with no intervening
  output corrections. K=1 and nominal use precisely the same windows.
- Separate zero-initialized full input-driven simulations are scored only on
  samples 50...199, after membership calibration. They use no state corrections.
- Report individual output RMSE and fixed-R normalized errors, paired K=2/K=1
  and nominal comparisons, and per-client errors. No outer score selects a
  restart, K, Q, bounds, assignment rule or other hyperparameter.
- On held-out suffixes, report whitened residual covariance/NIS and record-aware
  serial/past-command correlations for the selected MAP component. Component
  whitened residuals remain working-model diagnostics. No IID p-values.

## Numerical and interpretation audit

Report every restart, feasibility, analytic/finite-difference gradients, KKT
residuals for both plants and mixing weight, exact coupling, training occupancy,
mixture weight, parameter separation and best-reference objective difference.
Align restart component labels by minimum squared physical-log distance to the
selected restart; report partition agreement and parameter/objective spread,
including nonselected numerical failures. Inspect selected constrained-face
Hessian curvature. Numerical limits remain KKT <=1e-4, constraint violation
<=1e-8, log coupling <=1e-12, gradient relative error <=1e-5; violations remain
visible. A collapsed reference is retained if every split is worse than K=1.

Report gains and degradations on all five fleets in both covariance panels.
More training flexibility alone does not justify grouping. Consistent future
output improvement, especially longer-horizon/input-driven behavior, and stable
partitions would support compatible pooling. Failure leaves covariance, bias,
excitation and objective mismatch as alternative explanations; it does not
prove fleet homogeneity. K=2 is a minimal mechanism test, not a true family count.
No universal physical-readiness gate is added and historical gates are unchanged.

No latent states, client parameters, true matrices or labels leave the simulator
boundary for learning or this audit. Seeds 591–600 and 611–620 remain sealed.
No federated communication efficiency, latent-state estimation improvement,
controller gain or LPV/adaptive stability is established by this experiment.
