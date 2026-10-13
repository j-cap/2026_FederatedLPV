# Matrix-error sensitivity diagnostic

## Purpose and procedure

The original six-entry relative norm is coordinate dependent. This post-hoc
sensitivity analysis asks whether aggregate advantages depend on large matrix
entries dominating that norm. It does not replace the pre-existing endpoint.
No models are fitted, grouping is unchanged, and no trajectories are simulated.
Saved Global, FederatedLearned, and Smooth Local maps from primary seeds 371–380
are evaluated at the same eleven speeds. Reference Jacobians are recomputed by
the original equilibrium/Jacobian routine for evaluation only; all original
per-point errors reproduce within 4.13e-12.

Fixed entry scales are the RMS of saved reference entries from development
fleets 321–330, across 300 clients and eleven speeds. These references already
exist in the frozen smoothness development checkpoints. No primary or controller
results enter the scales, and no alternative scale rule is selected by its gains.
The rule is newly specified after the original study, so this is a diagnostic,
not a preregistered independent endpoint.

For entries (A11,A12,A21,A22,B1,B2), scales are
(0.946235,0.00867435,0.142843,0.935124,0.0268294,0.329858).
For each client-speed pair, the balanced error is the square root of the mean
of six squared entry errors divided by their respective fixed scales squared.
This is dimensionless and invariant to consistent changes in individual entry
units. It is not invariant to arbitrary changes or mixing of state coordinates,
and it is not a control-weighted norm or a stability guarantee. Each fixed-scale
error contributes equally relative to its development RMS magnitude. The
small A12 scale is fixed and positive; pointwise percentage error near a zero
entry is deliberately avoided.

Fleet means use the same full/seen/unseen client-speed pairs as the original
metric. Gains average paired fleet ratios; intervals resample ten fleet units
20,000 times, with fixed RNG seed 20261013. Entry-wise diagnostics report scaled
mean absolute error and physical-entry RMSE. Their gain percentages are unchanged
by division by the constant entry scale. Intervals are descriptive, pointwise,
and not multiplicity-adjusted. Clients remain nested within fleets.

## Aggregate findings

| Region | Grouped vs Global gain [%] | 95% interval | Wins |
|---|---:|---|---:|
| Full | 21.28 | [13.65,28.11] | 8/10 |
| Observed | 32.83 | [21.37,42.54] | 8/10 |
| Unobserved | 17.93 | [10.94,24.68] | 8/10 |

Against Smooth Local, balanced full/unobserved gains are 68.47% ([65.98,70.97])
and 72.95% ([70.62,75.29]), both with ten wins. At observed speeds the paired
gain is -5.74% ([-21.32,7.41]); an advantage there is not established.
The grouped-versus-Global aggregate conclusion survives this change in weighting.
The original full-envelope gain remains 23.74%, and historical bootstrap
intervals remain unchanged. This diagnostic has a different bootstrap RNG stream;
its recomputed original-metric intervals must not overwrite published intervals.

## Entry tradeoffs against Global

| Entry | Full-grid MAE gain [%] | Unobserved MAE gain [%] | Unobserved 95% interval | Unobserved wins |
|---|---:|---:|---|---:|
| A11 | 18.08 | 16.07 | [9.03,23.20] | 8/10 |
| A12 | 4.88 | 4.33 | [-1.35,9.50] | 6/10 |
| A21 | 27.16 | 23.30 | [14.43,32.16] | 8/10 |
| A22 | 31.33 | 27.28 | [17.29,36.20] | 8/10 |
| B1 | -30.12 | -30.90 | [-43.49,-18.03] | 0/10 |
| B2 | 18.08 | 14.63 | [7.95,22.02] | 8/10 |

Thus improvement is aggregate, not uniform across entries. B1 error is larger
in every fleet, while A12's gain interval includes zero. The diagnostic does
not isolate the cause of B1 deterioration or link it causally to controller
behavior. Retain the separate empirical control and feasibility evidence.

## Validation, commands, and manuscript recommendation

The execution checkpoint is 6c7c7de; independent summary verification is f94444c.
Two tests check unit-change invariance and the RMS definition. The independent
checker verifies source/output hashes, development scales, 9,900 unique
client-speed-method rows, 90 fleet summaries, and all 48 bootstrap comparisons.
Original model and result files are unchanged.

From the repository root:

```bat
conda activate federated_lpv
python paper\check_matrix_error_diagnostic.py
```

To reproduce the diagnostic outputs (same files, no model fitting):

```bat
python paper\matrix_error_diagnostic.py
python paper\check_matrix_error_diagnostic.py
```

For manuscript integration, define the development RMS scales in the metric
subsection and add a short results paragraph: the fixed-scale full/unobserved
advantages over Global persist, but B1 deteriorates and A12 has no clear gain.
Use the entry table in supplementary/reproducibility material if space is tight.
Do not replace the primary metric, imply improvement in every entry, or claim
that balanced matrix accuracy guarantees controller suitability. This report
does not modify the manuscript.
