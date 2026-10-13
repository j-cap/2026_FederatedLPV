# Review point 2: independent local regularization assumption

Protocol fixed before matched primary fitting. The previous LocalRegularized
method (called **Prior-centered Local** in the revised paper) is a diagnostic:
its L3 prior nearly fits its two or three matrix targets already. With the
implemented numerical diagonal, the richer correction satisfies

```
delta = solve(Phi.T Phi + (alpha+epsilon) I,
              Phi.T (Y - Phi Theta0) - epsilon Theta0)
N delta = -epsilon/(alpha+epsilon) N Theta0
```

Here N is an orthonormal row basis for the local design nullspace. Thus the
additional coefficient directions retain the prior, apart from numerical
shrinkage. Near-equality to Restricted Local does not establish failure of
other local regularizers.

**Smooth Local** instead penalizes the average squared second derivative of
the six fitted functions with respect to z=(v-20)/10 within the fixed L7 span.
It uses no L3 prior, group map, or family information. The factor uses exact
derivatives of the raw seven functions, the existing QR transform, and
401-point trapezoidal quadrature over the fixed envelope. Constant and affine
functions form its unpenalized space. The target loss is divided by the
number of observed speeds; epsilon=1e-16 only regularizes numerical solves.
The two-speed affine-recovery test showed that 1e-12 introduced a small
coefficient-shrinkage bias at the weakest curvature candidate; reducing the
numerical term was fixed before any development or primary results were run.

The common strength is selected from the seven candidates in
`code/config/paper_local_smoothness.json` on development fleets 321--330,
minimizing fleet-mean unobserved-speed matrix error with a one-standard-error
rule favoring the largest eligible strength. Development reference Jacobians
are used for this offline tuning. This resource is disclosed; the recipient
fit accepts only its local design and noisy targets. No primary outcome enters
selection. The selected strength is saved before primary fitting.

Original 10B partial calibration is reconstructed with its original RNG
order and checked against frozen Local and Global observed/unobserved fleet
means. Reconstructed targets and reference arrays are saved durably. The
full-coverage calibration and nonlinear recovery experiments are not rerun.
Primary fitting consumes the frozen review checkpoints for fleets 371--380;
there are no new fleets, group refits, or control rollouts. Results constitute
a matched post-review supplement, not independent confirmation.

Run from the repository root in the existing conda environment. Windows CMD:

```bat
set OPENBLAS_NUM_THREADS=1
set OMP_NUM_THREADS=1
python paper/local_baseline_validation.py --preflight
python paper/local_baseline_validation.py --run --workers 5
```

Commit/push execution checkpoints before post-fit auditing, as AGENTS.md
requires. Then run `python paper/local_baseline_validation.py --audit` to
check saved model predictions against freshly evaluated Jacobians, check
first-order stationarity, and audit the prior-centered nullspace identity.
The audit does not refit Smooth Local or group models or modify historical
results. It reconstructs the old L3 prior to check the centered-fit identity.

For Bash, prefix the run with
`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`. A completed rerun verifies hashes
and reuses frozen results; changed sources or frozen outputs cause failure.

## Completed matched result

Protocol was pushed as `1e36e5a` before execution. The execution checkpoint
was pushed as `51574d0` before auditing. The 15 review result files removed
by the October 13 Overleaf merge were recovered byte-for-byte from `c7777b8`;
that recovery is distinct from the reconstructed development calibration.

Development reconstruction agrees with frozen Local/Global region means
to at most 1.82e-11, within the predeclared 1e-8 tolerance. The best mean development candidate
was alpha=1, but the one-standard-error rule selected alpha=1e6, approaching
the affine-function limit. No candidate set was changed after seeing results.
The small numerical ridge was reduced during structural testing, before the
protocol was frozen or any development/primary results were evaluated.

| Method | Full error [%] | Observed [%] | Unobserved [%] |
| --- | ---: | ---: | ---: |
| Restricted Local | 49.562 | 1.930 | 65.568 |
| Prior-centered Local | 49.562 | 1.930 | 65.568 |
| Smooth Local | 7.489 | 1.630 | 9.458 |
| Global | 4.250 | 3.840 | 4.388 |
| Federated groups | 3.249 | 2.423 | 3.527 |

Mean paired grouped gains relative to Smooth Local are 56.20% full-envelope
(95% fleet-bootstrap CI 49.69--61.81%) and 62.33% unobserved-speed
(57.03--67.08%), with wins in all ten fleets. Observed-speed error favors
Smooth Local in every fleet; grouping is not uniformly superior. The
headlines, Table III, Figure 3, results, and conclusion now reflect this.

The old prior's observed-target residual has median 0.0020% and maximum
0.0708%. The nullspace identity is checked for all 300 clients. Its tiny
effect is framed as prior preservation rather than independent evidence
against local regularization. Historical numerical claims remain fully
verified; only the manuscript requirement to display the six redundant
gain/CI macros against Prior-centered Local was removed.

Saved Smooth Local coefficients satisfy the normalized stationarity audit
(maximum relative residual 4.36e-7) and reproduce independently recomputed
reference errors within 1e-10. Six structural/provenance tests pass. The
new evidence lock and `check_local_baseline_integration.py` protect the
audited summaries, generated numbers, and figure provenance without fitting.

The supplement uses existing primary fleets and is not independent
confirmation. Development simulator Jacobians provide offline tuning,
which is explicitly disclosed. Smooth Local has not been tested in the
separate scheduled-LQI confirmation study; those control claims retain
their original comparator set. Other local priors remain untested.

```bat
python paper/check_local_baseline_integration.py
python paper/plot_local_baseline.py --generate
```

The first command checks committed evidence without fitting; the second
recreates the revised coverage PDF/PNG. Original figures and input locks
remain intact. The revised working draft currently occupies seven IEEEtran
pages; final page-budget consolidation remains part of the manuscript pass.
All seven pages were visually checked. The final build has no overfull boxes,
unresolved references, or column-balance warning; minor underfull spacing
messages remain. The six new tests and five existing review-validation tests
pass, as do both historical evidence checks, the new integration check,
original figure preflight, and completed-run hash-only resume.
