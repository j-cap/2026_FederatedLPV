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
The audit does not refit models or modify historical results.

For Bash, prefix the run with
`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`. A completed rerun verifies hashes
and reuses frozen results; changed sources or frozen outputs cause failure.
