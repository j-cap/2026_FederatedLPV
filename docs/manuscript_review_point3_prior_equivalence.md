# Review point 3: local-baseline near-equivalence and one distinct prior

This pass strengthens the explanation and motivation of the matched baseline
supplement already completed in `0376170`. It reuses that frozen test rather
than counting it as another experiment or independent confirmation.

## Why the centered fit retains its prior

Let $\Theta_{i,0}$ be the restricted L3 estimate lifted into the L7 basis.
The implemented centered fit minimizes

$$
\|Y_i-\Phi_i\Theta\|_F^2
+\rho_i\|\Theta-\Theta_{i,0}\|_F^2
+\epsilon\|\Theta\|_F^2,
\qquad \rho_i>0,\quad\epsilon=10^{-12}.
$$

For $R_i=Y_i-\Phi_i\Theta_{i,0}$,
$\Delta_i=\widehat\Theta_i^{\mathrm P}-\Theta_{i,0}$, and
$P_i=\Phi_i^\top\Phi_i+(\rho_i+\epsilon)I$, its normal equations give

$$
P_i\Delta_i=\Phi_i^\top R_i-\epsilon\Theta_{i,0}.
$$

If the prior interpolates ($R_i=0$) and the numerical diagonal is absent
($\epsilon=0$), then the unique centered fit equals that prior for every
positive $\rho_i$. This is conditional exact equality, not a claim that the
two implemented estimators are identical.

For an orthonormal row basis $N_i$ of $\ker\Phi_i$,

$$
N_i\Delta_i=-\frac{\epsilon}{\rho_i+\epsilon}N_i\Theta_{i,0}.
$$

The extra coefficient directions retain the prior, apart from numerical
shrinkage. They receive no information from the recipient's local targets.
Each local L3 design has full row rank at the two or three distinct observed
speeds, so it can interpolate arbitrary six-entry matrix targets in the
unregularized case. The implemented L3 ridge and numerical lifting diagonal
make interpolation approximate. Existing audited residuals are median
0.0020% and maximum 0.0708% across 300 clients.

Small residuals alone do not guarantee small coefficient or unseen-speed
prediction changes under arbitrary conditioning. A general bound is

$$
\|\Delta_i\|_F\le
\frac{\|R_i\|_F}{2\sqrt{\rho_i+\epsilon}}
+\frac{\epsilon}{\rho_i+\epsilon}\|\Theta_{i,0}\|_F.
$$

It follows from the singular-value gain
$s/(s^2+\rho_i+\epsilon)\le1/(2\sqrt{\rho_i+\epsilon})$.
At an unseen speed, the corresponding prediction change is additionally
bounded by $\|\phi(v)\|_2\|\Delta_i\|_F$. Thus the manuscript links the
algebra to the observed matched result rather than asserting a universal
small-error guarantee. The existing frozen audit checks the correction and
nullspace identities for all 300 fits; the maximum nullspace discrepancy
is 1.41e-9 in coefficient units.

## What the distinct-prior test answers

The explicit evaluation question is whether sharing remains useful when
local extrapolation uses a different prior. Smooth Local penalizes function
curvature, not deviations from the recipient's L3 coefficient map.
Its unpenalized function space is $\operatorname{span}\{1,z\}$, whereas the
restricted span is $\operatorname{span}\{1,v^{-1},v^{-2}\}$. The priors
therefore differ as functions across the interval, even though a two-speed
recipient can fit its targets exactly under either low-dimensional span.

The existing seven-candidate, development-only one-standard-error selection
chooses alpha=1e6, approaching the affine speed-map limit. The recipient
still uses its original two or three speed targets (36 transitions per
speed); it receives no group map or family labels. Development reference
Jacobians are disclosed as an offline tuning resource. The selected strength
was saved before primary fitting. Regularization supplies a prior in
unobserved directions without restoring scheduling-data rank.

The frozen matched test already gives full-envelope errors of 7.489% for
Smooth Local and 3.249% for grouped fitting. The mean paired grouped gain is
56.20% (95% fleet-bootstrap CI 49.69--61.81%), with ten fleet wins. Smooth
Local has lower observed-speed error. Neither other local priors nor Smooth
Local's downstream LQI performance have been established by this test.

The manuscript now includes the normal and nullspace equations, the exact
equality conditions, the near-interpolation mechanism, the explicit test
question, and the distinct unpenalized function spaces. It retains the
existing table, coverage plot, paired intervals, and limitations. All frozen
experiment sources, fits, strengths, inputs, and numerical results remain
unchanged.

## Verification

From the repository root:

```bat
python paper/check_consolidation.py
python paper/check_review_integration.py
python paper/check_local_baseline_integration.py
```

These commands verify the historical and matched evidence without fitting
or simulations. The previous revision note records the completed experiment
and reproduction commands. This explanation pass needs no new simulation
run.

All three evidence checks passed, together with the 11 existing local and
review-validation tests. Six numerical sanity cases checked interpolating
prior equality and the perturbation bound at two- and three-speed supports
with three penalty strengths. All seven PDF pages were visually inspected;
the final build has no overfull boxes or unresolved-reference warnings.
The Smooth Local introduction and objective are kept together across the
column break. Minor IEEEtran flush-bottom underfull spacing remains.
