# 10H-N reconstruction: multihorizon refinement of frozen libraries

This fresh reconstruction follows reconstructed M, not the unavailable original
N. Commit/push the protocol before fitting and freeze/push all learned models
before P. Use only development seeds 601–605, panels 0.01 and 1, and M's two
separate ten-donor libraries. Keep the inherited bounds (0.1–10 times public
nominal effective coefficients), Q/R and eight coupled coordinates unchanged.
These bounds remain heuristic, not calibrated uncertainty intervals.

Refine each library's K1 and two components using the arithmetic mean of
sensor-R-normalized output MSE at horizons 1,5,20,50, common causal origins
20 through T-50 inclusive. Differentiate through the causal KF and each future
open-loop forecast. No future measured output enters predicted states. K1 uses
all donor records; each component uses its selected M restart's fixed soft
training memberships as record weights. Fixing those memberships is an explicit
reconstruction choice, not a claim about the missing N implementation.

Start at M's frozen coordinates, use joint SLSQP with the same coupled linear
coefficient bounds, max 200 iterations and ftol 1e-10. Divide each objective by
its initial loss for numerical scaling. Require success, normalized KKT <=1e-4,
constraint violation <=1e-8 and no training-loss increase. Otherwise retain
that original plant and record failure; do not silently declare a converged fit.

Select between the actual original/refined bundles and margins 0,0.1,0.25 on
the ten donor-validation clients only. Minimize mean log h=5,20,50 forecast-error
ratio against the original M K1 (a common fixed comparator). Ties within 1e-12
prefer original bundle, larger margin. Save the choice before outer scoring.
Recipients use only the 100-sample first-record prefix for plant choice, never
truth or labels. Compare paired outer errors to M, with two deployments averaged
per recipient first. No tuning to reproduce old metrics; confirmation stays sealed.
