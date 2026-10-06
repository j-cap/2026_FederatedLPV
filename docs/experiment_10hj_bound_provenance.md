# What the coefficient bounds mean

## Provenance

The uniform multipliers 0.35 and 3.0 first appear in Experiment 10H-G
(commit 29141cc). Section 49 describes them as fixed multipliers of the public
nominal design. There is no accompanying physical-range derivation or bound
sensitivity study. They were retained in 10H-H and 10H-I for controlled
comparisons. They are numerical guardrails, not calibrated vehicle priors.

Eight-coordinate coupling establishes realizability in the adopted positive
bicycle structure. It does not establish that the dimensions, normalized tire
stiffnesses or time constants describe the intended vehicle population.

## Implied intervals

For a uniform nine-coefficient box with multipliers a and b:

| Physical quantity, relative to public nominal | Implied interval |
| --- | --- |
| Front/rear axle distance | [a/b, b/a] |
| Wheelbase | [a/b, b/a] |
| Normalized stiffness C_f/m or C_r/m | [a/b, b/a] |
| Inertial ratio m/I_z | [a²/b, b²/a] |
| Front/rear relaxation length | [1/b, 1/a] |
| Actuator time constant | [1/b, 1/a] |

These are mathematical marginal ranges under exact coupling. They are not
jointly independent physical boxes. With the original [0.35, 3] box, the
public nominal geometry gives front distance 0.146–10.714 m, rear distance
0.181–13.286 m and wheelbase 0.327–24 m. The geometry ratios alone show why the
coefficient multipliers cannot be called a coherent passenger-vehicle envelope.
Widening that box is only an optimization/misspecification diagnostic.

The rear gain b_r=C_r/(m sigma_r) mixes normalized stiffness with tire
relaxation. Reaching its floor does not directly show that stiffness reaches
an independently justified minimum. Inspect C_r/m=b_r/d_r and sigma_r=1/d_r
as well as b_r. No absolute mass, inertia or stiffness is recoverable without
an additional independent scale measurement or prior.

## How to set actual engineering constraints

Use the physically interpretable log coordinates
(g, l_f, l_r, C_f/m, C_r/m, sigma_f, sigma_r, tau_delta).
Their transformation from the fitted eight log coordinates is linear:
log(C_f/m)=log(b_f)-log(d_f), log(C_r/m)=log(b_r)-log(d_r),
log(sigma_j)=-log(d_j), log(tau_delta)=-log(a_delta).
Documented intervals for these quantities therefore become linear inequalities
in the existing fit without changing its exact coupling. A wheelbase interval
is a joint geometry constraint and should be represented explicitly.

A defensible numerical specification needs:

| Quantity | Independent information needed |
| --- | --- |
| Geometry | Intended vehicle dimensions and a loading-dependent CG envelope |
| m/I_z | A defensible yaw radius-of-gyration envelope, I_z/m, rather than assumed exact mass |
| Normalized stiffness | Axle tire-force slope and loading envelope, with axle/per-tire convention stated |
| Relaxation lengths | Tire transient evidence over the speed/load range of interest |
| Steering time constant | Measured command-to-applied-steering response, including actuator delays |

Public model documentation supports these definitions, but published defaults
for an example model are not validated limits for this fleet. No new numerical
engineering bounds are adopted in 10H-J. The study neither reads simulator
truths nor reverse-engineers an admissible envelope from fitted outer scores.

## Primary references checked on 2026-10-06

- [MathWorks, Vehicle Body 3DOF](https://www.mathworks.com/help/vdynblks/ref/vehiclebody3dof.html):
  physical geometry/inertia, axle stiffness, load effects and relaxation-length
  definitions. Its defaults are example values, not population bounds.
- [MathWorks, What Is Residual Analysis?](https://www.mathworks.com/help/ident/ug/what-is-residual-analysis.html):
  validation through residual autocorrelation and correlation with past inputs.
- [MathWorks, Simulate and Predict Identified Model Output](https://www.mathworks.com/help/ident/ug/definition-simulation-and-prediction.html):
  distinction between measured-output-assisted prediction and simulation;
  longer-horizon validation for dynamic behavior.

The interval algebra above is derived from this repository's parameterization,
not imported from the references.
